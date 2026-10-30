"""Retrieve -> build prompt -> generate, timing each stage."""
import re
import time
from dataclasses import dataclass, field

from .ingest import Chunk, chunk_pages, load_pdfs
from .retriever import RETRIEVERS, tokenize

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions using only the numbered context passages you are given.
Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Keep the answer short: one to three sentences.
- After each fact, cite the passage it came from, like [1] or [2].
- If the passages do not contain the answer, reply exactly: "{NO_ANSWER}"
"""


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Chunk]
    timings_ms: dict = field(default_factory=dict)

    def cited(self) -> str:
        refs = sorted({f"{c.source} p.{c.page}" for c in self.sources})
        return "; ".join(refs)


class RAGPipeline:
    def __init__(self, pdf_dir="data/pdfs", retriever="bm25", top_k=3,
                 chunk_words=80, overlap=20, mode="llm", llm=None, min_score=None):
        t0 = time.perf_counter()
        self.chunks = chunk_pages(load_pdfs(pdf_dir), chunk_words, overlap)
        self.retriever = RETRIEVERS[retriever](self.chunks)
        self.index_ms = (time.perf_counter() - t0) * 1000
        self.top_k, self.mode = top_k, mode
        # below this top score the extractive baseline says "I don't know"
        self.min_score = min_score if min_score is not None else (4.0 if retriever == "bm25" else 0.12)
        if mode == "llm":
            from .llm import LLMClient
            self.llm = llm or LLMClient()

    def retrieve(self, question: str):
        return self.retriever.search(question, self.top_k)

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        t1 = time.perf_counter()
        text = self._generate(question, hits)
        t2 = time.perf_counter()
        return Answer(question, text, [c for c, _ in hits], {
            "retrieval": (t1 - t0) * 1000,
            "generation": (t2 - t1) * 1000,
            "total": (t2 - t0) * 1000,
        })

    def _generate(self, question, hits) -> str:
        if self.mode == "extractive":
            return self._extractive(question, hits)
        context = "\n\n".join(f"[{i}] ({c.source}, page {c.page})\n{c.text}"
                              for i, (c, _) in enumerate(hits, start=1))
        prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        return self.llm.chat(SYSTEM_PROMPT, prompt)

    def _extractive(self, question, hits) -> str:
        """No-LLM baseline: return the single sentence that overlaps the question most."""
        if not hits or hits[0][1] < self.min_score:
            return NO_ANSWER
        q = set(tokenize(question))
        best, best_score = None, 0.0
        for rank, (chunk, _) in enumerate(hits, start=1):
            # split on sentence ends followed by a capital letter, so "Rs. 5,000" stays whole
            for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", chunk.text):
                overlap = len(q & set(tokenize(sent)))
                score = overlap - 0.1 * rank  # small tie-break toward higher-ranked chunks
                if score > best_score:
                    best, best_score = f"{sent} [{rank}]", score
        return best or NO_ANSWER
