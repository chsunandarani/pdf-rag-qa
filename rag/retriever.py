"""Two lexical retrievers: BM25 (written from scratch) and TF-IDF cosine (scikit-learn)."""
import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its
of on or should the their there this to was what when where which who will with within you your
my me we our much many any""".split())


def tokenize(text: str) -> list[str]:
    # keep numbers like 5,000 together as 5000
    text = re.sub(r"(\d),(\d)", r"\1\2", text.lower())
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = self.lengths.mean()
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.chunks))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len)
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
            out += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in top]


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(c.text for c in chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        s = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in top]


RETRIEVERS = {"bm25": BM25Retriever, "tfidf": TfidfRetriever}
