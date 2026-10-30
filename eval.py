"""Measure retrieval quality, answer accuracy and response time on eval/questions.json.

    python eval.py --retrieval-only            # retriever / chunk-size comparison (no API key)
    python eval.py --mode extractive           # end-to-end, no-LLM baseline
    python eval.py --mode llm                  # end-to-end with the LLM (needs LLM_API_KEY)
    add --update-readme to write the result tables into README.md
"""
import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path

from rag import RAGPipeline
from rag.ingest import chunk_pages, load_pdfs
from rag.retriever import RETRIEVERS

ROOT = Path(__file__).parent
QUESTIONS = json.loads((ROOT / "eval/questions.json").read_text())
RESULTS = ROOT / "results"
LABEL = {"bm25": "BM25", "tfidf": "TF-IDF"}


# ---------- scoring helpers ----------

def norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"(\d),(\d)", r"\1\2", text)   # 15,000 -> 15000
    text = re.sub(r"[-\u2010-\u2015]", " ", text)  # 2-hour / 2‑hour (any dash) -> 2 hour
    return re.sub(r"\s+", " ", text)


def has_all(text: str, keywords: list[str]) -> bool:
    """Every keyword must appear; a keyword may list alternatives separated by '|'."""
    t = norm(text)
    return all(any(norm(alt) in t for alt in kw.split("|")) for kw in keywords)


def is_refusal(answer: str) -> bool:
    return bool(re.search(r"(don't|do not) know", norm(answer)))


def chunk_has_answer(chunk, q) -> bool:
    return chunk.source == q["source"] and has_all(chunk.text, q["keywords"])


def pct(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


# ---------- retrieval-only comparison ----------

def retrieval_metrics(retriever, answerable, max_k=5):
    ranks = []
    for q in answerable:
        hits = retriever.search(q["question"], max_k)
        rank = next((i for i, (c, _) in enumerate(hits, 1) if chunk_has_answer(c, q)), None)
        ranks.append(rank)
    n = len(ranks)
    return {
        "hit@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr@5": sum(1 / r for r in ranks if r) / n,
    }


def run_retrieval(pdf_dir):
    answerable = [q for q in QUESTIONS if q["source"]]
    pages = load_pdfs(pdf_dir)
    rows = []
    for size, overlap in [(40, 10), (80, 20), (150, 30)]:
        chunks = chunk_pages(pages, size, overlap)
        for name, cls in RETRIEVERS.items():
            r = cls(chunks)
            m = retrieval_metrics(r, answerable)
            t0 = time.perf_counter()
            for q in QUESTIONS:
                r.search(q["question"], 3)
            m["ms_per_query"] = (time.perf_counter() - t0) * 1000 / len(QUESTIONS)
            rows.append({"retriever": name, "chunk_words": size, "overlap": overlap,
                         "n_chunks": len(chunks), **m})
            print(f"{name:6s} chunk={size:3d}  chunks={len(chunks):3d}  "
                  f"hit@1={m['hit@1']:.0%} hit@3={m['hit@3']:.0%} hit@5={m['hit@5']:.0%} "
                  f"MRR={m['mrr@5']:.2f}  {m['ms_per_query']:.2f} ms/query")
    out = {"n_questions": len(answerable), "rows": rows}
    (RESULTS / "retrieval.json").write_text(json.dumps(out, indent=2))
    return out


# ---------- end-to-end ----------

def run_end_to_end(args):
    rag = RAGPipeline(args.pdfs, args.retriever, args.top_k, args.chunk_words,
                      args.overlap, mode=args.mode)
    per_q = []
    for i, q in enumerate(QUESTIONS, 1):
        a = rag.ask(q["question"])
        if q["source"]:
            correct = (not is_refusal(a.text)) and has_all(a.text, q["keywords"])
            retrieved = any(chunk_has_answer(c, q) for c in a.sources)
        else:
            correct, retrieved = is_refusal(a.text), None
        per_q.append({"id": q["id"], "question": q["question"], "answer": a.text,
                      "correct": correct, "answer_in_context": retrieved,
                      **{f"{k}_ms": round(v, 2) for k, v in a.timings_ms.items()}})
        print(f"[{i:2d}/{len(QUESTIONS)}] {'OK ' if correct else 'BAD'} {q['id']:4s} "
              f"{a.timings_ms['total']:7.1f} ms  {a.text[:90]}")
        if args.mode == "llm" and args.delay:
            time.sleep(args.delay)   # stay under free-tier rate limits (not counted in timings)

    ans = [r for r in per_q if not r["id"].startswith("n")]
    unans = [r for r in per_q if r["id"].startswith("n")]
    lat = {s: [r[f"{s}_ms"] for r in per_q] for s in ("retrieval", "generation", "total")}
    summary = {
        "mode": args.mode, "retriever": args.retriever, "top_k": args.top_k,
        "chunk_words": args.chunk_words, "overlap": args.overlap,
        "model": getattr(getattr(rag, "llm", None), "model", None),
        "provider": getattr(getattr(rag, "llm", None), "provider", None),
        "n_questions": len(per_q),
        "accuracy_overall": sum(r["correct"] for r in per_q) / len(per_q),
        "accuracy_answerable": sum(r["correct"] for r in ans) / len(ans),
        "correct_refusals": sum(r["correct"] for r in unans) / len(unans),
        "answer_in_context": sum(r["answer_in_context"] for r in ans) / len(ans),
        "latency_ms": {s: {"mean": statistics.mean(v), "p50": pct(v, 50), "p95": pct(v, 95)}
                       for s, v in lat.items()},
        "index_build_ms": rag.index_ms,
        "run_date": time.strftime("%Y-%m-%d"),
    }
    name = f"{args.mode}_{args.retriever}"
    (RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=2))
    with open(RESULTS / f"{name}_answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q[0]))
        w.writeheader()
        w.writerows(per_q)
    s = summary
    print(f"\nAccuracy {s['accuracy_overall']:.0%} overall | {s['accuracy_answerable']:.0%} answerable | "
          f"{s['correct_refusals']:.0%} correct refusals | answer in context {s['answer_in_context']:.0%}")
    print(f"Latency  p50 {s['latency_ms']['total']['p50']:.1f} ms | p95 {s['latency_ms']['total']['p95']:.1f} ms")
    return summary


# ---------- README tables ----------

def retrieval_table(data):
    lines = ["| Retriever | Chunk size (words) | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Time / query |",
             "|---|---|---|---|---|---|---|---|"]
    for r in data["rows"]:
        lines.append(f"| {LABEL[r['retriever']]} | {r['chunk_words']} "
                     f"| {r['n_chunks']} | {r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@5']:.0%} "
                     f"| {r['mrr@5']:.2f} | {r['ms_per_query']:.2f} ms |")
    return "\n".join(lines)


def e2e_table(runs):
    lines = ["| Setup | Accuracy (all 35) | Answerable (30) | Correct \"don't know\" (5) | Latency p50 | Latency p95 |",
             "|---|---|---|---|---|---|"]
    for s in runs:
        label = (f"LLM: {s['model']} ({s['provider']}) + {LABEL[s['retriever']]}"
                 if s["mode"] == "llm" else f"Extractive baseline + {LABEL[s['retriever']]}")
        L = s["latency_ms"]["total"]
        fmt = (lambda v: f"{v:.0f} ms") if s["mode"] == "llm" else (lambda v: f"{v:.1f} ms")
        lines.append(f"| {label} | {s['accuracy_overall']:.0%} | {s['accuracy_answerable']:.0%} "
                     f"| {s['correct_refusals']:.0%} | {fmt(L['p50'])} | {fmt(L['p95'])} |")
    return "\n".join(lines)


def latency_table(s):
    L = s["latency_ms"]
    rows = ["| Stage | Mean | p50 | p95 |", "|---|---|---|---|"]
    for stage in ("retrieval", "generation", "total"):
        rows.append(f"| {stage.title()} | {L[stage]['mean']:.1f} ms | {L[stage]['p50']:.1f} ms "
                    f"| {L[stage]['p95']:.1f} ms |")
    return "\n".join(rows)


def replace_block(text, tag, body):
    pattern = re.compile(rf"(<!-- {tag}:start -->).*?(<!-- {tag}:end -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"README is missing the {tag} markers")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text)


def update_readme():
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if (RESULTS / "retrieval.json").exists():
        text = replace_block(text, "RETRIEVAL", retrieval_table(json.loads((RESULTS / "retrieval.json").read_text())))
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*_*.json"))
            if not p.name.startswith("retrieval")]
    runs.sort(key=lambda s: s["mode"] != "llm")
    if runs:
        text = replace_block(text, "E2E", e2e_table(runs))
    llm = [s for s in runs if s["mode"] == "llm"]
    if llm:
        text = replace_block(text, "LATENCY",
                             f"Per-stage response time for the LLM run ({llm[0]['model']}, {llm[0]['run_date']}):\n\n"
                             + latency_table(llm[0]))
    readme.write_text(text, encoding="utf-8")
    print("README.md updated")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdfs", default=str(ROOT / "data/pdfs"))
    p.add_argument("--mode", choices=["llm", "extractive"], default="llm")
    p.add_argument("--retriever", choices=list(RETRIEVERS), default="bm25")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--chunk-words", type=int, default=80)
    p.add_argument("--overlap", type=int, default=20)
    p.add_argument("--delay", type=float, default=1.0, help="seconds between LLM calls")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--update-readme", action="store_true")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.retrieval_only:
        run_retrieval(args.pdfs)
    else:
        run_end_to_end(args)
    if args.update_readme:
        update_readme()


if __name__ == "__main__":
    main()
