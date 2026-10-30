"""Ask questions about the PDFs in data/pdfs from the command line.

    python app.py                                  # interactive, uses the LLM
    python app.py -q "How many sick days do I get?"
    python app.py --mode extractive                # no API key needed
"""
import argparse

from rag import RAGPipeline


def main():
    p = argparse.ArgumentParser(description="Question answering over your PDFs")
    p.add_argument("-q", "--question", help="ask one question and exit")
    p.add_argument("--pdfs", default="data/pdfs", help="folder of PDFs")
    p.add_argument("--retriever", choices=["bm25", "tfidf"], default="bm25")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--mode", choices=["llm", "extractive"], default="llm")
    p.add_argument("--show-context", action="store_true", help="print the retrieved passages")
    args = p.parse_args()

    rag = RAGPipeline(args.pdfs, args.retriever, args.top_k, mode=args.mode)
    print(f"Indexed {len(rag.chunks)} chunks in {rag.index_ms:.0f} ms ({args.retriever}, {args.mode})")

    def answer(q):
        a = rag.ask(q)
        print(f"\n{a.text}\n")
        print(f"  sources: {a.cited()}")
        print(f"  time: {a.timings_ms['total']:.0f} ms "
              f"(retrieval {a.timings_ms['retrieval']:.1f} ms, generation {a.timings_ms['generation']:.0f} ms)")
        if args.show_context:
            for i, c in enumerate(a.sources, 1):
                print(f"\n  [{i}] {c.source} p.{c.page}: {c.text[:200]}...")

    if args.question:
        answer(args.question)
        return
    print("Type a question (or 'exit').")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            answer(q)


if __name__ == "__main__":
    main()
