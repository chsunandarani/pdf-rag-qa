# PDF Question Answering with RAG

A small retrieval-augmented generation (RAG) app that answers questions about a set of PDFs and cites the file and page it got the answer from. I also built an evaluation script to check how often it gets the answer right and how long each answer takes, because without numbers it's hard to tell whether a change actually helped.

```
python app.py -q "How quickly must a security incident be reported?"
```
prints the answer with its citations, the files and pages it was drawn from, and how long retrieval and generation took.

## How it works

1. **Load and chunk** (`rag/ingest.py`): text is pulled from each PDF page with `pypdf` and split into 80-word chunks with a 20-word overlap. Chunks never cross a page, so every answer can point to an exact page.
2. **Retrieve** (`rag/retriever.py`): the question is matched against the chunks with BM25, which I wrote from scratch with NumPy. A TF-IDF retriever (scikit-learn) is included for comparison. The top 3 chunks are kept.
3. **Generate** (`rag/pipeline.py`, `rag/llm.py`): the chunks go into a prompt that tells the model to answer only from the passages, cite them as [1], [2], and say "I don't know based on the provided documents" when the answer isn't there. The client works with any OpenAI-compatible API (Groq, OpenAI, OpenRouter, or a local Ollama model) and uses only the Python standard library.
4. **Time each stage**: retrieval, generation and total time are recorded for every question.

There is also an `extractive` mode that skips the LLM and returns the best-matching sentence. It needs no API key and serves as a baseline for the LLM to beat.

## Data and test set

`data/pdfs/` holds three sample PDFs for a made-up company (an employee handbook, an IT security policy and a product user guide, 6 pages in total). I used made-up documents so that every answer has one clear, checkable source. You can drop your own PDFs into the folder and the app will index those instead.

`eval/questions.json` has 35 questions:
- 30 that can be answered from the PDFs, each with the file it comes from and the key facts the answer must contain (for example `18` for annual leave days)
- 5 that cannot be answered from the PDFs (like "Who is the CEO?"), where the right response is to say it doesn't know

Many questions are worded differently from the documents on purpose ("How much money do new joiners get to set up a home office?" vs. "one-time home office allowance") so that retrieval isn't just exact word matching.

**How answers are scored:** an answer to an answerable question counts as correct if it contains all the key facts. For the 5 unanswerable questions, the answer is correct only if the system says it doesn't know. Retrieval counts as a hit if a retrieved chunk from the right file contains the answer.

## Results

### Retrieval

Hit@k = share of the 30 answerable questions where a chunk containing the answer was in the top k results. MRR@5 = mean reciprocal rank of the first correct chunk.

<!-- RETRIEVAL:start -->
| Retriever | Chunk size (words) | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Time / query |
|---|---|---|---|---|---|---|---|
| BM25 | 40 | 42 | 77% | 93% | 100% | 0.86 | 0.04 ms |
| TF-IDF | 40 | 42 | 73% | 90% | 100% | 0.84 | 0.45 ms |
| BM25 | 80 | 22 | 97% | 100% | 100% | 0.98 | 0.03 ms |
| TF-IDF | 80 | 22 | 90% | 100% | 100% | 0.95 | 0.41 ms |
| BM25 | 150 | 12 | 93% | 100% | 100% | 0.97 | 0.03 ms |
| TF-IDF | 150 | 12 | 93% | 100% | 100% | 0.97 | 0.40 ms |
<!-- RETRIEVAL:end -->

What I took from this:
- **Chunk size mattered more than the choice of retriever.** With 40-word chunks the answer was often cut off from the words that matched the question. 80 words gave the best Hit@1, and 150 words didn't improve on it, so I kept 80.
- **BM25 did a little better than TF-IDF at Hit@1** and was faster per query on this corpus.
- **At k=3 the right chunk was retrieved for every answerable question**, so the remaining mistakes in the end-to-end results come from the answering step, not from search.

### End-to-end accuracy and response time

<!-- E2E:start -->
| Setup | Accuracy (all 35) | Answerable (30) | Correct "don't know" (5) | Latency p50 | Latency p95 |
|---|---|---|---|---|---|
| LLM: openai/gpt-oss-20b (groq) + BM25 | 100% | 100% | 100% | 670 ms | 980 ms |
| Extractive baseline + BM25 | 71% | 77% | 40% | 0.2 ms | 0.2 ms |
| Extractive baseline + TF-IDF | 74% | 83% | 20% | 0.6 ms | 0.8 ms |
<!-- E2E:end -->

The extractive baseline answers in well under a millisecond, but it often returns a neighbouring sentence instead of the one with the answer, and it is bad at saying "I don't know". It only refuses when the BM25 score is low, and a question like "What is the office Wi-Fi password?" still matches words in the security policy.

### Response time breakdown

<!-- LATENCY:start -->
Per-stage response time for the LLM run (openai/gpt-oss-20b, 2026-10-30):

| Stage | Mean | p50 | p95 |
|---|---|---|---|
| Retrieval | 0.6 ms | 0.6 ms | 1.0 ms |
| Generation | 750.3 ms | 669.1 ms | 979.4 ms |
| Total | 750.9 ms | 669.6 ms | 980.0 ms |
<!-- LATENCY:end -->

Retrieval takes well under a millisecond on this corpus, so almost all of the response time is the LLM call.

## Limitations and next steps

- BM25 is a keyword method, so it can miss answers phrased with completely different words. Adding embedding-based search (for example `sentence-transformers` with a hybrid BM25 + vector score) is the obvious next step.
- Scoring by key facts is simple and repeatable, but it can mark a correct answer wrong if the model words a number differently ("eighteen" vs "18"). I listed alternatives for the trickiest ones, and the full answers are saved in `results/*_answers.csv` for a manual check.
- 35 questions over 6 pages is a small test. The numbers show how the setups compare to each other, not how well this would do on a large document set.
- Scanned PDFs (images with no text layer) would need OCR first.

## Running it

```bash
pip install -r requirements.txt

# no API key needed
python app.py --mode extractive -q "How many sick days do I get?"
python eval.py --retrieval-only

# with an LLM (Groq has a free tier: https://console.groq.com)
export LLM_PROVIDER=groq          # or openai, openrouter, ollama
export LLM_API_KEY=your_key_here
python app.py                     # interactive
python eval.py --mode llm --update-readme
```

Set `LLM_MODEL` to try a different model. `--update-readme` writes the latest result tables into this file. On Windows, use `set` instead of `export`.

## Project layout

```
app.py               command-line app
eval.py              evaluation: retrieval metrics, accuracy, latency
rag/ingest.py        PDF loading and chunking
rag/retriever.py     BM25 and TF-IDF retrievers
rag/llm.py           OpenAI-compatible API client
rag/pipeline.py      retrieve -> prompt -> answer, with timings
eval/questions.json  35 test questions with expected facts
data/pdfs/           sample documents
results/             saved metrics (JSON) and every answer (CSV)
```
