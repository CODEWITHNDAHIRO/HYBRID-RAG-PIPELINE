# Hybrid RAG Pipeline

A production-style Retrieval-Augmented Generation system over real
technical documentation (FastAPI's docs): hybrid dense + sparse retrieval,
configurable chunking strategies, and citation-verified generation.

**Status:** Day 5 — dense (ChromaDB) + sparse (BM25) retrieval built and
validated against real data

## The problem

Most RAG portfolio projects are a single PDF and a LangChain quickstart.
This one is built around the production concerns that actually separate a
RAG engineer from someone who followed a tutorial: hybrid retrieval
(dense + sparse, not just embeddings), explicit chunking strategy
comparisons, and citation verification against hallucination.

## Corpus

15 real pages from FastAPI's official documentation (tutorial + advanced
topics: routing, validation, dependencies, security, databases, testing,
error handling). Chosen to span multiple distinct topics, so retrieval
has to find the *right* page among several plausible ones.

**Note:** the scraped corpus itself is not committed to this repo (see
[ADR 001](docs/decisions/001-corpus-not-committed.md)) — it's copyrighted
third-party documentation. Run the ingestion script to fetch it locally.

## Architecture (so far)

```
src/
  ingest.py            Fetches real doc pages, preserves structure as
                          lightweight Markdown (headings, code blocks, lists)
  chunking.py          Strategy 1 (fixed-size w/ overlap) and strategy 2
                          (structure-aware, heading breadcrumbs, recursive
                          fallback for oversized sections)
  embeddings.py         Local embedding model (sentence-transformers,
                          all-MiniLM-L6-v2) -- no API key required
  semantic_chunking.py    Strategy 3: sentence splitting + cosine-similarity
                          boundary detection
  tune_threshold.py        Diagnostic tool: measures the real similarity
                          distribution across the corpus to pick a data-
                          driven threshold instead of an arbitrary one
  vector_store.py           ChromaDB dense retrieval, one collection per
                          chunking strategy
  bm25_index.py               BM25 sparse (keyword) retrieval via rank_bm25
docs_corpus/
  raw/                   Scraped, structure-preserved text (gitignored)
  processed/               Chunk JSON per strategy, ChromaDB index, BM25
                          pickles (all gitignored -- derived from
                          copyrighted source text)
docs/decisions/            Architecture Decision Records
```

## Key design decisions

- **[ADR 001](docs/decisions/001-corpus-not-committed.md)** — the scraped
  corpus (and everything derived from it: chunks, embeddings, indexes) is
  gitignored, not committed; regenerate locally via the pipeline scripts.
- **[ADR 002](docs/decisions/002-structure-aware-chunking.md)** —
  structure-aware chunking splits along heading boundaries with a
  breadcrumb trail, recursively falling back to fixed-size splitting for
  oversized sections without ever losing section context.
- **[ADR 003](docs/decisions/003-local-embeddings.md)** — embeddings run
  locally via sentence-transformers instead of a hosted API: free,
  unlimited experimentation, no billing setup, honest quality tradeoff
  versus a frontier hosted model.
- **[ADR 004](docs/decisions/004-empirical-threshold-tuning.md)** — the
  semantic chunking similarity threshold was empirically measured from
  the real corpus distribution (15th percentile), not guessed. This fixed
  a real problem: the initial default (0.5) was flagging 83.7% of
  sentence pairs as boundaries, producing 671 near-useless tiny chunks;
  the tuned threshold (0.042) produced 192 reasonably-sized ones.

## Known issues (tracked, not yet fixed)

Text extraction has minor artifacts surfaced during Day 5 retrieval
testing: headings occasionally bleed into adjacent paragraph text without
a line break (which breaks `section_heading` detection in those specific
spots), missing spaces appear around some numbers, and MkDocs' decorative
"¶" permalink anchor symbols get captured as if they were real content.
Retrieval still returns correct, on-topic results despite this noise, but
it should be cleaned up before the evaluation phase, where exact text
quality matters more directly.

## Session log

**Day 1:** Built `ingest.py` — fetches 15 real FastAPI doc pages, strips
nav/script/footer chrome. Repo initialized, ADR 001 written (corpus not
committed, copyright).

**Day 2:** Upgraded `ingest.py` to preserve heading/code/list structure as
lightweight Markdown (was flat plain text before). Built `chunking.py`:
strategy 1 (fixed-size w/ overlap) and strategy 2 (structure-aware,
heading breadcrumbs, recursive fallback), both validated with synthetic
tests before running on real data. Real corpus: 432 fixed chunks (avg 782
chars). ADR 002 written.

**Day 3:** Built local embedding generation (`embeddings.py`) — no API
key needed, see ADR 003. Built strategy 3, semantic chunking
(`semantic_chunking.py`): sentence splitting with code-fence-aware
parsing (caught and fixed a real bug — original fence detection only
skipped marker lines, not the code between them), cosine-similarity
boundary detection validated against synthetic two-topic vectors. Real
corpus result: 671 semantic chunks, avg 159 chars — flagged as
suspiciously small for investigation.

**Day 4:** Diagnosed the Day 3 finding. Built `tune_threshold.py`:
measured the real distribution of consecutive-sentence similarities
(1,495 pairs). Finding: mean similarity was only 0.264, so the 0.5
threshold was flagging 83.7% of pairs as boundaries — noise, not signal.
Retuned to the 15th percentile of the real distribution (0.042). Result:
671 → 192 chunks, 159 → 558 chars avg, now comparable in scale to the
other strategies. ADR 004 written.

**Day 5:** Built the two halves of hybrid retrieval. `vector_store.py`
(ChromaDB dense retrieval) and `bm25_index.py` (BM25 sparse retrieval),
both validated with synthetic data before running on real chunks. Real
test queries against the actual corpus returned correct, on-topic results
for both (dense: query-params page for a query-parameter question;
sparse: handling-errors page for an HTTPException question). Surfaced the
text-extraction artifacts noted above during this testing.

**Day 6:** Built hybrid_search.py — Reciprocal Rank Fusion combining dense
and sparse results into one ranked list. RRF math validated with a
synthetic test before running on real data. Real query ("How do I raise
an HTTPException with a custom status code?") returned 5 results, all
correctly from handling-errors, all found by *both* dense and sparse
independently — strong agreement case. Hit and resolved a real workflow
issue: vector_store.py and bm25_index.py must each be run with an explicit
strategy argument per new chunking strategy; running without one silently
defaults to "fixed" rather than erroring, which caused a confusing
NotFoundError until traced back to the missing explicit argument.

**Day 7:** Fixed the text-extraction artifacts flagged on Day 5-6. Root
cause: MkDocs Material injects a hidden permalink anchor (renders as ¶)
inside every heading, which get_text() was collecting as real content;
and inline elements (<code>, <a>) sitting flush against surrounding text
were concatenating with zero space ("of404", "requestshttp://"). Fixed by
removing headerlink anchors before extraction and using an explicit space
separator + cleanup regex. Validated against synthetic HTML replicating
the exact real bug patterns before re-running the full pipeline.

**Day 8:** Built reranker.py — cross-encoder reranking pass on top of
RRF-fused results. Unlike our bi-encoder embedding model (query and chunk
encoded separately, then compared), a cross-encoder reads query+chunk
together for a more accurate but slower relevance score, so it only runs
on the small candidate pool from fusion (20), not the full corpus. Logic
validated with synthetic scores showing correct promotion of a
lower-RRF-rank-but-more-relevant candidate. Used a deliberately paraphrased
test query (no exact-vocabulary overlap with the docs) to give dense and
sparse retrieval a real chance to disagree, unlike Day 6's query.

**Day 9 (Phase 3 start):** Built generate.py — grounded generation with
inline citations. Retrieves + reranks via the full Phase 1-2 pipeline,
builds numbered source blocks, and prompts Claude to answer using ONLY
that context with [n] citations, refusing to guess if context is
insufficient. Citation parsing (regex extraction of [n] markers,
including multi-citation sentences and deduplication) validated with
synthetic answer text before running on a real query.

**Day 10 (Phase 3 complete):** Built citation_verification.py — LLM-as-
judge verification checking whether each cited claim is genuinely
supported by its source, not just that a citation number exists. Claim-
splitting logic (sentence-level, citation extraction, multi-citation
grouping, uncited-sentence skipping) and the 1-indexed-citation-to-
0-indexed-source mapping both validated with synthetic data before the
first live end-to-end run.

```bash
git clone https://github.com/CODEWITHNDAHIRO/hybrid-rag-pipeline.git
cd hybrid-rag-pipeline
pip install -r requirements.txt

python src/ingest.py                    # fetches the real corpus (~10-15s)
python src/chunking.py                  # builds strategies 1 & 2
python src/semantic_chunking.py         # builds strategy 3
python src/vector_store.py              # builds + test-queries the dense index
python src/bm25_index.py                # builds + test-queries the sparse index
```

## Roadmap

- [x] Phase 1, step 1 — Document ingestion (real pages, structure-preserved, metadata)
- [x] Phase 1, step 2 — Chunking strategies 1 & 2 (fixed-size, structure-aware)
- [x] Phase 1, step 2b — Semantic chunking (strategy 3) + local embedding model
- [x] Phase 2, step 1 — Dense retrieval (ChromaDB) + sparse retrieval (BM25)
- [x] Phase 2, step 2 — Fusion layer (Reciprocal Rank Fusion) + reranking
- [x] Phase 3 — Grounded generation with citation verification
- [ ] Phase 4 — Evaluation framework (golden Q&A dataset, faithfulness scoring,
      chunking strategy comparison)
- [ ] Phase 5 — API + dashboard
- [ ] Phase 6 — Portfolio polish

## Why this project

Second in a series of AI engineering portfolio projects, following a
model regression detection / eval CI system. This one focuses on
retrieval quality and grounding — the core RAG competency most demanded
in current AI engineering job descriptions.
