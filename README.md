# Hybrid RAG Pipeline

A production-style Retrieval-Augmented Generation system over real
technical documentation (FastAPI's docs): hybrid dense + sparse retrieval,
configurable chunking strategies, and citation-verified generation.

**Status:** Day 2 — chunking strategies 1 & 2 built (fixed-size, structure-aware)

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
  ingest.py       Fetches real doc pages, preserves structure as
                    lightweight Markdown (headings, code blocks, lists)
  chunking.py     Two chunking strategies: fixed-size w/ overlap,
                    structure-aware (heading breadcrumbs + recursive
                    fallback for oversized sections)
docs_corpus/
  raw/             Scraped, structure-preserved text (gitignored)
  processed/       Chunked outputs per strategy (gitignored)
docs/decisions/     Architecture Decision Records
```

## Key design decisions

- **[ADR 001](docs/decisions/001-corpus-not-committed.md)** — the scraped
  corpus is gitignored, not committed; regenerate locally via ingestion.
- **[ADR 002](docs/decisions/002-structure-aware-chunking.md)** —
  structure-aware chunking splits along heading boundaries with a
  breadcrumb trail, recursively falling back to fixed-size splitting for
  oversized sections without ever losing section context.

## Session log

**Day 1:** Built `ingest.py` — fetches 15 real FastAPI doc pages, strips
nav/script/footer chrome. Repo initialized, ADR 001 written (corpus not
committed, copyright).

**Day 2:** Upgraded `ingest.py` to preserve heading/code/list structure as
lightweight Markdown (was flat plain text before — needed for structure-
aware chunking to know where sections are). Built `chunking.py`: strategy
1 (fixed-size w/ overlap, validated with synthetic overlap-boundary
tests) and strategy 2 (structure-aware with heading breadcrumbs and
recursive fallback for oversized sections, validated against a synthetic
document with an artificially large section). Real corpus run: 432 fixed
chunks (avg 782 chars) vs [structured count TBD from your run]. ADR 002
written.

**Next (Day 3):** Strategy 3 — semantic chunking (embedding-based
boundary detection). Requires building the embeddings step first (Phase
1 step 3 in the roadmap), since semantic chunking uses embedding
similarity to detect topic shifts — so embeddings arrive earlier than
originally planned, pulled forward to support this.

## Setup

```bash
git clone https://github.com/CODEWITHNDAHIRO/hybrid-rag-pipeline.git
cd hybrid-rag-pipeline
pip install -r requirements.txt
python src/ingest.py   # fetches the real corpus, takes ~10-15 seconds
```

## Roadmap

- [x] Phase 1, step 1 — Document ingestion (real pages, cleaned, metadata)
- [x] Phase 1, step 2 — Configurable chunking strategies
  - [x] Strategy 1: fixed-size with overlap
  - [x] Strategy 2: structure-aware (heading breadcrumbs, recursive fallback)
  - [ ] Strategy 3: semantic chunking (embedding-based boundaries)
- [ ] Phase 1, step 3 — Embeddings + dense vector store
- [ ] Phase 1, step 4 — BM25 sparse index + deduplication
- [ ] Phase 2 — Hybrid retrieval (dense + sparse fusion, reranking)
- [ ] Phase 3 — Grounded generation with citation verification
- [ ] Phase 4 — Evaluation framework (golden Q&A dataset, faithfulness scoring)
- [ ] Phase 5 — API + dashboard
- [ ] Phase 6 — Portfolio polish

## Why this project

Second in a series of AI engineering portfolio projects, following a
model regression detection / eval CI system. This one focuses on
retrieval quality and grounding — the core RAG competency most demanded
in current AI engineering job descriptions.