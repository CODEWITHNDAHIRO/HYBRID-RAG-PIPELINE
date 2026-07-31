# Hybrid RAG Pipeline

A production-style Retrieval-Augmented Generation system over real
technical documentation (FastAPI's docs): hybrid dense + sparse retrieval,
configurable chunking strategies, and citation-verified generation.

**Status:** Day 1 — ingestion pipeline built

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
  ingest.py       Fetches real doc pages, strips nav/chrome, saves clean
                    text + metadata locally
docs_corpus/
  raw/             Scraped, cleaned text (gitignored, regenerate locally)
  processed/       Chunked/embedded versions (later phase, gitignored)
docs/decisions/     Architecture Decision Records
```

## Setup

```bash
git clone https://github.com/CODEWITHNDAHIRO/hybrid-rag-pipeline.git
cd hybrid-rag-pipeline
pip install -r requirements.txt
python src/ingest.py   # fetches the real corpus, takes ~10-15 seconds
```

## Roadmap

- [x] Phase 1, step 1 — Document ingestion (real pages, cleaned, metadata)
- [ ] Phase 1, step 2 — Configurable chunking strategies
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