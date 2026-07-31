# ADR 001: Don't commit the scraped document corpus to the repo

## Status
Accepted

## Context
This project's search corpus is real FastAPI documentation, fetched from
fastapi.tiangolo.com. That content is copyrighted by its authors. Fetching
it for personal, local use (to build and test a RAG pipeline) is
reasonable, similar to saving pages from a browser -- but committing that
scraped text to a public GitHub repository would mean redistributing
someone else's copyrighted documentation publicly, which is a different
and inappropriate thing to do.

## Decision
`docs_corpus/raw/` and `docs_corpus/processed/` are gitignored. Only the
pipeline code that generates them (`src/ingest.py` and later chunking/
embedding scripts) is committed. Anyone cloning the repo -- including
future me -- regenerates the corpus by running the ingestion script
locally.

## Consequences
- The repo is not "run and it just works" out of the box the way a repo
  with bundled sample data would be -- a `python src/ingest.py` step is
  required first. This is documented in the README setup instructions.
- This also has a practical benefit beyond the copyright concern: the
  pipeline is proven reproducible from source rather than depending on a
  stale, bundled data dump that could drift from the live docs over time.
