# ADR 002: Structure-aware chunking with recursive fixed-size fallback

## Status
Accepted

## Context
Fixed-size chunking (the baseline, see chunk_fixed_size) ignores document
structure entirely -- it can slice a heading away from its content, or
cut a code example in half. For well-structured docs like FastAPI's,
splitting along actual section boundaries should preserve more coherent,
self-contained chunks.

This required a prerequisite change: the original ingestion pipeline
(ingest.py) extracted flat plain text via BeautifulSoup's get_text(),
which discards all heading/structure information. Ingestion was updated
to walk the HTML tree and preserve headings as Markdown ('#'/'##'/'###'),
code blocks as fenced blocks, and list items with '- ' prefixes.

## Decision
Implemented `chunk_structured()`: splits text at heading boundaries using
a regex over the Markdown heading syntax, tracks a heading "breadcrumb"
(e.g. "Query Parameters > Required Parameters") via a level-indexed stack,
and produces one chunk per section *if* the section fits under
max_chunk_size. Oversized sections (e.g. the ~102k-character SQL databases
page) recursively fall back to fixed-size splitting, but every resulting
sub-chunk keeps the section breadcrumb prepended -- so no chunk ever loses
its section context, even after further splitting.

## Consequences
- Requires ingestion to preserve structure, meaning ingest.py and
  chunking.py are now coupled: chunking assumes Markdown-style headings
  exist in the raw text. This is documented here so it isn't a silent
  assumption.
- Chunk count and average size now differ meaningfully between strategies
  -- this is the first concrete data point toward the planned Phase 4
  chunking-strategy comparison against retrieval quality.
- The recursive fallback adds complexity but avoids the failure mode of
  "structure-aware chunking produces one giant unusable chunk for any
  long page" -- validated against a synthetic oversized section in
  testing before running on the real corpus.
