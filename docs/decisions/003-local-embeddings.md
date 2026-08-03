# ADR 003: Local embedding model (sentence-transformers) instead of a hosted API

## Status
Accepted

## Context
Semantic chunking (strategy 3) and, later, dense retrieval both need text
embeddings. The original reference spec for this kind of project assumes
a hosted embeddings API (e.g. OpenAI's text-embedding-3-small). Given
repeated friction earlier in this build series with API billing setup,
and given that embedding cost/latency at scale is a real production
concern regardless, a local model was evaluated as an alternative.

## Decision
Use `sentence-transformers` with the `all-MiniLM-L6-v2` model, running
entirely locally. No API key, no per-call cost, no network dependency at
inference time (only for the one-time model weight download).

## Consequences
- Lower embedding quality than a large hosted model -- an honest
  tradeoff, not free lunch. Worth naming directly if asked: "I chose a
  local model to avoid per-call cost and latency, at the cost of some
  retrieval quality versus a frontier embedding model."
- Enables free, unlimited local experimentation (re-embedding the full
  corpus repeatedly while tuning chunking strategies costs nothing).
- The boundary-detection algorithm (cosine similarity + threshold) was
  validated independently of the model itself, using synthetic vectors
  simulating a clear two-topic shift -- this caught the algorithm logic
  working correctly before ever touching a real embedding.
- A real bug was caught during synthetic testing of the sentence
  splitter: the original code-fence detection only skipped the ```
  marker lines themselves, not the code *between* them, letting code
  content leak into the sentence stream. Fixed by tracking fence state
  (in/out) across lines rather than checking each line in isolation.
