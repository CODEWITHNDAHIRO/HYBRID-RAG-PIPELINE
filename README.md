**Day 4:** Diagnosed why semantic chunking produced 671 tiny chunks
(159 chars avg). Built `tune_threshold.py`: measured the real distribution
of consecutive-sentence similarities across the corpus (1,495 pairs) instead
of trusting an arbitrary 0.5 default. Finding: mean similarity was only
0.264, so 0.5 was flagging 83.7% of pairs as boundaries — noise, not signal.
Retuned to the 15th percentile of the real distribution (0.042). Result:
671 → 192 chunks, 159 → 558 chars avg, now comparable in scale to the
fixed-size and structure-aware strategies. See ADR 004.

**Next (Day 5):** Build the vector store (ChromaDB) and BM25 sparse index —
the two halves of hybrid retrieval — indexing all three chunk sets so they
can be embedded and compared at retrieval time, not just at chunk-count level.