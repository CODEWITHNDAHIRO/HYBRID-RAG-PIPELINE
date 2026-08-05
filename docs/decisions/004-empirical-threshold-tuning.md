# ADR 004: Empirically tune the semantic chunking threshold, don't guess it

## Status
Accepted

## Context
Strategy 3 (semantic chunking) initially used an arbitrary similarity
threshold of 0.5, following common convention. Applied to the real
corpus, this produced 671 chunks averaging only 159 characters --
dramatically smaller and more numerous than strategy 1 (432 chunks, ~782
chars) or strategy 2, suggesting the threshold was miscalibrated rather
than reflecting genuine topic density.

## Investigation
Built `tune_threshold.py`: measures the real distribution of
consecutive-sentence cosine similarities across the entire corpus (1,495
sentence pairs), rather than assuming a fixed cutoff is meaningful across
different corpora or embedding models.

Findings: mean similarity across this corpus is only 0.264 (considerably
lower than the 0.5 threshold assumed). At threshold 0.5, 83.7% of all
sentence pairs were being flagged as topic boundaries -- meaning the
"boundary detector" was mostly just noise, not signal.

## Decision
Set the threshold to the 15th percentile of the corpus's actual
similarity distribution (0.042), rather than a fixed guess. This treats
"boundary" as a relative, corpus-specific concept -- only the most
dissimilar ~15% of consecutive-sentence pairs count as real topic shifts.

## Result
Re-running semantic chunking with the tuned threshold: 671 -> 192 chunks,
159 -> 558 chars average -- now in a comparable range to the other two
strategies, making a fair three-way comparison possible in the planned
Phase 4 evaluation.

## Consequences
- The 0.042 threshold is specific to this corpus and this embedding
  model (all-MiniLM-L6-v2). It should not be assumed to transfer to a
  different document set without re-running tune_threshold.py.
- This is a concrete example of a broader principle worth reusing
  elsewhere in this project (and in future ones): don't accept a
  parameter's default value without checking whether it's actually
  appropriate for your specific data. A convention borrowed from a
  tutorial or another corpus is not automatically correct here.
