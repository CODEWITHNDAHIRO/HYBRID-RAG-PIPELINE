"""
Day 4: threshold diagnostics for semantic chunking.

Instead of guessing a similarity threshold, measure the actual
distribution of consecutive-sentence similarities across the real corpus,
then pick a threshold based on percentiles of that real distribution --
e.g. "only the most dissimilar 15% of pairs count as a real topic
boundary" rather than an arbitrary fixed number.
"""
import numpy as np

from chunking import load_metadata, RAW_DIR
from semantic_chunking import split_into_sentences
from embeddings import embed_texts, cosine_similarity


def collect_consecutive_similarities() -> list[float]:
    """Walks every document, embeds its sentences, and returns the
    similarity score between every consecutive sentence pair across the
    whole corpus -- this is the real distribution we tune against."""
    metadata = load_metadata()
    all_similarities = []

    for doc in metadata:
        doc_id = doc["doc_id"]
        text = (RAW_DIR / f"{doc_id}.txt").read_text()
        sentences = split_into_sentences(text)

        if len(sentences) < 2:
            continue

        embeddings = embed_texts(sentences)
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[i - 1], embeddings[i])
            all_similarities.append(sim)

    return all_similarities


def summarize(similarities: list[float]) -> None:
    arr = np.array(similarities)
    percentiles = [5, 10, 15, 25, 50, 75, 90, 95]

    print(f"Total consecutive sentence pairs measured: {len(arr)}")
    print(f"Min: {arr.min():.3f}  Max: {arr.max():.3f}  Mean: {arr.mean():.3f}\n")

    print("Percentiles:")
    for p in percentiles:
        value = np.percentile(arr, p)
        print(f"  p{p:>2}: {value:.3f}")

    print(f"\nCurrent threshold (0.5) would flag {(arr < 0.5).sum()} / {len(arr)} "
          f"pairs as boundaries ({(arr < 0.5).mean():.1%} of all pairs).")

    # A reasonable data-driven default: treat the bottom ~15% of observed
    # similarities as real boundaries, not an arbitrary fixed cutoff.
    suggested = np.percentile(arr, 15)
    print(f"\nSuggested threshold (15th percentile of real data): {suggested:.3f}")
    print(f"At that threshold: {(arr < suggested).sum()} / {len(arr)} pairs flagged "
          f"({(arr < suggested).mean():.1%})")


if __name__ == "__main__":
    similarities = collect_consecutive_similarities()
    summarize(similarities)