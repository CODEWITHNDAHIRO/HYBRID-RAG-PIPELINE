"""
Local embedding generation via sentence-transformers. Runs entirely on
your machine -- no API key, no per-call cost, no network dependency once
the model weights are downloaded the first time.

Model: all-MiniLM-L6-v2 -- a small, fast, well-regarded general-purpose
embedding model. Not as strong as a large hosted model (OpenAI's
text-embedding-3, Voyage), but a reasonable, honest tradeoff for a
portfolio project: free, fast, and good enough to demonstrate the
mechanism correctly.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None  # lazy-loaded, see get_model()


def get_model() -> SentenceTransformer:
    """Loads the embedding model once and reuses it. Loading a model is
    slow (reads weights from disk/downloads them); doing this every call
    would be wasteful, so we cache it in a module-level variable."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embeds a list of strings, returns an (N, 384) array -- one 384-
    dimensional vector per input string. Batches internally for speed."""
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity: how similar the *direction* of two
    vectors is, ignoring magnitude. 1.0 = identical direction (very
    similar meaning), 0.0 = unrelated, negative = opposite."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    # Smoke test: semantically similar sentences should score high,
    # unrelated ones should score low.
    texts = [
        "FastAPI uses Python type hints for request validation.",
        "Type hints in Python are used by FastAPI to validate requests.",
        "The weather today is sunny with a light breeze.",
    ]
    vectors = embed_texts(texts)
    print(f"Embedded {len(texts)} texts, each with {vectors.shape[1]} dimensions.\n")

    sim_similar = cosine_similarity(vectors[0], vectors[1])
    sim_different = cosine_similarity(vectors[0], vectors[2])

    print(f"Similarity (paraphrased, same topic): {sim_similar:.3f}")
    print(f"Similarity (unrelated topic):          {sim_different:.3f}")
    assert sim_similar > sim_different, "Paraphrase should score higher than unrelated text"
    print("\nSanity check passed: paraphrase scored higher than unrelated text.")