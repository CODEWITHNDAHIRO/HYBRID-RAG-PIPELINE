"""
Phase 2, step 1 (continued): sparse retrieval via BM25.

BM25 scores documents by keyword overlap, weighted by term rarity --
this is what catches exact technical term matches (function names, error
codes, config keys) that semantic/dense search can blur past.
"""
import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

from chunking import Chunk, PROCESSED_DIR

BM25_INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"


def load_chunks(strategy: str) -> list[Chunk]:
    filename = f"chunks_{strategy}.json"
    path = PROCESSED_DIR / filename
    raw = json.loads(path.read_text())
    return [Chunk(**c) for c in raw]


def tokenize(text: str) -> list[str]:
    """BM25 operates on tokens (words), not raw strings. A simple
    lowercase + word-boundary split is enough for this corpus -- no
    stemming or stopword removal, since technical terms (exact function
    names, keywords) matter more here than they would in general prose."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def build_bm25_index(strategy: str) -> tuple[BM25Okapi, list[Chunk]]:
    """Builds a BM25 index over a chunk set and saves it (plus the chunk
    list, needed to map result positions back to actual content) to disk."""
    chunks = load_chunks(strategy)
    tokenized_corpus = [tokenize(c.content) for c in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    index_path = PROCESSED_DIR / f"bm25_{strategy}.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    return bm25, chunks


def load_bm25_index(strategy: str) -> tuple[BM25Okapi, list[Chunk]]:
    index_path = PROCESSED_DIR / f"bm25_{strategy}.pkl"
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def query_bm25(query: str, strategy: str, top_k: int = 5) -> list[dict]:
    bm25, chunks = load_bm25_index(strategy)
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    # Get the indices of the top_k highest scores, highest first.
    # argsort sorts ascending by default, so we reverse it.
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for i in top_indices:
        chunk = chunks[i]
        results.append({
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "score": float(scores[i]),   # higher = more relevant, for BM25
            "doc_id": chunk.doc_id,
            "section_heading": chunk.section_heading,
        })
    return results


if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "fixed"

    print(f"Building BM25 index for strategy '{strategy}'...")
    bm25, chunks = build_bm25_index(strategy)
    print(f"Indexed {len(chunks)} chunks.\n")

    test_query = "HTTPException status code"
    print(f"Test query: {test_query!r}\n")
    results = query_bm25(test_query, strategy, top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['doc_id']} | {r['section_heading']}")
        print(f"    {r['content'][:100]}...")