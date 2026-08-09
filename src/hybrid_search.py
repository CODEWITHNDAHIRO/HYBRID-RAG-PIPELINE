"""
Phase 2, step 2: hybrid search via Reciprocal Rank Fusion (RRF).

Combines dense (vector_store.py) and sparse (bm25_index.py) results into
a single ranked list. RRF works purely on rank position, not raw scores
-- which sidesteps the problem that cosine distance and BM25 scores live
on completely different, incomparable numeric scales.
"""
import sys
from vector_store import query_vector_store
from bm25_index import query_bm25

RRF_K = 60  # standard smoothing constant from the original RRF paper


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """RRF score for a document = sum, over every ranked list it appears
    in, of 1 / (k + rank). A doc appearing near the top of both lists
    scores higher than one appearing near the top of only one list.
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        chunk_id = result["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup[chunk_id] = result

    for rank, result in enumerate(sparse_results):
        chunk_id = result["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup.setdefault(chunk_id, result)

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused = []
    for chunk_id in ranked_ids:
        result = dict(chunk_lookup[chunk_id])
        result["rrf_score"] = scores[chunk_id]
        result["found_in_dense"] = any(r["chunk_id"] == chunk_id for r in dense_results)
        result["found_in_sparse"] = any(r["chunk_id"] == chunk_id for r in sparse_results)
        fused.append(result)

    return fused


def hybrid_search(query: str, strategy: str, top_k: int = 5, candidate_k: int = 10) -> list[dict]:
    """Runs both retrieval methods, fuses with RRF, returns the top_k."""
    dense_results = query_vector_store(query, strategy, top_k=candidate_k)
    sparse_results = query_bm25(query, strategy, top_k=candidate_k)
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_k]


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "structured"
    query = "How do I raise an HTTPException with a custom status code?"

    print(f"Hybrid search for: {query!r} (strategy={strategy})\n")
    results = hybrid_search(query, strategy, top_k=5)

    for r in results:
        sources = []
        if r["found_in_dense"]:
            sources.append("dense")
        if r["found_in_sparse"]:
            sources.append("sparse")
        print(f"[RRF={r['rrf_score']:.4f}] ({'+'.join(sources)}) {r['doc_id']}")
        print(f"    {r['content'][:100]}...")