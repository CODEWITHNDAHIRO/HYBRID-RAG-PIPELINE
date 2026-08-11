"""
Phase 2, step 4: reranking.

RRF fusion combines dense + sparse rankings, but it only knows rank
position -- it has no idea whether a chunk is *actually* relevant to the
query's meaning. A cross-encoder reranker fixes this: unlike our
embedding model (which encodes the query and each chunk *separately*,
then compares vectors), a cross-encoder reads the query and a candidate
chunk *together* in one pass and outputs a single relevance score. This
is slower (can't be precomputed/indexed like embeddings), which is why
it's only run on the small top-N candidate pool from fusion, not the
whole corpus.

Local model (cross-encoder/ms-marco-MiniLM-L-6-v2), same "free, no API
key" reasoning as ADR 003's embedding model choice.
"""
from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Scores every candidate against the query with the cross-encoder,
    re-sorts by that score, returns the top_k. Each returned dict gets a
    new 'rerank_score' field; original RRF ranking is preserved as
    'rrf_rank' so you can see how much reranking actually changed the
    order.
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [[query, c["content"]] for c in candidates]
    scores = model.predict(pairs)

    for candidate, score, original_rank in zip(candidates, scores, range(len(candidates))):
        candidate["rerank_score"] = float(score)
        candidate["rrf_rank"] = original_rank

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from hybrid_search import hybrid_search

    strategy = sys.argv[1] if len(sys.argv) > 1 else "structured"

    # A deliberately paraphrased query -- doesn't use the docs' exact
    # wording, to give dense and sparse retrieval a real chance to
    # disagree with each other (unlike Day 6's query, which shared exact
    # vocabulary with the docs and got unanimous agreement).
    query = "Why might my API endpoint reject the data someone sends it?"

    print(f"Query: {query!r} (strategy={strategy})\n")

    # Pull a wider candidate pool (20) from fusion before reranking down
    # to a final top 5 -- reranking only helps if it has real options to
    # choose between.
    candidates = hybrid_search(query, strategy, top_k=20, candidate_k=20)

    print("=== Before reranking (RRF order) ===")
    for i, c in enumerate(candidates[:5]):
        sources = ("dense" if c["found_in_dense"] else "") + ("+sparse" if c["found_in_sparse"] else "")
        print(f"{i}. [{sources}] {c['doc_id']}: {c['content'][:70]}...")

    reranked = rerank(query, candidates, top_k=5)

    print("\n=== After reranking (cross-encoder order) ===")
    for c in reranked:
        moved = c["rrf_rank"] - reranked.index(c)
        print(f"[score={c['rerank_score']:.2f}] (was rank {c['rrf_rank']}, moved {moved:+d}) "
              f"{c['doc_id']}: {c['content'][:70]}...")