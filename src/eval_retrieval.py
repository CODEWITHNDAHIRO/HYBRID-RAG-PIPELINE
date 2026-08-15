"""
Phase 4: retrieval evaluation.

Runs every golden Q&A question through hybrid search + reranking for
each of the three chunking strategies, and measures recall@k: did the
expected source doc(s) actually appear in the top-k retrieved results?
This is the real, empirical comparison the guide's Phase 4 calls for --
not an assumption about which chunking strategy is "better."

This entire evaluation runs on the local embedding/BM25/reranker
pipeline -- zero API cost, since no Claude calls are involved.
"""
import json
from pathlib import Path
from pydantic import BaseModel

from hybrid_search import hybrid_search
from reranker import rerank

EVAL_DATA_DIR = Path(__file__).parent.parent / "eval_data"
STRATEGIES = ["fixed", "structured", "semantic"]
TOP_K = 5


class GoldenQuestion(BaseModel):
    id: str
    question: str
    expected_doc_ids: list[str]
    difficulty: str
    notes: str


class QuestionResult(BaseModel):
    question_id: str
    strategy: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    hit: bool          # True if AT LEAST ONE expected doc was retrieved
    full_hit: bool       # True if ALL expected docs were retrieved (matters for multi-hop questions)


def load_golden_questions() -> list[GoldenQuestion]:
    raw = json.loads((EVAL_DATA_DIR / "golden_qa.json").read_text())
    return [GoldenQuestion(**q) for q in raw]


def evaluate_question(question: GoldenQuestion, strategy: str, top_k: int = TOP_K) -> QuestionResult:
    candidates = hybrid_search(question.question, strategy, top_k=20, candidate_k=20)
    reranked = rerank(question.question, candidates, top_k=top_k)

    retrieved_doc_ids = list(dict.fromkeys(r["doc_id"] for r in reranked))  # dedupe, preserve order
    expected_set = set(question.expected_doc_ids)
    retrieved_set = set(retrieved_doc_ids)

    return QuestionResult(
        question_id=question.id,
        strategy=strategy,
        expected_doc_ids=question.expected_doc_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        hit=bool(expected_set & retrieved_set),
        full_hit=expected_set.issubset(retrieved_set),
    )


def run_full_evaluation() -> dict[str, list[QuestionResult]]:
    """Runs every question against every strategy. Returns results
    grouped by strategy, so they can be compared side by side."""
    questions = load_golden_questions()
    results_by_strategy: dict[str, list[QuestionResult]] = {s: [] for s in STRATEGIES}

    for strategy in STRATEGIES:
        print(f"Evaluating strategy: {strategy}...")
        for question in questions:
            result = evaluate_question(question, strategy)
            results_by_strategy[strategy].append(result)

    return results_by_strategy


def print_comparison_report(results_by_strategy: dict[str, list[QuestionResult]]) -> None:
    print("\n=== Recall@5 by strategy ===\n")
    print(f"{'Strategy':<12} {'Hit rate (any)':<16} {'Full hit rate (all)':<20}")
    for strategy, results in results_by_strategy.items():
        hit_rate = sum(r.hit for r in results) / len(results)
        full_hit_rate = sum(r.full_hit for r in results) / len(results)
        print(f"{strategy:<12} {hit_rate:.0%}{'':<12} {full_hit_rate:.0%}")

    print("\n=== Per-question breakdown ===\n")
    questions = load_golden_questions()
    for q in questions:
        print(f"[{q.id}] ({q.difficulty}) {q.question}")
        print(f"    expected: {q.expected_doc_ids}")
        for strategy in STRATEGIES:
            result = next(r for r in results_by_strategy[strategy] if r.question_id == q.id)
            status = "HIT" if result.hit else "MISS"
            full = " (full)" if result.full_hit else ""
            print(f"    {strategy:<12} [{status}{full}] retrieved: {result.retrieved_doc_ids}")
        print()


if __name__ == "__main__":
    results = run_full_evaluation()
    print_comparison_report(results)