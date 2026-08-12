"""
Phase 3, step 1: grounded generation.

Takes a user question, retrieves + reranks the most relevant chunks
(everything built in Phase 1-2), and asks Claude to answer using ONLY
that context, citing which numbered source supports each claim. This is
what turns "a search engine" into "a system that answers questions."
"""
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

from hybrid_search import hybrid_search
from reranker import rerank

load_dotenv()
client = anthropic.Anthropic()

GROUNDED_SYSTEM_PROMPT = """You are a technical documentation assistant \
answering questions about FastAPI, using ONLY the numbered source \
excerpts provided below. Follow these rules strictly:

1. Answer using ONLY information present in the provided sources. Do not \
use any outside knowledge, even if you know the answer from training.
2. Cite the specific source number for every claim, using bracketed \
references like [1] or [2]. If a sentence draws on multiple sources, \
cite all of them: [1][3].
3. If the provided sources do not contain enough information to answer \
the question, say so explicitly. Do not guess or fill gaps with \
outside knowledge.
4. Be concise. Answer the question directly; don't restate the sources \
at length."""


def build_context_blocks(chunks: list[dict]) -> str:
    """Formats retrieved chunks as numbered source blocks for the prompt.
    The numbers here are what the model's [1][2] citations refer back to.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        heading = f" ({chunk['section_heading']})" if chunk.get("section_heading") else ""
        blocks.append(f"[{i}] Source: {chunk['doc_id']}{heading}\n{chunk['content']}")
    return "\n\n".join(blocks)


def parse_citations(answer_text: str) -> set[int]:
    """Extracts every citation number referenced in the answer, e.g.
    'FastAPI validates types [1][3].' -> {1, 3}."""
    matches = re.findall(r"\[(\d+)\]", answer_text)
    return {int(m) for m in matches}


def generate_answer(query: str, strategy: str = "structured", top_k: int = 5) -> dict:
    """The full pipeline: retrieve -> rerank -> generate a grounded,
    cited answer. Returns the answer plus the source chunks it was
    generated from, so citations can be verified afterward (Phase 3
    step 2, not built yet).
    """
    candidates = hybrid_search(query, strategy, top_k=20, candidate_k=20)
    top_chunks = rerank(query, candidates, top_k=top_k)

    context = build_context_blocks(top_chunks)
    user_message = f"Sources:\n\n{context}\n\nQuestion: {query}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=GROUNDED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text
    cited_sources = parse_citations(answer_text)

    return {
        "query": query,
        "answer": answer_text,
        "cited_source_numbers": sorted(cited_sources),
        "sources": top_chunks,   # index i in this list = citation number i+1
        "num_sources_provided": len(top_chunks),
        "num_sources_cited": len(cited_sources),
    }


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "structured"
    query = "How do I make a query parameter required in FastAPI?"

    result = generate_answer(query, strategy)

    print(f"Query: {result['query']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Cited {result['num_sources_cited']}/{result['num_sources_provided']} "
          f"provided sources: {result['cited_source_numbers']}\n")

    print("Sources provided:")
    for i, s in enumerate(result["sources"], start=1):
        cited_marker = "CITED" if i in result["cited_source_numbers"] else "unused"
        print(f"  [{i}] ({cited_marker}) {s['doc_id']}")