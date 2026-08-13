"""
Phase 3, step 2: citation verification.

generate.py produces answers with [n] citations, but a citation existing
doesn't prove the claim it's attached to is actually supported by that
source. This module splits an answer into individual claims, and for
each one, uses an LLM-as-judge call to check whether its cited source(s)
genuinely support it -- same pattern as Project 1's judge.py, applied to
claim-support instead of summary quality.
"""
import re
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()


class Claim(BaseModel):
    text: str
    cited_sources: list[int]   # which source numbers this claim cites, e.g. [1, 3]


class CitationVerdict(BaseModel):
    claim_text: str
    cited_sources: list[int]
    supported: bool
    reasoning: str


VERIFY_TOOL = {
    "name": "submit_verdict",
    "description": "Submit whether the claim is supported by the cited source(s).",
    "input_schema": {
        "type": "object",
        "properties": {
            "supported": {
                "type": "boolean",
                "description": "True if the source(s) genuinely support this specific claim",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the verdict",
            },
        },
        "required": ["supported", "reasoning"],
    },
}

VERIFY_SYSTEM_PROMPT = """You are checking whether a specific claim is \
genuinely supported by its cited source text. This is NOT about whether \
the claim is true in general -- it's specifically about whether THIS \
source text actually contains or directly implies THIS claim.

Mark supported=false if:
- The source doesn't mention the specific fact claimed
- The claim overstates or extrapolates beyond what the source says
- The claim is generic/true but not actually grounded in this specific source

Mark supported=true only if the source text genuinely contains or \
directly implies the claim."""


def split_answer_into_claims(answer_text: str) -> list[Claim]:
    """Splits an answer into sentence-level claims, each paired with the
    citation numbers attached to it. A sentence with no [n] markers at
    all is skipped -- there's nothing to verify if nothing was cited.
    """
    # Split on sentence boundaries, similar approach to Project 1's
    # sentence splitter -- not perfect, good enough for this purpose.
    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())

    claims = []
    for sentence in sentences:
        citation_numbers = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        if citation_numbers:
            # Strip the citation markers themselves out of the claim text
            # sent to the judge -- we want it judging the substance, not
            # parsing bracket syntax.
            clean_text = re.sub(r"\[\d+\]", "", sentence).strip()
            claims.append(Claim(text=clean_text, cited_sources=sorted(set(citation_numbers))))

    return claims


def verify_claim(claim: Claim, sources: list[dict]) -> CitationVerdict:
    """Checks one claim against its cited source(s) using an LLM judge."""
    # sources is 0-indexed (list position), citations are 1-indexed
    # (matching the [1][2] numbering shown to the user) -- this mapping
    # has to be handled carefully or verification checks the wrong chunk.
    cited_content = "\n\n".join(
        f"Source [{n}]: {sources[n - 1]['content']}"
        for n in claim.cited_sources
        if n - 1 < len(sources)
    )

    user_message = f"""Claim: {claim.text}

Cited source(s):
{cited_content}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=VERIFY_SYSTEM_PROMPT,
        tools=[VERIFY_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_call = next(b for b in response.content if b.type == "tool_use")
    return CitationVerdict(
        claim_text=claim.text,
        cited_sources=claim.cited_sources,
        supported=tool_call.input["supported"],
        reasoning=tool_call.input["reasoning"],
    )


def verify_all_citations(answer_text: str, sources: list[dict]) -> dict:
    """Verifies every claim in an answer, returns an aggregate report."""
    claims = split_answer_into_claims(answer_text)

    if not claims:
        return {"claims": [], "total_claims": 0, "supported_claims": 0, "citation_accuracy": None}

    verdicts = [verify_claim(claim, sources) for claim in claims]
    supported_count = sum(1 for v in verdicts if v.supported)

    return {
        "claims": [v.model_dump() for v in verdicts],
        "total_claims": len(verdicts),
        "supported_claims": supported_count,
        "citation_accuracy": supported_count / len(verdicts),
    }


if __name__ == "__main__":
    from generate import generate_answer

    query = "How do I make a query parameter required in FastAPI?"
    result = generate_answer(query, strategy="structured")

    print(f"Answer:\n{result['answer']}\n")

    report = verify_all_citations(result["answer"], result["sources"])

    print(f"Citation verification: {report['supported_claims']}/{report['total_claims']} "
          f"claims supported ({report['citation_accuracy']:.0%})\n")

    for claim in report["claims"]:
        status = "SUPPORTED" if claim["supported"] else "NOT SUPPORTED"
        print(f"[{status}] {claim['claim_text']}")
        print(f"    cited: {claim['cited_sources']} -- {claim['reasoning']}")