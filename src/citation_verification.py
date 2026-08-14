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
    cited_sources: list[int]


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


def _strip_markdown_structure(answer_text: str) -> str:
    """Removes Markdown structural lines (headings, horizontal rules,
    code fences + their content) before sentence-splitting.

    This is the fix for a real bug found on Day 10: naive sentence-
    splitting on '.', '!', '?' treated the period in a numbered heading
    like "### 2. With Query Validations" as a sentence boundary,
    producing a garbage pseudo-claim ("### 2.") that stole a citation
    marker away from the real sentence it belonged to. Same root cause
    pattern as the code-fence bug fixed in semantic_chunking.py on Day 7
    -- naive text processing not accounting for Markdown syntax.
    """
    lines = answer_text.split("\n")
    kept_lines = []
    inside_code_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            inside_code_fence = not inside_code_fence
            continue
        if inside_code_fence:
            continue

        if re.match(r"^#{1,6}\s", stripped):
            continue
        if re.match(r"^(-{3,}|_{3,}|\*{3,})$", stripped):
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines)


def split_answer_into_claims(answer_text: str) -> list[Claim]:
    """Splits an answer into sentence-level claims, each paired with the
    citation numbers attached to it.

    Two bugs were found and fixed here during real testing (Day 10-11):

    1. Markdown structure (headings, horizontal rules, code fences) was
       being sentence-split naively, e.g. the period in "### 2. With
       Query Validations" was mistaken for a sentence boundary,
       producing a garbage pseudo-claim. Fixed by stripping structural
       Markdown before sentence-splitting (_strip_markdown_structure).

    2. Even after that fix, a real citation like "...error. [1]" was
       still being split BETWEEN the period and the bracket, since a
       naive (?<=[.!?])\\s+ split fires on the whitespace immediately
       after the period, regardless of what follows. This stranded the
       citation marker at the start of the next chunk instead of the end
       of the sentence it belonged to. Fixed by inserting a boundary
       marker after any trailing citation bracket(s), not immediately
       after the terminating punctuation -- see the boundary_pattern
       below. A further edge case (a citation as the very last thing in
       the text, no trailing whitespace to match against) required
       padding the text with a trailing newline before processing.
    """
    cleaned = _strip_markdown_structure(answer_text)
    padded = cleaned.strip() + "\n"  # ensures a final citation always has trailing whitespace to match

    # Marks the boundary AFTER "terminator + any trailing [n] citations",
    # not immediately after the terminator itself.
    boundary_pattern = re.compile(r"([.!?](?:\s*\[\d+\])*)(\s+)")
    marked = boundary_pattern.sub(lambda m: m.group(1) + "\x00" + m.group(2), padded)
    pieces = [p.strip() for p in marked.split("\x00") if p.strip()]

    claims = []
    for piece in pieces:
        citation_numbers = [int(n) for n in re.findall(r"\[(\d+)\]", piece)]
        if citation_numbers:
            clean_text = re.sub(r"\[\d+\]", "", piece).strip()
            clean_text = re.sub(r"\s+", " ", clean_text)
            if clean_text:
                claims.append(Claim(text=clean_text, cited_sources=sorted(set(citation_numbers))))

    return claims


def verify_claim(claim: Claim, sources: list[dict]) -> CitationVerdict:
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