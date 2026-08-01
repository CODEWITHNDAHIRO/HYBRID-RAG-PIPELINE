"""
Phase 1, step 2: chunking strategies.

Splits ingested documents into smaller pieces for embedding/retrieval.
This module implements strategy 1 (fixed-size with overlap) -- the
baseline every other strategy gets compared against.
"""
import json
import re
from pathlib import Path
from pydantic import BaseModel

RAW_DIR = Path(__file__).parent.parent / "docs_corpus" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "docs_corpus" / "processed"


class Chunk(BaseModel):
    chunk_id: str        # stable id: "{doc_id}_{strategy}_{index}"
    doc_id: str           # which source document this came from
    strategy: str          # which chunking strategy produced this
    chunk_index: int        # position within the document (0, 1, 2, ...)
    content: str
    char_count: int
    section_heading: str | None = None  # breadcrumb path, e.g. "Query Parameters > Required Parameters"


def load_metadata() -> list[dict]:
    metadata_path = RAW_DIR / "_metadata.json"
    return json.loads(metadata_path.read_text())


def chunk_fixed_size(text: str, doc_id: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Splits text into fixed-size chunks with character overlap between
    consecutive chunks.

    Example with chunk_size=10, overlap=3 on "ABCDEFGHIJKLMNOP":
      chunk 0: "ABCDEFGHIJ"      (chars 0-10)
      chunk 1: "HIJKLMNOPQ"...   starts at 10-3=7, so it overlaps "HIJ"
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append(Chunk(
            chunk_id=f"{doc_id}_fixed_{index}",
            doc_id=doc_id,
            strategy="fixed",
            chunk_index=index,
            content=chunk_text,
            char_count=len(chunk_text),
        ))

        # Advance by (chunk_size - overlap), not chunk_size, so the next
        # chunk starts before this one ends -- that's what creates overlap.
        start += (chunk_size - overlap)
        index += 1

    return chunks


def chunk_all_documents(chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    metadata = load_metadata()
    all_chunks = []

    for doc in metadata:
        doc_id = doc["doc_id"]
        text_path = RAW_DIR / f"{doc_id}.txt"
        text = text_path.read_text()

        doc_chunks = chunk_fixed_size(text, doc_id, chunk_size, overlap)
        all_chunks.extend(doc_chunks)

    return all_chunks


# Matches a Markdown heading line: 1-4 '#' characters, a space, then text.
# re.MULTILINE makes '^' match the start of each line, not just the start
# of the whole string.
HEADING_PATTERN = re.compile(r"^(#{1,4}) (.+)$", re.MULTILINE)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Splits text at heading boundaries. Returns a list of
    (heading_path, section_content) tuples, where heading_path is a
    breadcrumb like "Query Parameters > Required Parameters" built from
    the current heading stack at each point in the document.
    """
    matches = list(HEADING_PATTERN.finditer(text))

    if not matches:
        # No headings at all -- treat the whole document as one section.
        return [("", text.strip())] if text.strip() else []

    sections = []
    heading_stack: dict[int, str] = {}  # level -> heading text, e.g. {1: "Title", 2: "Subsection"}

    for i, match in enumerate(matches):
        level = len(match.group(1))     # number of '#' characters
        heading_text = match.group(2).strip()

        # A new heading at level N replaces the stack entry at level N,
        # and clears out any deeper levels (e.g. a new H2 invalidates
        # whatever H3 subsection we were previously inside).
        heading_stack[level] = heading_text
        for deeper_level in [l for l in heading_stack if l > level]:
            del heading_stack[deeper_level]

        breadcrumb = " > ".join(heading_stack[lvl] for lvl in sorted(heading_stack))

        # This section's content runs from the end of this heading match
        # to the start of the next heading match (or end of text).
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_content = text[content_start:content_end].strip()

        if section_content:
            sections.append((breadcrumb, section_content))

    return sections


def chunk_structured(text: str, doc_id: str, max_chunk_size: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Splits text along heading boundaries. Sections that fit within
    max_chunk_size become one chunk each, preserving full section context.
    Oversized sections (e.g. a very long tutorial page) get recursively
    split with the fixed-size strategy, but each sub-chunk keeps the
    section's heading breadcrumb prepended -- so even a sub-chunk knows
    what section it came from.
    """
    sections = _split_into_sections(text)
    chunks = []
    index = 0

    for breadcrumb, content in sections:
        if len(content) <= max_chunk_size:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_structured_{index}",
                doc_id=doc_id,
                strategy="structured",
                chunk_index=index,
                content=content,
                char_count=len(content),
                section_heading=breadcrumb or None,
            ))
            index += 1
        else:
            # Section too big -- fall back to fixed-size splitting within
            # just this section, but keep the breadcrumb attached to each
            # piece so retrieval still knows the section context.
            sub_chunks = chunk_fixed_size(content, doc_id, max_chunk_size, overlap)
            for sub in sub_chunks:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_structured_{index}",
                    doc_id=doc_id,
                    strategy="structured",
                    chunk_index=index,
                    content=f"[{breadcrumb}]\n{sub.content}" if breadcrumb else sub.content,
                    char_count=sub.char_count,
                    section_heading=breadcrumb or None,
                ))
                index += 1

    return chunks


def chunk_all_documents_structured(max_chunk_size: int = 1200, overlap: int = 150) -> list[Chunk]:
    metadata = load_metadata()
    all_chunks = []

    for doc in metadata:
        doc_id = doc["doc_id"]
        text_path = RAW_DIR / f"{doc_id}.txt"
        text = text_path.read_text()

        doc_chunks = chunk_structured(text, doc_id, max_chunk_size, overlap)
        all_chunks.extend(doc_chunks)

    return all_chunks


def save_chunks(chunks: list[Chunk], filename: str = "chunks_fixed.json") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    out_path.write_text(json.dumps([c.model_dump() for c in chunks], indent=2))
    return out_path


if __name__ == "__main__":
    print("=== Strategy 1: fixed-size with overlap ===")
    fixed_chunks = chunk_all_documents()
    print(f"Produced {len(fixed_chunks)} chunks.")
    avg_fixed = sum(c.char_count for c in fixed_chunks) / len(fixed_chunks)
    print(f"Average chunk size: {avg_fixed:.0f} chars")
    save_chunks(fixed_chunks, "chunks_fixed.json")

    print("\n=== Strategy 2: structure-aware (heading-based) ===")
    structured_chunks = chunk_all_documents_structured()
    print(f"Produced {len(structured_chunks)} chunks.")
    avg_structured = sum(c.char_count for c in structured_chunks) / len(structured_chunks)
    print(f"Average chunk size: {avg_structured:.0f} chars")
    with_headings = sum(1 for c in structured_chunks if c.section_heading)
    print(f"Chunks with a section heading: {with_headings}/{len(structured_chunks)}")
    save_chunks(structured_chunks, "chunks_structured.json")

    print("\n=== Comparison ===")
    print(f"Fixed:      {len(fixed_chunks)} chunks, avg {avg_fixed:.0f} chars")
    print(f"Structured: {len(structured_chunks)} chunks, avg {avg_structured:.0f} chars")