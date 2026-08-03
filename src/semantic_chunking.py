"""
Phase 1, step 2 (continued): strategy 3, semantic chunking.

Splits text into sentences, embeds each one, and detects topic boundaries
by watching for drops in similarity between consecutive sentences.
Sentences between boundaries are grouped into one chunk.
"""
import re
import numpy as np
from pathlib import Path

from chunking import Chunk, load_metadata, RAW_DIR, PROCESSED_DIR, save_chunks
from embeddings import embed_texts, cosine_similarity

# If similarity between consecutive sentences drops below this, treat it
# as a topic boundary -- start a new chunk. Lower = more sensitive (more,
# smaller chunks); higher = less sensitive (fewer, larger chunks).
SIMILARITY_THRESHOLD = 0.5

# Never produce a chunk with fewer than this many sentences -- avoids
# pathological one-sentence chunks from noisy similarity dips.
MIN_SENTENCES_PER_CHUNK = 2


def split_into_sentences(text: str) -> list[str]:
    """A simple sentence splitter: breaks on '.', '!', '?' followed by
    whitespace, but not inside common abbreviations. Not perfect (no
    sentence splitter is, without a full NLP model), but good enough for
    technical documentation prose."""
    raw_pieces = re.split(r"\n+", text)
    sentences = []
    inside_code_fence = False

    for piece in raw_pieces:
        piece = piece.strip()
        if not piece:
            continue

        # Track fence state across lines -- a ``` line toggles us in or
        # out of a code block. Everything strictly between the opening
        # and closing fence must be skipped, not just the fence lines
        # themselves.
        if piece.startswith("```"):
            inside_code_fence = not inside_code_fence
            continue
        if inside_code_fence:
            continue

        if piece.startswith("#"):
            continue

        parts = re.split(r"(?<=[.!?])\s+", piece)
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences


def detect_boundaries(embeddings: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> list[int]:
    """Given N sentence embeddings, returns the list of sentence indices
    where a new chunk should start (always includes index 0).

    Walks consecutive pairs; wherever similarity drops below threshold,
    that's a boundary -- the *next* sentence starts a new chunk.
    """
    if len(embeddings) == 0:
        return []

    boundaries = [0]  # the first sentence always starts the first chunk

    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            boundaries.append(i)

    return boundaries


def group_sentences_by_boundaries(sentences: list[str], boundaries: list[int]) -> list[str]:
    """Turns a list of sentences + boundary indices into a list of chunk
    strings, merging any chunk that ended up too small (fewer than
    MIN_SENTENCES_PER_CHUNK sentences) into the next one."""
    raw_groups = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
        raw_groups.append(sentences[start:end])

    # Merge undersized groups forward into the next group.
    merged_groups: list[list[str]] = []
    pending: list[str] = []
    for group in raw_groups:
        pending.extend(group)
        if len(pending) >= MIN_SENTENCES_PER_CHUNK:
            merged_groups.append(pending)
            pending = []
    if pending:
        # Leftover at the end -- attach to the last group if one exists,
        # otherwise it's the only group there is.
        if merged_groups:
            merged_groups[-1].extend(pending)
        else:
            merged_groups.append(pending)

    return [" ".join(group) for group in merged_groups]


def chunk_semantic(text: str, doc_id: str, threshold: float = SIMILARITY_THRESHOLD) -> list[Chunk]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    embeddings = embed_texts(sentences)
    boundaries = detect_boundaries(embeddings, threshold)
    chunk_texts = group_sentences_by_boundaries(sentences, boundaries)

    chunks = []
    for index, content in enumerate(chunk_texts):
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_semantic_{index}",
            doc_id=doc_id,
            strategy="semantic",
            chunk_index=index,
            content=content,
            char_count=len(content),
        ))
    return chunks


def chunk_all_documents_semantic(threshold: float = SIMILARITY_THRESHOLD) -> list[Chunk]:
    metadata = load_metadata()
    all_chunks = []
    for doc in metadata:
        doc_id = doc["doc_id"]
        text = (RAW_DIR / f"{doc_id}.txt").read_text()
        all_chunks.extend(chunk_semantic(text, doc_id, threshold))
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_documents_semantic()
    print(f"Produced {len(chunks)} semantic chunks.")
    avg_size = sum(c.char_count for c in chunks) / len(chunks)
    print(f"Average chunk size: {avg_size:.0f} chars")
    save_chunks(chunks, "chunks_semantic.json")