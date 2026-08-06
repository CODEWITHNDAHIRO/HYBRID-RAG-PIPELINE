"""
Phase 2, step 1: dense retrieval via ChromaDB.

Stores every chunk's embedding (computed with our local sentence-
transformers model) in a persistent local vector store, and provides a
query function for finding the top-k most semantically similar chunks
to a new question.
"""
import json
from pathlib import Path
import chromadb

from chunking import Chunk, PROCESSED_DIR
from embeddings import embed_texts

CHROMA_DIR = PROCESSED_DIR / "chroma_db"


def get_chroma_client() -> chromadb.PersistentClient:
    """A PersistentClient writes to disk, so the index survives between
    script runs -- you don't have to re-embed everything every time you
    want to query it."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def load_chunks(strategy: str) -> list[Chunk]:
    """Loads a previously-saved chunk set (from chunking.py /
    semantic_chunking.py) for a given strategy."""
    filename = f"chunks_{strategy}.json"
    path = PROCESSED_DIR / filename
    raw = json.loads(path.read_text())
    return [Chunk(**c) for c in raw]


def build_vector_store(strategy: str) -> chromadb.Collection:
    """Embeds every chunk for a given strategy and stores it in a
    ChromaDB collection named after the strategy -- so "fixed",
    "structured", and "semantic" each get their own independent index,
    letting us compare retrieval quality across strategies later."""
    chunks = load_chunks(strategy)
    client = get_chroma_client()

    # Delete any existing collection with this name first, so re-running
    # this function doesn't just keep appending duplicate entries.
    try:
        client.delete_collection(name=strategy)
    except Exception:
        pass  # collection didn't exist yet, nothing to delete

    collection = client.create_collection(name=strategy)

    texts = [c.content for c in chunks]
    embeddings = embed_texts(texts)

    # ChromaDB wants embeddings as plain Python lists, not numpy arrays.
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{
            "doc_id": c.doc_id,
            "strategy": c.strategy,
            "chunk_index": c.chunk_index,
            "section_heading": c.section_heading or "",
        } for c in chunks],
    )

    return collection


def query_vector_store(query: str, strategy: str, top_k: int = 5) -> list[dict]:
    """Embeds the query and finds the top_k most similar chunks in the
    given strategy's collection."""
    client = get_chroma_client()
    collection = client.get_collection(name=strategy)

    query_embedding = embed_texts([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # ChromaDB returns parallel lists (ids[0], documents[0], distances[0],
    # metadatas[0]) rather than a list of result objects -- zip them
    # together into something easier to work with.
    output = []
    for chunk_id, doc, distance, meta in zip(
        results["ids"][0], results["documents"][0],
        results["distances"][0], results["metadatas"][0]
    ):
        output.append({
            "chunk_id": chunk_id,
            "content": doc,
            "distance": distance,     # lower = more similar, for ChromaDB's default metric
            "doc_id": meta["doc_id"],
            "section_heading": meta["section_heading"],
        })
    return output


if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "fixed"

    print(f"Building vector store for strategy '{strategy}'...")
    collection = build_vector_store(strategy)
    print(f"Indexed {collection.count()} chunks.\n")

    test_query = "How do I add a required query parameter?"
    print(f"Test query: {test_query!r}\n")
    results = query_vector_store(test_query, strategy, top_k=3)
    for r in results:
        print(f"[{r['distance']:.3f}] {r['doc_id']} | {r['section_heading']}")
        print(f"    {r['content'][:100]}...")