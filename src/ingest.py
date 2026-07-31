"""
Phase 1, step 1: document ingestion.

Fetches a curated list of real FastAPI documentation pages, strips out
navigation/sidebar/scripts, and saves clean text + metadata locally. This
is the corpus the rest of the RAG pipeline will index and search over.

Run this yourself locally -- it downloads real pages for your own use,
same as saving pages from a browser.
"""
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).parent.parent / "docs_corpus" / "raw"

# A curated set of FastAPI tutorial/advanced pages, chosen to span several
# distinct topics -- this matters later for the eval dataset, since we
# want questions that require finding the *right* topic among several
# plausible ones, not just the only page that exists.
PAGES = [
    ("first-steps", "https://fastapi.tiangolo.com/tutorial/first-steps/"),
    ("path-params", "https://fastapi.tiangolo.com/tutorial/path-params/"),
    ("query-params", "https://fastapi.tiangolo.com/tutorial/query-params/"),
    ("body", "https://fastapi.tiangolo.com/tutorial/body/"),
    ("query-params-str-validations", "https://fastapi.tiangolo.com/tutorial/query-params-str-validations/"),
    ("dependencies", "https://fastapi.tiangolo.com/tutorial/dependencies/"),
    ("security-first-steps", "https://fastapi.tiangolo.com/tutorial/security/first-steps/"),
    ("cors", "https://fastapi.tiangolo.com/tutorial/cors/"),
    ("middleware", "https://fastapi.tiangolo.com/tutorial/middleware/"),
    ("sql-databases", "https://fastapi.tiangolo.com/tutorial/sql-databases/"),
    ("background-tasks", "https://fastapi.tiangolo.com/tutorial/background-tasks/"),
    ("bigger-applications", "https://fastapi.tiangolo.com/tutorial/bigger-applications/"),
    ("testing", "https://fastapi.tiangolo.com/tutorial/testing/"),
    ("handling-errors", "https://fastapi.tiangolo.com/tutorial/handling-errors/"),
    ("response-model", "https://fastapi.tiangolo.com/tutorial/response-model/"),
]


@dataclass
class IngestedDoc:
    doc_id: str          # stable slug, e.g. "first-steps"
    title: str
    url: str
    content: str          # cleaned plaintext
    char_count: int


def fetch_page(url: str) -> str:
    """Downloads raw HTML for one page."""
    headers = {"User-Agent": "Mozilla/5.0 (personal RAG portfolio project)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text


def extract_clean_text(html: str) -> tuple[str, str]:
    """Parses HTML and returns (title, clean_body_text), stripping nav,
    sidebar, scripts, and other non-content chrome.

    FastAPI's docs use the MkDocs Material theme, where the actual article
    content lives inside <article> -- everything outside that (nav bars,
    table of contents sidebar, footer) is chrome we don't want in our
    search corpus.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    article = soup.find("article")
    if article is None:
        raise ValueError("Could not find <article> content -- page structure may have changed")

    # Remove elements that aren't real content even within the article
    # (e.g. "Edit this page" links, embedded nav).
    for unwanted in article.select("nav, .md-source-file, script, style"):
        unwanted.decompose()

    text = article.get_text(separator="\n", strip=True)
    # Collapse 3+ consecutive newlines down to 2, for readability.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return title, text


def ingest_all() -> list[IngestedDoc]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs = []

    for doc_id, url in PAGES:
        print(f"Fetching {doc_id} ...")
        try:
            html = fetch_page(url)
            title, content = extract_clean_text(html)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        doc = IngestedDoc(
            doc_id=doc_id,
            title=title,
            url=url,
            content=content,
            char_count=len(content),
        )
        docs.append(doc)

        # Save the clean text
        (RAW_DIR / f"{doc_id}.txt").write_text(content)
        print(f"  Saved {doc.char_count} chars")

        # Be a polite scraper -- don't hammer the server with rapid requests.
        time.sleep(0.5)

    # Save metadata for all docs together, for easy loading later.
    metadata = [asdict(d) for d in docs]
    (RAW_DIR / "_metadata.json").write_text(json.dumps(metadata, indent=2))

    return docs


if __name__ == "__main__":
    docs = ingest_all()
    print(f"\nIngested {len(docs)}/{len(PAGES)} pages successfully.")
    total_chars = sum(d.char_count for d in docs)
    print(f"Total corpus size: {total_chars:,} characters (~{total_chars // 4:,} tokens)")