"""
The LLM-facing document ingestion pipeline: fetches real FastAPI
documentation pages and saves clean, structure-preserved text locally.

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
    doc_id: str
    title: str
    url: str
    content: str
    char_count: int


def fetch_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (personal RAG portfolio project)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text


def _clean_element_text(tag) -> str:
    """Extracts text from a tag with a space inserted between every
    inline sub-element (fixes "of404" / "requestshttp://" style
    concatenation bugs), then collapses any resulting run of multiple
    spaces down to one.
    """
    text = tag.get_text(separator=" ", strip=True)
    text = re.sub(r" {2,}", " ", text)
    # Clean up spacing artifacts the separator can introduce around
    # punctuation, e.g. "item_id" and the quote around it should not
    # gain a stray space before a comma/period/closing paren.
    text = re.sub(r"\s+([,.\)\]:;!?])", r"\1", text)
    return text


def extract_clean_text(html: str) -> tuple[str, str]:
    """Parses HTML and returns (title, markdown_text), preserving
    heading/code/list structure while fixing two artifacts found during
    Day 5-6 retrieval testing:

    1. MkDocs Material injects a hidden "permalink" anchor (rendered as
       a pilcrow, ¶) inside every heading. Removed before text extraction.
    2. Inline elements (<code>, <a>) sitting flush against surrounding
       text with no HTML whitespace were being concatenated with zero
       space (e.g. "status code of404"). Fixed via _clean_element_text's
       explicit separator.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    article = soup.find("article")
    if article is None:
        raise ValueError("Could not find <article> content -- page structure may have changed")

    for unwanted in article.select("nav, .md-source-file, script, style, a.headerlink"):
        unwanted.decompose()

    heading_prefix = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}
    lines = []

    for tag in article.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li"]):
        tag_name = tag.name

        if tag_name in heading_prefix:
            text = _clean_element_text(tag)
            if text:
                lines.append(f"\n{heading_prefix[tag_name]} {text}\n")

        elif tag_name == "pre":
            code_text = tag.get_text()
            if code_text.strip():
                lines.append(f"```\n{code_text.strip()}\n```")

        elif tag_name == "li":
            text = _clean_element_text(tag)
            if text:
                lines.append(f"- {text}")

        elif tag_name == "p":
            text = _clean_element_text(tag)
            if text:
                lines.append(text)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return title, text.strip()


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

        doc = IngestedDoc(doc_id=doc_id, title=title, url=url, content=content, char_count=len(content))
        docs.append(doc)

        (RAW_DIR / f"{doc_id}.txt").write_text(content)
        print(f"  Saved {doc.char_count} chars")
        time.sleep(0.5)

    metadata = [asdict(d) for d in docs]
    (RAW_DIR / "_metadata.json").write_text(json.dumps(metadata, indent=2))
    return docs


if __name__ == "__main__":
    docs = ingest_all()
    print(f"\nIngested {len(docs)}/{len(PAGES)} pages successfully.")
    total_chars = sum(d.char_count for d in docs)
    print(f"Total corpus size: {total_chars:,} characters (~{total_chars // 4:,} tokens)")