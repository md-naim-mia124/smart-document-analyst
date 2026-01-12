# modules/doc_loader.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from pypdf import PdfReader
from docx import Document

def _clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text_from_pdf(path: Path, max_pages: Optional[int]) -> List[Tuple[int, str]]:
    """Extracts text page by page, returning a list of (page_num, page_text)."""
    reader = PdfReader(str(path))
    n_pages = len(reader.pages)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    pages: List[Tuple[int, str]] = []
    for i in range(n_pages):
        try:
            page_text = reader.pages[i].extract_text() or ""
            if page_text.strip():
                pages.append((i + 1, _clean(page_text)))
        except Exception:
            continue
    return pages

def extract_text_from_docx(path: Path) -> List[Tuple[int, str]]:
    """Extracts text from DOCX. Page numbers are not reliably available, so it returns as a single page."""
    doc = Document(str(path))
    full_text = "\n\n".join(_clean(p.text) for p in doc.paragraphs if p.text.strip())
    return [(1, full_text)]

def extract_text_from_txt(path: Path) -> List[Tuple[int, str]]:
    """Extracts text from TXT. Returns as a single page."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()
    return [(1, _clean(full_text))]


def extract_pages(file_path: str | Path, max_pages: Optional[int] = None) -> List[Tuple[int, str]]:
    """
    Extracts text and page numbers from a supported document.
    Returns a list of tuples: [(page_number, page_content), ...].
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path, max_pages)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext in {".txt", ".md"}:
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")