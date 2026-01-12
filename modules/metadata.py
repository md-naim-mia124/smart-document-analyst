# modules/meta_extractor.py


from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

# Heuristics focusing on front-matter
DATE_PAT  = re.compile(r"(?i)\b(?:\d{4}|[A-Z][a-z]+ \d{4})\b")
UNI_PAT   = re.compile(r"(?i)\b(university|institute|college|faculty|school|department)\b.*")
AUTH_PAT  = re.compile(
    r"(?i)\bby\s+([A-Z][A-Za-z\.\-'\s]{2,})\b|"
    r"(?:author|candidate|student|submitted\s+by)\s*[:\-–]\s*([A-Z][A-Za-z\.\-'\s]{2,})"
)

def _clean_lines(text: str) -> List[str]:
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    return [l for l in lines if l]

def _is_titleish(line: str) -> bool:
    # looks like a big heading / thesis title
    words = line.split()
    if len(words) < 3:
        return False
    caps_ratio = sum(1 for c in line if c.isupper()) / max(1, sum(1 for c in line if c.isalpha()))
    keywords = re.search(r"(?i)\b(study|effect|implement|e-?governance|arsenic|biosand|filter|cloud|thesis|dissertation|project|report)\b", line)
    return caps_ratio > 0.35 or bool(keywords)

def _merge_adjacent_title_lines(lines: List[str], window: int = 10) -> Optional[str]:
    """
    Merge consecutive title-looking lines from the first N lines.
    Prefer the longest merged candidate (captures multi-line titles).
    """
    candidates: List[str] = []
    N = min(len(lines), window)
    i = 0
    while i < N:
        if _is_titleish(lines[i]):
            j = i
            buf = [lines[i]]
            # pull following lines that also look like part of a title (short or titleish)
            while j + 1 < N and ( _is_titleish(lines[j+1]) or len(lines[j+1].split()) <= 6 ):
                buf.append(lines[j+1]); j += 1
            merged = " ".join(buf)
            # strip leading noise like "THESIS ENTITLED:"
            merged = re.sub(r"(?i)\b(thesis|dissertation|project|report)\s*(entitled|title)?\s*[:\-–]\s*", "", merged).strip()
            candidates.append(merged)
            i = j + 1
        else:
            i += 1
    if not candidates:
        return None
    # prefer the longest reasonable one (avoid absurd 300+ chars)
    candidates = [c for c in candidates if 15 <= len(c) <= 200]
    return max(candidates, key=len) if candidates else None

def extract_meta(front_text: str) -> Dict[str, Optional[str]]:
    """
    Parse first ~2–3 pages of text. Return {title, author, university, date} when found.
    """
    out: Dict[str, Optional[str]] = {}
    lines = _clean_lines(front_text)

    # Title: try merged multi-line first, then fallback to first titleish line
    title_merged = _merge_adjacent_title_lines(lines[:50])
    if title_merged:
        out["title"] = title_merged
    else:
        for l in lines[:40]:
            if _is_titleish(l):
                out["title"] = re.sub(r"(?i)\b(thesis|dissertation|project|report)\s*(entitled|title)?\s*[:\-–]\s*", "", l).strip()
                break

    # University / Department
    for l in lines[:60]:
        if UNI_PAT.search(l):
            out["university"] = l
            break

    # Author
    for l in lines[:80]:
        m = AUTH_PAT.search(l)
        if m:
            out["author"] = (m.group(1) or m.group(2) or "").strip()
            break
        if re.match(r"(?i)^\s*(author|candidate|student)\s*[:\-–]\s*[A-Z][A-Za-z\-\.'\s]{2,}", l):
            out["author"] = re.sub(r"(?i)^\s*(author|candidate|student)\s*[:\-–]\s*", "", l).strip()
            break

    # Date
    for l in lines[:80]:
        m = DATE_PAT.search(l)
        if m:
            out["date"] = m.group(0)
            break

    return out

