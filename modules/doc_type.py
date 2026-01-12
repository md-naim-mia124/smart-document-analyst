#modules\doc_type.py

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

HEAD_LIMIT = 5000

def _coerce_text(obj: Union[str, Tuple[str, Any], Dict[str, Any]]) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, tuple) and len(obj) >= 1:
        return obj[0] or ""
    if isinstance(obj, dict):
        return obj.get("text", "") or ""
    return str(obj)

def _has(headers: str, *terms: List[str]) -> bool:
    return any(t.lower() in headers for t in terms)

def detect_doc_type(text_or_tuple: Union[str, Tuple[str, Any], Dict[str, Any]], path: Optional[str] = None) -> Dict[str, Any]:
    """
    Robust detector:
      - accepts raw text, (text, page_map), or {'text': ..., 'page_map': ...}
      - returns {'type': str, 'confidence': float, 'hints': List[str], 'pages': Optional[int]}
    """
    text = _coerce_text(text_or_tuple)
    head = (text[:HEAD_LIMIT] if isinstance(text, str) else str(text)[:HEAD_LIMIT]).lower()

    hints: List[str] = []
    pages = None

    # Try to sniff page count from file name patterns like "... p. 123"
    if path:
        base = os.path.basename(path)
        m = re.search(r'(\d{1,4})\s*(pages|pp|p)\b', base.lower())
        if m:
            try:
                pages = int(m.group(1))
            except Exception:
                pages = None

    # Academic-like signals
    if "abstract" in head:
        hints.append("found 'Abstract'")
    if "introduction" in head:
        hints.append("found 'Introduction'")
    if any(k in head for k in ["method", "materials and methods", "methodology"]):
        hints.append("found 'Methodology'")
    if "results" in head:
        hints.append("found 'Results'")
    if "references" in head or "bibliography" in head:
        hints.append("found 'References'")

    # Heuristic classification
    if any(hints) and _has(head, "abstract", "introduction", "references"):
        return {"type": "academic_article", "confidence": 1.0, "hints": hints, "pages": pages}

    # Reports / manuals
    if _has(head, "executive summary", "table of contents", "glossary"):
        return {"type": "report_or_manual", "confidence": 0.8, "hints": hints, "pages": pages}

    # Presentations
    if _has(head, "agenda", "slide", "deck"):
        return {"type": "presentation", "confidence": 0.6, "hints": hints, "pages": pages}

    # Default
    return {"type": "document", "confidence": 0.4, "hints": hints, "pages": pages}

