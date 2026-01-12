# modules/toc_extractor.py

import re
from typing import List, Dict
import fitz  # PyMuPDF

def extract_toc(file_path: str) -> List[Dict]:
    """
    Extracts a Table of Contents from a PDF by analyzing font sizes.
    Headings are identified as text with a larger font size than the main body text.
    """
    toc = []
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF for TOC extraction: {e}")
        return []

    font_counts = {}
    # First pass: Determine the most common font size (likely body text)
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if b.get('type') == 0:  # It's a text block
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        size = round(s["size"])
                        font_counts[size] = font_counts.get(size, 0) + len(s.get("text", ""))
    
    if not font_counts:
        return []

    body_font_size = max(font_counts, key=font_counts.get)
    
    # Second pass: Extract headings
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if b.get('type') == 0:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        span_size = round(s["size"])
                        # A heading is larger than body text, not just a number, and has some length
                        if span_size > body_font_size and not s["text"].strip().isdigit() and len(s["text"].strip()) > 2:
                            title = s["text"].strip()
                            
                            # Basic hierarchy based on how much larger the font is
                            level = int((span_size - body_font_size) / 2)

                            # Avoid adding duplicates
                            if not any(item['title'] == title for item in toc):
                                toc.append({
                                    "title": title,
                                    "level": level,
                                    "page": page_num + 1,
                                    "children": []
                                })
    return toc