# # modules/validator.py

# import re
# from typing import List, Dict

# _num_pat  = re.compile(r"\b\d[\d\.,:/-]*\b")
# _sent_pat = re.compile(r"[^\.!?]+[\.!?]")  # naive sentence split

# def _present_in_evidence(needle: str, evidences: List[Dict]) -> int | None:
#     t = (needle or "").lower().strip()
#     if not t: return None
#     for i, e in enumerate(evidences, 1):
#         if t in (e.get("text") or "").lower():
#             return i
#     return None

# def _best_citation_for_sentence(s: str, evidences: List[Dict]) -> int | None:
#     s_low = (s or "").lower()
#     best_i, best_overlap = None, 0
#     for i, e in enumerate(evidences, 1):
#         et = (e.get("text") or "").lower()
#         overlap = 0
#         # cheap overlap: common numbers + shared tokens
#         for m in _num_pat.findall(s):
#             if m and m in et: overlap += 3
#         for w in set(re.findall(r"[a-z0-9]{3,}", s_low)):
#             if w in et: overlap += 1
#         if overlap > best_overlap:
#             best_overlap, best_i = overlap, i
#     return best_i if best_overlap > 0 else None

# def _attach_missing_citations(text: str, evidences: List[Dict], *, strict_all: bool) -> str:
#     out = []
#     for s in _sent_pat.findall(text):
#         s = s.strip()
#         if not s:
#             continue
#         if "[E" in s:
#             out.append(s)
#             continue
#         best = _best_citation_for_sentence(s, evidences)
#         if best:
#             out.append(f"{s} [E{best}]")
#         else:
#             # QA: flag; Long-form: keep fluent sentence unmodified
#             out.append(s if not strict_all else (s + " (not in the provided text)"))
#     return " ".join(out).strip()

# def _enforce_length(text: str, target_words: int) -> str:
#     if target_words <= 0: return text
#     words = text.split()
#     min_w = int(target_words * 0.85)
#     max_w = int(target_words * 1.15)
#     if len(words) > max_w:
#         return " ".join(words[:max_w])
#     return text

# def _ground_numbers_sentence_level(text: str, evidences: List[Dict]) -> str:
#     # If a sentence has numbers not present in evidence, tag once at sentence end.
#     out = []
#     for s in _sent_pat.findall(text):
#         s_clean = s.strip()
#         nums = _num_pat.findall(s_clean)
#         if not nums:
#             out.append(s_clean); continue
#         ok = any(_present_in_evidence(n, evidences) is not None for n in nums)
#         out.append(s_clean if ok else (s_clean + " [unverified number]"))
#     return " ".join(out)

# def validate_output(generated: str, evidences: List[Dict], *, mode: str, target_words: int) -> str:
#     """
#     - QA: strict citation enforcement per sentence, flags missing.
#     - Long forms (summary/review/novelty): lenient; preserve fluency; light number check; length envelope.
#     """
#     if not generated: return generated

#     # numbers checked at sentence level (don’t inject inside the number)
#     grounded = _ground_numbers_sentence_level(generated, evidences)

#     strict_all = (mode == "qa")
#     cited = _attach_missing_citations(grounded, evidences, strict_all=strict_all)

#     if mode in ("summary", "review", "novelty"):
#         cited = _enforce_length(cited, target_words)

#     return cited








# modules/validator.py

import re
from typing import List, Dict, Optional, Union

_num_pat  = re.compile(r"\b\d[\d\.,:/-]*\b")
_sent_pat = re.compile(r"[^\.!?]+[\.!?]")  # naive sentence split

# CHANGED: 'int | None' -> 'Optional[int]' for Python 3.9 compatibility
def _present_in_evidence(needle: str, evidences: List[Dict]) -> Optional[int]:
    t = (needle or "").lower().strip()
    if not t: return None
    for i, e in enumerate(evidences, 1):
        if t in (e.get("text") or "").lower():
            return i
    return None

# CHANGED: 'int | None' -> 'Optional[int]' for Python 3.9 compatibility
def _best_citation_for_sentence(s: str, evidences: List[Dict]) -> Optional[int]:
    s_low = (s or "").lower()
    best_i, best_overlap = None, 0
    for i, e in enumerate(evidences, 1):
        et = (e.get("text") or "").lower()
        overlap = 0
        # cheap overlap: common numbers + shared tokens
        for m in _num_pat.findall(s):
            if m and m in et: overlap += 3
        for w in set(re.findall(r"[a-z0-9]{3,}", s_low)):
            if w in et: overlap += 1
        if overlap > best_overlap:
            best_overlap, best_i = overlap, i
    return best_i if best_overlap > 0 else None

def _attach_missing_citations(text: str, evidences: List[Dict], *, strict_all: bool) -> str:
    out = []
    for s in _sent_pat.findall(text):
        s = s.strip()
        if not s:
            continue
        if "[E" in s:
            out.append(s)
            continue
        best = _best_citation_for_sentence(s, evidences)
        if best:
            out.append(f"{s} [E{best}]")
        else:
            # QA: flag; Long-form: keep fluent sentence unmodified
            out.append(s if not strict_all else (s + " (not in the provided text)"))
    return " ".join(out).strip()

def _enforce_length(text: str, target_words: int) -> str:
    if target_words <= 0: return text
    words = text.split()
    min_w = int(target_words * 0.85)
    max_w = int(target_words * 1.15)
    if len(words) > max_w:
        return " ".join(words[:max_w])
    return text

def _ground_numbers_sentence_level(text: str, evidences: List[Dict]) -> str:
    # If a sentence has numbers not present in evidence, tag once at sentence end.
    out = []
    for s in _sent_pat.findall(text):
        s_clean = s.strip()
        nums = _num_pat.findall(s_clean)
        if not nums:
            out.append(s_clean); continue
        ok = any(_present_in_evidence(n, evidences) is not None for n in nums)
        out.append(s_clean if ok else (s_clean + " [unverified number]"))
    return " ".join(out)

def validate_output(generated: str, evidences: List[Dict], *, mode: str, target_words: int) -> str:
    """
    - QA: strict citation enforcement per sentence, flags missing.
    - Long forms (summary/review/novelty): lenient; preserve fluency; light number check; length envelope.
    """
    if not generated: return generated

    # numbers checked at sentence level (don’t inject inside the number)
    grounded = _ground_numbers_sentence_level(generated, evidences)

    strict_all = (mode == "qa")
    cited = _attach_missing_citations(grounded, evidences, strict_all=strict_all)

    if mode in ("summary", "review", "novelty"):
        cited = _enforce_length(cited, target_words)

    return cited