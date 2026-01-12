#modules\citations.py

import re
from typing import List, Dict, Tuple
from modules.bm25 import BM25

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_NUM_PAT    = re.compile(r"\b\d[\d\.,:/-]*\b")

def _split_sentences(text: str) -> List[str]:
    s = text.strip()
    if not s:
        return []
    parts = _SENT_SPLIT.split(s)
    # merge very short fragments
    out, buf = [], ""
    for p in parts:
        if len(p.split()) < 3:
            buf = (buf + " " + p).strip()
            continue
        if buf:
            out.append(buf); buf = ""
        out.append(p.strip())
    if buf:
        out.append(buf)
    return out

def _best_evidence_idx(sentence: str, bm25: BM25) -> Tuple[int, float]:
    # BM25.topn returns list[(idx, score)]
    top = bm25.topn(sentence, 1)
    if not top:
        return -1, 0.0
    return top[0]

def attach_inline_citations(generated: str, evidences: List[Dict]) -> str:
    if not generated.strip():
        return generated
    ev_texts = [(e.get("text") or "").replace("\n", " ").strip() for e in evidences]
    if not any(ev_texts):
        return generated

    bm = BM25(ev_texts)
    sents = _split_sentences(generated)
    cited = []
    # estimate a dynamic threshold by sampling a simple query
    baseline_score = 0.0
    if ev_texts:
        probe = " ".join(ev_texts[0].split()[:12])
        _, baseline_score = _best_evidence_idx(probe, bm)
    # compose
    for sent in sents:
        s_clean = sent.strip()
        if not s_clean:
            continue
        # skip if already has [E#]
        if re.search(r"\[E\d+\]", s_clean):
            cited.append(s_clean); continue
        idx, score = _best_evidence_idx(s_clean, bm)
        # accept if meaningful or sentence contains numbers
        if idx >= 0 and (score >= (0.25 * max(baseline_score, 1e-6)) or len(s_clean.split()) >= 6):
            cited.append(f"{s_clean} [E{idx+1}]")
        else:
            cited.append(s_clean + " (not in the provided text)")
    return " ".join(cited)
