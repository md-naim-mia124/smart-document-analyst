# # modules/retriever.py


# from __future__ import annotations
# from typing import List, Dict, Tuple
# import numpy as np
# import re
# from collections import defaultdict
# from sentence_transformers import CrossEncoder
# import faiss

# from modules.bm25 import BM25
# from modules.vectorstore import VectorStore
# from modules.embedder import Embedder
# from config.settings import RAG_TOP_K, ALPHA_DENSE, MMR_LAMBDA, FINAL_E_BLOCKS

# try:
#     cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# except Exception as e:
#     print(f"Warning: Could not load CrossEncoder model. Re-ranking will be disabled. Error: {e}")
#     cross_encoder = None

# # --- All helper functions remain the same ---
# _WS = re.compile(r"\s+")
# _NONWORD = re.compile(r"[^A-Za-z0-9]+")
# def _nz(s: str) -> str: return _WS.sub(" ", (s or "").strip())
# def _words(s: str) -> set: return {t for t in _NONWORD.split((s or "").lower()) if len(t) > 2}
# _SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")
# def _sentences(s: str) -> List[str]:
#     s = _nz(s)
#     if not s: return []
#     parts = _SENT_SPLIT.split(s)
#     out, buf = [], ""
#     for p in parts:
#         if len(buf) + len(p) < 180: buf = (buf + " " + p).strip() if buf else p
#         else:
#             if buf: out.append(buf)
#             buf = p
#     if buf: out.append(buf)
#     return out
# def _jaccard(a: set, b: set) -> float:
#     if not a or not b: return 0.0
#     inter = len(a & b); union = len(a | b)
#     return inter / (union + 1e-12)
# _SYN = _words("context objective aim purpose method methodology approach materials results findings evaluation analysis discussion limitation future conclusion")
# def _slice_relevant(text: str, query: str, *, max_chars: int = 400) -> str:
#     qw = _words(query)
#     sents = _sentences(text)
#     if not sents: return text or ""
#     scored = sorted([(0.7*_jaccard(qw,_words(s)) + 0.3*_jaccard(_SYN,_words(s)) + (0.05 if re.search(r"\d",s) else 0), i, s) for i,s in enumerate(sents)], reverse=True)
#     chosen, total = [], 0
#     for _, _, s in scored:
#         if not s or s in chosen: continue
#         if total + len(s) + 1 > max_chars:
#             if not chosen: chosen.append(s[:max_chars].rstrip() + "…")
#             break
#         chosen.append(s); total += len(s) + 1
#         if len(chosen) >= 3: break
#     return " ".join(chosen).strip()
# def _normalize_scores(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
#     if not pairs: return pairs
#     vals = [s for _, s in pairs]
#     lo, hi = min(vals), max(vals)
#     if hi - lo < 1e-9: return [(i, 1.0) for i, _ in pairs]
#     return [(i, (s - lo) / (hi - lo + 1e-12)) for i, s in pairs]
# def _mmr(candidates: List[int], cand_embs: np.ndarray, q_emb: np.ndarray, k: int, lamb: float = 0.3) -> List[int]:
#     if not candidates or len(candidates) <= k: return candidates
#     C = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12)
#     q = q_emb.flatten(); q = q / (np.linalg.norm(q) + 1e-12)
#     simq = C @ q
#     selected, remaining = [], set(range(len(candidates)))
#     while len(selected) < k and remaining:
#         if not selected: j = max(remaining, key=lambda r: simq[r])
#         else: j = max(remaining, key=lambda r: lamb * simq[r] - (1-lamb) * max(C[r] @ C[s] for s in selected))
#         selected.append(j); remaining.remove(j)
#     return [candidates[j] for j in selected]
# def _rerank_with_cross_encoder(query: str, candidates: List[Tuple[int, str]]) -> List[int]:
#     if not cross_encoder or not query or not candidates: return [idx for idx, _ in candidates]
#     pairs = [[query, text] for _, text in candidates]
#     scores = cross_encoder.predict(pairs)
#     return [idx for idx, _ in sorted(zip([idx for idx, _ in candidates], scores), key=lambda x: x[1], reverse=True)]
# def _get_unique_sections(chunks: List[Dict]) -> List[str]:
#     return sorted(list({c.get("section", "Unknown") for c in chunks if c.get("section")}))

# def _section_weight(query: str, chunk_meta: Dict) -> float:
#     chunk_section = (chunk_meta.get("section") or "").lower()
#     q = query.lower()
#     if any(k in q for k in ["conclusion", "summary"]) and "conclusion" in chunk_section: return 2.0
#     if any(k in q for k in ["introduction", "background"]) and "introduction" in chunk_section: return 1.8
#     if any(k in q for k in ["method", "approach"]) and "method" in chunk_section: return 1.6
#     if any(k in q for k in ["result", "finding"]) and "result" in chunk_section: return 1.6
#     if any(k in q for k in ["future work", "limitation"]) and ("future" in chunk_section or "limitation" in chunk_section): return 1.8
#     if any(k in q for k in ["novelty", "contribution"]) and ("introduction" in chunk_section or "objective" in chunk_section): return 1.5
#     if any(p in chunk_section for p in ("abstract", "conclusion", "results", "method")): return 1.15
#     if any(n in chunk_section for n in ("acknowledg", "reference", "declaration")): return 0.2
#     return 1.0

# def _search_within_chunks(
#     question: str, store: VectorStore, embedder: Embedder, chunk_indices_to_search: List[int], *,
#     final_k: int = FINAL_E_BLOCKS
# ) -> List[Dict]:
#     if not chunk_indices_to_search: return []
#     q, q_emb = (question or "").strip(), embedder.encode([question or ""])
    
#     sub_chunks = {original_idx: store.get_chunk(original_idx) for original_idx in chunk_indices_to_search}
#     sub_texts = [chunk['text'] for chunk in sub_chunks.values()]
#     sub_embeddings = embedder.encode(sub_texts)
    
#     # Dense search on the subset
#     sub_index = faiss.IndexFlatIP(sub_embeddings.shape[1]); sub_index.add(sub_embeddings)
#     D, I = sub_index.search(q_emb.astype("float32"), k=min(len(sub_chunks), RAG_TOP_K * 2))
    
#     original_indices_list = list(sub_chunks.keys())
#     dense_pairs = [(original_indices_list[i], s) for i, s in zip(I[0], D[0]) if i != -1]
    
#     # BM25 on the subset
#     bm25 = BM25(sub_texts or [""])
#     bm25_local_results = bm25.topn(q, min(len(sub_texts), RAG_TOP_K * 2))
#     bm25_pairs = [(original_indices_list[i], s) for i, s in bm25_local_results]

#     # Blending and Re-ranking
#     union = defaultdict(float)
#     dmap = dict(_normalize_scores(dense_pairs)); bmap = dict(_normalize_scores(bm25_pairs))
#     for i in set(dmap.keys()).union(bmap.keys()):
#         score = ALPHA_DENSE * dmap.get(i, 0.0) + (1 - ALPHA_DENSE) * bmap.get(i, 0.0)
#         score *= _section_weight(q, store.get_chunk(i))
#         union[i] = score
    
#     sorted_candidates = sorted(union.items(), key=lambda x: x[1], reverse=True)
    
#     rerank_candidates = [(idx, store.get_chunk(idx)["text"]) for idx, _ in sorted_candidates[:RAG_TOP_K]]
#     reranked_indices = _rerank_with_cross_encoder(q, rerank_candidates)
    
#     # MMR and final evidence prep
#     cand_embs = embedder.encode([store.get_chunk(i)["text"] for i in reranked_indices]) if reranked_indices else []
#     chosen_indices = _mmr(reranked_indices, cand_embs, q_emb, k=final_k) if reranked_indices else []
    
#     evid, seen = [], set()
#     for idx in chosen_indices:
#         meta = store.get_chunk(idx)
#         snippet = _slice_relevant(meta.get("text", ""), q)
#         if not snippet or snippet.lower() in seen: continue
#         seen.add(snippet.lower())
#         evid.append({"id": idx, "text": snippet, "page": meta.get("page"), "section": meta.get("section")})
#     return evid

# def retrieve(
#     question: str, store: VectorStore, embedder: Embedder, *, final_k: int = FINAL_E_BLOCKS
# ) -> List[Dict]:
#     q_words = _words(question or "")
#     unique_sections = _get_unique_sections(store.chunks)
    
#     scored_sections = sorted([(_jaccard(q_words, _words(name)), name) for name in unique_sections], reverse=True)
    
#     # Select top sections, giving a bonus to sections that are substrings of the query
#     top_sections = {name for score, name in scored_sections if score > 0.1}
#     for sec in unique_sections:
#         if sec.lower() in question.lower(): top_sections.add(sec)
    
#     targeted_chunk_indices = [i for i, chunk in enumerate(store.chunks) if chunk.get("section") in top_sections] if top_sections else []

#     evidence = _search_within_chunks(question, store, embedder, targeted_chunk_indices, final_k=final_k)

#     # Fallback to global search only if targeted search is insufficient
#     if len(evidence) < final_k // 2:
#         global_evidence = _search_within_chunks(question, store, embedder, list(range(len(store.chunks))), final_k=final_k)
#         # Merge results, prioritizing evidence from the targeted search
#         merged = evidence + [item for item in global_evidence if item['id'] not in {e['id'] for e in evidence}]
#         evidence = merged[:final_k]

#     return evidence







from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import re
import os
from collections import defaultdict
from sentence_transformers import CrossEncoder
import faiss

from modules.bm25 import BM25
from modules.vectorstore import VectorStore
from modules.embedder import Embedder
from config.settings import RAG_TOP_K, ALPHA_DENSE, MMR_LAMBDA, FINAL_E_BLOCKS, PROJECT_ROOT

# --- Robust Cross-Encoder Loading (Auto-detects Local vs Online) ---
try:
    # 1. Try Local Path (Best for Docker/Offline)
    # This matches the folder name you just downloaded
    local_ce_path = os.path.join(PROJECT_ROOT, "models", "cross-encoder_ms-marco-MiniLM-L-6-v2")
    
    if os.path.exists(local_ce_path):
        print(f"[Retriever] Loading CrossEncoder from local: {local_ce_path}")
        cross_encoder = CrossEncoder(local_ce_path)
    else:
        # 2. Fallback to HuggingFace (Requires Internet)
        print("[Retriever] Local CrossEncoder not found. Downloading from HuggingFace...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

except Exception as e:
    print(f"Warning: Could not load CrossEncoder model. Re-ranking will be disabled. Error: {e}")
    cross_encoder = None

# --- All helper functions remain exactly the same ---
_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^A-Za-z0-9]+")

def _nz(s: str) -> str: return _WS.sub(" ", (s or "").strip())
def _words(s: str) -> set: return {t for t in _NONWORD.split((s or "").lower()) if len(t) > 2}
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")

def _sentences(s: str) -> List[str]:
    s = _nz(s)
    if not s: return []
    parts = _SENT_SPLIT.split(s)
    out, buf = [], ""
    for p in parts:
        if len(buf) + len(p) < 180: buf = (buf + " " + p).strip() if buf else p
        else:
            if buf: out.append(buf)
            buf = p
    if buf: out.append(buf)
    return out

def _jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / (union + 1e-12)

_SYN = _words("context objective aim purpose method methodology approach materials results findings evaluation analysis discussion limitation future conclusion")

def _slice_relevant(text: str, query: str, *, max_chars: int = 400) -> str:
    qw = _words(query)
    sents = _sentences(text)
    if not sents: return text or ""
    scored = sorted([(0.7*_jaccard(qw,_words(s)) + 0.3*_jaccard(_SYN,_words(s)) + (0.05 if re.search(r"\d",s) else 0), i, s) for i,s in enumerate(sents)], reverse=True)
    chosen, total = [], 0
    for _, _, s in scored:
        if not s or s in chosen: continue
        if total + len(s) + 1 > max_chars:
            if not chosen: chosen.append(s[:max_chars].rstrip() + "…")
            break
        chosen.append(s); total += len(s) + 1
        if len(chosen) >= 3: break
    return " ".join(chosen).strip()

def _normalize_scores(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    if not pairs: return pairs
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9: return [(i, 1.0) for i, _ in pairs]
    return [(i, (s - lo) / (hi - lo + 1e-12)) for i, s in pairs]

def _mmr(candidates: List[int], cand_embs: np.ndarray, q_emb: np.ndarray, k: int, lamb: float = 0.3) -> List[int]:
    if not candidates or len(candidates) <= k: return candidates
    C = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12)
    q = q_emb.flatten(); q = q / (np.linalg.norm(q) + 1e-12)
    simq = C @ q
    selected, remaining = [], set(range(len(candidates)))
    while len(selected) < k and remaining:
        if not selected: j = max(remaining, key=lambda r: simq[r])
        else: j = max(remaining, key=lambda r: lamb * simq[r] - (1-lamb) * max(C[r] @ C[s] for s in selected))
        selected.append(j); remaining.remove(j)
    return [candidates[j] for j in selected]

def _rerank_with_cross_encoder(query: str, candidates: List[Tuple[int, str]]) -> List[int]:
    if not cross_encoder or not query or not candidates: return [idx for idx, _ in candidates]
    pairs = [[query, text] for _, text in candidates]
    try:
        scores = cross_encoder.predict(pairs)
        return [idx for idx, _ in sorted(zip([idx for idx, _ in candidates], scores), key=lambda x: x[1], reverse=True)]
    except Exception as e:
        print(f"Rerank failed: {e}")
        return [idx for idx, _ in candidates]

def _get_unique_sections(chunks: List[Dict]) -> List[str]:
    return sorted(list({c.get("section", "Unknown") for c in chunks if c.get("section")}))

def _section_weight(query: str, chunk_meta: Dict) -> float:
    chunk_section = (chunk_meta.get("section") or "").lower()
    q = query.lower()
    if any(k in q for k in ["conclusion", "summary"]) and "conclusion" in chunk_section: return 2.0
    if any(k in q for k in ["introduction", "background"]) and "introduction" in chunk_section: return 1.8
    if any(k in q for k in ["method", "approach"]) and "method" in chunk_section: return 1.6
    if any(k in q for k in ["result", "finding"]) and "result" in chunk_section: return 1.6
    if any(k in q for k in ["future work", "limitation"]) and ("future" in chunk_section or "limitation" in chunk_section): return 1.8
    if any(k in q for k in ["novelty", "contribution"]) and ("introduction" in chunk_section or "objective" in chunk_section): return 1.5
    if any(p in chunk_section for p in ("abstract", "conclusion", "results", "method")): return 1.15
    if any(n in chunk_section for n in ("acknowledg", "reference", "declaration")): return 0.2
    return 1.0

def _search_within_chunks(
    question: str, store: VectorStore, embedder: Embedder, chunk_indices_to_search: List[int], *,
    final_k: int = FINAL_E_BLOCKS
) -> List[Dict]:
    if not chunk_indices_to_search: return []
    q, q_emb = (question or "").strip(), embedder.encode([question or ""])
    
    sub_chunks = {original_idx: store.get_chunk(original_idx) for original_idx in chunk_indices_to_search}
    sub_texts = [chunk['text'] for chunk in sub_chunks.values()]
    sub_embeddings = embedder.encode(sub_texts)
    
    # Dense search on the subset
    sub_index = faiss.IndexFlatIP(sub_embeddings.shape[1]); sub_index.add(sub_embeddings)
    D, I = sub_index.search(q_emb.astype("float32"), k=min(len(sub_chunks), RAG_TOP_K * 2))
    
    original_indices_list = list(sub_chunks.keys())
    dense_pairs = [(original_indices_list[i], s) for i, s in zip(I[0], D[0]) if i != -1]
    
    # BM25 on the subset
    bm25 = BM25(sub_texts or [""])
    bm25_local_results = bm25.topn(q, min(len(sub_texts), RAG_TOP_K * 2))
    bm25_pairs = [(original_indices_list[i], s) for i, s in bm25_local_results]

    # Blending and Re-ranking
    union = defaultdict(float)
    dmap = dict(_normalize_scores(dense_pairs)); bmap = dict(_normalize_scores(bm25_pairs))
    for i in set(dmap.keys()).union(bmap.keys()):
        score = ALPHA_DENSE * dmap.get(i, 0.0) + (1 - ALPHA_DENSE) * bmap.get(i, 0.0)
        score *= _section_weight(q, store.get_chunk(i))
        union[i] = score
    
    sorted_candidates = sorted(union.items(), key=lambda x: x[1], reverse=True)
    
    rerank_candidates = [(idx, store.get_chunk(idx)["text"]) for idx, _ in sorted_candidates[:RAG_TOP_K]]
    reranked_indices = _rerank_with_cross_encoder(q, rerank_candidates)
    
    # MMR and final evidence prep
    cand_embs = embedder.encode([store.get_chunk(i)["text"] for i in reranked_indices]) if reranked_indices else []
    chosen_indices = _mmr(reranked_indices, cand_embs, q_emb, k=final_k) if reranked_indices else []
    
    evid, seen = [], set()
    for idx in chosen_indices:
        meta = store.get_chunk(idx)
        snippet = _slice_relevant(meta.get("text", ""), q)
        if not snippet or snippet.lower() in seen: continue
        seen.add(snippet.lower())
        evid.append({"id": idx, "text": snippet, "page": meta.get("page"), "section": meta.get("section")})
    return evid

def retrieve(
    question: str, store: VectorStore, embedder: Embedder, *, final_k: int = FINAL_E_BLOCKS
) -> List[Dict]:
    q_words = _words(question or "")
    unique_sections = _get_unique_sections(store.chunks)
    
    scored_sections = sorted([(_jaccard(q_words, _words(name)), name) for name in unique_sections], reverse=True)
    
    # Select top sections, giving a bonus to sections that are substrings of the query
    top_sections = {name for score, name in scored_sections if score > 0.1}
    for sec in unique_sections:
        if sec.lower() in question.lower(): top_sections.add(sec)
    
    targeted_chunk_indices = [i for i, chunk in enumerate(store.chunks) if chunk.get("section") in top_sections] if top_sections else []

    evidence = _search_within_chunks(question, store, embedder, targeted_chunk_indices, final_k=final_k)

    # Fallback to global search only if targeted search is insufficient
    if len(evidence) < final_k // 2:
        global_evidence = _search_within_chunks(question, store, embedder, list(range(len(store.chunks))), final_k=final_k)
        # Merge results, prioritizing evidence from the targeted search
        merged = evidence + [item for item in global_evidence if item['id'] not in {e['id'] for e in evidence}]
        evidence = merged[:final_k]

    return evidence

# --- ADDED: retrieve_multi (Required by Summarizer) ---
def retrieve_multi(
    queries: List[str], 
    full_text: str, # Kept for compatibility
    store: VectorStore, 
    embedder: Embedder, 
    *, 
    mode: str = "summary", 
    top_k: int = 12, 
    final_k: int = 6, 
    final_k_each: int = 2
) -> List[Dict]:
    """
    Runs multiple queries and merges unique evidence chunks.
    Used specifically by the Summarizer to cover different aspects (methods, results, etc.).
    """
    unique_evidence = {}
    
    for q in queries:
        # Run standard retrieve for each sub-query
        results = retrieve(q, store, embedder, final_k=final_k_each)
        for item in results:
            if item['id'] not in unique_evidence:
                unique_evidence[item['id']] = item

    # Convert back to list
    combined_evidence = list(unique_evidence.values())

    # Fallback: if we found very little, do a generic broad search
    if len(combined_evidence) < 3:
        fallback = retrieve("summary main points overview", store, embedder, final_k=final_k)
        for item in fallback:
            if item['id'] not in unique_evidence:
                combined_evidence.append(item)
                unique_evidence[item['id']] = item

    # Limit to final_k
    return combined_evidence[:final_k]