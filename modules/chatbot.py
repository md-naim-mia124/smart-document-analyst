#modules\chatbot


from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import streamlit as st

from config.settings import TOP_K, FINAL_E_BLOCKS
from modules.embedder import Embedder
from modules.vectorstore import VectorStore
from modules.llm_interface import get_llm
from modules.retriever import retrieve
from modules.generator import answer_q_and_a_decomposed, generate_novelty_analysis, generate_summary_map_reduce
from modules.validator import validate_output
from modules.doc_loader import extract_pages
from modules.doc_type import detect_doc_type
from modules.timer import timer
from modules.metadata import extract_meta

@st.cache_resource(show_spinner=False)
def _get_embedder_cached() -> Embedder:
    return Embedder()

@st.cache_resource(show_spinner=False)
def _get_llm_cached():
    return get_llm()

class Chatbot:
    def __init__(self, doc_id: str, file_path: str):
        self.doc_id = doc_id
        self.file_path = file_path
        self.store = VectorStore(doc_id)
        if not self.store.load():
            raise ValueError(f"No index found for '{doc_id}'. Please build or re-upload the document.")
        self.embedder = _get_embedder_cached()
        self.llm = _get_llm_cached()
        pages = extract_pages(self.file_path)
        self.doc_text = "\n\n".join(p_text for _, p_text in pages)
        self.doc_type = detect_doc_type(self.doc_text, self.file_path)
        self.meta = extract_meta(self.doc_text[:4000])


    def _intent(self, q: str) -> str:
        qt = q.lower().strip()
        if any(k in qt for k in ["summarize", "summary", "review", "critique"]): return "summary_review"
        if any(k in qt for k in ["novelty", "contribution"]): return "novelty"
        return "qa"


    def _is_title_question(self, q: str) -> bool:
        t = q.lower()
        return ("title" in t or "name" in t) and any(k in t for k in ["thesis", "paper", "document"])

    def _emit_evidence_panel(self, evid: List[Dict]):
        st.session_state["__last_evidence__"] = evid

    @timer
    def ask(self, question: str, *, max_new_tokens: int, target_words: int, verbosity: str, strict: bool, progress_placeholder=None):
        q = (question or "").strip()
        mode = self._intent(q)


        if self._is_title_question(q) and (self.meta.get("title") or "").strip():
            title_txt = self.meta["title"].strip()
            evid = retrieve(title_txt, self.store, self.embedder, final_k=2)
            raw = f"The document title appears to be: **{title_txt}**."
            out = validate_output(raw, evid, mode="qa", target_words=60)
            self._emit_evidence_panel(evid)
            return out


        
        if mode == "summary_review":
            task = "Create a comprehensive summary"
            if "review" in q:
                task = "Create a critical review with sections for Introduction, Methods, Results, and Conclusion"
            all_chunks = self.store.chunks
            raw_text, all_evidence = generate_summary_map_reduce(
                task, all_chunks, 
                target_words=target_words, max_new_tokens=max_new_tokens,
                progress_placeholder=progress_placeholder
            )
            out = validate_output(raw_text, all_evidence, mode="summary", target_words=target_words)
            self._emit_evidence_panel(all_evidence)
            return out

        # -------------------------------------------------------------------

        # Q&A mode
        if mode == "qa":
            raw_text, evid = answer_q_and_a_decomposed(question, self.store, self.embedder)
            out = validate_output(raw_text, evid, mode=mode, target_words=target_words)
            self._emit_evidence_panel(evid)
            return out
        
        # Novelty analysis mode
        if mode == "novelty":
            raw_text, evid = generate_novelty_analysis(
                question, self.store, self.embedder,
                target_words=target_words, max_new_tokens=max_new_tokens
            )
            out = validate_output(raw_text, evid, mode=mode, target_words=target_words)
            self._emit_evidence_panel(evid)
            return out
        
        return "Could not determine the correct action for the query."
