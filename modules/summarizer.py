# modules/summarizer.py


from __future__ import annotations
from typing import List, Dict, Optional

from modules.generator import generate
from modules.retriever import retrieve, retrieve_multi
from modules.embedder import Embedder
from modules.vectorstore import VectorStore

class Summarizer:
    """
    Evidence-first summarizer.
    - Keeps the last evidence used in `self.last_evidence` so the UI can show it.
    """
    def __init__(self, store: VectorStore, embedder: Embedder, full_text: str):
        self.store: VectorStore = store
        self.embedder: Embedder = embedder
        self.full_text: str = full_text or ""
        self.last_evidence: Optional[List[Dict]] = None  # populated on summarize()

    def _retrieve_overview(self) -> List[Dict]:
        """
        Retrieve evidence in multiple buckets to cover: context/purpose, methods, findings, limits/conclusion.
        """
        queries = [
            "abstract OR overview OR introduction OR background",
            "objective OR aim OR purpose",
            "method OR methodology OR materials and methods OR approach",
            "result OR findings OR evaluation OR analysis",
            "limitation OR future work OR conclusion"
        ]
        try:
            evid = retrieve_multi(
                queries, self.full_text, self.store, self.embedder,
                mode="summary", top_k=12, final_k=6, final_k_each=2
            )
        except TypeError:
            # Backward-compatible single-query fallback
            evid = retrieve("document overview: context, objectives, methods, findings, limits",
                            self.full_text, self.store, self.embedder,
                            top_k=12, final_k=6)
        return evid or []

    def summarize(
        self,
        target_words: int = 500,
        verbosity: str = "detailed",
        max_new_tokens: int = 900
    ) -> str:
        evid = self._retrieve_overview()
        self.last_evidence = evid  # record for the UI
        return generate(
            "summary",
            "document overview",
            evidences=evid,
            target_words=target_words,
            verbosity=verbosity,
            strict=True,
            max_tokens=max_new_tokens
        )
