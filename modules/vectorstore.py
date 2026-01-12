# modules/vectorstore.py

# import os, pickle
# import numpy as np
# import faiss
# from typing import List, Dict
# from config.settings import INDEX_DIR
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def _ensure_dir(p):
#     os.makedirs(os.path.dirname(p), exist_ok=True)

# class VectorStore:
#     def __init__(self, doc_id: str):
#         safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)
#         self.index_path = os.path.join(INDEX_DIR, f"{safe}_index.faiss")
#         self.meta_path  = os.path.join(INDEX_DIR, f"{safe}_meta.pkl")
#         self.index = None
#         self.chunks: List[Dict] = []

#     def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
#         """Uses a robust recursive splitter to better preserve semantic boundaries."""
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""],
#         )
#         return text_splitter.split_text(text)

#     def build_index(self, embeddings: np.ndarray, chunk_dicts: List[Dict]):
#         """Builds index from embeddings and a list of chunk dictionaries with metadata."""
#         dim = embeddings.shape[1]
#         _ensure_dir(self.index_path)
        
#         norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
#         vecs  = (embeddings / norms).astype("float32")
        
#         self.index = faiss.IndexFlatIP(dim)
#         self.index.add(vecs)
        
#         self.chunks = chunk_dicts
#         self.save()

#     def save(self):
#         _ensure_dir(self.index_path)
#         faiss.write_index(self.index, self.index_path)
#         with open(self.meta_path, "wb") as f:
#             pickle.dump(self.chunks, f)

#     def load(self) -> bool:
#         if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
#             self.index  = faiss.read_index(self.index_path)
#             with open(self.meta_path, "rb") as f:
#                 self.chunks = pickle.load(f)
#             return True
#         return False

#     def search_dense(self, q_emb: np.ndarray, top_k: int = 10):
#         if self.index is None:
#             raise RuntimeError("Index not built/loaded.")
#         D, I = self.index.search(q_emb.astype("float32"), top_k)
#         res = []
#         for score, idx in zip(D[0].tolist(), I[0].tolist()):
#             if 0 <= idx < len(self.chunks):
#                 res.append((idx, float(score)))
#         return res

#     def get_chunk(self, idx: int) -> Dict:
#         return self.chunks[idx]





# aws serializing faild

# # modules/vectorestore.py
# import os, json
# import numpy as np
# import faiss
# from typing import List, Dict
# from config.settings import INDEX_DIR
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def _ensure_dir(p):
#     os.makedirs(os.path.dirname(p), exist_ok=True)

# class VectorStore:
#     def __init__(self, doc_id: str):
#         safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)
#         self.index_path = os.path.join(INDEX_DIR, f"{safe}_index.faiss")
#         self.meta_path  = os.path.join(INDEX_DIR, f"{safe}_meta.json")
#         self.index = None
#         self.chunks: List[Dict] = []

#     def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
#         """Uses a robust recursive splitter to better preserve semantic boundaries."""
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""],
#         )
#         return text_splitter.split_text(text)

#     def build_index(self, embeddings: np.ndarray, chunk_dicts: List[Dict]):
#         """Builds index from embeddings and saves metadata as JSON to save memory."""
#         dim = embeddings.shape[1]
#         _ensure_dir(self.index_path)
        
#         norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
#         vecs  = (embeddings / norms).astype("float32")
        
#         self.index = faiss.IndexFlatIP(dim)
#         self.index.add(vecs)
        
#         self.chunks = chunk_dicts
#         self.save()

#     def save(self):
#         _ensure_dir(self.index_path)
#         faiss.write_index(self.index, self.index_path)
#         # JSON is safer for low RAM environments like t3.micro
#         with open(self.meta_path, "w", encoding="utf-8") as f:
#             json.dump(self.chunks, f, ensure_ascii=False)

#     def load(self) -> bool:
#         if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
#             try:
#                 self.index = faiss.read_index(self.index_path)
#                 with open(self.meta_path, "r", encoding="utf-8") as f:
#                     self.chunks = json.load(f)
#                 return True
#             except Exception as e:
#                 print(f"Index loading failed: {e}")
#                 return False
#         return False

#     def search_dense(self, q_emb: np.ndarray, top_k: int = 10):
#         if self.index is None:
#             raise RuntimeError("Index not built/loaded.")
#         D, I = self.index.search(q_emb.astype("float32"), top_k)
#         res = []
#         for score, idx in zip(D[0].tolist(), I[0].tolist()):
#             if 0 <= idx < len(self.chunks):
#                 res.append((idx, float(score)))
#         return res

#     def get_chunk(self, idx: int) -> Dict:
#         return self.chunks[idx]






#chatgpt solution---->
# modules/vectorstore.py

import os
import json
from typing import List, Dict, Tuple

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import INDEX_DIR

# =========================
# SAFETY CONSTANTS
# =========================
DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 50

MAX_CONTEXT_CHARS = 3000      # 🔒 hard LLM payload cap
MAX_PREVIEW_CHARS = 200       # 🔒 evidence preview only


# =========================
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# =========================
class VectorStore:
    def __init__(self, doc_id: str):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)

        self.index_path = os.path.join(INDEX_DIR, f"{safe}.faiss")
        self.meta_path  = os.path.join(INDEX_DIR, f"{safe}.json")

        self.index: faiss.Index | None = None
        self.chunks: List[Dict] = []

    # =========================
    # TEXT SPLITTING (SAFE)
    # =========================
    def split(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "],
        )
        return splitter.split_text(text)

    # =========================
    # BUILD VECTOR INDEX
    # =========================
    def build_index(
        self,
        embeddings: np.ndarray,
        chunk_dicts: List[Dict],
    ):
        """
        chunk_dicts structure (MANDATORY):
        {
            "id": int,
            "page": int,
            "source": str,
            "text": str   # FULL text stored ONLY here
        }
        """

        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings are empty")

        dim = embeddings.shape[1]
        _ensure_dir(self.index_path)

        # Normalize for cosine similarity (IP index)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        vectors = (embeddings / norms).astype("float32")

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        self.chunks = chunk_dicts
        self._save()

    # =========================
    def _save(self):
        _ensure_dir(self.index_path)
        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    # =========================
    def load(self) -> bool:
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            return False

        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            return True
        except Exception as e:
            print(f"[VectorStore] load failed: {e}")
            return False

    # =========================
    # VECTOR SEARCH
    # =========================
    def search(
        self,
        q_embedding: np.ndarray,
        top_k: int = 6,
    ) -> List[Tuple[int, float]]:

        if self.index is None:
            raise RuntimeError("Index not built or loaded")

        D, I = self.index.search(q_embedding.astype("float32"), top_k)

        results: List[Tuple[int, float]] = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.chunks):
                results.append((int(idx), float(score)))

        return results

    # =========================
    # SAFE CONTEXT + EVIDENCE BUILDER
    # =========================
    def build_llm_context(
        self,
        search_results: List[Tuple[int, float]],
    ) -> Dict:
        """
        Returns:
        {
            "context": str,      # sent to LLM
            "evidence": list     # shown in UI
        }
        """

        context_text = ""
        evidence = []

        for idx, score in search_results:
            chunk = self.chunks[idx]
            text = chunk["text"]

            if len(context_text) + len(text) > MAX_CONTEXT_CHARS:
                break

            context_text += text.strip() + "\n\n"

            evidence.append({
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "preview": text[:MAX_PREVIEW_CHARS],
                "score": round(score, 4),
            })

        return {
            "context": context_text.strip(),
            "evidence": evidence,
        }

    # =========================
    def get_chunk(self, idx: int) -> Dict:
        return self.chunks[idx]
