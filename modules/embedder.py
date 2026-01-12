#modules\embedder.py

import os
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config.settings import EMBED_MODEL

class Embedder:
    def __init__(self, model_path: str = EMBED_MODEL, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"[Embedder ERROR] Local model folder not found: {model_path}\n"
                "Download once with:\n"
                "  from sentence_transformers import SentenceTransformer\n"
                "  m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')\n"
                "  m.save(r'C:/Mr. Stranger/pdf_chatbot/models/all-MiniLM-L6-v2')\n"
                "and set EMBED_MODEL accordingly in config/settings.py"
            )
        self.model = SentenceTransformer(model_path, device=device)
        self.dim   = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=normalize)
        if not normalize:
            # optional: L2 normalize manually
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
        return embs.astype("float32")
