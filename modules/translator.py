# modules/translator.py


from __future__ import annotations
from typing import List
import os
import re

try:
    import torch
except Exception:
    torch = None

# Sentence chunking regex
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


class Translator:
    """
    Local-only translator using M2M100 (facebook/m2m100_418M).

    - Primary load: local directory path (ENV: TRANSLATION_MODEL) or default "models/m2m100_418M"
    - No remote downloads (local_files_only=True)
    - Uses GPU if available, otherwise CPU
    - Requires: transformers, sentencepiece, torch
    """

    LANG_MAP = {
        "en": "English",
        "fr": "French",
        "bn": "Bengali",
        "ru": "Russian",
        "tr": "Turkish",
        "ar": "Arabic",
        "es": "Spanish",
        "hi": "Hindi",
        "de": "German",
        "pt": "Portuguese",
        "zh": "Chinese",
    }

    def __init__(self, model_name: str | None = None, device: str = "auto"):
        self.available = False
        self.last_error = ""
        self._tokenizer = None
        self._model = None

        env_path = (os.getenv("TRANSLATION_MODEL", "") or "").strip()
        default_path = "models/m2m100_418M"
        resolved = env_path or (model_name or default_path)
        resolved = os.path.normpath(os.path.expanduser(resolved))
        self._model_path = resolved

        self._device = "cpu"
        if device == "auto" and torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
            self._device = "cuda"
        elif device in ("cpu", "cuda"):
            self._device = device

        self._load_local_only()

    def _load_local_only(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, local_files_only=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path, local_files_only=True)
            if self._device == "cuda":
                self._model = self._model.to("cuda")
            self.available = True
        except Exception as e:
            self.last_error = f"Local load failed from '{self._model_path}': {e}"
            self._tokenizer = None
            self._model = None
            self.available = False

    @staticmethod
    def _chunk(text: str, max_chars: int = 1600) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        parts: List[str] = []
        cur, cur_len = [], 0
        for sent in _split_sentences(text):
            if cur_len + len(sent) > max_chars and cur:
                parts.append(" ".join(cur))
                cur, cur_len = [sent], len(sent)
            else:
                cur.append(sent)
                cur_len += len(sent)
        if cur:
            parts.append(" ".join(cur))
        return parts

    def translate(self, text: str, target_lang: str = "en", source_lang: str | None = "en") -> str:
        """Return translated text; if unavailable, return input unchanged."""
        if not isinstance(text, str) or not text.strip():
            return text
        if not self.available or self._tokenizer is None or self._model is None:
            return text

        tgt = target_lang if target_lang in self.LANG_MAP else "en"
        src = source_lang if (source_lang in self.LANG_MAP) else None

        # Set language codes for M2M100
        try:
            if src:
                self._tokenizer.src_lang = src
            self._tokenizer.tgt_lang = tgt
        except Exception:
            pass

        # Force target language to avoid drifting into another language
        try:
            bos_id = self._tokenizer.get_lang_id(tgt)
        except Exception:
            bos_id = None

        out_chunks: List[str] = []
        for ch in self._chunk(text):
            try:
                enc = self._tokenizer(ch, return_tensors="pt", padding=True, truncation=True)
                if self._device == "cuda":
                    enc = {k: v.to("cuda") for k, v in enc.items()}

                gen_kwargs = dict(max_length=2048, num_beams=4)
                if bos_id is not None:
                    gen_kwargs["forced_bos_token_id"] = bos_id

                gen = self._model.generate(**enc, **gen_kwargs)
                out = self._tokenizer.batch_decode(gen, skip_special_tokens=True)
                out_chunks.append(out[0] if out else ch)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                out_chunks.append(ch)

        return "\n".join(out_chunks)

    def translate_sectionwise(self, text: str, target_lang: str = "en") -> str:
        return self.translate(text, target_lang=target_lang, source_lang="en")

