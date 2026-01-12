# modules/llm_interface.py

from __future__ import annotations

import os
from typing import Optional
from dataclasses import dataclass

from config.settings import (
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_THREADS,
    LLM_N_BATCH,
    LLM_PROVIDER,
)

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

@dataclass
class GenParams:
    max_tokens: int = 512
    temperature: float = 0.35
    top_p: float = 0.9
    repeat_penalty: float = 1.15
    seed: int = 42

def _safe_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default

class LocalLLM:
    """
    llama-cpp-python wrapper with context budgeting to prevent
    'Requested tokens exceed context window' errors.
    """
    def __init__(self, model_path: str = LLM_MODEL_PATH):
        if Llama is None:
            raise RuntimeError("llama_cpp not installed. Please install llama-cpp-python.")
        self.n_ctx = _safe_int(LLM_N_CTX, 4096)
        self.model = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=_safe_int(LLM_N_THREADS, 4),
            n_batch=_safe_int(LLM_N_BATCH, 128),
            use_mlock=False,
            verbose=True,
        )
        try:
            print(f"Local LLM loaded: {model_path} (ctx={self.n_ctx}, threads={LLM_N_THREADS}, batch={LLM_N_BATCH})")
        except Exception:
            pass

    def _token_len(self, s: str) -> int:
        try:
            # llama.cpp expects bytes for tokenize
            return len(self.model.tokenize(s.encode("utf-8")))
        except Exception:
            # rough fallback
            return max(1, len(s) // 4)

    def _budget(self, prompt: str, params: GenParams) -> GenParams:
        """
        Ensure prompt_tokens + max_tokens <= n_ctx - safety.
        If overflow, reduce max_tokens; if still overflow, hard-trim prompt head (rare, due to upstream evidence caps).
        """
        safety = 32
        ptok = self._token_len(prompt)
        room = self.n_ctx - safety - ptok
        max_tok = params.max_tokens
        if room < 64:
            # trim generation but keep at least 64 tokens if possible
            max_tok = max(64, room)
        else:
            max_tok = min(max_tok, room)
        if max_tok < 64:
            max_tok = 64  # never go below this; llama will still guard
        return GenParams(
            max_tokens=max_tok,
            temperature=params.temperature,
            top_p=params.top_p,
            repeat_penalty=params.repeat_penalty,
            seed=params.seed
        )

    def ask(self, prompt: str, params: Optional[GenParams] = None) -> str:
        if params is None:
            params = GenParams()
        formatted = f"[INST] {prompt.strip()} [/INST]"
        params = self._budget(formatted, params)
        print(f"\n--- GENERATION STARTED (Input Len: {len(prompt)}) ---") # Debug print

        resp = self.model(
            prompt=formatted,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            repeat_penalty=params.repeat_penalty,
            seed=params.seed
        )
        
        print("--- GENERATION FINISHED ---n") # Debug print
        
        return resp["choices"][0]["text"].strip()

class OpenAIProvider:
    """
    Optional HTTP provider; shares interface with LocalLLM.
    Enable via LLM_PROVIDER='openai' and OPENAI_API_KEY env var.
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        # from openai import OpenAI
        # self.client = OpenAI(api_key=key)
        self.model = model
        self.key = key

    def ask(self, prompt: str, params: Optional[GenParams] = None) -> str:
        if params is None:
            params = GenParams()
        # Stubbed to avoid hard dependency; uncomment if SDK present.
        # out = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=params.temperature,
        #     top_p=params.top_p,
        #     max_tokens=params.max_tokens
        # )
        # return out.choices[0].message.content.strip()
        raise RuntimeError("OpenAIProvider is stubbed. Install SDK and uncomment calls.")

def get_llm():
    provider = (LLM_PROVIDER or "local").lower()
    if provider == "openai":
        try:
            return OpenAIProvider()
        except Exception as e:
            print(f"[warn] Falling back to local LLM because OpenAI provider failed: {e}")
    return LocalLLM()






