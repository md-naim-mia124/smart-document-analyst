#config\settings.py


import os

# Project paths (resolve relative to this file by default)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
INDEX_DIR    = os.path.join(PROJECT_ROOT, "indexes")
UPLOAD_DIR   = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# UI / Behavior
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]
SHOW_LATENCY = True

# Retrieval/Generation knobs (can be overridden via ENV)
RAG_TOP_K     = int(os.getenv("RAG_TOP_K", "10"))        # candidates before MMR
FINAL_E_BLOCKS= int(os.getenv("FINAL_E_BLOCKS", "5"))    # evidence blocks shown to LLM
ALPHA_DENSE   = float(os.getenv("ALPHA_DENSE", "0.65"))  # dense vs lexical mix
MMR_LAMBDA    = float(os.getenv("MMR_LAMBDA", "0.3"))    # diversity
MAX_EVID_TOKS = int(os.getenv("MAX_EVID_TOKS", "900"))   # rough, word-approx



# Embedding model (local path recommended)
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    os.path.join(PROJECT_ROOT, "models", "all-MiniLM-L6-v2")
)


# LLM model path (GGUF) & runtime
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "mistral-7b-instruct-v0.1.Q5_K_M.gguf")
)
LLM_N_CTX     = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", max(os.cpu_count() or 4, 4)))
LLM_N_BATCH   = int(os.getenv("LLM_N_BATCH", "128"))

# Provider switch: "local" or "openai" (if you want to route heavy jobs)
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "local")
# Set OPENAI_API_KEY in env if using openai provider

# Misc
TOP_K = RAG_TOP_K
