# README.md

```markdown
# Smart Document Analyst

Smart Document Analyst is a fully offline, privacy-first multilingual Retrieval-Augmented Generation (RAG) system. Engineered for strict factual grounding, it allows users to perform Q&A, summarization, novelty detection, and critical reviews on complex documents without relying on external cloud APIs. 

The entire ecosystem is containerized and deployed on AWS with a robust DevOps pipeline including Nginx reverse proxy and Prometheus/Grafana monitoring.

## 🚀 Key Features

* **Fully Offline Inference:** Runs locally on CPU/Edge devices using `Mistral-7B-Instruct-v0.1` (Quantized). Zero data leakage to third-party APIs.
* **Hybrid Retrieval System:** Combines Dense Vector Search (FAISS + BAAI/MiniLM embeddings) with BM25 lexical search and MMR (Maximal Marginal Relevance) to ensure high precision and minimal context redundancy.
* **Multilingual Neural Translation:** Integrated offline translation using the `M2M100` model, allowing document analysis and output generation across multiple languages.
* **Strict Evidence Grounding:** Answers are forcefully grounded using `[E#]` citations. Includes a post-generation validation module to prune hallucinations and verify numeric data.
* **Production-Ready Deployment:** Fully Dockerized architecture. Orchestrated via `docker-compose` with Nginx handling routing and Grafana + Prometheus integrated for real-time system monitoring.
* **Proprietary Evaluation Framework:** Includes a custom LLM-as-a-judge module designed to rigorously evaluate generated outputs against 6 specific parameters on a 10-point scale.

## ⚙️ Architecture & Tech Stack

* **LLM Core:** Llama.cpp (Mistral-7B Q5_K_M), M2M100 (Translation)
* **Embeddings & Reranking:** SentenceTransformers (`all-MiniLM-L6-v2`), Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
* **Vector Store:** FAISS (Cosine/IP)
* **Web UI:** Streamlit
* **DevOps:** Docker, Nginx, Prometheus, Grafana, AWS

## 📦 Local Installation (CPU-Friendly)

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/smart-document-analyst.git](https://github.com/your-username/smart-document-analyst.git)
cd smart-document-analyst
```

**2. Download Offline Models:**
Since this system prioritizes privacy, models must be downloaded locally to the `models/` directory. Run the following Python snippet:

```python
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

base_path = r'models'
os.makedirs(base_path, exist_ok=True)

# Download Embedding Model
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
m.save(os.path.join(base_path, 'all-MiniLM-L6-v2'))

# Download Cross-Encoder (For High Precision Reranking)
c = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
c.model.save_pretrained(os.path.join(base_path, 'cross-encoder_ms-marco-MiniLM-L-6-v2'))
c.tokenizer.save_pretrained(os.path.join(base_path, 'cross-encoder_ms-marco-MiniLM-L-6-v2'))
```

**3. Configure Environment:**
Set the paths in `config/settings.py` or via environment variables:
```bash
export EMBED_MODEL="models/all-MiniLM-L6-v2"
export LLM_MODEL_PATH="models/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
export LLM_N_CTX="2048"
export LLM_N_THREADS="6"
export RAG_TOP_K="10"
```

**4. Run Locally (Streamlit):**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🐳 Docker & Production Deployment

To run the full production stack (App + Nginx + Prometheus + Grafana) via Docker:

```bash
docker-compose up --build -d
```
* The Streamlit UI will be accessible via Nginx.
* Monitor system health and LLM metrics via the deployed Grafana dashboard.

## 🛡️ Important Notes
* **Strict Grounding:** If you notice hallucinated numbers or facts, ensure the "Strict Grounding" toggle is turned ON in the Streamlit UI.
* **Performance:** Temperature is forced low by default to prioritize factual accuracy over creative writing.
```

***
