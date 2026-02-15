
# Scalable RAG System for Henkel Adhesive Products

A Retrieval-Augmented Generation (RAG) system that answers technical questions about Henkel LOCTITE adhesive products by retrieving information from Safety Data Sheets (SDS) and generating answers with LLMs.

## Architecture

```
  Streamlit (port 8501)
       │
       ▼
  Nginx LB (port 8000)
    ┌───┴───┐
    ▼       ▼
  api-1   api-2    (internal port 8000 each)
    │       │
    ├───────┤
    ▼       ▼
 PostgreSQL    Shared embeddings volume (read-only)
```

5 containers: `nginx`, `api-1`, `api-2`, `streamlit`, `db`

- **API layer is stateless** — conversations persist in PostgreSQL, auth is HTTP Basic, and each API instance builds its own in-process RAG cache from the shared FAISS/embeddings volume.
- **Nginx** load-balances across the two API instances using `least_conn`.

## Overview

This project builds an end-to-end RAG pipeline over 36 Henkel LOCTITE product PDFs. The system:

1. **Parses and chunks** PDF safety data sheets using `unstructured`, with text cleaning and section-aware chunking.
2. **Embeds** text chunks using sentence transformer models and stores them in a FAISS vector index for fast similarity search.
3. **Retrieves** relevant chunks at query time using FAISS cosine similarity, with optional cross-encoder reranking and product-scoped search.
4. **Generates** answers by augmenting an LLM prompt with retrieved context — supports both Google Gemma model and the Google Gemini API.
5. **Serves** the pipeline through a FastAPI backend (2 instances + Nginx LB) and a Streamlit chat UI.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A Google Gemini API key

### Setup

```bash
# Clone the repository
git clone https://github.com/TracyWu7724/Scalable_LLM.git && cd Scalable_LLM

# Create .env from example
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY

# Build and start all 5 containers
docker compose up --build
```

Then open `http://localhost:8501` for the Streamlit UI, or call the API directly at `http://localhost:8000`.

### Verify

```bash
# Health check (via Nginx → one of the API instances)
curl http://localhost:8000/health
# → {"status": "ok"}
```

### Rebuild embeddings from PDFs (optional)

```python
from data_preprocessing import DataPreprocessor, EmbedProcess

# Step 1: Parse PDFs into a JSONL corpus
preprocessor = DataPreprocessor(input_dir="Data", output_path="corpus.jsonl")
preprocessor.save_corpus()

# Step 2: Generate embeddings and FAISS index
data = EmbedProcess.from_corpus("corpus.jsonl", embed_model_name="BAAI/bge-base-en-v1.5")
data.save_embeddings(output_dir="embeddings")
```

## Project Structure

```
Scalable_RAG/
├── nginx/
│   └── nginx.conf              # Nginx load balancer config
├── api/
│   ├── Dockerfile              # Streamlit container
│   └── app.py                  # Streamlit chat UI
├── server/
│   ├── Dockerfile              # FastAPI container
│   ├── requirements.txt        # Python dependencies (API only)
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── auth.py                 # HTTP Basic auth
│   ├── schemas.py              # Pydantic models
│   ├── services/
│   │   └── faiss_rag.py        # RAG pipeline (FAISS + Gemini)
│   └── db/
│       ├── __init__.py
│       ├── models.py           # SQLAlchemy models
│       ├── session.py          # Async DB engine
│       └── crud.py             # DB operations
├── pipelines/                  # Offline data tooling
│   ├── data/                   # Source PDFs and datasets
│   ├── embeddings/             # Pre-built FAISS indices
│   └── data_preprocessing.py   # PDF → embeddings pipeline
├── docker-compose.yml          # 5-service deployment
├── .env.example                # Required environment variables
└── README.md
```

## Evaluation

### Retrieval: Embedding Model Comparison

MAP@10 and NDCG@10 across four embedding models using global FAISS retrieval.

![Embedding Model Comparison](assets/embed_comparison.png)

### Retrieval: Top-K Parameter Sensitivity

Effect of varying the top-K retrieval parameter on MAP@10 and NDCG@10.

![Top-K Comparison](assets/topk_comparison.png)

### Retrieval: Ranking Strategy Comparison

Comparison of no-rerank, global rerank, and product-based rerank strategies.

![Ranking Strategy Comparison](assets/rerank_strategy_comparison.png)

### Generation: Model Performance (with and without RAG)

BERTScore and ROUGE metrics across local Gemma and Gemini API models, with and without RAG context.

![Performance Comparison](assets/performance_comparison.png)

### Generation: Strategy Comparison

BERTScore and ROUGE across baseline, global rerank, and product-based rerank strategies for Gemini models.

![Rerank Comparison](assets/rerank_comparison.png)

### End-to-End Latency

Generation time comparison between Gemini API and local Gemma model.

![E2E Latency](assets/E2E_latency.png)


## Future TODO
1. add multiple LLM endpoints
2. add multiple different price-level LLM providers