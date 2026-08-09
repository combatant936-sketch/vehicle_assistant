# 🚗 Vehicle Diagnostic Assistant (Agentic RAG)

An advanced, enterprise-grade **Agentic Retrieval-Augmented Generation (RAG)** application for vehicle diagnostics. Built with **LangGraph**, **FastAPI**, **ChromaDB**, **SQLite Search**, **PostgreSQL**, and **Grafana**, this system assists users in diagnosing vehicle issues using a hybrid search pipeline, document grading, query rewriting, automated LLM-as-a-judge evaluation, and real-time operational monitoring.

---

## 🛠️ Technology Stack & Component Breakdown

Every technology in this repository is strictly aligned with the source code:

| Technology / Library | Role & Purpose in Project |
| :--- | :--- |
| **FastAPI & Uvicorn** | High-performance Python web framework (`project/app.py`) running Uvicorn ASGI server hosting REST API endpoints (`/question`, `/feedback`). |
| **LangGraph (v2 Agentic RAG)** | Stateful orchestration graph (`StateGraph` in `project/versions/v2/rag.py`) managing the agentic lifecycle: **Hybrid Search ➔ Document Grading ➔ Query Rewriting ➔ Answer Generation / Fallback**. |
| **LangChain & ChatOpenAI** | Abstraction layer connecting to LLM providers via Groq API base URL (`https://api.groq.com/openai/v1`), executing structured prompt templates and chains. |
| **ChromaDB** | Vector database (`project/chroma_db`) storing dense vector embeddings of vehicle diagnostic issues for semantic similarity retrieval. |
| **ONNX Runtime & Tokenizers** | Local embedding execution (`embedder.py`) using an ONNX-quantized `Xenova/all-MiniLM-L6-v2` model downloaded via `download.py` for fast CPU-based embeddings without API costs. |
| **sqlitesearch (BM25 SQLite)** | SQLite-backed full-text keyword search engine (`project/issues.db`) with field boosting (`issue_name`, `obd_code`, `symptoms`, `diagnostic_steps`, etc.). |
| **minsearch** | Lightweight memory-based search engine used in early RAG prototyping (`project/ingest.py`). |
| **Reciprocal Rank Fusion (RRF)** | Hybrid search ranker combining top search results from dense vector search (ChromaDB) and sparse text search (SQLite BM25). |
| **PostgreSQL & psycopg2** | Relational database (`project/db.py`) storing detailed conversation logs, response time metrics, token usage, cost tracking, relevance evaluation results, and user feedback (+1 / -1). |
| **Grafana** | Real-time operational dashboard visualizing token usage, response latency, LLM cost, relevance scores, and feedback trends (`project/dashboard.json`). |
| **Groq API** | LLM inference API running models such as `llama-3.1-8b-instant` or `openai/gpt-oss-120b`. |
| **Docker & Docker Compose** | Multi-container environment orchestrating `app` (FastAPI), `postgres` (PostgreSQL 16), and `grafana` (Grafana Latest). |
| **`uv` (by Astral)** | Blazing-fast Python package and project environment manager used in `Dockerfile` (`python:3.14-slim`) and dependency resolution (`pyproject.toml`, `uv.lock`). |
| **Jupyter Notebook** | Environment for synthetic data generation, search index experimentation, and offline RAG benchmarking (`project/data/data-creation-and-evaluation.ipynb`). |

---

## 📁 Detailed Codebase & File Architecture

Here is the exact responsibility of every file in the repository:

### Root Files
- **`Dockerfile`**: Docker build definition using `python:3.14-slim` and `uv` package installer to build and run the FastAPI app (`uvicorn project.app:app --host 0.0.0.0 --port 8000 --reload`).
- **`docker-compose.yaml`**: Orchestrates 3 services (`postgres` on 5432, `app` on 8001:8000, and `grafana` on 3000).
- **`pyproject.toml` & `uv.lock`**: Python project configuration and exact dependency lockfile.
- **`download.py`**: Downloads ONNX weights (`model.onnx`) and `tokenizer.json` for `Xenova/all-MiniLM-L6-v2` from Hugging Face Hub into `models/Xenova/all-MiniLM-L6-v2/`.
- **`embedder.py`**: Implements custom `Embedder` class using `onnxruntime` and `tokenizers` to compute 384-dimensional dense text embeddings locally on CPU.

### `project/` Directory
- **`project/app.py`**: FastAPI application.
  - `POST /question`: Receives user question, generates UUID `conversation_id`, executes `rag.query()`, saves result to PostgreSQL, and returns response.
  - `POST /feedback`: Receives `{ "conversation_id": "...", "feedback": 1 | -1 }` and saves to PostgreSQL.
- **`project/db.py`**: PostgreSQL database connector (`get_db_connection`) and schema manager (`init_db`, `save_conversation`, `save_feedback`).
- **`project/init.py`**: Automated provisioning script for Grafana. Creates `ProgrammaticSA` Service Account, generates API token, configures PostgreSQL datasource, dynamically updates datasource UIDs in `project/dashboard.json`, and imports the dashboard.
- **`project/dashboard.json`**: Pre-configured Grafana dashboard JSON (`UID: ar2zr8`) containing 7 panels.
- **`project/ingest.py`**: Loader script for building an in-memory `minsearch` index from `data/data.csv`.
- **`project/ingest_sqlite.py`**: Loader script connecting to the SQLite full-text search index at `project/issues.db`.
- **`project/issues.db`**: SQLite database for BM25 text search.
- **`project/versions/v2/rag.py`**: Production **LangGraph Agentic RAG** graph:
  - **`RAGState`**: TypedDict schema tracking query, rewritten query, documents, generation, relevance score, retries, and max retries.
  - **`hybrid_search`**: Merges SQLite BM25 text search (with field boosts: `issue_name: 2.98`, `component: 2.86`, `system: 2.25`, `diagnostic_steps: 2.11`, `likely_causes: 2.04`, `symptoms: 1.71`, `severity: 1.19`, `diy_or_mechanic: 1.07`, `obd_code: 0.88`) and ChromaDB vector search using Reciprocal Rank Fusion (RRF).
  - **`grade_documents`**: LLM node evaluating retrieved document relevance scores (0.0 to 1.0) and keeping relevant context.
  - **`rewrite_query`**: LLM query rewriter node invoked when retrieval relevance falls below threshold.
  - **`generate_answer`**: Constructs answer strictly from retrieved context and executes `evaluate_relevance` (LLM-as-a-Judge rating `RELEVANT`, `PARTLY_RELEVANT`, `NON_RELEVANT`) and calculates token costs.
  - **`generate_fallback`**: Returns a structured fallback response when no context matches after retries.

### `project/data/` Directory
- **`data/data.csv`**: Synthetic dataset containing 50 diagnostic issue records (OBD codes, symptoms, likely causes, diagnostic steps, diy/mechanic recommendation).
- **`data/ground-truth-retrieval.csv`**: Benchmark ground-truth QA dataset mapping questions to `issue_id`s.
- **`data/data-creation-and-evaluation.ipynb`**: Complete Jupyter notebook demonstrating dataset generation, retrieval benchmarking (Hit Rate & MRR), and RAG evaluation.

---

## 🏗️ System Architecture & Workflow

```
[ User Request ]
       │
       ▼
[ FastAPI App (/question) ]
       │
       ▼
[ LangGraph Agentic Pipeline (v2/rag.py) ]
  ├── 1. Hybrid Search (ChromaDB Vector + SQLite BM25 -> RRF Ranker)
  ├── 2. Grade Documents (LLM evaluates retrieved context relevance)
  ├── 3. Query Rewrite (If relevance score < threshold, rewrites query & retries)
  ├── 4. Generate Answer / Fallback (Constructs answer strictly from context)
  └── 5. Auto-Evaluate Relevance (LLM-as-a-Judge rates output relevance)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ PostgreSQL Database ]                 [ JSON API Response ]
  (Saves tokens, cost, latency,            (Returns conversation_id & answer)
   eval score & user feedback)                   │
       │                                         ▼
       ▼                                 [ User Feedback ]
[ Grafana Dashboard ]                       (POST /feedback)
  (Real-time analytics & graphs)                 │
                                                 └──► Saved to Postgres
```

---

## 📋 Prerequisites

Ensure you have the following installed on your host machine:

- **Docker** & **Docker Compose** (v2.0+)
- **Git**
- **Groq API Key** (Get a free key from [Groq Console](https://console.groq.com/))

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/combatant936-sketch/vehicle_assistant.git
cd vehicle-assistant
```

### Step 2: Configure Environment Variables
Create a `.env` file in the root directory by copying `.env.example`:

```bash
cp .env.example .env
```

Open `.env` and set your configuration parameters:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
AI_MODEL=llama-3.1-8b-instant
MODEL_BASE_URL=https://api.groq.com/openai/v1
POSTGRES_HOST=postgres
POSTGRES_DB=vehicle_assistant
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_PORT=5432
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
SQLITESEARCHDB=./issues.db
CHROMA_COLLECTION=obd_diagnostics
CHROMA_DB_DIR=./chroma_db
```

### Step 3: Build & Start Services
Launch all containers using Docker Compose:

```bash
docker compose up -d --build
```

This starts 3 containers:
- **`postgres`**: Relational DB on port `5432`
- **`app`**: FastAPI RAG Server on port `8001` (mapped to internal `8000`)
- **`grafana`**: Monitoring Dashboard on port `3000`

### Step 4: Initialize Database & Grafana Dashboard
Run the setup commands inside the app container to set up tables, datasources, and dashboards:

**1. Initialize PostgreSQL Tables:**
```bash
docker compose exec app python -c "from project import db; db.init_db()"
```

**2. Initialize Grafana Datasource & Dashboard:**
```bash
docker compose exec app python project/init.py
```

---

## 📖 API Usage & Endpoints

### 1. Ask a Diagnostic Question (`POST /question`)

**Request**:
```bash
curl -X POST http://localhost:8001/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What causes engine misfires?"
  }'
```

**Response**:
```json
{
  "conversation_id": "07e72c1f-3fc1-4163-b182-67a23de7d0f8",
  "question": "What causes engine misfires?",
  "answera": "Based on the provided CONTEXT, engine misfires can be caused by worn spark plugs, faulty ignition coils, low fuel pressure, or vacuum leaks..."
}
```

### 2. Submit User Feedback (`POST /feedback`)

**Request**:
```bash
curl -X POST http://localhost:8001/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "07e72c1f-3fc1-4163-b182-67a23de7d0f8",
    "feedback": 1
  }'
```
*(Use `1` for positive / thumbs up, `-1` for negative / thumbs down)*

**Response**:
```json
{
  "message": "Feedback received: 1"
}
```

---

## 📊 Analytics & Grafana Dashboard

Open your browser and navigate to:
```
http://localhost:3000
```
- **Login**: `admin` / `admin` (or the credentials set in `.env`)
- Navigate to **Dashboards ➔ Vehicle assistant** (`/d/ar2zr8/vehicle-assistant`).

### Dashboard Metrics Tracked:
- 💬 **Last Conversations Table**: Live feed of questions, answers, and relevance ratings.
- 🎯 **Relevancy Gauge**: Visual breakdown of `RELEVANT`, `PARTLY_RELEVANT`, and `NON_RELEVANT` ratings.
- ⚡ **Response Time**: Full execution latency tracking for the agentic RAG flow.
- 🪙 **Token Consumption**: Real-time breakdown of prompt vs completion tokens.
- 💵 **Cost Estimation**: Groq LLM API cost calculated per conversation.
- 👍 **User Feedback**: Aggregated thumbs up (+1) vs thumbs down (-1) metrics.

---

## ⚙️ Useful Management Commands

```powershell
# View live application logs
docker compose logs -f app

# Rebuild and restart application after code changes
docker compose up -d --build --force-recreate app

# Query PostgreSQL conversations table directly
docker compose exec postgres psql -U user -d vehicle_assistant -c "SELECT question, relevance, total_tokens, groq_cost FROM conversations;"

# Stop all services
docker compose down
```
