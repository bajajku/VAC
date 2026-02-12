# VAC - Veteran Affairs Canada AI Chatbot

**Develop and evaluate a trauma-informed LLM-based chatbot that is empathetic, safe, ethical, and context-aware for military populations.**

This document provides a complete analysis of the codebase, the **Jury of LLMs** concept, work completed, work to be done, and a step-by-step guide to run the full pipeline.

---

## Table of Contents
1. [Jury of LLMs Concept](#jury-of-llms-concept)
2. [Architecture Overview](#architecture-overview)
3. [Work Completed](#work-completed)
4. [Work to Be Done](#work-to-be-done)
5. [Step-by-Step Pipeline Guide](#step-by-step-pipeline-guide)
6. [API Reference](#api-reference)

---

## Jury of LLMs Concept

### What It Is

The **Jury of LLMs** is an ensemble evaluation system where **multiple LLMs independently evaluate** RAG (Retrieval-Augmented Generation) responses in parallel. Their outputs are combined using configurable **voting strategies**, producing a more robust and reliable assessment than a single model.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     JURY OF LLMs EVALUATION FLOW                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Query → RAG System → Response + Context Documents                 │
│                                    │                                    │
│                                    ▼                                    │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │  JURY DELIBERATION (Parallel)                                 │     │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │     │
│   │  │ LLM 1    │  │ LLM 2    │  │ LLM 3    │  ... (configurable) │     │
│   │  │ (Granite)│  │ (GPT-OSS)│  │ (Mistral)│                    │     │
│   │  └────┬─────┘  └────┬─────┘  └────┬─────┘                    │     │
│   │       │             │             │                           │     │
│   │       └─────────────┼─────────────┘                           │     │
│   │                     ▼                                         │     │
│   │           Voting Strategy (weighted/majority/unanimous)        │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                    │                                    │
│                                    ▼                                    │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │  JUDGE LLM (Optional - Presiding Judge)                       │     │
│   │  Reviews jury deliberations → Final verdict & reasoning       │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                    │                                    │
│                                    ▼                                    │
│              Evaluation Report (scores, pass/fail, improvements)         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| **Jury Class** | `backend/models/jury.py` | Orchestrates multiple LLMs, runs parallel deliberation, applies voting strategies |
| **RAGEvaluator** | `backend/models/rag_evaluator.py` | Uses Jury to evaluate RAG responses across 14 criteria |
| **JudgeLLM** | `backend/tests/test_evaluation.py` | Presiding Judge that synthesizes jury verdicts into a final decision |
| **EvaluationSystem** | `backend/tests/test_evaluation.py` | End-to-end evaluation: RAG → Jury → Judge → Report |

### Voting Strategies (Jury)

- **`majority`** – Most common response wins  
- **`weighted`** – Provider-weighted average (used for evaluation scores)  
- **`unanimous`** – All must agree; otherwise returns aggregated result  
- **`first_valid`** – First successful response  
- **`average_score`** – Numeric average for scoring  
- **`consensus`** – Common themes across responses  

### Evaluation Criteria (14 Total)

| Category | Criteria |
|----------|----------|
| **Quality** | `retrieval_relevance`, `hallucination`, `noise_robustness`, `negative_rejection`, `information_integration`, `counterfactual_robustness`, `completeness` |
| **Safety** | `privacy_breach`, `malicious_use`, `security_breach` |
| **Domain** | `out_of_domain`, `brand_damage` |
| **Mental Health** | `empathy`, `sensitivity` |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VAC SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │  Frontend   │     │  Backend    │     │  Evaluation │                    │
│  │  (Next.js)  │────►│  API :8000  │     │  API :8001  │                    │
│  │  :3000      │     │  (FastAPI)  │     │  (FastAPI)  │                    │
│  └─────────────┘     └──────┬──────┘     └──────┬──────┘                    │
│                             │                   │                            │
│         ┌───────────────────┼───────────────────┘                            │
│         │                   │                                                │
│         ▼                   ▼                                                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │  MongoDB    │     │  RAG Agent  │     │  Jury +     │                    │
│  │  (Feedback, │     │  LangGraph  │     │  Judge LLMs │                    │
│  │   Sessions) │     │  Chroma DB  │     │             │                    │
│  └─────────────┘     └─────────────┘     └─────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
VAC/
├── backend/                    # Python FastAPI backend
│   ├── api.py                  # Main RAG + Auth API (port 8000)
│   ├── evaluation_server.py    # Jury evaluation API (port 8001)
│   ├── preprocess_data.py      # Offline data preprocessing
│   ├── main.py                 # Interactive CLI
│   ├── config/                 # Settings, constants, MongoDB, OAuth
│   ├── core/                   # RAG application core
│   ├── models/                 # Jury, RAGEvaluator, RAGAgent, LLM, etc.
│   ├── scripts/
│   │   ├── data_cleaning/      # LLM-based document cleaning
│   │   └── data_collection/    # Web crawler, scraper, JSON parser
│   ├── tests/                  # test_evaluation.py (EvaluationSystem)
│   ├── utils/                  # Helpers, auth, prompt, retriever
│   └── readme/                 # Additional documentation
├── frontend/                   # Next.js React app
│   └── src/
│       ├── app/                # Pages: chat, auth, feedback
│       ├── components/         # ChatInput, Feedback, SessionManager
│       └── services/           # feedbackService, sessionService
└── chroma_db/                  # Vector database (generated)
```

---

## Work Completed

### 1. RAG System
- LangGraph-based RAG agent with retriever tool
- Chroma vector DB with multiple retrieval strategies (similarity, MMR, hybrid, ensemble)
- Document preprocessing (basic + LLM-enhanced cleaning)
- Recursive text splitting, re-ranking, result fusion

### 2. Jury of LLMs
- `Jury` class with parallel deliberation (`deliberate`, `adeliberate`)
- Six voting strategies
- Factory and presets (`JuryPresets.diverse_jury`, etc.)
- `RAGEvaluator` using Jury across 14 criteria
- Weighted scoring and improvement suggestions

### 3. Judge Oversight
- `JudgeLLM` for final verdicts from jury deliberations
- Jury agreement level (high/medium/low)
- Optional judge mode (`use_judge=True`) in EvaluationSystem

### 4. Evaluation Pipeline
- `RAGEvaluationPipeline` for end-to-end RAG evaluation
- Batch and async evaluation
- Comprehensive report generation
- `TestCaseGenerator` for retrieval, safety, mental health, hallucination tests

### 5. Evaluation API Server
- Standalone FastAPI server on port 8001
- Endpoints: `/api/evaluate`, `/api/evaluate/batch`, `/api/jury/info`, `/api/dashboard/stats`
- MongoDB storage for jury evaluations

### 6. Main RAG API
- FastAPI on port 8000
- Auth (register, login, JWT, Google OAuth)
- Query, enhanced query, document upload, feedback
- Chat sessions, rate limiting

### 7. Frontend
- Next.js 15 with React 19
- Chat UI with streaming, markdown, sources
- Feedback with detailed ratings
- Auth (login, signup, Google OAuth)
- Session management

### 8. Data Pipeline
- Web crawler, scraper, JSON parser
- LLM-based data cleaner
- Offline preprocessing script

---

## Work to Be Done

### High Priority
1. **Update LLM Base URLs** – Evaluation config uses hardcoded `http://100.96.237.56:8000-8003`. Move to `.env` and support different deployments.
2. **Context Document Extraction** – `_extract_context_documents` in `evaluation_pipeline.py` is a placeholder; implement real retrieval capture.
3. **Frontend–Evaluation Integration** – Connect chat UI to evaluation API (run evaluations from dashboard or after chat).

### Medium Priority
4. **RETRIEVAL_DIVERSITY Criterion** – Referenced in `TestCaseGenerator` but not defined in `EvaluationCriteria`; add or remove usage.
5. **MongoDB Optional Mode** – Evaluation server should degrade gracefully when MongoDB is unavailable.
6. **OAuth Client IDs** – Configure Google OAuth in `.env` for production auth.

### Lower Priority
7. **Docker Compose** – One-command setup for backend, frontend, MongoDB, Chroma.
8. **Evaluation Dashboard UI** – Dedicated frontend for Jury/Judge reports and trends.
9. **Batch Evaluation CLI** – Script to run batch evaluations from command line.
10. **Prompt Optimization Loop** – Integrate `PromptOptimizer` into a full evaluation→optimize→re-evaluate cycle.

---

## Step-by-Step Pipeline Guide

### Prerequisites
- **Python 3.12+**
- **Node.js 18+**
- **MongoDB** (local or Atlas)
- API keys for LLM providers (OpenAI, Together, OpenRouter, or self-hosted)

---

### Step 1: Clone and Navigate

```bash
cd c:\Users\parth\Desktop\CMI\CAAI_VAC\VAC\VAC
```

---

### Step 2: Backend Setup

```bash
cd backend
```

Create virtual environment (optional but recommended):
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

Install dependencies:
```bash
pip install -e .
# OR
pip install -r requirements.txt
```

---

### Step 3: Configure Environment

```bash
cp env.example .env
```

Edit `.env`:

```env
# Required for RAG
TOGETHER_API_KEY=your_api_key_here

# Skip auto-processing on startup (recommended after first run)
SKIP_AUTO_PROCESSING=true

# LLM for RAG
LLM_PROVIDER=chatopenai
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free

# MongoDB (for feedback, sessions, jury evaluations)
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=vac_feedback

# Optional
CHUNK_SIZE=800
CHUNK_OVERLAP=100
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

### Step 4: Data Preparation

**Option A – Use Existing Data**

If `scripts/data_cleaning/cleaned_data/` has preprocessed JSON files, skip to Step 5.

**Option B – Preprocess Raw Data**

```bash
# Enhanced processing (LLM cleaning, ~2–5 min)
python preprocess_data.py --auto

# Basic processing (faster)
python preprocess_data.py --auto --basic

# Custom chunking
python preprocess_data.py --auto --chunk-size 1000 --chunk-overlap 200
```

**Option C – Collect New Data**

```bash
# Run web crawler/scraper (adjust scripts as needed)
# Output goes to scripts/data_collection/crawl_results/
# Then run preprocess_data.py on the output
```

---

### Step 5: Start MongoDB

**Local:**
```bash
# Windows (if installed as service)
net start MongoDB

# Linux/Mac
brew services start mongodb-community
# or
sudo systemctl start mongod
```

**Atlas:** Set `MONGODB_URL` in `.env` to your Atlas connection string.

---

### Step 6: Start Backend API (RAG + Auth)

```bash
cd backend
python api.py
# OR
uvicorn api:app_api --host 0.0.0.0 --port 8000
```

API: `http://localhost:8000`

---

### Step 7: Start Evaluation Server (Jury of LLMs)

In a **new terminal**:

```bash
cd backend
uvicorn evaluation_server:app --host 0.0.0.0 --port 8001 --reload
```

Evaluation API: `http://localhost:8001`

---

### Step 8: Configure Jury/Judge LLMs

Edit `backend/tests/test_evaluation.py` in `EvaluationSystem.initialize_evaluation_system()` and `initialize_judge_llm()` to match your LLM setup:

- **Jury members** – 3 models (e.g. Granite, GPT-OSS, Mistral) via OpenRouter/self-hosted
- **Judge** – Single stronger model (e.g. Llama-3-70B) for final verdicts

Replace `base_url` and `api_key` with your endpoints and keys.

---

### Step 9: Frontend Setup

In a **new terminal**:

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the frontend:

```bash
npm run dev
```

Frontend: `http://localhost:3000`

---

### Step 10: Run the Pipeline

1. **Register/Login** – Go to `http://localhost:3000/auth/login`
2. **Chat** – Ask questions at `http://localhost:3000/chat`
3. **Feedback** – Submit feedback on answers
4. **Evaluation** – Call evaluation API:

   **Single evaluation:**
   ```bash
   curl -X POST "http://localhost:8001/api/evaluate" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is an operational stress injury?", "use_judge": true}'
   ```

   **Batch evaluation:**
   ```bash
   curl -X POST "http://localhost:8001/api/evaluate/batch" \
     -H "Content-Type: application/json" \
     -d '{"questions": ["What is PTSD?", "How can I get support?"], "use_judge": true}'
   ```

   **Jury info:**
   ```bash
   curl "http://localhost:8001/api/jury/info"
   ```

---

### Step 11: Optional – Interactive Evaluation

```bash
cd backend
python main.py
# Follow menu for evaluation demos
```

Or run the evaluation script directly:

```bash
cd backend
python -c "
from tests.test_evaluation import EvaluationSystem
eval_sys = EvaluationSystem()
eval_sys.initialize_rag_agent()
report = eval_sys.evaluate_single_question('What is operational stress injury?', use_judge=True)
print(f'Score: {report.overall_score}, Pass: {report.overall_pass_fail}')
"
```

---

## Quick Reference: Services and Ports

| Service              | Port | Command                                           |
|----------------------|------|---------------------------------------------------|
| Backend RAG API      | 8000 | `uvicorn api:app_api --host 0.0.0.0 --port 8000`  |
| Evaluation API       | 8001 | `uvicorn evaluation_server:app --host 0.0.0.0 --port 8001` |
| Frontend             | 3000 | `npm run dev` (in frontend/)                      |
| MongoDB              | 27017| Default MongoDB port                              |

---

## API Reference

### Main RAG API (port 8000)
- `POST /query` – Basic RAG query
- `POST /query-enhanced` – Enhanced retrieval (hybrid, ensemble, etc.)
- `POST /auth/register`, `POST /auth/login` – Auth
- `POST /feedback` – User feedback
- `GET /stats` – System stats

### Evaluation API (port 8001)
- `POST /api/evaluate` – Single evaluation
- `POST /api/evaluate/batch` – Batch evaluation
- `GET /api/evaluations` – List evaluations
- `GET /api/evaluations/{id}` – Get evaluation details
- `GET /api/jury/info` – Jury composition
- `GET /api/dashboard/stats` – Aggregated stats
- `GET /health` – Health check

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API startup slow | Set `SKIP_AUTO_PROCESSING=true`, run `preprocess_data.py --auto` once |
| MongoDB connection failed | Start MongoDB; check `MONGODB_URL` in `.env` |
| Jury evaluation fails | Check LLM `base_url` and `api_key` in `test_evaluation.py` |
| No documents in Chroma | Run `preprocess_data.py --auto` and ensure cleaned JSON exists |
| Frontend can't reach API | Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `.env.local` |

---

## License

MIT License. See repository for details.
