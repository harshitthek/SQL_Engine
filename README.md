---
title: Text To SQL
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
python_version: "3.11"
app_file: deployment/app.py
pinned: false
license: mit
---

# Neural Text-to-SQL Engine

Production-grade Text-to-SQL semantic parser fine-tuned on the [Spider benchmark](https://yale-lily.github.io/spider) using QLoRA with **Qwen2.5-Coder-1.5B**. Supports multi-GPU training (NVIDIA Dual T4 DDP), FastAPI inference API with rate limiting, and Gradio interactive UI with multi-database connectivity.

---

## Architecture

```
┌─────────────┐       HTTP        ┌──────────────────┐      Model     ┌──────────────────┐
│  Gradio UI  │ ──────────────▶   │   FastAPI API    │ ─────────────▶ │  Qwen2.5-Coder   │
│  (port 7860)│   /v1/tosql       │   (port 8000)    │  generate_sql  │  Text2SQL Engine  │
│             │ ◀──────────────   │   Rate Limiting  │ ◀───────────── │  (GPU/MPS/CPU)    │
│  Database   │       SQL         │   Health Check   │      SQL       │                   │
│  Execution  │                   │   Request Queue  │                │                   │
└─────────────┘                   └──────────────────┘                └──────────────────┘
```

> **Design Rule**: Database credentials and SQL execution stay in Gradio. Model inference stays in FastAPI. They communicate only via HTTP.

---

## 📁 Repository Structure

```text
SQL_Engine/
├── Makefile                          # Unified targets: install, serve, test, docker, etc.
├── Dockerfile                        # Multi-stage production Docker image
├── docker-compose.yml                # Two-service deployment (API + UI)
├── pyproject.toml                    # Project metadata, pytest, ruff, mypy config
├── requirements.txt                  # Deployment dependencies (inference + serving)
├── requirements-train.txt            # Full training dependencies (includes deployment)
├── .env_example                      # Environment variable template
├── .gitignore                        # Comprehensive exclusion rules
├── .dockerignore                     # Docker build context exclusions
│
├── deployment/                       # Production Deployment Stack
│   ├── api.py                        # FastAPI REST API (rate limiting, health, inference lock)
│   ├── prediction.py                 # Text2SQLEngine with KaggleHub model resolution
│   ├── database.py                   # SQLAlchemy DB manager (SQLite, PostgreSQL, MySQL, Supabase)
│   ├── gradio_app.py                 # Gradio UI (connection manager, SQL execution, themes)
│   ├── api_client.py                 # FastAPI HTTP client with typed exceptions
│   ├── app.py                        # Orchestrator: auto-starts FastAPI then launches Gradio
│   ├── test_api.py                   # API tests (711 lines)
│   ├── test_prediction.py            # Inference engine tests
│   ├── test_database_integration.py  # Database layer tests
│   ├── test_gradio_app.py            # Gradio UI tests (1,501 lines)
│   ├── test_api_client.py            # Client tests
│   └── test_app.py                   # Orchestrator tests
│
├── data/                             # Data Layer
│   ├── spider_data/                  # Raw Spider benchmark (git-ignored)
│   ├── processed_arrow/              # Preprocessed HF Arrow dataset (git-ignored)
│   ├── reports/                      # JSON audit & distribution reports
│   └── processing/                   # Data processing modules
│       ├── complexity.py             # SQL AST hardness classifier
│       ├── prompt_templates.py       # Prompt template library
│       ├── schema_serializer.py      # SQLite DDL extraction
│       ├── dataset_builder.py        # Arrow dataset creator
│       ├── tokenizer_utils.py        # Tokenizer wrapper & visualizer
│       ├── sanity_checker.py         # Validation & inspection
│       ├── sample_rows_study.py      # Token inflation study
│       └── run_pipeline.py           # Unified runner (Tasks 02-09)
│
├── src/                              # Core ML Engine
│   ├── training/                     # QLoRA Training
│   │   ├── train_qlora.py            # 4-bit NF4 fine-tuning with PEFT + TRL
│   │   ├── eval_metrics.py           # Exact Match evaluation & mid-training callback
│   │   └── merge_model.py            # LoRA merge & SHA-256 metadata export
│   ├── inference/                    # Inference Engine
│   │   └── engine.py                 # Text2SQLEngine (used by eval scripts)
│   └── evaluation/                   # SQL Evaluator
│       └── evaluator.py              # EM, EX, timeout, sandbox, error categorization
│
├── scripts/                          # Orchestration Utilities
│   ├── run_eval.py                   # Batch evaluation on Spider dev set
│   ├── smoke_test_pipeline.py        # Fast training smoke test
│   ├── generate_notebook.py          # Kaggle notebook generator
│   └── run_*.py                      # Individual task runners
│
├── tests/                            # Test Suite
│   ├── test_pipeline.py              # Data pipeline tests
│   └── test_evaluator.py             # SQL evaluator tests (47 tests)
│
├── notebooks/                        # Kaggle Training Notebook
│   └── sql_engine.ipynb              # Dual T4 DDP training (Tasks 01-10)
│
├── models/                           # Trained Weights (git-ignored, Docker volume)
│   └── text2sql-qwen-pytorch-v1-v1/
│       ├── qlora-adapter/            # LoRA adapter checkpoints
│       └── text2sql-v1/              # Merged standalone model
│
├── artifacts/                        # Evaluation Reports
│   ├── eval_error_analysis.md        # Top failure patterns
│   ├── eval_metrics.json             # EM & EX scores
│   └── dev_predictions.json          # Model predictions
│
└── configs/                          # Environment Specs
    └── kaggle_env_spec.txt           # Kaggle Dual T4 hardware audit
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Deployment only (inference + serving)
make install

# Full (deployment + training)
make install-train
```

### 2. Start the Server

```bash
# Start FastAPI inference API (port 8000)
make serve

# Start Gradio UI (port 7860, auto-starts FastAPI)
make gradio

# Or use Docker
make docker-up
```

### 3. API Usage

```bash
# Health check
curl http://localhost:8000/health

# Generate SQL
curl -X POST http://localhost:8000/v1/tosql \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many students are there?",
    "schema": "CREATE TABLE students (id INT, name TEXT, age INT);",
    "dialect": "sqlite"
  }'

# Interactive API docs
open http://localhost:8000/docs
```

### 4. Tests

```bash
# Core tests (pipeline + evaluator)
make test

# Deployment tests (API, Gradio, DB)
make test-deployment

# All tests
make test-all
```

### 5. Training on Kaggle (Dual T4 GPUs)

1. Upload `notebooks/sql_engine.ipynb` to Kaggle
2. Select **GPU T4 x 2** accelerator
3. Set `"SMOKE_TEST": False` in CONFIG cell for full training
4. Click **Run All** — DDP launches automatically

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t sql-engine .

# Run (mount model weights)
docker run -p 8000:8000 -p 7860:7860 \
  -v ./models:/app/models \
  sql-engine

# Or with docker compose (separate API + UI services)
docker compose up
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEXT2SQL_MODEL_PATH` | Auto-detected | Path to merged model weights |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI port |
| `FASTAPI_URL` | `http://127.0.0.1:8000` | URL Gradio uses to reach FastAPI |
| `GRADIO_HOST` | `0.0.0.0` | Gradio bind host |
| `GRADIO_PORT` | `7860` | Gradio port |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `RATE_LIMIT_PER_MINUTE` | `10/minute` | Per-IP rate limit |
| `RATE_LIMIT_BURST` | `3/10seconds` | Burst rate limit |
| `RELOAD_SECRET` | *(none)* | Secret for `/v1/reload` endpoint |

Copy `.env_example` to `.env` and customize:
```bash
cp .env_example .env
```

---

## 📊 Evaluation Results

| Metric | Score |
|--------|-------|
| Exact Match (EM) | 44.38% |
| Execution Accuracy (EX) | 65.00% |

Evaluated on 160 Spider dev examples across all complexity tiers.

---

## License

MIT
