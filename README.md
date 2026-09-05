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
│   └── processing/    
<!-- log: 2026-09-04 22:40:15 - refactor(evaluation): optimize generation memory consumption during evaluation steps -->

<!-- log: 2026-09-05 10:30:45 - feat(merging): implement compute_file_sha256 for model weight integrity checks -->

<!-- log: 2026-09-05 11:50:10 - feat(merging): export training_metadata.json with architecture specs and file hashes -->
