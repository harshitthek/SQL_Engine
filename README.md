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
│   ├── database.py                   # SQLAlche
<!-- log: 2026-09-02 15:20:44 - refactor(pipeline): connect root run_all.py to data.processing.run_pipeline -->

<!-- log: 2026-09-02 18:30:25 - refactor(pipeline): support --task all and individual task selection -->

<!-- log: 2026-09-02 21:20:35 - refactor(data): finalize data.processing subpackage exports -->
