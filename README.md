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
├── Makefile   
<!-- log: 2026-08-29 22:45:00 - refactor(schema): add type hints and docstrings across schema serializer -->

<!-- log: 2026-08-30 13:00:22 - data: record empirical token count inflation for 0, 1, 3, and 5 sample rows -->

<!-- log: 2026-08-30 16:10:05 - refactor(study): add row data sanitization and truncation safeguards -->
