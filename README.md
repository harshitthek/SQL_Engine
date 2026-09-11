---
title: Text To SQL
emoji: ⚡
sdk: gradio
sdk_version: 6.26.0
app_file: deployment/app.py
---

<div align="center">

# ⚡ Neural Text-to-SQL Engine

**Production-grade Text-to-SQL semantic parser fine-tuned on the Spider benchmark with Qwen2.5-Coder-1.5B, featuring multi-database execution, rate-limited FastAPI serving, and an interactive Gradio workstation.**

[![Live Web App](https://img.shields.io/badge/Live%20Demo-Cloudflare%20Workers-F38020?logo=cloudflare&logoColor=white)](https://text-to-sql.here-2007.workers.dev/)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Spaces%20ZeroGPU-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Pernav/Text_to_SQL)
[![Kaggle Model](https://img.shields.io/badge/Kaggle-harshitxdev%2Ftext2sql--qwen-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/models/harshitxdev/text2sql-qwen)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Spider%20Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/pernavjain/natural-language-to-sql)
[![Benchmark](https://img.shields.io/badge/Benchmark-Yale%20Spider-8B0000)](https://yale-lily.github.io/spider)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-6.0%2B-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
[![Tests](https://img.shields.io/badge/Tests-233%20Passing-brightgreen?logo=pytest&logoColor=white)](#-testing--quality-assurance)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-black?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Live Web App**](https://text-to-sql.here-2007.workers.dev/) • [**Hugging Face Space**](https://huggingface.co/spaces/Pernav/Text_to_SQL) • [**Trained Weights**](https://www.kaggle.com/models/harshitxdev/text2sql-qwen) • [**Architecture**](#-system-architecture) • [**Quickstart**](#-quickstart)

</div>

---

## 🌐 Interactive Demos & Artifacts

| Platform | Resource | Description |
| :--- | :--- | :--- |
| **Cloudflare Workers** | [**text-to-sql.here-2007.workers.dev**](https://text-to-sql.here-2007.workers.dev/) | Global edge-routed live web application with low latency. |
| **Hugging Face Spaces** | [**Pernav/Text_to_SQL**](https://huggingface.co/spaces/Pernav/Text_to_SQL) | Cloud workstation powered by ZeroGPU direct inference. |
| **Kaggle Models** | [**harshitxdev/text2sql-qwen**](https://www.kaggle.com/models/harshitxdev/text2sql-qwen) | Standalone merged FP16 PyTorch weights and tokenizer. |
| **Kaggle Datasets** | [**pernavjain/natural-language-to-sql**](https://www.kaggle.com/datasets/pernavjain/natural-language-to-sql) | Preprocessed Spider benchmark dataset with serialized DDL. |
| **Yale LILY** | [**Spider Benchmark**](https://yale-lily.github.io/spider) | 10,181 natural language questions across 200 complex relational schemas. |

---

## 👥 Authors & Core Team

This project is a collaborative effort bringing together deep learning model engineering and full-stack cloud systems architecture:

- **Pushkar Jain / Pernav Jain** ([@here-2007](https://github.com/here-2007) / [@Pernav](https://huggingface.co/spaces/Pernav/Text_to_SQL) / [@pernavjain](https://www.kaggle.com/pernavjain))
  - **Project Creator & Full-Stack Architect**: Designed the end-to-end multi-tier system architecture and published the Spider benchmark dataset on Kaggle ([pernavjain/natural-language-to-sql](https://www.kaggle.com/datasets/pernavjain/natural-language-to-sql)).
  - **Cloud Deployment**: Deployed and maintained the Hugging Face Space ([Pernav/Text_to_SQL](https://huggingface.co/spaces/Pernav/Text_to_SQL)) and configured Cloudflare Workers edge routing ([text-to-sql.here-2007.workers.dev](https://text-to-sql.here-2007.workers.dev/)).
  - **Frontend Engineering**: Built the Gradio workstation UI featuring the JetBrains Mono terminal design system, connection management, dark/light themes, and client-side `localStorage` state syncing.
  - **API Gateway**: Built the production FastAPI REST service with SlowAPI rate limiting, concurrency queue locks, and typed client SDKs.

- **Harshit Sharma** ([@harshitthek](https://github.com/harshitthek) / [@harshitxdev](https://www.kaggle.com/harshitxdev))
  - **Machine Learning & Model Training**: Orchestrated QLoRA fine-tuning on Dual NVIDIA T4 GPUs using Distributed Data Parallelism (DDP) with 4-bit NF4 quantization, 8-bit Paged AdamW, and cosine decay. Published the trained weights to [harshitxdev/text2sql-qwen](https://www.kaggle.com/models/harshitxdev/text2sql-qwen).
  - **Inference Optimization**: Implemented memory-efficient CPU loading (`low_cpu_mem_usage=True`) for sub-3GB RAM execution, standalone weight merging, and SHA-256 artifact verification.
  - **Database Integration**: Developed the passwordless Supabase (API) connection mode with OpenAPI schema introspection and SQL Management API execution.
  - **Testing & Tooling**: Built isolated SQLite test fixtures and cross-platform Windows/POSIX developer portability.

---

## 🏗️ System Architecture

The engine strictly separates **model inference** from **database credentials and query execution**, guaranteeing that database passwords and sensitive schema details never pass to the model serving layer.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Edge Routing (Cloudflare Workers)                      │
│                    https://text-to-sql.here-2007.workers.dev                    │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   Hugging Face Spaces / Container Environment                   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      Gradio UI Workstation (:7860)                      │   │
│   │  • JetBrains Mono Terminal UI (Emerald/Zinc) with Dark/Light Themes     │   │
│   │  • Connection Manager (SQLite, PostgreSQL, MySQL, Supabase API)         │   │
│   │  • Interactive Schema Browser & Query History (localStorage Synced)     │   │
│   │  • Database Credential Redaction & Dialect Harmonizer                   │   │
│   └──────────────────────┬────────────────────────────▲─────────────────────┘   │
│                          │                            │                         │
│             HTTP /v1/tosql (Question + Schema)        │ Formatted SQL           │
│                          ▼                            │                         │
│   ┌───────────────────────────────────────────────────┴─────────────────────┐   │
│   │                     FastAPI Inference API (:8000)                       │   │
│   │  • Strict Pydantic v2 Request/Response Validation                       │   │
│   │  • SlowAPI Rate Limiting (10 req/min, 3 req/10s burst per IP)           │   │
│   │  • Synchronized Inference Queue Lock (Timeout: 30s)                     │   │
│   │  • Health & Concurrency Metrics (/health: queue depth, latency)         │   │
│   └──────────────────────┬──────────────────────────────────────────────────┘   │
│                          │                                                      │
│                          ▼                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                   Neural Text2SQL Engine (PyTorch)                      │   │
│   │  • Fine-Tuned Qwen2.5-Coder-1.5B (KaggleHub Auto-Resolution)            │   │
│   │  • Multi-Device Support: CUDA (bf16/fp16), Apple MPS, CPU (float32)     │   │
│   │  • Memory Optimization: low_cpu_mem_usage=True (<3GB RAM footprint)    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼ SQL Query Execution (Direct from UI)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Relational Database Layer                            │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   ┌──────────────┐  │
│    │    SQLite    │    │  PostgreSQL  │    │    MySQL     │   │ Supabase API │  │
│    │  (Local DB)  │    │ (SQLAlchemy) │    │ (SQLAlchemy) │   │ (REST/HTTPS) │  │
│    └──────────────┘    └──────────────┘    └──────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles
1. **Zero Credential Exposure**: Database connection strings, passwords, and service tokens are kept exclusively within the Gradio client process and are never transmitted to the inference backend.
2. **Synchronized Inference Locking**: GPU/CPU text generation is serialized using `asyncio.Lock` with a 30-second queue timeout, preventing memory thrashing and Out-Of-Memory (OOM) crashes under concurrent load.
3. **Multi-Dialect Harmonization**: Generated queries are post-processed through an AST transformer that automatically maps SQLite functions (`IFNULL`, `strftime`) to target dialects like PostgreSQL/Supabase (`COALESCE`, `EXTRACT`) and normalizes quoted identifiers.

---

## 📊 The Data Pipeline (Tasks 02–09)

The preprocessing pipeline transforms the raw Spider benchmark into high-density training records through a 10-stage preparation workflow:

```
Raw Spider Dataset ──▶ AST Hardness Audit ──▶ Schema DDL Extraction ──▶ Sample Rows Study ──▶ Token Distribution ──▶ Filter & Export Arrow
(train + dev.json)       (Task 02)                (Task 04)                 (Task 05)               (Task 06)              (Tasks 07 & 08)
```

| Task | Module | Purpose & Output |
| :--- | :--- | :--- |
| **Task 02: Complexity Audit** | [`complexity.py`](data/processing/complexity.py) | Parses SQL ASTs into hardness tiers (`easy`, `medium`, `hard`, `extra-hard`) based on joins, subqueries, group by, and set operators. |
| **Task 03: Prompt Templates** | [`prompt_templates.py`](data/processing/prompt_templates.py) | Formulates structured prompt variations (`markdown`, `chatml`, `code_comment`) isolating questions, schema context, and target SQL. |
| **Task 04: Schema Serializer** | [`schema_serializer.py`](data/processing/schema_serializer.py) | Introspects SQLite system catalogs to generate standardized DDL declarations with primary and foreign key references. |
| **Task 05: Sample Rows Study** | [`sample_rows_study.py`](data/processing/sample_rows_study.py) | Evaluates token inflation vs. model accuracy when embedding 3 sample rows per table into prompt context. |
| **Task 06: Token Distribution** | [`tokenizer_utils.py`](data/processing/tokenizer_utils.py) | Tokenizes the entire dataset using Qwen2.5 and computes sequence length percentiles to identify the optimal cutoff. |
| **Tasks 07 & 08: Arrow Export** | [`dataset_builder.py`](data/processing/dataset_builder.py) | Filters records beyond the 95th percentile cutoff (1,024 tokens) and serializes the dataset into memory-mapped Hugging Face Arrow format. |
| **Task 09: Sanity Checking** | [`sanity_checker.py`](data/processing/sanity_checker.py) | Verifies rendered prompt formatting, token lengths, and absence of target SQL leakage in input sequences. |

### Dataset Complexity Distribution (Spider Benchmark)
```
┌────────────┬─────────────┬───────────┐
│ Hardness   │ Train Count │ Dev Count │
├────────────┼─────────────┼───────────┤
│ Easy       │ 1,784       │ 248       │
│ Medium     │ 2,684       │ 446       │
│ Hard       │ 1,744       │ 174       │
│ Extra-Hard │ 1,388       │ 166       │
├────────────┼─────────────┼───────────┤
│ Total      │ 7,600       │ 1,034     │
└────────────┴─────────────┴───────────┘
```

---

## 🧠 QLoRA Training & Optimization

The model was fine-tuned on Kaggle using dual NVIDIA Tesla T4 GPUs with **Distributed Data Parallelism (DDP)** via `accelerate.notebook_launcher`.

```
Base Model: Qwen2.5-Coder-1.5B (1.54B Parameters)
├── Quantization: 4-bit NormalFloat4 (NF4) with Double Quantization
├── Adapter: Low-Rank Adaptation (PEFT LoRA)
│   ├── Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
│   ├── Rank (r): 16 | Scaling Alpha: 32 | Dropout: 0.05
│   └── Trainable Parameters: ~18.4M (1.2% of base model)
├── Optimizer: 8-bit Paged AdamW (paged_adamw_8bit) — 75% VRAM Reduction
├── Learning Rate: 2e-4 with Cosine Annealing & 5% Linear Warmup
├── Effective Batch Size: 16 (4 per device × 4 gradient accumulation steps)
└── Mid-Training Evaluation: Greedy Exact Match (EM) every 200 steps on 500 Spider dev samples
```

### Hyperparameter Cheat Sheet

| Parameter | Configuration | Engineering Rationale |
| :--- | :--- | :--- |
| **Base Architecture** | `Qwen/Qwen2.5-Coder-1.5B` | Superior code reasoning and syntax precision in the sub-3B class. |
| **Quantization** | `bitsandbytes` 4-bit NF4 | Keeps base model in ~1.2 GB VRAM per GPU without precision collapse. |
| **LoRA Target Modules** | All 7 attention & MLP projections | Ensures adaptation across both attention heads and feed-forward feature routing. |
| **Sequence Length** | 1,024 tokens | Covers 95% of Spider schemas without memory fragmentation. |
| **Optimizer** | `paged_adamw_8bit` | Pages memory to system RAM during gradient spikes to eliminate OOMs. |
| **DataLoader** | `num_workers=2`, `pin_memory=True` | Asynchronous DMA streaming prevents GPU starvation between forward passes. |
| **Weight Merging** | `merge_and_unload()` | Fuses LoRA delta weights directly into base FP16 tensors for zero-overhead inference. |

---

## 📈 Evaluation & Benchmark Suite

The evaluation suite implements strict semantic parsing metrics on the Spider development set, moving beyond string matching to verify true query execution semantics.

### 1. Exact Match (EM) Normalization
- **Column-Order Invariance**: Automatically sorts `SELECT` column projections alphabetically, ensuring `SELECT name, age` and `SELECT age, name` evaluate identically.
- **SQL Cleanup**: Strips markdown backticks, semicolons, comments, and extra whitespace before comparison.

### 2. Execution Accuracy (EX) & Multiset Matching
- **Multiset Comparison**: Treats query results as unordered multisets (bags of rows) to ensure row ordering does not penalize valid queries unless `ORDER BY` is required.
- **Column Permutation Search**: Employs an exact column permutation algorithm with multiset column pruning to compare results when columns appear in different projection sequences.
- **Float Rounding & Type Handling**: Compares floating-point values within tolerance (`0.001`) while preserving strings with leading zeros (e.g. zip codes).

### 3. Sandbox Security & Timeout Guardrail
- **Read-Only Enforcement**: Rejects queries attempting database modifications (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`). Inspects subqueries and CTEs for embedded writes.
- **3-Second Query Timeout**: Wraps database execution in a `concurrent.futures.ThreadPoolExecutor` that cancels execution and invokes `sqlite3.Connection.interrupt()` if execution exceeds 3.0 seconds.

### 4. Error Taxonomy
The evaluator classifies query failures into five actionable categories:
- `SyntaxError`: SQL grammar or lexical parsing failures.
- `RuntimeError`: Missing tables, nonexistent columns, or dialect execution faults.
- `EmptyResult`: Query executed successfully but produced 0 rows when gold SQL produced data.
- `Timeout`: Execution exceeded the 3.0-second safety window (e.g. infinite recursive CTEs).
- `SandboxViolation`: Query attempted write operations or multi-statement injection.

---

## 🔌 Multi-Database Connectivity & Dialects

The Gradio workstation includes an integrated database connection manager supporting four execution targets:

| Database | Connection Method | Features |
| :--- | :--- | :--- |
| **SQLite** | Local file / In-memory | Bundled sample database (`sample_company.db`), isolated test fixtures, zero configuration. |
| **PostgreSQL** | SQLAlchemy (`psycopg2`) | Real-time schema introspection via `information_schema`, connection pooling, full ACID support. |
| **MySQL** | SQLAlchemy (`pymysql`) | Standard host/port credentials, table catalog introspection, foreign key resolution. |
| **Supabase (API)** | HTTPS Management REST API | **Passwordless connection**: Connect using your Supabase Project Ref and Personal Access Token. Executes SQL via the Management API and introspects schemas via OpenAPI. |

### Real-Time Dialect Harmonization
Queries generated by the model are automatically harmonized before execution:
```sql
-- SQLite Source
SELECT IFNULL(salary, 0) FROM employees WHERE strftime('%Y', hire_date) = '2024';

-- Automatically Harmonized for PostgreSQL & Supabase
SELECT COALESCE(salary, 0) FROM employees WHERE EXTRACT(YEAR FROM hire_date) = 2024;
```

---

## 🚀 Quickstart

### Option 1: Local Installation

#### 1. Prerequisites
- Python 3.11+
- Git

#### 2. Clone and Install Dependencies
```bash
git clone https://github.com/here-2007/SQL_Engine.git
cd SQL_Engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install deployment dependencies
pip install -r requirements.txt
```

#### 3. Run the Application
The orchestrator starts the FastAPI inference server in the background and launches the Gradio UI:
```bash
python deployment/app.py
```
- **Gradio UI**: Open [http://127.0.0.1:7860](http://127.0.0.1:7860)
- **FastAPI Documentation**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Option 2: Docker & Docker Compose

Run the entire production stack (FastAPI backend + Gradio frontend) in isolated containers:

```bash
# Start both services
docker compose up --build

# Or run in detached mode
docker compose up -d
```

To run individual services:
```bash
docker compose up api   # FastAPI backend only
docker compose up ui    # Gradio frontend only (waits for healthy API)
```

---

### Option 3: Developer Makefile

The project includes a cross-platform Makefile supporting Windows, Linux, and macOS:

```bash
make install          # Install deployment requirements
make install-train    # Install training + GPU libraries
make test             # Run core pipeline + evaluator tests
make test-deployment  # Run API, Gradio, and database integration tests
make test-all         # Run all 233+ tests
make serve            # Start FastAPI inference API on port 8000
make gradio           # Start Gradio UI on port 7860
make lint             # Check code with Ruff linter
make format           # Auto-format codebase with Ruff
make clean            # Cross-platform removal of caches and artifacts
```

---

## 📡 REST API Reference

The FastAPI service exposes interactive Swagger docs at `http://127.0.0.1:8000/docs`.

### 1. Health & Concurrency Check
```http
GET /health
```
**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "model": "text2sql-v1",
  "device": "cpu",
  "version": "1.0.0",
  "queue_depth": 0,
  "total_requests": 42,
  "last_inference_latency_ms": 142.5
}
```

### 2. Natural Language to SQL
```http
POST /v1/tosql
Content-Type: application/json
```
**Request Body**:
```json
{
  "question": "Find the names of all departments with a budget greater than 500000.",
  "schema": "CREATE TABLE department (department_id INT PRIMARY KEY, name TEXT, budget FLOAT);",
  "dialect": "sqlite"
}
```
**Response (`200 OK`)**:
```json
{
  "request_id": "9f1c7d2e-4b8a-4e2a-9f5b-1c3d5e7a9b1c",
  "sql": "SELECT name FROM department WHERE budget > 500000;",
  "model": "text2sql-v1",
  "generation_time_ms": 184.2
}
```

**Machine-Readable Error Codes**:
- `422 VALIDATION_ERROR`: Missing or malformed question or schema.
- `429 RATE_LIMITED`: Exceeded 10 requests/minute or 3 requests/10-second burst. Includes `Retry-After` header.
- `503 SERVICE_BUSY`: Inference queue wait time exceeded 30 seconds.
- `500 INFERENCE_FAILED`: Model generation error.

---

## 📁 Repository Layout

```text
SQL_Engine/
├── Makefile                          # Cross-platform build & run automation
├── Dockerfile                        # Multi-stage production container build
├── docker-compose.yml                # Two-tier service composition (API + UI)
├── pyproject.toml                    # Pytest, Ruff linter, and project metadata
├── requirements.txt                  # Production serving dependencies
├── requirements-train.txt            # GPU fine-tuning & training dependencies
├── .env_example                      # Template environment variable configuration
│
├── deployment/                       # Production Serving Stack
│   ├── api.py                        # FastAPI service with rate limiting & queue metrics
│   ├── api_client.py                 # Resilient HTTP client with typed exception hierarchy
│   ├── app.py                        # Service orchestrator (manages FastAPI subprocess)
│   ├── database.py                   # Multi-database manager & AST dialect harmonizer
│   ├── gradio_app.py                 # Gradio workstation UI (JetBrains Mono terminal)
│   ├── prediction.py                 # Text2SQLEngine model loader & KaggleHub resolver
│   ├── test_api.py                   # API endpoint tests (711 lines)
│   ├── test_api_client.py            # Client exception handling & retry tests
│   ├── test_app.py                   # Orchestrator lifecycle & process tests
│   ├── test_database_integration.py  # SQLite, PostgreSQL, MySQL, Supabase tests
│   ├── test_gradio_app.py            # UI workstation test suite (1,540 lines)
│   └── test_prediction.py            # Inference engine & prompt formatter tests
│
├── data/                             # Data Layer & Preprocessing
│   ├── processing/                   # Preprocessing modules (Tasks 02–09)
│   │   ├── complexity.py             # SQL AST hardness classifier
│   │   ├── dataset_builder.py        # Arrow dataset serializer
│   │   ├── prompt_templates.py       # Multi-style prompt formatters
│   │   ├── run_pipeline.py           # Unified pipeline runner (Tasks 02–09)
│   │   ├── sample_rows_study.py      # Token inflation empirical study
│   │   ├── sanity_checker.py         # Leakage & prompt validation
│   │   ├── schema_serializer.py      # SQLite DDL extraction
│   │   └── tokenizer_utils.py        # Qwen2.5 tokenizer wrapper & visualizer
│   └── reports/                      # JSON audit and token distribution reports
│
├── src/                              # Core Machine Learning Engine
│   ├── training/                     # QLoRA Fine-Tuning
│   │   ├── train_qlora.py            # 4-bit NF4 training with PEFT, TRL & DDP
│   │   ├── eval_metrics.py           # Mid-training Exact Match evaluation callback
│   │   └── merge_model.py            # LoRA weight merging & SHA-256 checksum export
│   ├── inference/                    # Inference Utilities
│   │   └── engine.py                 # Standalone Text2SQLEngine
│   └── evaluation/                   # Spider Evaluation Suite
│       └── evaluator.py              # EM, EX, query timeout, and sandbox guardrail
│
├── tests/                            # Core Test Suite
│   ├── conftest.py                   # Isolated SQLite test fixtures (auto-provisioned)
│   ├── test_evaluator.py             # SQL evaluator test suite (400+ lines)
│   └── test_pipeline.py              # Data pipeline unit tests
│
├── scripts/                          # Automation & Utilities
│   ├── run_eval.py                   # Batch evaluation runner on Spider dev set
│   ├── smoke_test_pipeline.py        # Rapid 2-step end-to-end training smoke test
│   └── generate_notebook.py          # Jupyter notebook generator for Kaggle DDP
│
└── notebooks/                        # Jupyter Notebooks
    └── sql_engine.ipynb              # Complete Kaggle Dual T4 DDP training notebook
```

---

## 🧪 Testing & Quality Assurance

The test suite covers the complete application lifecycle, from data processing and AST hardness parsing to database drivers and Gradio UI components.

```bash
# Run all tests
pytest -v

# Run with coverage report
pytest --cov=deployment --cov=src --cov=data
```

### Test Suite Breakdown
- **Core Pipeline & Evaluator Tests** (`tests/`): **50 tests** covering SQLite resolution, AST column sorting, multiset equality, timeout handling, and sandbox violation trapping.
- **Deployment & API Tests** (`deployment/`): **180 tests** covering FastAPI endpoints, SlowAPI rate limiting, concurrency queue depth, database drivers, Supabase Management API introspection, and Gradio component state management.
- **Total Passing Tests**: **230+ tests** passing with 0 failures.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📚 Acknowledgements & References

- **Yale LILY Lab**: For introducing the [Spider Benchmark](https://yale-lily.github.io/spider) (*Tao Yu et al., EMNLP 2018*).
- **Qwen Team (Alibaba Cloud)**: For the outstanding [Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) foundation model.
- **Hugging Face**: For the `transformers`, `peft`, `trl`, and `datasets` libraries enabling efficient QLoRA fine-tuning.
- **Gradio Team**: For the flexible interactive UI framework.
