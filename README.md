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
┌─────────────┐       HTTP        ┌──────────────────┐
## Prompt Engineering
Markdown, ChatML, and Code Comment prompt templates.

<!-- log: 2026-08-28 18:05:39 - refactor(prompting): standardize schema and question formatting delimiters -->
