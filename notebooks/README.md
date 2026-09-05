# Notebooks Directory

This directory contains Jupyter Notebooks for interactive model training and experimentation.

- **`sql_engine.ipynb`**: The primary self-contained Kaggle notebook implementing Tasks 01–10 (QLoRA 4-bit NF4 fine-tuning with PEFT + TRL).
  - Designed for Kaggle **Dual T4 GPUs** (`GPU T4 x 2`) with automatic DDP parallelization via `accelerate.notebook_launcher`.
  - Also works seamlessly on single GPUs (T4, P100, A100) or local CPU/MPS.
  - Generates merged standalone model weights with SHA-256 integrity checksums.
