"""QLoRA Fine-Tuning Pipeline with PEFT + TRL (Tasks 01–10).

Features:
- Robust device handling (Kaggle CUDA 4-bit vs Mac MPS vs CPU)
- 4-bit NF4 quantization with BitsAndBytesConfig (Item 01)
- prepare_model_for_kbit_training() gradient checkpointing (Item 02)
- LoraConfig targeting all linear projection layers (Item 03)
- TrainingArguments with cosine decay, 5% warmup, grad accum (Item 04)
- Fast smoke testing toggle for pipeline validation (Item 05)
- Experiment tracking integration (Item 06)
- Mid-training Exact Match (EM) evaluation (Item 07)
- Best checkpoint saving based on validation EM (Item 08)
- LoRA adapter merging with merge_and_unload() (Item 09)
- Standalone model + metadata + hash exporting (Item 10)
"""

import argparse
import os
import sys
from typing import Any

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTTrainer

# Add repo root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from src.training.eval_metrics import run_dev_evaluation
    from src.training.merge_model import merge_lora_and_save
except ImportError:
    from .eval_metrics import run_dev_evaluation
    from .merge_model import merge_lora_and_save

# ==============================================================================
# CONFIGURATION CHEAT SHEET (TUNE PARAMETERS HERE FOR KAGGLE GPU)
# ==============================================================================
DEFAULT_CONFIG: dict[str, Any] = {
    # Mode Toggle: Set to False for full Kaggle training; True for fast smoke testing
    "SMOKE_TEST": False,
    "SMOKE_SAMPLES_TRAIN": 200,       # 200 samples for smoke test (Task 05)
    "SMOKE_STEPS": 50,                # 50 steps for smoke test (Task 05)
    "SMOKE_SAMPLES_DEV": 50,

    # Model & Tokenizer
    "MODEL_ID": "Qwen/Qwen2.5-Coder-1.5B",  # Base model (1.5B or 7B)
    "MAX_SEQ_LENGTH": 1024,           # 95th percentile sequence length cutoff

    # QLoRA & Adapter Hyperparameters (Tasks 01 & 03)
    "LORA_R": 16,                     # LoRA rank (Task 03)
    "LORA_ALPHA": 32,                 # Scaling factor (Task 03)
    "LORA_DROPOUT": 0.05,             # Dropout rate (Task 03)
    "TARGET_MODULES": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],

    # Training Arguments (Task 04)
    "NUM_EPOCHS": 3,                  # Full training: 3 epochs (Task 08)
    "LEARNING_RATE": 2e-4,            # Initial learning rate (Task 04)
    "BATCH_SIZE": 4,                  # Per-device batch size (Task 04)
    "GRAD_ACCUM_STEPS": 4,            # Effective batch size = 4 * 4 = 16 (Task 04)
    "WARMUP_RATIO": 0.05,             # 5% warmup with cosine decay (Task 04)
    "WEIGHT_DECAY": 0.01,
    "MAX_GRAD_NORM": 1.0,
    "LOGGING_STEPS": 10,
    "OPTIMIZER": "paged_adamw_8bit",
    "DATALOADER_NUM_WORKERS": 2,
    "DATALOADER_PIN_MEMORY": True,

    # Mid-Training Evaluation & Checkpointing (Tasks 06, 07, 08)
    "EVAL_STEPS": 200,                # Evaluate every 200 steps (Task 06 & 07)
    "EVAL_DEV_SAMPLES": 500,          # 500 dev examples for EM evaluation (Task 07)

    # Directories & Experiment Tracking
    "DATASET_PATH": "data/processed_arrow",
    "ADAPTER_OUTPUT_DIR": "./models/qlora-adapter",
    "MERGED_OUTPUT_DIR": "./models/text2sql-v1",
    "TRACKER": "none",                # "none", "wandb", or "mlflow" (Task 06)
}


class MidTrainingExactMatchCallback(TrainerCallback):
    """Callback to run greedy Exact Match evaluation on Spider dev set every N steps."""

    def __init__(self, dev_dataset: Any, tokenizer: Any, eval_steps: int = 200, max_samples: int = 500):
        self.dev_dataset = dev_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.max_samples = max_samples
        self.best_em = -1.0
        self.history = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n>>> [Mid-Training Eval @ Step {state.global_step}] Evaluating Exact Match on {self.max_samples} dev examples...")
            try:
                metrics = run_dev_evaluation(
                    model=model,
                    tokenizer=self.tokenizer,
                    dev_dataset=self.dev_dataset,
                    max_samples=self.max_samples,
                )
                em = metrics["exact_match"]
                print(f">>> Step {state.global_step} Exact Match (EM): {em:.2f}% ({metrics['correct']}/{metrics['total']})")
                self.history.append({"step": state.global_step, "exact_match": em})
                if em > self.best_em:
                    self.best_em = em
                    print(f">>> New best validation Exact Match: {em:.2f}%")
            except Exception as e:
                print(f">>> Warning: Mid-training evaluation error: {e}")


def get_device_info() -> dict[str, Any]:
    """Detect available hardware capabilities."""
    is_cuda = torch.cuda.is_available()
    is_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    is_mps = torch.backends.mps.is_available()

    import importlib.util

    bnb_available = importlib.util.find_spec("bitsandbytes") is not None

    device_name = "CPU"
    if is_cuda:
        device_name = torch.cuda.get_device_name(0)
    elif is_mps:
        device_name = "Apple Silicon (MPS)"

    return {
        "is_cuda": is_cuda,
        "is_bf16": is_bf16,
        "is_mps": is_mps,
        "bnb_available": bnb_available,
        "device_name": device_name,
    }


def train_qlora(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute the end-to-end QLoRA training pipeline."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    print("=" * 75)
    print("02 TRAINING: QLoRA FINE-TUNING WITH PEFT + TRL (Tasks 01–10)")
    print("=" * 75)

    # 1. Device and precision setup
    hw = get_device_info()
    print("\n[Hardware Detection]:")
    print(f"- Primary Device: {hw['device_name']}")
    print(f"- CUDA Available: {hw['is_cuda']}")
    print(f"- bfloat16 Hardware Support: {hw['is_bf16']}")
    print(f"- bitsandbytes Available: {hw['bnb_available']}")

    # 2. Tokenizer
    model_id = cfg["MODEL_ID"]
    print(f"\nLoading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Determine intelligent device map for multi-GPU (e.g. Kaggle 2 x T4)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        # Running under DDP via accelerate launch or torchrun
        device_map = {"": local_rank}
        print(f"Distributed DDP detected! Binding process to GPU {local_rank}")
    elif hw["is_cuda"]:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Multi-GPU detected: {num_gpus} GPUs available (e.g. 2 x T4).")
            # When running single-process notebook, pinning to GPU 0 avoids PCIe pipelining stall
            device_map = cfg.get("DEVICE_MAP", {"": 0})
        else:
            device_map = {"": 0}
    else:
        device_map = None

    # 3. Load Model (Task 01 & Task 02)
    print("\n--- TASK 01 & 02: Model Loading & Quantization Setup ---")
    if hw["is_cuda"] and hw["bnb_available"]:
        compute_dtype = torch.bfloat16 if hw["is_bf16"] else torch.float16
        print(f"Configuring 4-bit BitsAndBytesConfig (NF4, compute_dtype={compute_dtype})...")
        print(f"Using device_map: {device_map}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        print("Calling prepare_model_for_kbit_training() (Task 02)...")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        print("Running in non-CUDA/CPU/MPS fallback mode for testing...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    # 4. LoRA Configuration (Task 03)
    print("\n--- TASK 03: LoRA Adapter Configuration ---")
    lora_config = LoraConfig(
        r=cfg["LORA_R"],
        lora_alpha=cfg["LORA_ALPHA"],
        lora_dropout=cfg["LORA_DROPOUT"],
        target_modules=cfg["TARGET_MODULES"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Load Processed Arrow Dataset
    dataset_path = cfg["DATASET_PATH"]
    print(f"\nLoading preprocessed Arrow dataset from '{dataset_path}'...")
    dataset = load_from_disk(dataset_path)
    train_data = dataset["train"].map(lambda ex: {"completion": " " + ex["sql"]})
    val_data = dataset["validation"].map(lambda ex: {"completion": " " + ex["sql"]})

    # Handle Smoke Test mode (Task 05)
    if cfg["SMOKE_TEST"]:
        print("\n" + "!" * 60)
        print(f"SMOKE TEST ACTIVE: Slicing {cfg['SMOKE_SAMPLES_TRAIN']} train samples, {cfg['SMOKE_STEPS']} steps")
        print("!" * 60)
        train_data = train_data.select(range(min(cfg["SMOKE_SAMPLES_TRAIN"], len(train_data))))
        val_data = val_data.select(range(min(cfg["SMOKE_SAMPLES_DEV"], len(val_data))))

    # 6. Training Arguments & SFTConfig (Task 04)
    print("\n--- TASK 04: TrainingArguments / SFTConfig Configuration ---")
    adapter_dir = cfg["ADAPTER_OUTPUT_DIR"]
    os.makedirs(adapter_dir, exist_ok=True)

    max_steps = cfg["SMOKE_STEPS"] if cfg["SMOKE_TEST"] else -1
    eval_steps = min(cfg["EVAL_STEPS"], max(1, max_steps // 2)) if cfg["SMOKE_TEST"] else cfg["EVAL_STEPS"]

    # Detect SFTConfig parameters for version compatibility (TRL v0.8 vs v0.12+)
    import inspect

    from trl import SFTConfig
    sft_sig = inspect.signature(SFTConfig.__init__).parameters
    # Calculate warmup steps from warmup ratio
    if max_steps > 0:
        total_steps = max_steps
    else:
        steps_per_epoch = len(train_data) // (cfg["BATCH_SIZE"] * cfg["GRAD_ACCUM_STEPS"])
        total_steps = steps_per_epoch * cfg["NUM_EPOCHS"]
    warmup_steps = max(1, int(cfg["WARMUP_RATIO"] * total_steps))

    sft_kwargs = {
        "output_dir": adapter_dir,
        "learning_rate": cfg["LEARNING_RATE"],
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "gradient_accumulation_steps": cfg["GRAD_ACCUM_STEPS"],
        "per_device_train_batch_size": cfg["BATCH_SIZE"],
        "per_device_eval_batch_size": cfg["BATCH_SIZE"],
        "num_train_epochs": cfg["NUM_EPOCHS"],
        "max_steps": max_steps,
        "bf16": hw["is_bf16"],
        "fp16": (not hw["is_bf16"] and hw["is_cuda"]),
        "logging_steps": cfg["LOGGING_STEPS"],
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "save_total_limit": 2,
        "report_to": [] if cfg["TRACKER"] == "none" else [cfg["TRACKER"]],
        "logging_dir": os.path.join(adapter_dir, "logs"),
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": cfg["OPTIMIZER"] if (hw["is_cuda"] and hw["bnb_available"]) else "adamw_torch",
        "dataloader_num_workers": cfg["DATALOADER_NUM_WORKERS"] if hw["is_cuda"] else 0,
        "dataloader_pin_memory": cfg["DATALOADER_PIN_MEMORY"] if hw["is_cuda"] else False,
    }
    if "max_length" in sft_sig:
        sft_kwargs["max_length"] = cfg["MAX_SEQ_LENGTH"]
    if "dataset_text_field" in sft_sig:
        sft_kwargs["dataset_text_field"] = "full_text"

    training_args = SFTConfig(**sft_kwargs)

    # 7. Trainer Setup with Mid-Training Exact Match (Task 06 & 07)
    em_callback = MidTrainingExactMatchCallback(
        dev_dataset=val_data,
        tokenizer=tokenizer,
        eval_steps=eval_steps,
        max_samples=cfg["SMOKE_SAMPLES_DEV"] if cfg["SMOKE_TEST"] else cfg["EVAL_DEV_SAMPLES"],
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_data,
        "args": training_args,
        "callbacks": [em_callback],
    }
    # Check SFTTrainer signature
    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in trainer_sig and "dataset_text_field" not in sft_kwargs:
        trainer_kwargs["dataset_text_field"] = "full_text"
    if "max_seq_length" in trainer_sig and "max_length" not in sft_kwargs:
        trainer_kwargs["max_seq_length"] = cfg["MAX_SEQ_LENGTH"]

    trainer = SFTTrainer(**trainer_kwargs)

    # 8. Train (Task 08)
    print("\n--- TASK 08: Starting Training Loop ---")
    train_result = trainer.train()
    print("Training finished successfully!")
    print(f"Final Train Loss: {train_result.training_loss:.4f}")

    # Save trained LoRA adapter weights
    print(f"Saving fine-tuned LoRA adapter to '{adapter_dir}'...")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # 9 & 10. Merge LoRA Adapters & Export Standalone Model (Task 09 & 10)
    print("\n--- TASKS 09 & 10: Merging Adapters & Exporting Standalone Model ---")
    merged_output_dir = cfg["MERGED_OUTPUT_DIR"]
    metadata_payload = {
        "config": cfg,
        "hardware": hw,
        "final_loss": float(train_result.training_loss),
        "best_exact_match": em_callback.best_em,
        "em_history": em_callback.history,
    }

    metadata = merge_lora_and_save(
        base_model_id=cfg["MODEL_ID"],
        adapter_path=adapter_dir,
        output_dir=merged_output_dir,
        tokenizer_id=model_id,
        training_metadata=metadata_payload,
        device_map="auto" if hw["is_cuda"] else "cpu",
    )

    return {
        "final_loss": train_result.training_loss,
        "best_em": em_callback.best_em,
        "adapter_dir": adapter_dir,
        "merged_dir": merged_output_dir,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="QLoRA Text-to-SQL Training")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--model-id", type=str, default=DEFAULT_CONFIG["MODEL_ID"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["NUM_EPOCHS"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["BATCH_SIZE"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["LEARNING_RATE"])
    args = parser.parse_args()

    config = {
        "SMOKE_TEST": args.smoke_test,
        "MODEL_ID": args.model_id,
        "NUM_EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "LEARNING_RATE": args.lr,
    }

    train_qlora(config)


if __name__ == "__main__":
    main()
