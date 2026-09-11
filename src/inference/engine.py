"""Text-to-SQL Inference Engine.

Loads merged standalone model (or base + LoRA adapter) and generates
deterministic SQLite queries from natural language questions and schema DDL.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_default_model_path() -> str:
    """Find the best available local model path."""
    candidates = [
        os.path.join(REPO_ROOT, "models/text2sql-qwen-pytorch-v1-v1/text2sql-v1"),
        os.path.join(REPO_ROOT, "models/text2sql-v1"),
        os.path.join(REPO_ROOT, "models/smoke-test-v1"),
        "Qwen/Qwen2.5-Coder-1.5B",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


class Text2SQLEngine:
    """Production Text-to-SQL inference generator."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        self.model_path = model_path or get_default_model_path()

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            if self.device == "cuda":
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif self.device == "mps":
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        print(f"Loading Text-to-SQL model from: {self.model_path}")
        print(f"Target Device: {self.device} | Precision: {self.torch_dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if self.device == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if self.device in ("mps", "cpu") and device_map is None:
            self.model = self.model.to(self.device)

        self.model.eval()
        print("Model successfully loaded and ready for inference.")

    def format_prompt(self, question: str, schema: str) -> str:
        """Construct prompt matching the training prompt distribution."""
        return (
            "You are an expert SQL engineer. Given the database schema, "
            "write the exact SQLite query that answers the user question.\n\n"
            f"### Database Schema:\n{schema.strip()}\n\n"
            f"### Question:\n{question.strip()}\n\n"
            "### SQL:\n"
        )

    def generate_sql(
        self,
        question: str,
        schema: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """Generate SQL query for the given question and schema."""
        prompt = self.format_prompt(question, schema)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature if temperature > 0.0 else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        sql = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean any trailing markdown delimiters or comments
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        return sql.strip()
