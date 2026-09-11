import os

import kagglehub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Text2SQLEngine:

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        if model_path is not None:
            self.model_path = model_path
        elif os.environ.get("TEXT2SQL_MODEL_PATH"):
            self.model_path = os.environ["TEXT2SQL_MODEL_PATH"]
        else:
            candidate_paths = [
                os.path.expanduser("~/.cache/kagglehub/models/pernavjain/text2sql-qwen/pyTorch/v1/2/text2sql-v1"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "text2sql-v1"),
                os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "text2sql-qwen-pytorch-v1-v1", "text2sql-v1")),
            ]
            found_path = next((p for p in candidate_paths if os.path.isdir(p)), None)
            if found_path:
                self.model_path = found_path
            else:
                import glob
                cached_matches = glob.glob(os.path.expanduser("~/.cache/kagglehub/models/pernavjain/text2sql-qwen/pyTorch/v1/*/text2sql-v1"))
                if cached_matches:
                    self.model_path = cached_matches[0]
                else:
                    dl_path = kagglehub.model_download("pernavjain/text2sql-qwen/pyTorch/v1")
                    sub_path = os.path.join(dl_path, "text2sql-v1")
                    self.model_path = sub_path if os.path.isdir(sub_path) else dl_path

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
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if self.device in ("mps", "cpu") and device_map is None:
            self.model = self.model.to(self.device)

        self.model.eval()
        print("Model successfully loaded and ready for inference.")

    def format_prompt(self, question: str, schema: str, dialect: str = "sqlite") -> str:
        dialect_display = {
            "postgresql": "PostgreSQL",
            "postgres": "PostgreSQL",
            "supabase": "PostgreSQL",
            "supabase_api": "PostgreSQL",
            "supabase (api)": "PostgreSQL",
            "mysql": "MySQL",
            "sqlite": "SQLite",
        }.get((dialect or "sqlite").strip().lower(), (dialect or "sqlite").strip().title())

        return (
            "You are an expert SQL engineer. Given the database schema, "
            f"write the exact {dialect_display} query that answers the user question.\n\n"
            f"### Database Schema:\n{schema.strip()}\n\n"
            f"### Question:\n{question.strip()}\n\n"
            "### SQL:\n"
        )

    def generate_sql(
        self,
        question: str,
        schema: str,
        dialect: str = "sqlite",
        max_new_tokens: int = 512,
        temperature: float = 0.5,
    ) -> str:
        prompt = self.format_prompt(question, schema, dialect=dialect)
        max_ctx = min(getattr(self.tokenizer, "model_max_length", 4096), 4096)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_ctx,
        ).to(self.device)

        do_sample = bool(temperature > 0.0)
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        sql = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean any markdown code delimiters or commentary
        sql = sql.strip()
        if "```sql" in sql:
            sql = sql.split("```sql", 1)[1]
            if "```" in sql:
                sql = sql.split("```", 1)[0]
        elif "```" in sql:
            sql = sql.split("```", 1)[1]
            if "```" in sql:
                sql = sql.split("```", 1)[0]
        sql = sql.strip()

        # If there's commentary following a semicolon, keep only the SQL statement
        if ";" in sql:
            first_stmt = sql.split(";")[0].strip() + ";"
            after_semi = sql[sql.index(";") + 1:].strip()
            first_word = after_semi.split()[0].upper() if after_semi.split() else ""
            if first_word and first_word not in ("SELECT", "WITH", "EXPLAIN"):
                sql = first_stmt

        # Apply dialect harmonization post-processing
        from database import adapt_sql_dialect
        sql = adapt_sql_dialect(sql, dialect=dialect)

        return sql.strip()
