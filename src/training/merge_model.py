"""LoRA Adapter Merging and Model Export (Tasks 09 & 10).

Merges trained LoRA adapters into the base model using merge_and_unload(),
saves the standalone model + tokenizer, computes SHA-256 hashes, and logs
training metadata.
"""

import datetime
import hashlib
import json
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def merge_lora_and_save(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    tokenizer_id: str | None = None,
    training_metadata: dict[str, Any] | None = None,
    torch_dtype: torch.dtype | None = None,
    device_map: str = "auto",
) -> dict[str, Any]:
    """Merge LoRA adapter into base model and export complete standalone model.

    Tasks 09 & 10 implementation.
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_id = tokenizer_id or base_model_id

    # Determine precision
    if torch_dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    print(f"Loading base model '{base_model_id}' with dtype {torch_dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"Attaching LoRA adapter from '{adapter_path}'...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Executing merge_and_unload() to fold adapter weights into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged standalone model to '{output_dir}'...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    print(f"Saving tokenizer to '{output_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Compute SHA-256 checksums of saved model files
    print("Computing file hashes for artifacts...")
    file_hashes = {}
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath):
            file_hashes[filename] = compute_file_sha256(filepath)

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "base_model": base_model_id,
        "adapter_path": adapter_path,
        "output_directory": output_dir,
        "merged_dtype": str(torch_dtype),
        "exported_files": list(file_hashes.keys()),
        "file_sha256_checksums": file_hashes,
        "training_details": training_metadata or {},
    }

    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"SUCCESS: Standalone model & metadata exported to '{output_dir}'")
    print(f"Metadata file: {metadata_path}")
    return metadata
