#!/usr/bin/env python3
"""
finetune.py
Production-Grade LoRA Fine-Tuning & Model Versioning Engine for Alisa:
  - Trains Apple Silicon MLX (mlx-lm) or Hugging Face PEFT LoRA on All-Rounder instruction datasets.
  - Automatically manages model versioning (v1, v2, v3...) via ModelVersionManager.
  - Automatically checks and prepares train/valid/test datasets via DatasetBuilder.
  - Supports incremental continuous learning (resuming from previous active adapter)
    OR starting fresh from the base model (--fresh).
  - Registers version metadata and sets newly fine-tuned model as ACTIVE.
"""

import argparse
import os
import platform
import re
import subprocess
import sys
from typing import Optional

from dataset_builder import DatasetBuilder
from model_manager import ModelVersionManager, clean_model_id


def count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def run_mlx_lora(
    model_id: str,
    data_dir: str,
    adapter_dir: str,
    iters: int,
    batch_size: int,
    lora_layers: int,
    max_seq_length: int = 1024,
    grad_checkpoint: bool = True,
    resume_from_adapter: Optional[str] = None,
) -> None:
    os.makedirs(adapter_dir, exist_ok=True)
    clean_id = clean_model_id(model_id)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        clean_id,
        "--train",
        "--data",
        data_dir,
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(lora_layers),
        "--max-seq-length",
        str(max_seq_length),
        "--mask-prompt",
        "--adapter-path",
        adapter_dir,
    ]

    if resume_from_adapter:
        # Check if resume_from_adapter has adapters.safetensors
        resume_safetensors = (
            resume_from_adapter
            if resume_from_adapter.endswith(".safetensors")
            else os.path.join(resume_from_adapter, "adapters.safetensors")
        )
        if os.path.exists(resume_safetensors):
            cmd.extend(["--resume-adapter-file", resume_safetensors])
            print(f"[finetune] Resuming incremental training from prior adapter: {resume_safetensors}")

    if grad_checkpoint:
        cmd.append("--grad-checkpoint")

    print(f"\n[finetune] Launching Apple Silicon Metal LoRA training:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"[finetune] ✓ LoRA adapter weights saved to {adapter_dir}")


def run_hf_peft(
    model_id: str,
    data_dir: str,
    adapter_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] Running Hugging Face PEFT LoRA fine-tuning on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_dir, "train.jsonl"),
            "validation": os.path.join(data_dir, "valid.jsonl"),
        },
    )

    def format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    formatted_ds = dataset.map(format_chat)

    training_args = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=(device == "cuda"),
        report_to="none",
        dataset_text_field="text",
        max_seq_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_ds["train"],
        eval_dataset=formatted_ds["validation"],
        processing_class=tokenizer,
        args=training_args,
    )
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[finetune] ✓ Hugging Face PEFT LoRA adapter saved to {adapter_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Alisa All-Rounder LLM with automated model versioning.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model ID (e.g., Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct, meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mlx", "hf"],
        default="auto",
        help="Training framework: 'mlx' (Apple Silicon MLX), 'hf' (Transformers + PEFT), or 'auto'",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--adapter-dir", default=None, help="Explicit output path for LoRA adapter (if omitted, auto-versions)")
    parser.add_argument("--iters", type=int, default=600, help="Training iterations for MLX LoRA")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for HF PEFT")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device training batch size (default: 1 for low memory)")
    parser.add_argument("--lora-layers", type=int, default=16, help="Number of transformer layers to adapt in MLX")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum sequence length (default: 1024)")
    parser.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--fresh", action="store_true", help="Train from fresh base model, do not resume from active adapter")
    parser.add_argument("--notes", default="", help="Optional description note for this model version")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for HF PEFT")
    args = parser.parse_args()

    args.model = clean_model_id(args.model)

    # 1. Ensure All-Rounder Datasets are ready
    train_path = os.path.join(args.data_dir, "train.jsonl")
    valid_path = os.path.join(args.data_dir, "valid.jsonl")
    test_path = os.path.join(args.data_dir, "test.jsonl")

    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        print("[finetune] Training datasets not found. Generating balanced All-Rounder dataset...")
        builder = DatasetBuilder(output_dir=args.data_dir)
        builder.build_all_rounder_dataset()

    train_count = count_jsonl_lines(train_path)
    valid_count = count_jsonl_lines(valid_path)
    test_count = count_jsonl_lines(test_path)
    print(f"[finetune] Dataset verified: {train_count} train | {valid_count} valid | {test_count} test examples.")

    if train_count == 0:
        print("[finetune] Error: train.jsonl is empty. Please add data or run dataset_builder.py first.")
        sys.exit(1)

    # 2. Setup Versioning Manager
    version_mgr = ModelVersionManager()
    active_version = version_mgr.get_active_version()
    print(f"[finetune] Current Active Version: {active_version.get('version_id')} ({active_version.get('name')})")

    if args.adapter_dir:
        target_adapter_dir = args.adapter_dir
        new_version_id = "custom"
    else:
        new_version_id, target_adapter_dir = version_mgr.prepare_new_version(
            base_model=args.model,
            notes=args.notes or f"Trained on {train_count} All-Rounder samples",
        )
        print(f"[finetune] 📦 Allocating new model version: {new_version_id} -> {target_adapter_dir}")

    # Determine resume adapter
    resume_adapter = None
    if not args.fresh:
        resume_adapter = version_mgr.get_active_adapter_path()

    # 3. Detect Backend
    backend = args.backend
    if backend == "auto":
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        try:
            import mlx_lm  # noqa: F401
            backend = "mlx" if is_apple_silicon else "hf"
        except ImportError:
            backend = "hf"

    print(f"[finetune] Framework: {backend.upper()} | Base Model: {args.model}")

    # 4. Run Training
    if backend == "mlx":
        run_mlx_lora(
            model_id=args.model,
            data_dir=args.data_dir,
            adapter_dir=target_adapter_dir,
            iters=args.iters,
            batch_size=args.batch_size,
            lora_layers=args.lora_layers,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=not args.no_grad_checkpoint,
            resume_from_adapter=resume_adapter,
        )
    else:
        run_hf_peft(
            model_id=args.model,
            data_dir=args.data_dir,
            adapter_dir=target_adapter_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    # 5. Register and Activate Version
    if new_version_id != "custom":
        version_mgr.register_and_activate_version(
            version_id=new_version_id,
            version_dir=target_adapter_dir,
            base_model=args.model,
            train_samples=train_count,
            valid_samples=valid_count,
            test_samples=test_count,
            metrics={"iters": args.iters if backend == "mlx" else args.epochs, "backend": backend},
            notes=args.notes or f"Fine-tuned {args.model} with {backend.upper()}",
        )
        print(f"\n🎉 Successfully created and activated Model Version: {new_version_id}!")
        print(f"   • Active Adapter: {target_adapter_dir}")
        print(f"   • Rollback command: python3 model_manager.py --rollback")
        print(f"   • Reset to fresh base: python3 model_manager.py --rollback-fresh")
    else:
        print(f"\n[finetune] Saved custom adapter to {target_adapter_dir}")


if __name__ == "__main__":
    main()
