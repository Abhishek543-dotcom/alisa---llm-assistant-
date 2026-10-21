#!/usr/bin/env python3
"""
finetune.py
Fine-tunes a small 1B-1.5B parameter local LLM (Qwen/Qwen2.5-1.5B-Instruct or
meta-llama/Llama-3.2-1B-Instruct) on your personal email instruction dataset
using either:
  - Apple Silicon MLX (`mlx-lm`) for ultra-fast local unified-memory LoRA training
  - Hugging Face `transformers` + `peft` + `trl` (PyTorch MPS / CUDA / CPU)
"""

import argparse
import os
import platform
import re
import subprocess
import sys


def run_mlx_lora(
    model_id: str,
    data_dir: str,
    adapter_dir: str,
    iters: int,
    batch_size: int,
    lora_layers: int,
    max_seq_length: int = 1024,
    grad_checkpoint: bool = True,
    resume: bool = True,
) -> None:
    os.makedirs(adapter_dir, exist_ok=True)
    clean_model_id = re.sub(r"\s+", "", model_id.strip())
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        clean_model_id,
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
    resume_file = os.path.join(adapter_dir, "adapters.safetensors")
    if resume and os.path.exists(resume_file):
        cmd.extend(["--resume-adapter-file", resume_file])
        print(f"[finetune] Resuming incremental training from existing adapter: {resume_file}")

    if grad_checkpoint:
        cmd.append("--grad-checkpoint")
    print(f"[finetune] Running MLX LoRA fine-tuning:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[finetune] MLX LoRA adapters saved to {adapter_dir}")


def run_hf_peft(model_id: str, data_dir: str, adapter_dir: str, epochs: int, batch_size: int, lr: float) -> None:
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
    print(f"[finetune] Hugging Face PEFT LoRA adapters saved to {adapter_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a lightweight model on your Gmail dataset.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model ID (e.g., Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct, meta-llama/Llama-3.2-1B-Instruct, SmolLM2-360M-Instruct)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mlx", "hf"],
        default="auto",
        help="Training framework: 'mlx' (Apple Silicon MLX), 'hf' (Transformers + PEFT), or 'auto'",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--adapter-dir", default="adapters/personal_lora", help="Output path for trained LoRA adapters")
    parser.add_argument("--iters", type=int, default=600, help="Training iterations for MLX LoRA")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for HF PEFT")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device training batch size (default: 1 for low memory)")
    parser.add_argument("--lora-layers", type=int, default=16, help="Number of transformer layers to adapt in MLX")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum sequence length (default: 1024)")
    parser.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing adapter, start fresh")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for HF PEFT")
    args = parser.parse_args()
    args.model = re.sub(r"\s+", "", args.model.strip())

    backend = args.backend
    if backend == "auto":
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        try:
            import mlx_lm  # noqa: F401
            backend = "mlx" if is_apple_silicon else "hf"
        except ImportError:
            backend = "hf"

    if backend == "mlx":
        run_mlx_lora(
            args.model,
            args.data_dir,
            args.adapter_dir,
            args.iters,
            args.batch_size,
            args.lora_layers,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=not args.no_grad_checkpoint,
            resume=not args.no_resume,
        )
    else:
        run_hf_peft(args.model, args.data_dir, args.adapter_dir, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
