#!/usr/bin/env python3
"""
chat_assistant.py
Hybrid RAG + LoRA Personal Assistant:
  1. Queries your local ChromaDB vector index for the top-k relevant emails (exact facts, dates, senders)
  2. Feeds the retrieved context into your fine-tuned Qwen2.5-1.5B-Instruct / Llama-3.2-1B-Instruct LoRA model
"""

import argparse
import os
import platform
import re
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer


def retrieve_email_context(query: str, db_dir: str, embedder: SentenceTransformer, top_k: int = 5) -> str:
    if not os.path.exists(db_dir):
        return ""
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name="personal_emails")
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    docs: List[str] = results.get("documents", [[]])[0]
    if not docs:
        return ""
    formatted = "\n\n---\n\n".join(f"[Email Snippet {i+1}]\n{doc}" for i, doc in enumerate(docs))
    return formatted


def generate_mlx(model_id: str, adapter_dir: str, messages: List[dict], max_tokens: int = 512) -> str:
    from mlx_lm import generate, load

    adapter_arg = adapter_dir if os.path.exists(adapter_dir) else None
    model, tokenizer = load(model_id, adapter_path=adapter_arg)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def generate_hf(model_id: str, adapter_dir: str, messages: List[dict], max_tokens: int = 512) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    if os.path.exists(adapter_dir):
        model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    else:
        model = base_model

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3, do_sample=True)
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions or draft emails with your Hybrid RAG + LoRA LLM.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model ID")
    parser.add_argument("--adapter-dir", default="adapters/personal_lora", help="Trained LoRA adapter directory")
    parser.add_argument("--db-dir", default="chroma_db", help="ChromaDB directory")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model ID")
    parser.add_argument("--backend", choices=["auto", "mlx", "hf"], default="auto", help="Inference backend")
    parser.add_argument("--top-k", type=int, default=5, help="Number of emails to retrieve via RAG")
    parser.add_argument("--query", default=None, help="Single question to ask (launches interactive shell if omitted)")
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

    embedder = SentenceTransformer(args.embedding_model)

    def answer_question(user_query: str) -> str:
        context = retrieve_email_context(user_query, args.db_dir, embedder, top_k=args.top_k)
        system_content = (
            "You are Abhishek Tiwari's personal AI assistant trained on his email archive. "
            "Use the retrieved email snippets below to answer accurately with exact dates, names, and facts, "
            "while matching Abhishek's personal communication style.\n\n"
            f"RETRIEVED EMAIL CONTEXT:\n{context or 'No matching emails found.'}"
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query},
        ]
        if backend == "mlx":
            return generate_mlx(args.model, args.adapter_dir, messages)
        return generate_hf(args.model, args.adapter_dir, messages)

    if args.query:
        print(answer_question(args.query))
        return

    print("Personal Email LLM + RAG Assistant ready. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        print("\nAssistant >", answer_question(q))


if __name__ == "__main__":
    main()
