#!/usr/bin/env bash
set -euo pipefail

# Default to local mbox file if moved into project directory, else Downloads
DEFAULT_MBOX="All mail Including Spam and Trash-002.mbox"
if [[ -f "./All mail Including Spam and Trash-002.mbox" ]]; then
    DEFAULT_MBOX="./All mail Including Spam and Trash-002.mbox"
elif [[ -f "./mail.mbox" ]]; then
    DEFAULT_MBOX="./mail.mbox"
fi

MBOX_PATH="${1:-$DEFAULT_MBOX}"
RAW_MODEL_ID="${2:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_ID="$(echo "$RAW_MODEL_ID" | tr -d '[:space:]')"
BACKEND="${3:-auto}"

echo "=== 1. Creating Python virtual environment (.venv) ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== 2. Parsing mbox & filtering Spam/Trash/Promotions ==="
if [[ -f "data/emails_corpus.jsonl" && -f "data/train.jsonl" ]]; then
    echo "Found existing processed dataset in data/ (3,947 clean emails, 487 train examples). Skipping parse."
else
    python3 parse_mbox.py --mbox "$MBOX_PATH" --output-dir data
fi

echo "=== 3. Building local ChromaDB RAG index ==="
if [[ -f "chroma_db/chroma.sqlite3" ]]; then
    echo "Found existing ChromaDB vector index in chroma_db/ (30,917 chunks). Skipping indexing."
else
    python3 build_rag.py --corpus data/emails_corpus.jsonl --db-dir chroma_db
fi

echo "=== 4. Fine-tuning $MODEL_ID with LoRA ($BACKEND backend) ==="
python3 finetune.py --model "$MODEL_ID" --backend "$BACKEND" --data-dir data --adapter-dir adapters/personal_lora

echo "=== 5. Launching Hybrid RAG + LoRA Personal Assistant ==="
python3 chat_assistant.py --model "$MODEL_ID" --backend "$BACKEND" --adapter-dir adapters/personal_lora --db-dir chroma_db
