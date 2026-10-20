# Personal Email LLM & Hybrid RAG Project

This project turns your Gmail Takeout archive (`/Users/abhishektiwari/Downloads/All mail Including Spam and Trash-002.mbox` or local `All mail Including Spam and Trash-002.mbox`) into a **Hybrid Local AI Assistant** that combines:
1. **LoRA Fine-Tuning** on [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (lightweight, ~490M params, memory-friendly for MacBooks) or [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) / [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) using Apple Silicon [mlx-lm](https://github.com/ml-explore/mlx-lm) or Hugging Face [PEFT](https://github.com/huggingface/peft) so the model learns your personal writing style, tone, and reply patterns.
2. **Local RAG Vector Search** using [ChromaDB](https://github.com/chroma-core/chroma) and [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings so your assistant retrieves exact dates, names, subjects, and historical facts from your emails without hallucinating.

## Project Structure

* `parse_mbox.py`: Streams your `.mbox` archive, strips out `Spam`, `Trash`, `Category Promotions`, `Category Updates`, `Category Social`, `Category Forums`, and automated `noreply` newsletters, and outputs `data/emails_corpus.jsonl` (for RAG) plus `data/train.jsonl` and `data/valid.jsonl` (for LoRA instruction tuning).
* `build_rag.py`: Chunks `data/emails_corpus.jsonl` with email header metadata and indexes them into a persistent local `./chroma_db` vector database.
* `finetune.py`: Fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` (default, low VRAM) or `Qwen/Qwen2.5-1.5B-Instruct` with LoRA adapters using either `mlx-lm` (Apple Silicon unified memory) or Hugging Face `transformers` + `peft` + `trl`.
* `chat_assistant.py`: Interactive CLI and single-query tool that retrieves the top-k matching emails from `./chroma_db` and generates grounded answers using your fine-tuned LoRA adapter.
* `run_pipeline.sh`: End-to-end bash script to create `.venv`, install `requirements.txt`, parse the `.mbox`, build the RAG index, fine-tune the model, and start the assistant.

## Quick Start

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Switching Base Models or Frameworks

* **Train with Qwen2.5-0.5B-Instruct on Apple Silicon (`mlx-lm`, default & low-RAM friendly)**:
  ```bash
  python3 finetune.py --model Qwen/Qwen2.5-0.5B-Instruct --backend mlx
  ```
* **Train with Llama-3.2-1B-Instruct on Apple Silicon (`mlx-lm`)**:
  ```bash
  python3 finetune.py --model meta-llama/Llama-3.2-1B-Instruct --backend mlx
  ```
* **Train with Qwen2.5-1.5B-Instruct on PyTorch MPS/CPU (`peft`)**:
  ```bash
  python3 finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --backend hf
  ```
