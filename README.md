# Autonomous Personal AI Assistant (Low-Resource & Voice-Enabled)

A zero-hardcoding, fully autonomous local AI assistant optimized for Apple Silicon (MacBook Air / M-Series) combining:
1. **Zero Hardcoding Dynamic Memory** ([`memory_manager.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/memory_manager.py)): Learns your profile, contacts, and chronological career progression directly from your data without any hardcoded assumptions.
2. **Universal Autonomous Learner** ([`auto_learner.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/auto_learner.py)): Drop ANY file (`.pdf`, `.txt`, `.csv`, `.json`, `.py`, `.mbox`, etc.) into `inbox/` or use `/learn`—it automatically parses, indexes into ChromaDB, extracts knowledge, and generates fine-tuning training pairs for future model training.
3. **Ultra-Low Resource Voice Engine** ([`voice_engine.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/voice_engine.py)):
   * **Text-to-Speech (TTS):** Uses macOS native `/usr/bin/say` (0 RAM, 0 GPU overhead, non-blocking).
   * **Speech-to-Text (STT):** Uses Apple Silicon `mlx-whisper` (tiny model, ~75MB) running directly on Metal GPU for near-instant microphone voice transcription.
4. **Dynamic ChromaDB Vector Search** ([`rag_engine.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/rag_engine.py)): Attach or detach vector search on the fly (`/attach`, `/detach`).
5. **Temporal Awareness & Recency Priority**: Parses timestamps so newer records always take precedence (e.g. knowing your latest company from the newest emails).
6. **Incremental Re-exports & Continuous Learning**: Export new emails or dump new files anytime; it deduplicates, embeds only new content, and supports resuming LoRA fine-tuning.

## In-Chat Commands

Inside the assistant, you can run:
* `/speak` — Speak your question hands-free via microphone (Apple Silicon MLX Whisper)
* `/voice` — Toggle spoken voice output (TTS via macOS say)
* `/learn <file or text>` — Ingest and learn any file (`.pdf`, `.txt`, `.csv`, etc.) or note
* `/scan` — Ingest all files dropped in the `inbox/` directory
* `/attach` or `/rag on` — Attach ChromaDB vector search
* `/detach` or `/rag off` — Detach ChromaDB (assistant answers purely from memory & model)
* `/rag status` — Check ChromaDB connection status and chunk count
* `/memory` — View dynamically discovered memory profile and learned knowledge
* `/remember <key>: <value>` — Save a custom fact to persistent memory
* `/help` — List all available commands
* `exit` — Quit assistant

## Quick Start

```bash
source .venv/bin/activate
python3 chat_assistant.py --no-adapter
```

To start directly with spoken voice enabled:
```bash
python3 chat_assistant.py --no-adapter --voice
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
