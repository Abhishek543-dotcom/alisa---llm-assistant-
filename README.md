# Alisa — Abhishek's Autonomous Personal Voice AI Assistant

A zero-hardcoding, fully autonomous local AI assistant named **Alisa** optimized for Apple Silicon (MacBook Air / M-Series) combining:
1. **Wake-Word & Hands-Free Interaction** ([`voice_engine.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/voice_engine.py)):
   * **Wake Word:** Say `"Alisa"` or `"Hey Alisa"` anytime to wake her up like Alexa or Siri.
   * **Wake Chime:** Plays macOS native alert sound (`Tink.aiff`) on activation.
   * **Text-to-Speech (TTS):** Uses macOS native `/usr/bin/say -v Samantha` (0 RAM, 0 GPU overhead, crystal-clear female voice).
   * **Speech-to-Text (STT):** Uses Apple Silicon `mlx-whisper` (`whisper-tiny`) running directly on Metal GPU for ~80ms in-memory speech transcription (zero ffmpeg dependency).
2. **Zero Hardcoding Dynamic Memory** ([`memory_manager.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/memory_manager.py)): Learns Abhishek's profile, contacts, and chronological career progression directly from your data without any hardcoded assumptions.
3. **Universal Autonomous Learner** ([`auto_learner.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/auto_learner.py)): Drop ANY file (`.pdf`, `.txt`, `.csv`, `.json`, `.py`, `.mbox`, etc.) into `inbox/` or use `/learn`—it automatically parses, indexes into ChromaDB, extracts knowledge, and generates fine-tuning training pairs for future model training.
4. **Dynamic ChromaDB Vector Search** ([`rag_engine.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/rag_engine.py)): Attach or detach vector search on the fly (`/attach`, `/detach`).
5. **Temporal Awareness & Recency Priority**: Parses timestamps so newer records always take precedence (e.g. knowing your latest company from the newest emails).

---

## Quick Start: Launching Alisa

### 1. Hands-Free Wake-Word Mode (Alexa / Siri Experience)
```bash
./alisa.sh --listen
# OR
source .venv/bin/activate
python3 alisa.py --listen
```
Say **"Alisa"** (you'll hear the wake chime), then ask your question! Or say it all in one breath: *"Alisa, where do I work?"*

### 2. Interactive Console Mode (Chat + Push-to-Talk)
```bash
./alisa.sh
# OR
source .venv/bin/activate
python3 alisa.py
```

---

## Commands Inside Alisa

* `/listen` — Enter hands-free continuous Alexa/Siri mode (say "Alisa" to trigger)
* `/speak` — Push-to-talk mic mode (records 5s with MLX Whisper)
* `/voice` — Toggle spoken voice output on or off
* `/learn <file or text>` — Ingest and learn any file (`.pdf`, `.txt`, `.csv`, etc.) or note
* `/scan` — Ingest all files dropped in the `inbox/` directory
* `/attach` or `/rag on` — Attach ChromaDB vector search
* `/detach` or `/rag off` — Detach ChromaDB (answers purely from memory & model)
* `/rag status` — Check ChromaDB connection status and chunk count
* `/memory` — View dynamically discovered memory profile and learned knowledge
* `/remember <key>: <value>` — Save a custom fact to persistent memory
* `/help` — List all available commands
* `exit` — Quit Alisa

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
