# Alisa — Abhishek's Autonomous All-Rounder Personal Voice AI Assistant

A zero-hardcoding, fully autonomous local AI assistant named **Alisa** optimized for Apple Silicon (MacBook Air / M-Series) combining:
1. **Wake-Word & Hands-Free Interaction** ([`voice_engine.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/voice_engine.py)):
   * **Wake Word:** Say `"Alisa"` or `"Hey Alisa"` anytime to wake her up like Alexa or Siri.
   * **Wake Chime:** Plays macOS native alert sound (`Tink.aiff`) on activation.
   * **Text-to-Speech (TTS):** Uses macOS native `/usr/bin/say -v Samantha` (0 RAM, 0 GPU overhead, crystal-clear female voice).
   * **Speech-to-Text (STT):** Uses Apple Silicon `mlx-whisper` (`whisper-tiny`) running directly on Metal GPU for ~80ms in-memory speech transcription.
2. **Model Versioning & Instant Rollback System** ([`model_manager.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/model_manager.py)):
   * **Version v0 (Fresh Base):** Guaranteed clean untuned baseline model.
   * **Auto-Versioning (v1, v2, v3...):** Automatically increments version after every fine-tune or inbox ingestion run.
   * **One-Click Hot-Swap & Rollback:** Revert to previous version or reset to fresh base instantly inside the console or from the menu.
3. **Standalone Base Model Downloader** ([`model_manager.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/model_manager.py)):
   * One-click download and cache base models (Qwen2.5 0.5B, 1.5B, Coder, Llama 3.2 1B, 3B, Whisper, or any Hugging Face repo).
4. **All-Rounder Multi-Task Instruction Engine** ([`dataset_builder.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/dataset_builder.py)):
   * Expands Alisa beyond email into a complete software engineering, debugging, reasoning, system orchestration, and life assistant.
   * Clean 3-way split: `train.jsonl` (80%), `valid.jsonl` (10%), and `test.jsonl` (10%) with SHA-256 deduplication.
5. **Universal Autonomous Learner** ([`auto_learner.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/auto_learner.py)):
   * Drop ANY file (`.pdf`, `.txt`, `.csv`, `.json`, `.py`, `.sh`, `.mbox`, etc.) into `inbox/` or use `/learn`.
   * Indexes into ChromaDB, tracks processed files in `data/processed_files.json`, and updates balanced datasets.
6. **Zero Hardcoding Dynamic Memory** ([`memory_manager.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/memory_manager.py)):
   * Learns Abhishek's profile, skills, contacts, and chronological career progression directly from your data without any hardcoded assumptions.

---

## Quick Start: Launching Alisa

Both `./alisa.sh` and `./alisha.sh` work identically:

### 1. Production Dashboard & Control Center
```bash
./alisha.sh
# or
./alisa.sh
```

### 2. Direct CLI Commands & Shortcuts
```bash
# Generate executive system & productivity briefing:
./alisha.sh --briefing

# Vocalize executive briefing:
./alisha.sh --speak-briefing

# Launch hands-free Alexa/Siri listening mode directly:
./alisha.sh --listen

# Download any base model into local cache:
./alisha.sh --download-model Qwen/Qwen2.5-1.5B-Instruct

# Inspect all model versions and active status:
./alisha.sh --versions

# Roll back to the previous model version:
./alisha.sh --rollback

# Roll back directly to the clean Fresh Base Model (v0):
./alisha.sh --fresh

# Run system hardware & pipeline diagnostics:
./alisha.sh --diag
```

---

## Commands Inside Alisa

* `/briefing` — Generate proactive executive system briefing (power, memory, disk, AI status)
* `/good` — Record positive feedback on previous response (RLHF human alignment)
* `/bad <correction>` — Correct previous response and queue for fine-tuning alignment
* `/episodes` or `/diary` — View chronological daily milestones and activity diary
* `/workflow [name]` — View or save developer execution procedures (e.g., `/workflow deploy = git pull | pip install`)
* `/feedback` — View human feedback alignment statistics

* `/version` or `/models` — View all model versions, creation dates, training samples, and active status
* `/rollback` — Hot-swap back to previous model version without exiting
* `/fresh` — Revert instantly to untuned Fresh Base Model (v0)
* `/switch <vX>` — Switch active model to specific version (e.g., `/switch v1`)
* `/download <model_id>` — Download and cache any Hugging Face model
* `/listen` — Enter hands-free continuous Alexa/Siri mode (say "Alisa" to trigger, "stop" to sleep)
* `/speak` — Push-to-talk mic mode (records 5s with MLX Whisper)
* `/voice` — Toggle spoken voice output on or off
* `/battery` — Check macOS battery percentage and power status
* `/volume [0-100]` — Inspect or adjust system output volume
* `/stats` — View live RAM, CPU load, and free disk storage
* `/solutions` — View lifelong documented error resolutions and fixes
* `/history` — Review past conversation interactions
* `/run <command>` — Execute bash commands directly in the workspace
* `/learn <file or text>` — Ingest and learn any file (`.pdf`, `.txt`, `.csv`, etc.) or note
* `/scan` — Ingest all files dropped in the `inbox/` directory
* `/attach` or `/rag on` — Attach ChromaDB vector search
* `/detach` or `/rag off` — Detach ChromaDB (answers purely from memory & model)
* `/memory` — View dynamically discovered memory profile and learned knowledge
* `/remember <key>: <value>` — Save a custom fact to persistent memory
* `/help` — List all available commands
* `exit` — Quit Alisa
