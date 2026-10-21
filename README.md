# Alisa — Abhishek's Autonomous All-Rounder Personal Voice AI Assistant

[![Platform: macOS Apple Silicon](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon%20(M1--M4)-black?style=flat&logo=apple)](https://apple.com)
[![GPU: Metal GPU Accelerated (MLX)](https://img.shields.io/badge/Engine-Metal%20GPU%20(MLX%20Unified%20Memory)-blue)](https://github.com/ml-explore/mlx)
[![Voice: Ultra-Fast STT + 0 RAM TTS](https://img.shields.io/badge/Voice-MLX%20Whisper%20%2B%20macOS%20Samantha-success)](https://github.com/ml-explore/mlx-examples)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A zero-hardcoding, fully autonomous, privacy-first local AI assistant named **Alisa** engineered specifically for macOS and Apple Silicon. Alisa combines local Metal GPU-accelerated LLM reasoning, continuous self-improving memory, hands-free Alexa/Siri voice activation, autonomous document ingestion, and macOS system orchestration.

---

## 🚀 1-Line Universal Mac Installer

Give this single command to any Mac user. They paste it into their Terminal and press Enter — everything is installed, configured, pre-cached, and ready automatically:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Abhishek543-dotcom/alisa---llm-assistant-/main/install.sh)"
```

### Verified Live Status
The installer script is live on GitHub and responding with `HTTP 200 OK`:
```bash
curl -sI https://raw.githubusercontent.com/Abhishek543-dotcom/alisa---llm-assistant-/main/install.sh
```

*What the installer does automatically:*
1. **Hardware & Metal Acceleration:** Detects Apple Silicon (`arm64` M1/M2/M3/M4) and validates Metal GPU execution.
2. **Environment & Tooling:** Verifies Python 3.10+ and Git (prompts or configures via Homebrew).
3. **Repository Deployment:** Clones Alisa into `~/alisa` and prepares an isolated virtual environment (`.venv`).
4. **Base Model Pre-Caching:** Downloads `Qwen2.5-0.5B-Instruct`, `Whisper Tiny` (Metal STT), and `BAAI/bge-small-en-v1.5` embeddings.
5. **Brain & Dataset Initialization:** Initializes Model Versioning Registry (`v0` Fresh Base) and builds All-Rounder datasets.
6. **Global Terminal Access:** Creates `alisa` and `alisha` global CLI shortcuts in `~/.local/bin` (and adds to `$PATH`).
7. **Verification Chime:** Plays native alert chime and confirms installation with crystal-clear Samantha voice output.

---

## 🏛️ System Architecture

```
                               ┌─────────────────────────────────────────────────────────────┐
                               │     ALISA OMNI-CAPABLE PERSONAL AI ASSISTANT CORE           │
                               └──────────────────────────────┬──────────────────────────────┘
                                                              │
               ┌──────────────────────────────┬───────────────┴──────────────┬──────────────────────────────┐
               ▼                              ▼                              ▼                              ▼
    ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
    │     VOICE ENGINE     │       │   PROACTIVE AGENT    │       │   COGNITIVE MEMORY   │       │  SYSTEM ORCHESTRATION│
    │   voice_engine.py    │       │  proactive_agent.py  │       │  memory_manager.py   │       │   system_tools.py    │
    ├──────────────────────┤       ├──────────────────────┤       ├──────────────────────┤       ├──────────────────────┤
    │• Say "Alisa" Wake    │       │• Executive Briefing  │       │• Semantic Profile    │       │• Volume & Audio Ctrls│
    │• MLX Whisper (~80ms) │       │• Hardware Telemetry  │       │• Episodic Daily Diary│       │• Battery Telemetry   │
    │• Native macOS TTS    │       │• Anomaly Detection   │       │• Procedural Runbooks │       │• RAM / CPU / Disk    │
    │• Zero Python RAM TTS │       │• Voice / Console Mode│       │• Human Feedback/RLHF │       │• Shell /run Execution│
    └──────────┬───────────┘       └──────────┬───────────┘       └──────────┬───────────┘       └──────────┬───────────┘
               │                              │                              │                              │
               └──────────────────────────────┼──────────────────────────────┴──────────────────────────────┘
                                              ▼
                               ┌──────────────────────────────┐
                               │   CHAT INFERENCE ASSISTANT   │
                               │      chat_assistant.py       │
                               └──────────────┬───────────────┘
                                              │
               ┌──────────────────────────────┼──────────────────────────────┐
               ▼                              ▼                              ▼
    ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
    │  KNOWLEDGE INGESTION │       │ DATASET BUILDER SFT  │       │ MODEL VERSION LEDGER │
    │   auto_learner.py    │       │  dataset_builder.py  │       │   model_manager.py   │
    ├──────────────────────┤       ├──────────────────────┤       ├──────────────────────┤
    │• inbox/ Drop Folder  │       │• 80/10/10 Split      │       │• v0: Fresh Base      │
    │• PDF, Code, CSV, Doc │       │• Coding & Reasoning  │       │• v1, v2, v3... LoRA  │
    │• Streaming MBOX Scan │       │• SHA-256 Dedup       │       │• Instant Hot Rollback│
    │• ChromaDB Vector RAG │       │• Preference Pairs    │       │• Multi-Model Download│
    └──────────────────────┘       └──────────────────────┘       └──────────────────────┘
```

---

## 🌟 Core Features & Modules

### 1. Hands-Free Wake-Word & Voice Engine ([`voice_engine.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/voice_engine.py))
* **Wake Word Detection:** Say `"Alisa"` or `"Hey Alisa"` anytime to activate, exactly like Siri or Alexa.
* **Apple Silicon Metal STT:** Uses `mlx-whisper` (`whisper-tiny`, ~75MB) running directly on Metal GPU for ~80ms in-memory speech transcription (zero ffmpeg dependency).
* **Zero Overhead Speech Synthesis (TTS):** Uses macOS native `/usr/bin/say -v Samantha` running in an independent OS subprocess (0 MB Python RAM overhead, 0 GPU memory overhead).
* **Wake Chime:** Plays native macOS audio feedback (`Tink.aiff`) on activation.

### 2. Proactive Intelligence & Daily Executive Briefing ([`proactive_agent.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/proactive_agent.py))
* Alisa is not purely reactive; she proactively summarizes your workspace and system health:
  * **System Telemetry:** Live RAM % used, CPU load, free disk space, battery % and charging status.
  * **Hardware Anomaly Alerts:** Proactively flags low battery (<20%), low disk space (<15GB), or high memory pressure (>90%).
  * **AI State & Milestones:** Active model version, ChromaDB chunks, and recent episodic milestones.
  * **Vocalized & Visual Output:** Outputs a visual dashboard or speaks the briefing via Samantha TTS.
* Run via CLI (`./alisha.sh --briefing`), menu option `[3]`, or inside chat (`/briefing`).

### 3. Multi-Tier Cognitive Memory Engine ([`memory_manager.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/memory_manager.py))
* **Tier 1: Semantic & Profile Memory (`data/profile_memory.json`):** Discovered profile, chronological career progression, CTC, contacts, and personal preferences without hardcoding.
* **Tier 2: Episodic Memory Diary (`data/episodic_memory.json`):** Chronological diary of daily activities, sessions, and milestones. Type `/episodes` or `/diary` to see what was accomplished on any day.
* **Tier 3: Procedural Memory & Developer Runbooks (`data/procedural_memory.json`):** Saves repeatable engineering workflows and execution procedures (e.g. `/workflow deploy = git pull | pip install | ./alisha.sh`).
* **Tier 4: Human Feedback & RLHF Alignment (`data/preference_pairs.jsonl`):** Type `/good` to reinforce great responses or `/bad <correction>` to teach Alisa preferred answers.
* **Tier 5: Lifelong Error Memory (`data/solution_memory.json`):** Documents how past technical errors and bugs were resolved so they are never repeated.

### 4. Model Versioning & Instant Rollback System ([`model_manager.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/model_manager.py))
* **Version `v0` (Fresh Base):** Guaranteed clean untuned baseline model.
* **Auto-Incrementing Versions (`v1`, `v2`, `v3`...):** Every fine-tune run or data ingestion allocates an immutable version directory in `adapters/versions/v{N}` and records metadata in `data/model_registry.json`.
* **Instant Hot-Swap Rollback:**
  * Revert to previous version: `./alisha.sh --rollback` or `/rollback` in chat.
  * Revert to clean base model: `./alisha.sh --fresh` or `/fresh` in chat.
  * Switch to any version: `/switch <vX>` in chat.
* **Standalone Model Downloader:** Download models cleanly with `./alisha.sh --download-model [MODEL_ID]` or menu option `[4]`.

### 5. All-Rounder Instruction Engine ([`dataset_builder.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/dataset_builder.py))
* Expands Alisa beyond email into a complete software engineering, debugging, and productivity assistant:
  1. **Software Engineering & Coding:** Python functions, Bash scripts, data structures, algorithms, debugging.
  2. **Personal Context:** Grounded in discovered profile memory and custom notes.
  3. **Ingested Documents:** Multi-turn Q&A, summaries, and code breakdowns from `inbox/`.
  4. **Episodic & Procedural:** Daily diary events and developer runbooks.
  5. **Human Alignment:** Reinforcement pairs from user feedback (`/good`, `/bad`).
  6. **System Orchestration:** Volume, battery, diagnostics, app launching.
* **Production Controls:** Balanced 80/10/10 split (`train.jsonl`, `valid.jsonl`, `test.jsonl`), SHA-256 deduplication, and ChatML schema validation.

### 6. Universal Document Learner ([`auto_learner.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/auto_learner.py))
* Drop any file (`.pdf`, `.txt`, `.md`, `.csv`, `.json`, `.py`, `.sh`, `.mbox`, etc.) into `inbox/` or use `/learn`.
* Automatically indexes chunks into ChromaDB vector store.
* Tracks file signatures in `data/processed_files.json` to avoid duplicate re-processing.
* Large archive protection: routes `.mbox` files through streaming spam/trash filters.

### 7. macOS System Orchestration ([`system_tools.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/system_tools.py))
* **Audio Controls:** Query volume (`/volume`), set volume (`/volume 50`), mute, or unmute via AppleScript.
* **Battery Telemetry:** Real-time percentage and charging state via `pmset`.
* **System Telemetry:** Live RAM usage, CPU load, and free disk storage via `psutil`.
* **Safe Terminal Execution:** Run bash commands safely via `/run <command>`.
* **App Launcher:** Launch macOS applications directly from natural language.

---

## 📜 Complete Commands Reference

### Interactive Chat Commands

| Command | Category | Description | Example |
| :--- | :--- | :--- | :--- |
| `/briefing` | **Executive** | Generate comprehensive daily morning/evening briefing | `/briefing` |
| `/good` *(or `/+1`)* | **Alignment** | Record positive feedback on previous response (RLHF) | `/good` |
| `/bad <correction>` | **Alignment** | Correct previous answer and queue for model fine-tuning | `/bad Use list comprehension instead` |
| `/feedback` | **Alignment** | View human feedback positive vs corrected stats | `/feedback` |
| `/episodes` *(or `/diary`)* | **Memory** | View chronological timeline of daily milestones | `/episodes` |
| `/workflow [name]` | **Memory** | View all or specific saved developer runbooks | `/workflow deploy` |
| `/workflow <name> = <steps>` | **Memory** | Save new execution runbook (steps separated by `\|`) | `/workflow test = pytest \| ./alisha.sh --diag` |
| `/version` *(or `/models`)* | **Versioning** | View table of all model versions and active status | `/version` |
| `/rollback` | **Versioning** | Hot-swap back to previous model version | `/rollback` |
| `/fresh` *(or `/base`)* | **Versioning** | Revert instantly to untuned Fresh Base Model (v0) | `/fresh` |
| `/switch <vX>` | **Versioning** | Switch active model to specific version | `/switch v1` |
| `/download <model_id>` | **Models** | Download and cache any Hugging Face model | `/download Qwen/Qwen2.5-1.5B-Instruct` |
| `/listen` | **Voice** | Enter continuous hands-free Alexa/Siri mode | `/listen` |
| `/speak` | **Voice** | Push-to-talk mic mode (records 5s with MLX Whisper) | `/speak` |
| `/voice` | **Voice** | Toggle spoken voice output on or off | `/voice off` |
| `/battery` | **System** | Check macOS battery percentage and power source | `/battery` |
| `/volume [0-100]` | **System** | Inspect or adjust macOS system output volume | `/volume 60` |
| `/stats` | **System** | View live RAM, CPU load, and free disk storage | `/stats` |
| `/run <command>` | **System** | Execute bash terminal command directly | `/run git status` |
| `/learn <file or text>` | **Ingestion** | Ingest and learn any file or note | `/learn resume.pdf` |
| `/scan` | **Ingestion** | Ingest all new files placed in `inbox/` | `/scan` |
| `/attach` / `/detach` | **RAG** | Attach or detach ChromaDB vector search | `/detach` |
| `/rag status` | **RAG** | Check ChromaDB status and chunk count | `/rag status` |
| `/memory` | **Memory** | View dynamically discovered cognitive memory | `/memory` |
| `/remember <k>: <v>` | **Memory** | Save a custom fact to persistent memory | `/remember favorite_editor: VS Code` |
| `/solutions` | **Memory** | View lifelong troubleshooting resolutions | `/solutions` |
| `/history` | **Memory** | View recent conversation turns | `/history` |
| `exit` | **Session** | Quit Alisa | `exit` |

---

## 💻 Terminal CLI Usage & Shortcuts

After running the 1-line installer, you can run Alisa from **ANY** terminal window using `alisa` or `alisha`:

```bash
# Launch interactive dashboard & control center:
alisa
# (or ./alisha.sh / ./alisa.sh from the repository directory)

# Generate instant executive briefing:
alisa --briefing

# Vocalize executive briefing with voice:
alisa --speak-briefing

# Launch hands-free Alexa/Siri listening mode directly:
alisa --listen

# Download any base model into local cache:
alisa --download-model Qwen/Qwen2.5-1.5B-Instruct

# Inspect all model versions and active status:
alisa --versions

# Roll back to the previous model version:
alisa --rollback

# Roll back directly to clean Fresh Base Model (v0):
alisa --fresh

# Run system hardware & pipeline diagnostics:
alisa --diag

# Ask a single question directly:
alisa --query "Write a Python script to monitor CPU usage"
```

---

## 🧠 Model Training & Fine-Tuning Guide

Alisa supports ultra-fast incremental LoRA fine-tuning directly on Apple Silicon unified memory via `mlx-lm`, as well as PyTorch PEFT fallback:

### 1. Fine-Tune on Apple Silicon Metal GPU (Default & Recommended)
```bash
python3 finetune.py --model Qwen/Qwen2.5-0.5B-Instruct --backend mlx --iters 600
```
* **Auto-Versioning:** Automatically creates version `v{N}`, saves weights in `adapters/versions/v{N}`, and sets it as `[ACTIVE]`.
* **Dataset Preparation:** Automatically reads and validates `data/train.jsonl` and `data/valid.jsonl` (generated by `dataset_builder.py`).

### 2. Fine-Tune Starting Fresh from Base Model (No Resume)
```bash
python3 finetune.py --model Qwen/Qwen2.5-0.5B-Instruct --backend mlx --fresh --notes "Fresh baseline training"
```

### 3. Fine-Tune with Larger Models
```bash
# Qwen 2.5 1.5B Instruct (Higher reasoning & coding performance)
python3 finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --backend mlx

# Llama 3.2 1B Instruct
python3 finetune.py --model meta-llama/Llama-3.2-1B-Instruct --backend mlx
```

---

## 🔬 Production Diagnostics Suite ([`diagnostics.py`](file:///Users/abhishektiwari/Documents/dev/personal-email-llm/diagnostics.py))

Run anytime to verify all 6 subsystems:
```bash
./alisha.sh --diag
# or
python3 diagnostics.py
```

```
======================================================================
🔬 Alisa All-Rounder Voice AI — Production System Diagnostics
======================================================================
[1/6] Checking Audio Output (Speakers / TTS)...
  ✓ macOS native audio output (afplay): WORKING
  ✓ macOS speech synthesis (Samantha): WORKING
[2/6] Checking Audio Input (Microphone / sounddevice)...
  ✓ Microphone capture: WORKING (Sampled RMS: 0.00366)
[3/6] Checking Apple Silicon Metal GPU & MLX...
  ✓ MLX Core Device: Device(gpu, 0) (Metal GPU Accelerated)
  ✓ MLX Whisper Metal GPU inference: WORKING (80ms warm latency)
[4/6] Checking Model Versioning & Weights...
  ✓ Model Registry: 2 version(s) recorded.
  ✓ Active Version: v1 (Fine-Tuned v1)
  ✓ Base Model: Qwen/Qwen2.5-0.5B-Instruct
  ✓ Active LoRA Adapter: adapters/versions/v1
[5/6] Checking All-Rounder Datasets (train / valid / test)...
  ✓ train.jsonl :    15 valid examples
  ✓ valid.jsonl :     1 valid examples
  ✓ test.jsonl  :     1 valid examples
[6/6] Checking Vector Database & Memory...
  ✓ ChromaDB Storage: 'personal_emails' active (1 chunks)
  ✓ Discovered Career: Active
  ✓ Solved Issues Memory: Documented resolutions active
  ✓ Conversation History: Active
======================================================================
🎉 ALL PRODUCTION CHECKS VERIFIED: Alisa is 100% Operational!
======================================================================
```

---

## 📂 Project Directory Structure

```
.
├── alisa.sh               # Production interactive dashboard & control center
├── alisha.sh              # Seamless executable symlink to alisa.sh
├── install.sh             # 1-Line universal Mac installer script
├── alisa.py               # Single-command Python launcher
├── chat_assistant.py      # Core conversational assistant & hot-swap engine
├── model_manager.py       # Model downloader & versioning rollback ledger
├── dataset_builder.py     # All-Rounder 80/10/10 dataset generator & dedup engine
├── proactive_agent.py     # Executive daily briefing & hardware anomaly watcher
├── memory_manager.py      # Multi-tier cognitive memory (Episodic, Rules, RLHF)
├── voice_engine.py        # Wake-word engine, MLX Whisper STT, native TTS
├── auto_learner.py        # Universal document parser & inbox ingestion scanner
├── build_rag.py           # ChromaDB vector index builder
├── rag_engine.py          # Modular vector retrieval engine with recency priority
├── system_tools.py        # macOS volume, battery, RAM/CPU, shell runner
├── parse_mbox.py          # Streaming spam/trash filtered MBOX parser
├── diagnostics.py         # 6-point production health check suite
├── requirements.txt       # Dependencies with Apple Silicon constraints
├── adapters/              # Immutable LoRA model versions
│   ├── active             # Symlink to currently active adapter
│   ├── personal_lora      # Compatibility link
│   └── versions/          # Version storage (v1, v2, v3...)
├── chroma_db/             # Persistent local ChromaDB vector store
├── inbox/                 # File drop folder (PDF, Code, Text, MBOX)
└── data/                  # Dynamic knowledge & dataset storage
    ├── model_registry.json      # Model version ledger & rollback tracking
    ├── profile_memory.json      # Discovered profile & career progression
    ├── episodic_memory.json     # Chronological daily diary & milestones
    ├── procedural_memory.json   # Saved developer workflows & runbooks
    ├── preference_pairs.jsonl   # RLHF / DPO human feedback pairs
    ├── solution_memory.json     # Lifelong error troubleshooting fixes
    ├── processed_files.json     # Ingested file signatures & hashes
    ├── train.jsonl              # 80% instruction fine-tuning dataset
    ├── valid.jsonl              # 10% validation dataset
    └── test.jsonl               # 10% test dataset
```

---

## 🔒 Privacy & Architecture Guarantees

* **100% Local & Offline Capable:** Zero conversational data, personal emails, documents, or voice audio are ever sent to external cloud APIs.
* **Apple Silicon Optimized:** Leverages unified memory architecture so weights are shared between CPU and Metal GPU with 0 copy overhead.
* **Zero Python RAM Overhead for Speech:** Audio generation bypasses heavy Python TTS libraries, piping clean strings directly to macOS native subsystem.
