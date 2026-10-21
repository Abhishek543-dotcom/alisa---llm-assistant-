#!/usr/bin/env bash
# ==============================================================================
# alisa.sh (alisha.sh) — Production-Grade Autonomous All-Rounder Personal Voice AI
# Author: Abhishek Tiwari
# Architecture: Apple Silicon Metal GPU (MLX) + Multi-Tier Cognitive Memory
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color Codes
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
MAGENTA="\033[0;35m"
RED="\033[0;31m"
RESET="\033[0m"

# Ensure Virtual Environment is active
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
else
    echo -e "${RED}Error: .venv not found. Please run ./run_pipeline.sh first to set up the environment.${RESET}"
    exit 1
fi

# ==============================================================================
# CLI Argument Pass-Through & Direct Shortcuts
# ==============================================================================
if [[ $# -gt 0 ]]; then
    case "$1" in
        --briefing)
            python3 proactive_agent.py --silent
            exit 0
            ;;
        --speak-briefing)
            python3 proactive_agent.py --speak
            exit 0
            ;;
        --download|--download-model)
            if [[ -n "$2" ]]; then
                python3 model_manager.py --download "$2"
            else
                python3 model_manager.py --download-menu
            fi
            exit 0
            ;;
        --versions|--version|--models)
            python3 model_manager.py --list
            exit 0
            ;;
        --rollback)
            python3 model_manager.py --rollback
            exit 0
            ;;
        --fresh|--rollback-fresh)
            python3 model_manager.py --rollback-fresh
            exit 0
            ;;
        --diag|--diagnostics)
            python3 diagnostics.py
            exit 0
            ;;
        *)
            python3 alisa.py "$@"
            exit 0
            ;;
    esac
fi

# ==============================================================================
# Dynamic Dashboard Header
# ==============================================================================
print_header() {
    clear
    echo -e "${CYAN}${BOLD}"
    cat << "EOF"
    ___    __    ____ _____ ___ 
   /   |  / /   /  _// ___//   |
  / /| | / /    / /  \__ \/ /| |
 / ___ |/ /____/ /  ___/ / ___ |
/_/  |_/_____/___/ /____/_/  |_|
EOF
    echo -e "${RESET}"
    echo -e "${BOLD}${BLUE}✨ Abhishek Tiwari's Autonomous All-Rounder Voice AI Assistant${RESET}"
    echo -e "${CYAN}─────────────────────────────────────────────────────────────────${RESET}"

    # Extract dynamic live metrics
    local batt_str
    batt_str="$(pmset -g batt 2>/dev/null | grep -o '[0-9]*%' | head -1 || echo 'N/A')"
    local ram_pct
    ram_pct="$(python3 -c 'import psutil; print(int(psutil.virtual_memory().percent))' 2>/dev/null || echo 'N/A')"
    local active_model_ver
    active_model_ver="$(python3 -c '
from model_manager import ModelVersionManager
v = ModelVersionManager().get_active_version()
bm = v.get("base_model", "Qwen2.5-0.5B").split("/")[-1]
vid = v.get("version_id", "v0")
print(f"{bm} [{vid}]")
' 2>/dev/null || echo 'Qwen2.5-0.5B [v0]')"
    local chunks_count
    chunks_count="$(python3 -c '
import chromadb
try:
    c = chromadb.PersistentClient("chroma_db").get_collection("personal_emails").count()
    print(f"{c:,} chunks")
except Exception:
    print("0 chunks")
' 2>/dev/null || echo '0 chunks')"
    local sol_count
    sol_count="$(python3 -c 'from memory_manager import MemoryManager; print(len(MemoryManager().load_solutions()))' 2>/dev/null || echo '0')"
    local latest_co
    latest_co="$(python3 -c '
from memory_manager import MemoryManager
co = MemoryManager().get_latest_company()
print(co if co else "Ready to discover")
' 2>/dev/null || echo 'Ready to discover')"

    echo -e " 💻 ${BOLD}System:${RESET} Apple Silicon Metal  |  ${BOLD}RAM:${RESET} ${ram_pct}%  |  ${BOLD}Battery:${RESET} ${batt_str}"
    echo -e " 🧠 ${BOLD}Model:${RESET} ${active_model_ver}  |  ${BOLD}Vector DB:${RESET} ${chunks_count}"
    echo -e " 👤 ${BOLD}Career:${RESET} ${latest_co}  |  ${BOLD}Troubleshoot Solutions:${RESET} ${sol_count} documented"
    echo -e "${CYAN}─────────────────────────────────────────────────────────────────${RESET}"
}

# ==============================================================================
# Interactive Menu Loop
# ==============================================================================
while true; do
    print_header
    echo -e "${BOLD}Select an operation:${RESET}\n"
    echo -e "  ${GREEN}[1] 💬 Talk with Alisa${RESET}            (Interactive Console + Hot-Swap + Voice)"
    echo -e "  ${YELLOW}[2] 🎙️  Hands-Free Wake-Word Mode${RESET}  (Say 'Alisa' to wake, 'Stop' to sleep)"
    echo -e "  ${CYAN}[3] 🌅 Executive Daily Briefing${RESET}    (Proactive health, alerts, and model status)"
    echo -e "  ${BLUE}[4] 📥 Download Base Model${RESET}        (Download & cache Qwen2.5, Llama, Whisper)"
    echo -e "  ${MAGENTA}[5] 🏛️  Model Versions & Rollback${RESET}  (Inspect versions, rollback, switch)"
    echo -e "  ${CYAN}[6] 🧠 Train & Fine-Tune Model${RESET}     (Auto-versioned LoRA on Apple Silicon GPU)"
    echo -e "  ${GREEN}[7] 📚 Auto-Learner & Ingest${RESET}       (Scan inbox/ or add any PDF/code/doc)"
    echo -e "  ${YELLOW}[8] 📅 Episodic Diary & Milestones${RESET} (View daily timeline of accomplishments)"
    echo -e "  ${BLUE}[9] 🛠️  Developer Runbooks${RESET}        (Inspect procedural execution procedures)"
    echo -e "  ${MAGENTA}[10] 👤 Discovered Profile Memory${RESET}  (View timeline, skills, discovered facts)"
    echo -e "  ${GREEN}[11] 🔬 Run System Diagnostics${RESET}    (Verify Mic, Speakers, GPU, DB, Models)"
    echo -e "  ${RED}[12] 🚪 Exit Alisa${RESET}\n"

    read -rp "Enter choice [1-12]: " choice

    case "$choice" in
        1)
            echo -e "\n${GREEN}Starting Alisa Interactive Console...${RESET}"
            python3 alisa.py
            read -rp "Press Enter to return to menu..."
            ;;
        2)
            echo -e "\n${YELLOW}Activating Microphone & Wake-Word Listening...${RESET}"
            python3 alisa.py --listen
            read -rp "Press Enter to return to menu..."
            ;;
        3)
            echo -e "\n${CYAN}--- Executive Daily Briefing ---${RESET}"
            python3 proactive_agent.py --silent
            read -rp "Speak briefing with voice? (y/N): " spk
            if [[ "$spk" =~ ^[Yy]$ ]]; then
                python3 proactive_agent.py --speak
            fi
            read -rp "Press Enter to return to menu..."
            ;;
        4)
            echo -e "\n${BLUE}--- Download Base Model ---${RESET}"
            python3 model_manager.py --download-menu
            read -rp "Press Enter to return to menu..."
            ;;
        5)
            echo -e "\n${MAGENTA}--- Model Versioning & Rollback Center ---${RESET}"
            python3 model_manager.py --menu
            read -rp "Press Enter to return to menu..."
            ;;
        6)
            echo -e "\n${CYAN}--- Incremental LoRA Fine-Tuning & Versioning ---${RESET}"
            echo "This will create a new immutable model version (e.g. v1, v2...) on your All-Rounder dataset."
            read -rp "Enter optional note for this version (or press Enter): " vnote
            read -rp "Start fresh from base model (v0) or resume from active version? [F]resh / [R]esume (default: Resume): " rchoice
            fresh_flag=""
            if [[ "$rchoice" =~ ^[Ff] ]]; then
                fresh_flag="--fresh"
            fi
            read -rp "Proceed with training on Apple Silicon GPU? (y/N): " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                python3 finetune.py --model Qwen/Qwen2.5-0.5B-Instruct --backend mlx --data-dir data $fresh_flag --notes "$vnote"
            fi
            read -rp "Press Enter to return to menu..."
            ;;
        7)
            echo -e "\n${GREEN}--- Universal Auto-Learner ---${RESET}"
            echo "1. Ingest all files dropped in 'inbox/' (skips unchanged files)"
            echo "2. Ingest specific file or note"
            echo "3. Rebuild All-Rounder dataset from existing sources"
            read -rp "Choose [1, 2, or 3]: " lchoice
            if [[ "$lchoice" == "1" ]]; then
                python3 auto_learner.py --inbox inbox
            elif [[ "$lchoice" == "2" ]]; then
                read -rp "Enter path to file or text note: " payload
                python3 -c "
from auto_learner import AutoLearner
import os
l = AutoLearner()
p = '$payload'
if os.path.exists(p):
    print(l.learn_file(p))
else:
    print(l.learn_text(p))
"
            elif [[ "$lchoice" == "3" ]]; then
                python3 dataset_builder.py
            fi
            read -rp "Press Enter to return to menu..."
            ;;
        8)
            echo -e "\n${YELLOW}--- Episodic Diary & Daily Milestones ---${RESET}"
            python3 -c "from memory_manager import MemoryManager; m=MemoryManager(); print(m.get_recent_episodes_summary(days=7))"
            read -rp "Press Enter to return to menu..."
            ;;
        9)
            echo -e "\n${BLUE}--- Developer Workflows & Procedural Runbooks ---${RESET}"
            python3 -c "
from memory_manager import MemoryManager
procs = MemoryManager().load_procedures()
if not procs:
    print('No procedures saved yet. Create one inside the console with: /workflow <name> = <step1> | <step2>')
for p in procs.values():
    print(f'• {p[\"name\"]} (Trigger: \"{p.get(\"trigger\", \"\")}\")')
    for i, step in enumerate(p.get(\"steps\", [])):
        print(f'    {i+1}. {step}')
"
            read -rp "Press Enter to return to menu..."
            ;;
        10)
            echo -e "\n${MAGENTA}--- Dynamically Discovered Career & Profile Memory ---${RESET}"
            python3 -c "from memory_manager import MemoryManager; m=MemoryManager(); print(m.get_summary())"
            read -rp "Press Enter to return to menu..."
            ;;
        11)
            python3 diagnostics.py
            read -rp "Press Enter to return to menu..."
            ;;
        12|q|Q|exit)
            echo -e "\n${CYAN}Goodbye Abhishek! Alisa will be right here when you return.${RESET}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}Invalid option.${RESET}"
            sleep 1
            ;;
    esac
done
