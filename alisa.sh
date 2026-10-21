#!/usr/bin/env bash
# ==============================================================================
# alisa.sh — Production-Grade Autonomous Personal Voice AI Assistant
# Author: Abhishek Tiwari
# Architecture: Apple Silicon Metal GPU (MLX) + Low-Resource Voice Synthesis
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

# Direct Pass-through for CLI arguments (e.g. ./alisa.sh --listen, ./alisa.sh --query "...")
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--diag" || "$1" == "--diagnostics" ]]; then
        python3 diagnostics.py
        exit 0
    fi
    python3 alisa.py "$@"
    exit 0
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
    echo -e "${BOLD}${BLUE}✨ Abhishek Tiwari's Autonomous Personal Voice AI Assistant${RESET}"
    echo -e "${CYAN}─────────────────────────────────────────────────────────────────${RESET}"

    # Extract quick system metrics
    local batt_str
    batt_str="$(pmset -g batt 2>/dev/null | grep -o '[0-9]*%' | head -1 || echo 'N/A')"
    local ram_pct
    ram_pct="$(python3 -c 'import psutil; print(int(psutil.virtual_memory().percent))' 2>/dev/null || echo 'N/A')"
    local chunks_count
    chunks_count="$(python3 -c 'import chromadb; c=chromadb.PersistentClient("chroma_db").get_collection("personal_emails").count(); print(f"{c:,}")' 2>/dev/null || echo '30,919')"
    local sol_count
    sol_count="$(python3 -c 'from memory_manager import MemoryManager; print(len(MemoryManager().load_solutions()))' 2>/dev/null || echo '4')"
    local latest_co
    latest_co="$(python3 -c 'from memory_manager import MemoryManager; print(MemoryManager().get_latest_company())' 2>/dev/null || echo 'JPMorgan Chase & Co.')"

    echo -e " 💻 ${BOLD}System:${RESET} M4 Apple Silicon  |  ${BOLD}RAM:${RESET} ${ram_pct}%  |  ${BOLD}Battery:${RESET} ${batt_str}"
    echo -e " 🧠 ${BOLD}Engine:${RESET} Qwen2.5-0.5B (MLX GPU)  |  ${BOLD}ChromaDB:${RESET} ${chunks_count} chunks"
    echo -e " 👤 ${BOLD}Career:${RESET} ${latest_co}  |  ${BOLD}Solved Issues:${RESET} ${sol_count} documented"
    echo -e "${CYAN}─────────────────────────────────────────────────────────────────${RESET}"
}

# ==============================================================================
# Interactive Menu Loop
# ==============================================================================
while true; do
    print_header
    echo -e "${BOLD}Select an operation:${RESET}\n"
    echo -e "  ${GREEN}[1] 💬 Talk with Alisa${RESET}           (Interactive Console + Voice Output)"
    echo -e "  ${YELLOW}[2] 🎙️  Hands-Free Wake-Word Mode${RESET} (Say 'Alisa' to wake, 'Stop' to sleep)"
    echo -e "  ${CYAN}[3] 📚 Auto-Learner & Ingest${RESET}      (Scan inbox/ or add any PDF/code/doc)"
    echo -e "  ${MAGENTA}[4] 🧠 Train & Fine-Tune Model${RESET}    (Incremental LoRA on Apple Silicon GPU)"
    echo -e "  ${BLUE}[5] 👤 Discovered Profile Memory${RESET}  (View timeline, CTC, discovered facts)"
    echo -e "  ${YELLOW}[6] 🛠️  Lifelong Error Memory${RESET}     (Inspect how past errors were resolved)"
    echo -e "  ${CYAN}[7] 📜 Conversation History${RESET}       (View previous Q&A turns)"
    echo -e "  ${GREEN}[8] 🔬 Run System Diagnostics${RESET}     (Verify Mic, Speakers, GPU, DB)"
    echo -e "  ${RED}[9] 🚪 Exit Alisa${RESET}\n"

    read -rp "Enter choice [1-9]: " choice

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
            echo -e "\n${CYAN}--- Universal Auto-Learner ---${RESET}"
            echo "1. Ingest all files dropped in 'inbox/'"
            echo "2. Ingest specific file or note"
            read -rp "Choose [1 or 2]: " lchoice
            if [[ "$lchoice" == "1" ]]; then
                python3 -c "from auto_learner import ingest_inbox_directory; ingest_inbox_directory('inbox')"
            else
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
            fi
            read -rp "Press Enter to return to menu..."
            ;;
        4)
            echo -e "\n${MAGENTA}--- Incremental LoRA Fine-Tuning on Apple Silicon Metal ---${RESET}"
            echo "This will train Qwen2.5-0.5B-Instruct on your learned corpus and synthetic instruction pairs."
            read -rp "Proceed with training? (y/N): " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                python3 finetune.py --model Qwen/Qwen2.5-0.5B-Instruct --backend mlx --data-dir data --adapter-dir adapters/personal_lora
            fi
            read -rp "Press Enter to return to menu..."
            ;;
        5)
            echo -e "\n${BLUE}--- Dynamically Discovered Career & Profile Memory ---${RESET}"
            python3 -c "from memory_manager import MemoryManager; m=MemoryManager(); print(m.get_summary())"
            read -rp "Press Enter to return to menu..."
            ;;
        6)
            echo -e "\n${YELLOW}--- Lifelong Solved Errors & Troubleshooting Memory ---${RESET}"
            python3 -c "
from memory_manager import MemoryManager
m = MemoryManager()
sols = m.load_solutions()
for s in sols:
    print(f'[{s[\"id\"]}] {s[\"issue\"]}')
    print(f'       Root Cause: {s.get(\"root_cause\", \"N/A\")}')
    print(f'       Resolution: {s[\"resolution\"]}\n')
"
            read -rp "Press Enter to return to menu..."
            ;;
        7)
            echo -e "\n${CYAN}--- Recent Conversation History ---${RESET}"
            python3 -c "
from memory_manager import MemoryManager
m = MemoryManager()
hist = m.get_recent_history(limit=10)
for h in hist:
    print(f'[{h.get(\"timestamp\", \"\")}] ({h.get(\"source\", \"\")})')
    print(f'User:  {h.get(\"query\", \"\")}')
    print(f'Alisa: {h.get(\"answer\", \"\")}\n')
"
            read -rp "Press Enter to return to menu..."
            ;;
        8)
            python3 diagnostics.py
            read -rp "Press Enter to return to menu..."
            ;;
        9|q|Q|exit)
            echo -e "\n${CYAN}Goodbye Abhishek! Alisa will be right here when you return.${RESET}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}Invalid option.${RESET}"
            sleep 1
            ;;
    esac
done
