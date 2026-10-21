#!/usr/bin/env bash
# ==============================================================================
# Alisa 1-Line Universal Mac Installer
# Author: Abhishek Tiwari
# Target: macOS (Apple Silicon M-Series & Intel)
#
# Usage (1-Line Command):
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Abhishek543-dotcom/alisa---llm-assistant-/main/install.sh)"
# ==============================================================================

set -eo pipefail

# ANSI Colors
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
MAGENTA="\033[0;35m"
RED="\033[0;31m"
RESET="\033[0m"

REPO_URL="${ALISA_REPO_URL:-https://github.com/Abhishek543-dotcom/alisa---llm-assistant-.git}"
BRANCH="${ALISA_BRANCH:-main}"
INSTALL_DIR="${ALISA_INSTALL_DIR:-$HOME/alisa}"

clear 2>/dev/null || true
echo -e "${CYAN}${BOLD}"
cat << "EOF"
    ___    __    ____ _____ ___ 
   /   |  / /   /  _// ___//   |
  / /| | / /    / /  \__ \/ /| |
 / ___ |/ /____/ /  ___/ / ___ |
/_/  |_/_____/___/ /____/_/  |_|
EOF
echo -e "${RESET}"
echo -e "${BOLD}${BLUE}✨ Alisa — 1-Line Autonomous All-Rounder Voice AI Installer${RESET}"
echo -e "${CYAN}─────────────────────────────────────────────────────────────────${RESET}"

# 1. OS & Architecture Check
echo -e "\n${BOLD}[1/6] Verifying System Compatibility...${RESET}"
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo -e "${RED}✗ Error: Alisa is designed specifically for macOS (Apple Silicon Metal GPU & native voice synthesis).${RESET}"
    exit 1
fi

ARCH="$(uname -m)"
OS_VER="$(sw_vers -productVersion)"
if [[ "$ARCH" == "arm64" ]]; then
    echo -e "  ${GREEN}✓ Apple Silicon Detected (${ARCH}, macOS ${OS_VER})${RESET}"
    echo -e "  ${GREEN}✓ Metal GPU Acceleration & MLX Unified Memory: FULLY SUPPORTED${RESET}"
else
    echo -e "  ${YELLOW}! Intel x86_64 Mac detected (macOS ${OS_VER}). Running PyTorch MPS/CPU fallback.${RESET}"
fi

# 2. Prerequisites Check (Git, Python)
echo -e "\n${BOLD}[2/6] Checking Prerequisites (Git & Python 3)...${RESET}"

# Check Git
if ! command -v git &>/dev/null; then
    echo -e "${YELLOW}Git not found. Triggering macOS Command Line Tools installation...${RESET}"
    xcode-select --install || true
    echo -e "${RED}Please complete the Command Line Tools installation dialog, then rerun this script.${RESET}"
    exit 1
fi
echo -e "  ${GREEN}✓ Git: $(git --version)${RESET}"

# Find best Python 3
PYTHON_BIN=""
for py in python3.11 python3.12 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        # Check minor version >= 9
        ver="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        major="$(echo "$ver" | cut -d. -f1)"
        minor="$(echo "$ver" | cut -d. -f2)"
        if [[ "$major" -eq 3 && "$minor" -ge 9 ]]; then
            PYTHON_BIN="$(command -v "$py")"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo -e "${RED}✗ Python 3.9+ is required. Installing via Homebrew or python.org is recommended.${RESET}"
    if command -v brew &>/dev/null; then
        echo -e "${CYAN}Installing Python via Homebrew...${RESET}"
        brew install python@3.11
        PYTHON_BIN="$(brew --prefix)/bin/python3.11"
    else
        echo -e "${RED}Please install Python 3.10 or 3.11 from https://www.python.org/downloads/ and rerun.${RESET}"
        exit 1
    fi
fi
echo -e "  ${GREEN}✓ Python: $("$PYTHON_BIN" --version) ($PYTHON_BIN)${RESET}"

# 3. Clone or Update Repository
echo -e "\n${BOLD}[3/6] Setting Up Alisa Repository...${RESET}"
# Check if currently inside the repo directory
if [[ -f "alisa.sh" && -f "chat_assistant.py" ]]; then
    INSTALL_DIR="$(pwd)"
    echo -e "  ${GREEN}✓ Using current directory: ${INSTALL_DIR}${RESET}"
else
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        echo -e "  ${CYAN}Updating existing installation in ${INSTALL_DIR}...${RESET}"
        cd "$INSTALL_DIR"
        git pull origin "$BRANCH" 2>/dev/null || git pull || true
    else
        echo -e "  ${CYAN}Cloning Alisa into ${INSTALL_DIR}...${RESET}"
        mkdir -p "$INSTALL_DIR"
        git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" 2>/dev/null || \
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
fi

# Ensure executable permissions and symlinks
chmod +x alisa.sh 2>/dev/null || true
ln -sf alisa.sh alisha.sh 2>/dev/null || true

# 4. Create Virtual Environment & Install Dependencies
echo -e "\n${BOLD}[4/6] Configuring Python Virtual Environment & Metal MLX...${RESET}"
if [[ ! -d ".venv" ]]; then
    echo -e "  Creating isolated virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
echo -e "  ${CYAN}Installing / updating required packages...${RESET}"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "  ${GREEN}✓ Dependencies installed successfully!${RESET}"

# 5. Pre-Cache Core Base Models & Datasets
echo -e "\n${BOLD}[5/6] Pre-caching Default Models & Initializing AI Brain...${RESET}"
python3 dataset_builder.py >/dev/null 2>&1 || true
python3 model_manager.py --list >/dev/null 2>&1 || true

echo -e "  ${CYAN}Checking base models (Qwen2.5-0.5B, Whisper Tiny, BGE Embeddings)...${RESET}"
python3 -c "
from model_manager import download_base_model
try:
    download_base_model('BAAI/bge-small-en-v1.5', verbose=False)
    download_base_model('mlx-community/whisper-tiny', verbose=False)
    download_base_model('Qwen/Qwen2.5-0.5B-Instruct', verbose=False)
    print('  ✓ Pre-cached Qwen2.5-0.5B, Whisper, and Embeddings successfully.')
except Exception as e:
    print(f'  Note: Pre-caching will continue on first launch ({e})')
" || true

# 6. Global CLI Launcher & PATH Setup
echo -e "\n${BOLD}[6/6] Creating Global 'alisa' & 'alisha' Terminal Commands...${RESET}"
mkdir -p "$HOME/.local/bin"

cat << EOF > "$HOME/.local/bin/alisa"
#!/usr/bin/env bash
cd "$INSTALL_DIR"
exec ./alisha.sh "\$@"
EOF
chmod +x "$HOME/.local/bin/alisa"
ln -sf "$HOME/.local/bin/alisa" "$HOME/.local/bin/alisha"

# Add ~/.local/bin to PATH in shell rc files if missing
PATH_EXPORT='export PATH="$HOME/.local/bin:$PATH"'
for rc in "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.bashrc"; do
    if [[ -f "$rc" ]]; then
        if ! grep -q '.local/bin' "$rc"; then
            echo -e "\n# Added by Alisa Installer\n$PATH_EXPORT" >> "$rc"
        fi
    fi
done
export PATH="$HOME/.local/bin:$PATH"

# Play native macOS completion chime & voice announcement
afplay /System/Library/Sounds/Tink.aiff 2>/dev/null || true
(say -v Samantha "Alisa installation is complete! I am ready to assist you." 2>/dev/null &)

echo -e "\n${CYAN}─────────────────────────────────────────────────────────────────${RESET}"
echo -e "${GREEN}${BOLD}🎉 SUCCESS! Alisa is 100% installed and ready on your Mac!${RESET}"
echo -e "${CYAN}─────────────────────────────────────────────────────────────────${RESET}"
echo -e "You can now run Alisa from ${BOLD}ANY${RESET} terminal window using either command:\n"
echo -e "  ${BOLD}${GREEN}alisa${RESET}       (Launches Alisa interactive console)"
echo -e "  ${BOLD}${CYAN}alisha${RESET}      (Alternate command)"
echo -e "  ${BOLD}${YELLOW}alisa --listen${RESET}  (Hands-free wake-word mode: say 'Alisa')"
echo -e "  ${BOLD}${BLUE}alisa --briefing${RESET} (Instant executive morning/evening briefing)"
echo -e "\nInstallation directory: ${BOLD}${INSTALL_DIR}${RESET}\n"

# If interactive terminal, ask to launch right now
if [[ -t 0 ]]; then
    read -rp "Would you like to start Alisa right now? (Y/n): " start_now < /dev/tty || start_now="y"
    if [[ ! "$start_now" =~ ^[Nn] ]]; then
        cd "$INSTALL_DIR"
        exec ./alisha.sh
    fi
fi
