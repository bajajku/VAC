#!/usr/bin/env bash
#
# ╔══════════════════════════════════════════════════════════════════╗
# ║          VAC Backend — One-Command Startup Script               ║
# ║  Installs all deps (no sudo) & launches API + vLLM + ngrok     ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash start_backend.sh          # Install deps & launch everything
#   bash start_backend.sh --stop   # Tear down the tmux session
#   bash start_backend.sh --status # Check if services are running
#
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these as needed
# ─────────────────────────────────────────────────────────────────────
VLLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
VLLM_PORT=8001
API_PORT=8000
TMUX_SESSION="vac-backend"
NGROK_VERSION="v3-stable"

# Resolve the backend directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOCAL_BIN="${HOME}/.local/bin"
NGROK_BIN="${LOCAL_BIN}/ngrok"

# ─────────────────────────────────────────────────────────────────────
# COLORS & HELPERS
# ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[  OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[FAIL]${NC}  $*"; }
header()  { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}\n"; }

# ─────────────────────────────────────────────────────────────────────
# --stop / --status FLAGS
# ─────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        tmux kill-session -t "${TMUX_SESSION}"
        success "Stopped tmux session '${TMUX_SESSION}'"
    else
        warn "No tmux session '${TMUX_SESSION}' found"
    fi
    exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        success "Session '${TMUX_SESSION}' is running"
        tmux list-panes -t "${TMUX_SESSION}" -F "  Pane #{pane_index}: #{pane_current_command} (#{pane_pid})"
    else
        warn "No tmux session '${TMUX_SESSION}' found"
    fi
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────
header "Pre-flight Checks"

# tmux is required
if ! command -v tmux &>/dev/null; then
    error "tmux is not installed. Please ask your admin to install it, or:"
    echo "       conda install -c conda-forge tmux   (if conda is available)"
    exit 1
fi
success "tmux found: $(tmux -V)"

# Python 3
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    error "Python 3.10+ is required but not found in PATH"
    exit 1
fi
success "Python found: $($PYTHON --version)"

# GPU check
if command -v nvidia-smi &>/dev/null; then
    success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/         /'
else
    warn "nvidia-smi not found — vLLM requires an NVIDIA GPU with CUDA"
    warn "Continuing anyway (vLLM will fail to start if no GPU is available)"
fi

# Check for existing session
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    warn "Tmux session '${TMUX_SESSION}' already exists!"
    read -rp "       Kill it and start fresh? [y/N]: " answer
    if [[ "${answer,,}" == "y" ]]; then
        tmux kill-session -t "${TMUX_SESSION}"
        success "Killed old session"
    else
        info "Attaching to existing session..."
        tmux attach-session -t "${TMUX_SESSION}"
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# LOAD .env FILE
# ─────────────────────────────────────────────────────────────────────
header "Loading Environment"

ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    # Source .env, handling spaces around = signs
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments & empty lines
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        # Remove leading/trailing whitespace, normalize key=value
        clean="$(echo "$line" | sed 's/[[:space:]]*=[[:space:]]*/=/')"
        if [[ "$clean" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            export "$clean"
        fi
    done < "$ENV_FILE"
    success "Loaded .env from ${ENV_FILE}"
else
    warn ".env file not found at ${ENV_FILE}"
fi

# HUGGING_FACE_TOKEN — needed for gated models
HF_TOKEN="${HUGGING_FACE_TOKEN:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
if [[ -z "$HF_TOKEN" ]]; then
    warn "No Hugging Face token found (HUGGING_FACE_TOKEN / HF_TOKEN)"
    read -rp "       Enter your HF token (or press Enter to skip): " HF_TOKEN
fi
if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    success "Hugging Face token set"
fi

# NGROK_AUTH_TOKEN
NGROK_TOKEN="${NGROK_AUTH_TOKEN:-${NGROK_AUTHTOKEN:-}}"
if [[ -z "$NGROK_TOKEN" ]]; then
    warn "No ngrok auth token found (NGROK_AUTH_TOKEN / NGROK_AUTHTOKEN)"
    read -rp "       Enter your ngrok auth token (or press Enter to skip): " NGROK_TOKEN
fi
if [[ -n "$NGROK_TOKEN" ]]; then
    export NGROK_AUTHTOKEN="$NGROK_TOKEN"
    success "ngrok auth token set"
fi

# ─────────────────────────────────────────────────────────────────────
# PYTHON VIRTUAL ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────
header "Python Virtual Environment"

if [[ -d "$VENV_DIR" && -f "${VENV_DIR}/bin/activate" ]]; then
    success "Virtual environment exists at ${VENV_DIR}"
else
    info "Creating virtual environment at ${VENV_DIR}..."
    $PYTHON -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

# Activate
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
success "Activated venv ($(python --version))"

# Upgrade pip
info "Upgrading pip..."
pip install --upgrade pip --quiet
success "pip upgraded"

# ─────────────────────────────────────────────────────────────────────
# INSTALL BACKEND DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────
# header "Installing Backend Dependencies"

# REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
# if [[ -f "$REQUIREMENTS" ]]; then
#     info "Installing from requirements.txt..."
#     pip install -r "$REQUIREMENTS" --quiet
#     success "Backend dependencies installed"
# else
#     warn "requirements.txt not found at ${REQUIREMENTS}"
# fi

# ─────────────────────────────────────────────────────────────────────
# INSTALL vLLM
# ─────────────────────────────────────────────────────────────────────
header "Installing vLLM"

if python -c "import vllm" 2>/dev/null; then
    VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    success "vLLM already installed (v${VLLM_VER})"
else
    info "Installing vLLM (this may take a few minutes)..."
    pip install vllm
    if python -c "import vllm" 2>/dev/null; then
        VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
        success "vLLM installed successfully (v${VLLM_VER})"
    else
        error "vLLM installation failed. Check CUDA compatibility."
        error "You may need: pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# INSTALL NGROK (standalone binary, no sudo)
# ─────────────────────────────────────────────────────────────────────
header "Installing ngrok"

# Check if ngrok is already available
NGROK_CMD=""
if command -v ngrok &>/dev/null; then
    NGROK_CMD="ngrok"
    success "ngrok found in PATH: $(ngrok version 2>/dev/null || echo 'unknown version')"
elif [[ -x "$NGROK_BIN" ]]; then
    NGROK_CMD="$NGROK_BIN"
    success "ngrok found at ${NGROK_BIN}"
else
    info "Downloading ngrok (no sudo required)..."
    mkdir -p "$LOCAL_BIN"

    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)  NGROK_ARCH="amd64" ;;
        aarch64) NGROK_ARCH="arm64" ;;
        *)
            error "Unsupported architecture: ${ARCH}"
            exit 1
            ;;
    esac

    NGROK_URL="https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-${NGROK_VERSION}-linux-${NGROK_ARCH}.tgz"
    TMPFILE=$(mktemp /tmp/ngrok-XXXXXX.tgz)

    if command -v wget &>/dev/null; then
        wget -q -O "$TMPFILE" "$NGROK_URL"
    elif command -v curl &>/dev/null; then
        curl -sL -o "$TMPFILE" "$NGROK_URL"
    else
        error "Neither wget nor curl found. Cannot download ngrok."
        exit 1
    fi

    tar -xzf "$TMPFILE" -C "$LOCAL_BIN"
    rm -f "$TMPFILE"
    chmod +x "$NGROK_BIN"
    NGROK_CMD="$NGROK_BIN"
    success "ngrok installed to ${NGROK_BIN}"
fi

# Configure ngrok auth token
if [[ -n "${NGROK_AUTHTOKEN:-}" ]]; then
    info "Configuring ngrok auth token..."
    "$NGROK_CMD" config add-authtoken "$NGROK_AUTHTOKEN" 2>/dev/null || true
    success "ngrok auth token configured"
fi

# Ensure ~/.local/bin is in PATH for tmux panes
export PATH="${LOCAL_BIN}:${PATH}"

# ─────────────────────────────────────────────────────────────────────
# PRE-DOWNLOAD THE vLLM MODEL (optional but avoids first-run delay)
# ─────────────────────────────────────────────────────────────────────
header "Checking Model Availability"

info "Verifying model '${VLLM_MODEL}' is accessible..."
if python -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
try:
    path = snapshot_download('${VLLM_MODEL}', token=token, local_files_only=True)
    print(f'Model cached at: {path}')
except Exception:
    print('Model not cached locally, will download on first vLLM start')
" 2>/dev/null; then
    success "Model check complete"
else
    warn "Could not verify model cache (huggingface_hub may not be installed)"
    info "vLLM will download the model on first start if needed"
fi

# ─────────────────────────────────────────────────────────────────────
# LAUNCH TMUX SESSION
# ─────────────────────────────────────────────────────────────────────
header "Launching Services in tmux"

# Build the activation prefix for tmux panes
ACTIVATE_CMD="source ${VENV_DIR}/bin/activate"
ENV_EXPORTS=""
[[ -n "${HF_TOKEN:-}" ]] && ENV_EXPORTS+="export HF_TOKEN='${HF_TOKEN}'; export HUGGING_FACE_HUB_TOKEN='${HF_TOKEN}'; "
[[ -n "${NGROK_AUTHTOKEN:-}" ]] && ENV_EXPORTS+="export NGROK_AUTHTOKEN='${NGROK_AUTHTOKEN}'; "
ENV_EXPORTS+="export PATH='${LOCAL_BIN}:\$PATH'; "

# Pane commands
CMD_API="cd ${SCRIPT_DIR} && ${ACTIVATE_CMD} && ${ENV_EXPORTS} echo -e '${GREEN}[Core API]${NC} Starting on port ${API_PORT}...' && uvicorn api:app_api --host 0.0.0.0 --port ${API_PORT}"

CMD_VLLM="cd ${SCRIPT_DIR} && ${ACTIVATE_CMD} && ${ENV_EXPORTS} echo -e '${GREEN}[vLLM]${NC} Starting ${VLLM_MODEL} on port ${VLLM_PORT}...' && vllm serve ${VLLM_MODEL} --port ${VLLM_PORT} --host 0.0.0.0"

CMD_NGROK="sleep 5 && ${ENV_EXPORTS} echo -e '${GREEN}[ngrok]${NC} Tunneling port ${API_PORT}...' && ${NGROK_CMD} http ${API_PORT}"

# Create tmux session with first pane (Core API)
tmux new-session -d -s "${TMUX_SESSION}" -n "backend" bash -c "${CMD_API}; exec bash"

# Split horizontally for vLLM
tmux split-window -h -t "${TMUX_SESSION}:backend" bash -c "${CMD_VLLM}; exec bash"

# Split the right pane vertically for ngrok
tmux split-window -v -t "${TMUX_SESSION}:backend.1" bash -c "${CMD_NGROK}; exec bash"

# Even out the pane layout
tmux select-layout -t "${TMUX_SESSION}:backend" main-vertical

# Select the first pane
tmux select-pane -t "${TMUX_SESSION}:backend.0"

# ─────────────────────────────────────────────────────────────────────
# LAUNCH BANNER
# ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║           🚀 VAC Backend — All Services Launched        ║${NC}"
echo -e "${BOLD}${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}${CYAN}║${NC}  Core API  │ http://localhost:${API_PORT}                      ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  vLLM      │ http://localhost:${VLLM_PORT}  (${VLLM_MODEL}) ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  ngrok     │ Tunneling port ${API_PORT} (check ngrok pane)     ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}${CYAN}║${NC}  tmux session: ${BOLD}${TMUX_SESSION}${NC}                              ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  Switch panes:  ${BOLD}Ctrl-b + arrow keys${NC}                   ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  Detach:        ${BOLD}Ctrl-b + d${NC}                             ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  Re-attach:     ${BOLD}tmux attach -t ${TMUX_SESSION}${NC}               ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  Stop all:      ${BOLD}bash start_backend.sh --stop${NC}            ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Attach to the session
info "Attaching to tmux session..."
tmux attach-session -t "${TMUX_SESSION}"
