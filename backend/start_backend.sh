#!/usr/bin/env bash
#
# ╔══════════════════════════════════════════════════════════════════╗
# ║          VAC Backend — One-Command Startup Script               ║
# ║  Installs all deps (no sudo) & launches API + vLLM + ngrok     ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash start_backend.sh            # Install deps & launch everything
#   bash start_backend.sh --stop     # Stop all services
#   bash start_backend.sh --status   # Check if services are running
#   bash start_backend.sh --logs     # Tail all service logs
#   bash start_backend.sh --logs api # Tail a specific service log
#
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these as needed
# ─────────────────────────────────────────────────────────────────────
VLLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
VLLM_PORT=8001
API_PORT=8000
NGROK_VERSION="v3-stable"

# Resolve the backend directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOCAL_BIN="${HOME}/.local/bin"
NGROK_BIN="${LOCAL_BIN}/ngrok"

# Directories for logs and PID files
LOG_DIR="${SCRIPT_DIR}/logs"
PID_DIR="${SCRIPT_DIR}/.pids"

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

# Check if a service is running by its PID file
is_running() {
    local name="$1"
    local pidfile="${PID_DIR}/${name}.pid"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Stop a specific service
stop_service() {
    local name="$1"
    local pidfile="${PID_DIR}/${name}.pid"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            # Wait up to 5 seconds for graceful shutdown
            for _ in $(seq 1 10); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            success "Stopped ${name} (PID ${pid})"
        else
            warn "${name} was not running (stale PID ${pid})"
        fi
        rm -f "$pidfile"
    else
        warn "No PID file for ${name}"
    fi
}

# ─────────────────────────────────────────────────────────────────────
# --stop FLAG
# ─────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    header "Stopping Services"
    for svc in ngrok vllm api; do
        stop_service "$svc"
    done
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# --status FLAG
# ─────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--status" ]]; then
    header "Service Status"
    any_running=false
    for svc in api vllm ngrok; do
        pidfile="${PID_DIR}/${svc}.pid"
        if [[ -f "$pidfile" ]]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                success "${svc} is running (PID ${pid})"
                any_running=true
            else
                warn "${svc} is NOT running (stale PID file)"
                rm -f "$pidfile"
            fi
        else
            warn "${svc} is NOT running (no PID file)"
        fi
    done
    echo ""
    if $any_running; then
        info "View logs: bash start_backend.sh --logs"
        info "Stop all:  bash start_backend.sh --stop"
    fi
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# --logs FLAG
# ─────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--logs" ]]; then
    svc="${2:-}"
    if [[ -n "$svc" ]]; then
        logfile="${LOG_DIR}/${svc}.log"
        if [[ -f "$logfile" ]]; then
            tail -f "$logfile"
        else
            error "Log file not found: ${logfile}"
            exit 1
        fi
    else
        # Tail all log files
        logfiles=()
        for f in api vllm ngrok; do
            if [[ -f "${LOG_DIR}/${f}.log" ]]; then
                logfiles+=("${LOG_DIR}/${f}.log")
            fi
        done
        if [[ ${#logfiles[@]} -eq 0 ]]; then
            error "No log files found in ${LOG_DIR}"
            exit 1
        fi
        tail -f "${logfiles[@]}"
    fi
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────
header "Pre-flight Checks"

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

# Check for already-running services
any_already_running=false
for svc in api vllm ngrok; do
    if is_running "$svc"; then
        any_already_running=true
    fi
done

if $any_already_running; then
    warn "Some services are already running!"
    echo ""
    bash "${BASH_SOURCE[0]}" --status 2>/dev/null || true
    echo ""
    read -rp "       Stop them and start fresh? [y/N]: " answer
    if [[ "${answer,,}" == "y" ]]; then
        bash "${BASH_SOURCE[0]}" --stop 2>/dev/null || true
        success "Stopped old services"
    else
        info "Exiting. Use --stop to stop services, or --status to check."
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# LOAD .env FILE
# ─────────────────────────────────────────────────────────────────────
header "Loading Environment"

ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        clean="$(echo "$line" | sed 's/[[:space:]]*=[[:space:]]*/=/')"
        
        var_name="${clean%%=*}"
        var_value="${clean#*=}"
        
        # Strip leading/trailing double quotes
        var_value="${var_value%\"}"
        var_value="${var_value#\"}"
        # Strip leading/trailing single quotes
        var_value="${var_value%\'}"
        var_value="${var_value#\'}"
        
        if [[ "$var_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            export "${var_name}=${var_value}"
        fi
    done < "$ENV_FILE"
    success "Loaded .env from ${ENV_FILE}"
else
    warn ".env file not found at ${ENV_FILE}"
fi

# HUGGING_FACE_TOKEN
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
# INSTALL uv (if not present)
# ─────────────────────────────────────────────────────────────────────
header "Package Manager (uv)"

if command -v uv &>/dev/null; then
    success "uv found: $(uv --version)"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    if command -v uv &>/dev/null; then
        success "uv installed: $(uv --version)"
    else
        error "Failed to install uv. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# INSTALL BACKEND DEPENDENCIES (via uv)
# ─────────────────────────────────────────────────────────────────────
header "Installing Backend Dependencies"

cd "$SCRIPT_DIR"

if [[ -f "pyproject.toml" ]]; then
    info "Running uv sync (using pyproject.toml + uv.lock)..."
    if uv sync 2>&1; then
        success "Dependencies installed via uv sync"
    else
        warn "uv sync failed — falling back to installing packages individually..."
        # Extract package names from pyproject.toml dependencies and install one by one
        uv pip install --quiet pip 2>/dev/null || true
        FAILED_PKGS=()
        while IFS= read -r pkg; do
            # Strip quotes, whitespace, trailing comma
            pkg=$(echo "$pkg" | sed 's/^[[:space:]]*"//;s/"[[:space:]]*,*$//')
            [[ -z "$pkg" || "$pkg" == "]" || "$pkg" == "[" ]] && continue
            if ! uv pip install "$pkg" --quiet 2>/dev/null; then
                warn "Skipped: ${pkg} (install failed)"
                FAILED_PKGS+=("$pkg")
            fi
        done < <(sed -n '/^dependencies = \[/,/^\]/p' pyproject.toml | grep -v 'dependencies' | grep -v '^\]')
        if [[ ${#FAILED_PKGS[@]} -gt 0 ]]; then
            warn "Failed packages: ${FAILED_PKGS[*]}"
            warn "You may need to install these manually"
        else
            success "All packages installed individually"
        fi
    fi
else
    warn "pyproject.toml not found — skipping dependency installation"
fi

# Activate the venv that uv created/managed
VENV_DIR="${SCRIPT_DIR}/.venv"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    success "Activated venv ($(python --version))"
else
    error "No virtual environment found at ${VENV_DIR}. uv sync should have created one."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# INSTALL vLLM
# ─────────────────────────────────────────────────────────────────────
header "Installing vLLM"

if python -c "import vllm" 2>/dev/null; then
    VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    success "vLLM already installed (v${VLLM_VER})"
else
    info "Installing vLLM (this may take a few minutes)..."
    if uv pip install vllm "openai<1.60.0" 2>&1; then
        if python -c "import vllm" 2>/dev/null; then
            VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
            success "vLLM installed successfully (v${VLLM_VER})"
        else
            warn "vLLM installed but import failed — will try to launch anyway"
        fi
    else
        warn "vLLM installation failed. Check CUDA compatibility."
        warn "You may need: uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129"
        warn "Continuing without vLLM..."
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# INSTALL NGROK (standalone binary, no sudo)
# ─────────────────────────────────────────────────────────────────────
header "Installing ngrok"

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

export PATH="${LOCAL_BIN}:${PATH}"

# ─────────────────────────────────────────────────────────────────────
# PRE-DOWNLOAD THE vLLM MODEL
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
# LAUNCH SERVICES AS BACKGROUND PROCESSES
# ─────────────────────────────────────────────────────────────────────
header "Launching Services"

mkdir -p "$LOG_DIR" "$PID_DIR"

# ── 1. Core API ──────────────────────────────────────────────────────
info "Starting Core API on port ${API_PORT}..."
cd "$SCRIPT_DIR"
nohup "${VENV_DIR}/bin/uvicorn" api:app_api \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    >> "${LOG_DIR}/api.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "${PID_DIR}/api.pid"

# Verify it started
sleep 1
if kill -0 "$API_PID" 2>/dev/null; then
    success "Core API started (PID ${API_PID}) → ${LOG_DIR}/api.log"
else
    error "Core API failed to start. Check ${LOG_DIR}/api.log"
    tail -5 "${LOG_DIR}/api.log" 2>/dev/null | sed 's/^/         /'
    exit 1
fi

# ── 2. vLLM Server ──────────────────────────────────────────────────
info "Starting vLLM (${VLLM_MODEL}) on port ${VLLM_PORT}..."
nohup "${VENV_DIR}/bin/vllm" serve "$VLLM_MODEL" \
    --port "$VLLM_PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    >> "${LOG_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "${PID_DIR}/vllm.pid"

sleep 1
if kill -0 "$VLLM_PID" 2>/dev/null; then
    success "vLLM started (PID ${VLLM_PID}) → ${LOG_DIR}/vllm.log"
else
    error "vLLM failed to start. Check ${LOG_DIR}/vllm.log"
    tail -5 "${LOG_DIR}/vllm.log" 2>/dev/null | sed 's/^/         /'
    # Don't exit — API can still work without vLLM
    warn "Continuing without vLLM..."
fi

# ── 3. ngrok Tunnel ──────────────────────────────────────────────────
info "Starting ngrok tunnel for port ${API_PORT}..."
nohup "$NGROK_CMD" http --domain=dassie-skilled-mako.ngrok-free.app "$API_PORT" \
    --log=stdout \
    >> "${LOG_DIR}/ngrok.log" 2>&1 &
NGROK_PID=$!
echo "$NGROK_PID" > "${PID_DIR}/ngrok.pid"

sleep 2
if kill -0 "$NGROK_PID" 2>/dev/null; then
    success "ngrok started (PID ${NGROK_PID}) → ${LOG_DIR}/ngrok.log"
    # Try to extract the public URL from ngrok API
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
        | python -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null \
        || echo "check ngrok log")
    if [[ "$NGROK_URL" != "check ngrok log" ]]; then
        success "ngrok public URL: ${NGROK_URL}"
    else
        info "ngrok URL not yet available — check: curl http://localhost:4040/api/tunnels"
    fi
else
    error "ngrok failed to start. Check ${LOG_DIR}/ngrok.log"
    tail -5 "${LOG_DIR}/ngrok.log" 2>/dev/null | sed 's/^/         /'
    warn "Continuing without ngrok..."
fi

# ─────────────────────────────────────────────────────────────────────
# LAUNCH BANNER
# ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║           🚀 VAC Backend — All Services Launched        ║${NC}"
echo -e "${BOLD}${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}${CYAN}║${NC}  Core API  │ http://localhost:${API_PORT}                      ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  vLLM      │ http://localhost:${VLLM_PORT}  (${VLLM_MODEL}) ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  ngrok     │ ${NGROK_URL:-check ngrok log}                    ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}${CYAN}║${NC}  View logs:   ${BOLD}bash start_backend.sh --logs${NC}              ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  View one:    ${BOLD}bash start_backend.sh --logs api${NC}          ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  Status:      ${BOLD}bash start_backend.sh --status${NC}            ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}║${NC}  Stop all:    ${BOLD}bash start_backend.sh --stop${NC}              ${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Services are running in the background. Your terminal is free to use.${NC}"
echo ""
