#!/bin/bash
# -*- coding: utf-8 -*-
# Fun-ASR vLLM One-Click Install & Start Script
# Usage: curl -sSL https://raw.githubusercontent.com/chicogong/Fun-ASR-vllm/main/install.sh | bash
#
# Options (via environment variables):
#   FUNASR_MODEL_DIR      - Model directory (default: FunAudioLLM/Fun-ASR-MLT-Nano-2512)
#   FUNASR_VLLM_MODEL_DIR - vLLM model (default: yuekai/Fun-ASR-MLT-Nano-2512-vllm)
#   FUNASR_PORT           - API port (default: 8080)
#   FUNASR_INSTALL_DIR    - Install directory (default: ~/Fun-ASR-vllm)
#   SKIP_INSTALL          - Skip git clone if set (default: false)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "¨X¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨["
echo "¨U       Fun-ASR vLLM - One-Click Install & Start             ¨U"
echo "¨U       Multilingual ASR with RTF ~0.02                      ¨U"
echo "¨^¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨a"
echo -e "${NC}"

# Configuration
INSTALL_DIR="${FUNASR_INSTALL_DIR:-$HOME/Fun-ASR-vllm}"
REPO_URL="https://github.com/chicogong/Fun-ASR-vllm.git"
export FUNASR_MODEL_DIR="${FUNASR_MODEL_DIR:-FunAudioLLM/Fun-ASR-MLT-Nano-2512}"
export FUNASR_VLLM_MODEL_DIR="${FUNASR_VLLM_MODEL_DIR:-yuekai/Fun-ASR-MLT-Nano-2512-vllm}"
export FUNASR_DEVICE="${FUNASR_DEVICE:-cuda:0}"
export FUNASR_VLLM_GPU_MEM="${FUNASR_VLLM_GPU_MEM:-0.4}"
export FUNASR_VLLM_MAX_MODEL_LEN="${FUNASR_VLLM_MAX_MODEL_LEN:-8192}"
PORT="${FUNASR_PORT:-8080}"

# Step 1: Check prerequisites
echo -e "${GREEN}[1/5]${NC} Checking prerequisites..."

# Check Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python not found. Please install Python 3.10+"
    exit 1
fi
PYTHON_CMD=$(command -v python3 || command -v python)
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "  Python: ${YELLOW}$PYTHON_VERSION${NC}"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  GPU: ${YELLOW}$GPU_INFO${NC}"
else
    echo -e "${YELLOW}[WARN]${NC} nvidia-smi not found. GPU may not be available."
fi

# Step 2: Clone or update repository
echo -e "${GREEN}[2/5]${NC} Setting up repository..."

if [ -z "$SKIP_INSTALL" ]; then
    if [ -d "$INSTALL_DIR/.git" ]; then
        echo "  Updating existing installation..."
        cd "$INSTALL_DIR"
        git pull --quiet origin main 2>/dev/null || true
    else
        echo "  Cloning repository to $INSTALL_DIR..."
        git clone --quiet "$REPO_URL" "$INSTALL_DIR" 2>/dev/null || {
            echo -e "${YELLOW}[WARN]${NC} Clone failed, directory may exist. Trying to use existing..."
        }
        cd "$INSTALL_DIR"
    fi
else
    echo "  Skipping clone (SKIP_INSTALL set)"
    cd "$INSTALL_DIR"
fi

# Step 3: Install dependencies
echo -e "${GREEN}[3/5]${NC} Installing dependencies..."

# Try to find new_tts conda environment
PYTHON_CMD="python"
if [ -f "/opt/conda/envs/new_tts/bin/python" ]; then
    PYTHON_CMD="/opt/conda/envs/new_tts/bin/python"
    echo "  Using conda environment: new_tts"
    echo "  Python path: $PYTHON_CMD"
elif command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/envs/new_tts/bin/python" ]; then
        PYTHON_CMD="$CONDA_BASE/envs/new_tts/bin/python"
        echo "  Using conda environment: new_tts"
        echo "  Python path: $PYTHON_CMD"
    fi
fi

# Install Python packages (only if not using pre-configured env)
if [ "$PYTHON_CMD" = "python" ]; then
    echo "  Installing Python packages..."
    pip install -q --upgrade pip 2>/dev/null || true
    pip install -q vllm>=0.10.0 fastapi uvicorn[standard] python-multipart torchaudio funasr>=1.2.7 2>/dev/null || {
        echo -e "${YELLOW}[WARN]${NC} Some packages may have failed. Trying requirements.txt..."
        pip install -q -r requirements.txt 2>/dev/null || true
    }
else
    echo "  Dependencies already installed in conda environment"
fi

# Step 4: Kill existing processes
echo -e "${GREEN}[4/5]${NC} Checking for existing processes..."
pkill -f "api_server.py" 2>/dev/null && echo "  Stopped existing api_server" || true
sleep 1

# Step 5: Start server
echo -e "${GREEN}[5/5]${NC} Starting API server..."
echo ""
echo -e "  ${BLUE}Configuration:${NC}"
echo -e "    Model:      ${YELLOW}$FUNASR_MODEL_DIR${NC}"
echo -e "    vLLM Model: ${YELLOW}$FUNASR_VLLM_MODEL_DIR${NC}"
echo -e "    Device:     ${YELLOW}$FUNASR_DEVICE${NC}"
echo -e "    GPU Memory: ${YELLOW}$FUNASR_VLLM_GPU_MEM${NC}"
echo -e "    Port:       ${YELLOW}$PORT${NC}"
echo ""

# Start in foreground or background based on TTY
if [ -t 0 ]; then
    echo -e "${YELLOW}[NOTE]${NC} First startup may take 1-2 minutes to download models..."
    echo -e "${GREEN}[INFO]${NC} Server starting at ${BLUE}http://0.0.0.0:$PORT${NC}"
    echo -e "${GREEN}[INFO]${NC} Press Ctrl+C to stop"
    echo ""
    cd "$INSTALL_DIR" && $PYTHON_CMD api_server.py
else
    echo -e "${GREEN}[INFO]${NC} Starting in background mode..."
    echo -e "${GREEN}[INFO]${NC} Using Python: $PYTHON_CMD"
    # Start with explicit Python path
    cd "$INSTALL_DIR"
    nohup $PYTHON_CMD api_server.py > /tmp/funasr_server.log 2>&1 &
    SERVER_PID=$!
    echo -e "${GREEN}[INFO]${NC} Server PID: $SERVER_PID"
    echo -e "${GREEN}[INFO]${NC} Log file: /tmp/funasr_server.log"
    echo ""
    echo -e "${BLUE}Waiting for server to start...${NC}"
    
    for i in {1..120}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo ""
            echo -e "${GREEN}¨X¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨[${NC}"
            echo -e "${GREEN}¨U  Server started successfully!                              ¨U${NC}"
            echo -e "${GREEN}¨^¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨a${NC}"
            echo ""
            echo -e "  API URL:     ${BLUE}http://localhost:$PORT${NC}"
            echo -e "  Health:      ${BLUE}http://localhost:$PORT/health${NC}"
            echo -e "  ASR (JSON):  ${BLUE}POST http://localhost:$PORT/asr${NC}"
            echo -e "  ASR (File):  ${BLUE}POST http://localhost:$PORT/asr/file${NC}"
            echo ""
            echo -e "  ${YELLOW}Example:${NC}"
            echo '  curl -X POST http://localhost:8080/asr/file -F "file=@audio.wav"'
            echo ""
            exit 0
        fi
        printf "."
        sleep 1
    done
    
    echo ""
    echo -e "${RED}[ERROR]${NC} Server failed to start. Check log: /tmp/funasr_server.log"
    tail -20 /tmp/funasr_server.log
    exit 1
fi
