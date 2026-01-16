#!/bin/bash
# -*- coding: utf-8 -*-
# Fun-ASR vLLM API Server Startup Script
# One-click start for MLT multilingual ASR service

set -e

# Configuration (can be overridden by environment variables)
export FUNASR_MODEL_DIR="${FUNASR_MODEL_DIR:-FunAudioLLM/Fun-ASR-MLT-Nano-2512}"
export FUNASR_VLLM_MODEL_DIR="${FUNASR_VLLM_MODEL_DIR:-yuekai/Fun-ASR-MLT-Nano-2512-vllm}"
export FUNASR_DEVICE="${FUNASR_DEVICE:-cuda:0}"
export FUNASR_VLLM_GPU_MEM="${FUNASR_VLLM_GPU_MEM:-0.4}"
export FUNASR_VLLM_MAX_MODEL_LEN="${FUNASR_VLLM_MAX_MODEL_LEN:-8192}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Fun-ASR vLLM API Server Startup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Model:      ${YELLOW}${FUNASR_MODEL_DIR}${NC}"
echo -e "vLLM Model: ${YELLOW}${FUNASR_VLLM_MODEL_DIR}${NC}"
echo -e "Device:     ${YELLOW}${FUNASR_DEVICE}${NC}"
echo -e "GPU Memory: ${YELLOW}${FUNASR_VLLM_GPU_MEM}${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if conda is available
ENV_NAME="funasr-vllm"
if command -v conda &> /dev/null; then
    echo -e "${GREEN}[INFO]${NC} Checking conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    
    if conda env list | grep -q "^${ENV_NAME} "; then
        conda activate "$ENV_NAME"
        echo -e "${GREEN}[INFO]${NC} Activated: $ENV_NAME"
    else
        echo -e "${YELLOW}[WARN]${NC} $ENV_NAME environment not found"
        echo -e "${YELLOW}[INFO]${NC} Run 'curl -sSL https://raw.githubusercontent.com/chicogong/Fun-ASR-vllm/main/install.sh | bash' to setup"
        exit 1
    fi
fi

# Check Python version
echo -e "${GREEN}[INFO]${NC} Python: $(python --version)"

# Check GPU
echo -e "${GREEN}[INFO]${NC} Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo -e "${YELLOW}[WARN]${NC} nvidia-smi not found"
fi

# Kill any existing api_server process
echo -e "${GREEN}[INFO]${NC} Checking for existing processes..."
pkill -f "api_server.py" 2>/dev/null || true
sleep 1

# Start server
echo ""
echo -e "${GREEN}[INFO]${NC} Starting API server on http://0.0.0.0:8080 ..."
echo -e "${YELLOW}[NOTE]${NC} First startup may take 1-2 minutes to download and load models"
echo ""

python api_server.py
