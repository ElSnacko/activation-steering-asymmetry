#!/usr/bin/env bash
# Setup script for RunPod (or any fresh Ubuntu+CUDA instance).
# Usage: bash setup_runpod.sh
# Prereqs: git, CUDA driver, HF_TOKEN env var (for Mistral gated model)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/activation-steering-llm}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ── 1. System deps ─────────────────────────────────────────────────────────────
log "Installing system packages..."
apt-get update -qq && apt-get install -y -qq git git-lfs curl tmux nvtop jq > /dev/null
git lfs install --skip-repo > /dev/null

# ── 2. uv (fast pip replacement, used by LLM-Refusal-Evaluation submodule) ────
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# ── 3. Clone / update repo ────────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "Cloning repo to $REPO_DIR..."
    git clone --recurse-submodules https://github.com/ElSnacko/activation-steering-llm.git "$REPO_DIR"
else
    log "Repo already at $REPO_DIR — pulling latest..."
    git -C "$REPO_DIR" pull
    git -C "$REPO_DIR" submodule update --init --recursive
fi
cd "$REPO_DIR"

# ── 4. Python deps (main package) ─────────────────────────────────────────────
log "Installing activation-steering package..."
pip install -q -e ".[bayesian]"
pip install -q transformers>=5.3.0 accelerate datasets scipy matplotlib tqdm openai

# vLLM — needed for local model generation (not for judge, which uses DeepSeek API)
log "Installing vllm..."
pip install -q vllm==0.17.1

# ── 5. LLM-Refusal-Evaluation submodule venv ──────────────────────────────────
log "Setting up LLM-Refusal-Evaluation submodule venv..."
cd "$REPO_DIR/LLM-Refusal-Evaluation"
uv sync
cd "$REPO_DIR"

# ── 6. Capability probe set ────────────────────────────────────────────────────
if [[ ! -f data/capability_questions.json ]]; then
    log "Building MMLU capability probe set..."
    python scripts/build_capability_probe.py
fi

# ── 7. Download models from HuggingFace ───────────────────────────────────────
mkdir -p "$MODELS_DIR"
pip install -q huggingface_hub[cli]

download_model() {
    local model_id="$1"
    local dest="$MODELS_DIR/$model_id"
    if [[ -d "$dest" && -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
        log "Model $model_id already at $dest — skipping."
    else
        log "Downloading $model_id → $dest ..."
        huggingface-cli download "$model_id" --local-dir "$dest"
    fi
}

download_model "Qwen/Qwen3.5-9B"
download_model "mistralai/Mistral-7B-Instruct-v0.2"  # requires HF_TOKEN if gated

# ── 8. Environment variables ──────────────────────────────────────────────────
cat >> ~/.bashrc << 'EOF'

# activation-steering-llm
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,max_split_size_mb:512"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
EOF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,max_split_size_mb:512"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ── 9. Verify DEEPSEEK_API_KEY ────────────────────────────────────────────────
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    warn "DEEPSEEK_API_KEY is not set — judge scoring will fail."
    warn "Set it before running: export DEEPSEEK_API_KEY=sk-..."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
log ""
log "Setup complete. To run the full pipeline:"
log ""
log "  cd $REPO_DIR"
log "  export DEEPSEEK_API_KEY=sk-..."
log "  export QWEN_MODEL=$MODELS_DIR/Qwen/Qwen3.5-9B"
log "  export MISTRAL_MODEL=$MODELS_DIR/mistralai/Mistral-7B-Instruct-v0.2"
log "  bash run_pipeline.sh 2>&1 | tee outputs/pipeline.log"
log ""
log "Baseline judge scores are in baselines/ (committed in the repo)."
