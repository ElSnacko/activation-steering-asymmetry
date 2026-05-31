# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Activation steering toolkit for LLMs — extracts activations, computes steering vectors (MD/RMD/WRMD), and modifies refusal behavior at inference time. Implements methodology from [arXiv:2512.16602](https://arxiv.org/abs/2512.16602).

## Build & Development Commands

```bash
# Install (editable mode)
pip install -e .
pip install -e ".[dev]"    # includes pytest, black, isort, mypy

# Formatting
black --line-length 100 --target-version py311 src/ scripts/
isort --profile black --line-length 100 src/ scripts/

# Type checking
mypy src/

# Tests (no test suite yet — validation is done via inference pipeline)
```

## Architecture

**Package** (`src/activation_steering/`): Core library (extraction, computation, steering,
routing, capability, KL, dynamic-layer, merge, analysis, utils).
**Scripts** (`scripts/`): CLI entry points — the sequential pipeline plus the analysis
scripts behind the result docs.
**Data** (`data/`): Built-in evaluation data (capability questions).
**Submodule** (`LLM-Refusal-Evaluation/`): External judge scoring system (has its own venv via `uv`).

### Pipeline Flow

Each script consumes the previous step's output files:

1. `LLM-Refusal-Evaluation` → judge scores (baseline evaluation)
2. `scripts/extract_activations.py` → `activations_*.pt` (labeled hidden states)
3. `scripts/compute_wrmd.py` → `steering_vectors_*.pt` (MD default, WRMD for large datasets)
4. `scripts/find_best_layers.py` → `layer_correlations.json` (effective layers)
5. `scripts/optimize_alpha.py` → `optimization_summary.json` (optimal steering strength, optional `--capability-eval`)
6. `scripts/eval_capability.py` → capability preservation check (MMLU-style MCQ, no judge needed)
7. `scripts/test_steering.py` → interactive testing with steered model
8. `scripts/merge_steering.py` → permanent weight modification (optional)

### Core Mechanics

- **Steering equation**: `h' = h + α · v` (alpha typically -5.0 to 0.0; negative reduces refusal)
- **Dynamic steering**: PyTorch forward hooks intercept activations at selected layers — weights untouched
- **Static steering**: Merges `α · v` into bias terms permanently (MLP `down_proj`, attn `o_proj`, or both)
- **Judge scores**: >0.1 = refusal, <-0.1 = compliant, [-0.1, 0.1] = uncertain (filtered out)

### Key Classes

| Class | Module | Role |
|-------|--------|------|
| `ActivationExtractor` | extraction.py | Extract hidden states via forward hooks |
| `WRMDCalculator` | computation.py | Compute steering vectors (MD/RMD/WRMD) |
| `SteeringHook` | steering.py | Runtime activation modification (single component) |
| `SteeringHookGroup` | steering.py | Dual-component (attn+mlp) steering orchestration |
| `DynamicSteeringLayer` | dynamic_layer.py | Custom layer wrapper for dynamic steering |
| `DynamicSteeringSubmodule` | dynamic_layer.py | Custom submodule wrapper (attn/mlp) for dynamic steering |
| `CapabilityResult` | capability.py | MCQ-based capability preservation eval |

### Output Structure

All outputs go to `outputs/{model_name}/{run_id}/{step}/`. Model name is auto-extracted (e.g., `Qwen/Qwen3.5-9B` → `qwen3.5-9b`). Run ID is a timestamp (`YYYYMMDD-HHMMSS`). Override with `--output-dir` and `--run-id`.

## Code Style

- Black formatter, line length 100, target Python 3.11
- isort with black profile
- Package source in `src/` layout

## Important Notes

- Python >=3.11 required
- All model loading uses `torch_dtype=torch.bfloat16` and `device_map="auto"`
- Qwen3.5 MoE models need nightly vllm + transformers from source
- GPUs <80GB VRAM may need `enforce_eager=True`
- The `LLM-Refusal-Evaluation/` submodule has its own dependencies managed via `uv`

## Commit Conventions

- Do NOT include any reference to Claude Code in commit messages (no "Co-Authored-By" or similar attribution)
