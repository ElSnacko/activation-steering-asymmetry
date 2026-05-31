# Activation Steering

A Python toolkit for extracting activations, computing steering vectors, and analyzing effective layers for modifying LLM refusal behavior. Implements methodology from [arXiv:2512.16602](https://arxiv.org/abs/2512.16602).

Unlike simple "abliteration" approaches that treat refusal as a single monolithic behavior, this toolkit provides:

- **Domain-aware steering** — per-category vectors that target specific harm types (violence, privacy, etc.) independently
- **Dual-component extraction** — extracts both attention and MLP activations, with per-category component selection (MLP wins 10/13 categories for Qwen3.5-9B)
- **Bootstrap stability analysis** — quantifies whether your steering vectors are reliable or artifacts of small sample sizes
- **Bootstrap convergence analysis** — fits learning curves to predict how many samples each category needs, and whether instability is statistical (fixable with more data) or geometric (intrinsic to the model)
- **Activation-based routing** — classifies prompts using the steering vectors themselves (no external classifier needed), with residual projection and z-score normalization for accurate multi-class routing
- **Principled degradation** — stable categories get category-specific steering, unstable categories fall back to global steering, benign prompts get no steering

## Findings (Qwen3.5-9B & Mistral-7B, BeaverTails)

Across both models, MLP activation steering toward compliance does **not** unlock harmful
content at coherence-preserving strength: after FP-correction, **Qwen3.5-9B produces 0
genuinely harmful complies** across all 12 categories and **Mistral-7B produces ~2 total** —
even with per-category vectors, multi-layer steering, and the SiLU-gated dynamic/momentum
modes. The ceiling is structural, and the project pins down two reasons:

- **The refusal axis is layer-local.** The *difference* vector between refused and complied
  activations rotates ~90° between consecutive layers (Qwen 93.4°, Mistral 88.8°; tight
  bootstrap CIs), even though raw activations barely change. A perturbation injected at one
  layer is near-orthogonal to the next layer's refusal axis, so single-vector steering can't
  saturate a representation that reorients at every depth.
- **Steering is asymmetric.** Pushing toward *more* refusal is cheap (KL below the random
  floor) and effective; pushing toward compliance is non-functional at safe alpha and
  degenerates output before it complies. Refusal is an attractor that is cheap to deepen and
  expensive to escape.

Full write-ups, numbers, and a reproduced-verification report:

- [`docs/results_qwen3_5_9b_beavertails.md`](docs/results_qwen3_5_9b_beavertails.md)
- [`docs/results_mistral_7b_beavertails.md`](docs/results_mistral_7b_beavertails.md)
- [`docs/verification_pod_report.md`](docs/verification_pod_report.md) — 12/12 claims reproduced

> **Reproduce it:** baseline judge scores are committed under `baselines/`, so you can start
> mid-pipeline. The end-to-end run used in the write-ups is `bash run_pipeline.sh` (see
> `setup_runpod.sh` for environment setup).

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Why MD is Default](#why-md-is-default)
- [Inner Workings](#inner-workings)
- [Dependencies](#dependencies)

For detailed CLI arguments, per-step walkthroughs, file format specs, library API, and troubleshooting, see the **[full usage guide](docs/USAGE.md)**.

---

## Quick Start

Get started in 5 steps:

### 1. Clone and Setup

```bash
git clone --recurse-submodules https://github.com/ElSnacko/activation-steering-llm
cd activation-steering-llm
pip install -e .
pip install uv
```

### 2. Generate Baseline Evaluation

```bash
cd LLM-Refusal-Evaluation
uv run python -m src.compute_refusal_score --config configs/your_model.yaml
cd ..
```

### 3. Extract Activations

```bash
python scripts/extract_activations.py \
    --model Qwen/Qwen3.5-9B \
    --results-dir LLM-Refusal-Evaluation/results/baseline \
    --components attn+mlp
```

The `--components attn+mlp` flag extracts both attention and MLP activations, enabling per-category component selection (recommended). Use `--components attn` or `--components mlp` for single-component extraction.

### 4. Compute Steering Vectors & Find Best Layers

```bash
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_*.pt \
    --component mlp

python scripts/find_best_layers.py \
    --activations outputs/.../activations_*.pt \
    --steering-vectors outputs/.../steering_vectors_md_mlp.pt \
    --component mlp \
    --top-k 5
```

### 5. Optimize Alpha & Test

```bash
python scripts/optimize_alpha.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors outputs/.../steering_vectors_md.pt \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/baseline \
    --top-k 3 --num-prompts 100

python scripts/test_steering.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors outputs/.../steering_vectors_md.pt \
    --correlations outputs/.../layer_correlations.json \
    --top-k 3 --alpha -2.5
```

For permanent steering, category-specific vectors, `--stable-categories` multi-category optimization, capability evaluation, GGUF export, and all CLI options, see the **[full usage guide](docs/USAGE.md)**.

---

## Installation

### Requirements

- Python >= 3.11
- PyTorch >= 2.0.0 with CUDA support
- Transformers >= 4.30.0
- 16GB+ GPU memory (for 9B models)

### Install

```bash
git clone --recurse-submodules https://github.com/ElSnacko/activation-steering-llm.git
cd activation-steering-llm

# If already cloned without submodules
git submodule update --init --recursive

# Install
pip install -e .
pip install -e ".[bayesian]"  # adds Optuna for Bayesian alpha optimization
pip install uv                # for LLM-Refusal-Evaluation submodule
```

---

## Project Structure

```
activation_steering/
├── src/activation_steering/      # Main package
│   ├── extraction.py             # ActivationExtractor, cross-arch attn discovery
│   ├── computation.py            # WRMDCalculator, multi-rank decomposition
│   ├── analysis.py               # Layer correlation analysis
│   ├── steering.py               # SteeringHook + SteeringHookGroup for runtime steering
│   ├── routing.py                # CategoryRouter, residual vector routing
│   ├── capability.py             # Capability preservation (MCQ + perplexity)
│   ├── kl_divergence.py          # KL divergence measurement
│   ├── dynamic_layer.py          # DynamicSteeringLayer + DynamicSteeringSubmodule
│   ├── merge_steering_into_weights.py  # Permanent weight merging
│   └── utils.py                  # Output utilities
├── scripts/                      # CLI entry points (pipeline + analysis)
│   ├── extract_activations.py    # Step 1: Extract activations (attn, mlp, or both)
│   ├── compute_wrmd.py           # Step 2: Compute steering vectors (per-component)
│   ├── find_best_layers.py       # Step 3: Find effective layers (per-category)
│   ├── optimize_alpha.py         # Step 4: Find optimal alpha (Bayesian)
│   ├── eval_capability.py        # Step 4b: Capability preservation check
│   ├── test_steering.py          # Step 5: Test steering interactively
│   ├── calibrate_router.py       # Step 6: Category router calibration (optional)
│   ├── merge_steering.py         # Optional: Merge into weights
│   ├── export_to_gguf.py         # Optional: GGUF export
│   ├── analyze_axis_rotation.py  # Analysis: cross-layer rotation (+ bootstrap_axis_rotation.py)
│   ├── analyze_causal_propagation.py  # Analysis: perturbation propagation across layers
│   ├── random_direction_control.py    # Analysis: geometric-privilege control
│   ├── fp_correction_qa.py       # Analysis: false-positive correction of judge comply
│   └── hedge_*_mistral.py        # Analysis: hedge-regime geometry (Mistral)
├── data/                         # Capability-probe evaluation data
├── baselines/                    # Committed judge scores (start the pipeline mid-stream)
├── results/                      # Computed steering vectors & analysis JSONs
├── docs/                         # Usage guide, methodology, results write-ups
├── run_pipeline.sh               # End-to-end reproduction of the published experiment
├── LLM-Refusal-Evaluation/       # External submodule (judge scoring)
└── outputs/                      # Run artifacts (auto-created; verification JSONs committed)
```

### Output Directory Structure

```
outputs/
└── {model_name}/
    └── {run_id}/
        ├── extract_activations/
        │   └── activations_*.pt
        ├── compute_wrmd/
        │   ├── steering_vectors_*.pt
        │   ├── bootstrap_stability.json
        │   ├── bootstrap_convergence.json
        │   └── ...
        ├── find_best_layers/
        │   └── layer_correlations.json
        ├── optimize_alpha/
        │   └── optimization_summary.json
        ├── calibrate_router/
        │   ├── calibration.json
        │   └── router_roc.png
        └── eval_capability/
            └── capability_comparison.json
```

---

## Pipeline Overview

```
1.  LLM-Refusal-Evaluation           → judge scores (baseline)
2.  extract_activations.py           → activations_*.pt  (attn+mlp dual-component)
3.  compute_wrmd.py                  → steering_vectors_*.pt  (MD, per-component)
4.  find_best_layers.py              → layer_correlations.json (per-category)
5.  optimize_alpha.py                → optimization_summary.json (Bayesian)
6.  eval_capability.py               → capability check (MCQ + perplexity + KL)
7.  calibrate_router.py              → calibration.json (category routing, optional)
8.  test_steering.py                 → interactive testing (dynamic or fixed-alpha)
9.  merge_steering.py                → permanent model (optional)
```

Each stage consumes outputs from previous stages. See the **[full usage guide](docs/USAGE.md)** for all options and detailed walkthroughs.

---

## Why MD is Default

The default steering vector method is **MD** (Mean Difference), not WRMD (Weighted Ridge Mean Difference). While WRMD is theoretically superior — it accounts for covariance structure and is equivalent to Fisher's Linear Discriminant under Gaussian assumptions — it is less stable than MD at typical dataset sizes.

**The problem is covariance inversion.** WRMD computes `v = (C + lambda*I)^(-1) * (mean_refusal - mean_compliant)`, where `C` is a `[hidden_size, hidden_size]` covariance matrix estimated from compliant samples. For a model with `hidden_size=3584`, this matrix has ~6.4 million parameters, but you typically have only 200-800 compliant samples — far fewer than the ~3584 needed for a well-conditioned estimate. The ridge term prevents numerical blowup, but doesn't prevent the inversion from amplifying noise in the ~3000 undersampled dimensions.

**Bootstrap stability confirms this.** With typical BeaverTails-sized datasets (~50 samples per category, ~400 total refusal samples), WRMD bootstrap spreads are consistently 20-40% higher than MD. Per-category WRMD vectors in the "moderate" to "unreliable" range (5-15 deg+) often fall to "stable" (<5 deg) with MD.

**When to use WRMD instead:** WRMD becomes worthwhile when you have roughly as many samples as dimensions (`N ~ hidden_size`), so ~3500+ for a 3584-dimensional model. You can also try increasing `--lambda-ridge` (e.g., 1.0 or 10.0) to shrink WRMD toward MD. Use `--bootstrap-stability --compare-methods` to empirically compare stability at your dataset size.

**RMD** (Ridge Mean Difference without weighting) has the same covariance inversion issue as WRMD. MD avoids it entirely: `v = mean(refusal) - mean(compliant)`, with no matrix inversion, no eigenvalue sensitivity, and proportional response to input perturbations.

---

## Inner Workings

### Dynamic Steering

This toolkit uses **dynamic steering** to modify model behavior at inference time without changing model weights. During generation, PyTorch forward hooks intercept activations at selected transformer layers and shift them along precomputed steering vectors:

**Rank-1 (single direction):**
```
h' = h + alpha * v
```

**Multi-rank (multiple directions):**
```
h' = h + alpha_1 * v_1 + alpha_2 * v_2 + ... + alpha_k * v_k
```

Where `h` is the original hidden state, `v` are raw (unnormalized) steering vectors, and `alpha` values control per-direction steering strength. Negative alpha reduces refusal; positive alpha increases it. The perturbation magnitude is `|alpha| * ||v||`, so alpha scales the vector's natural magnitude — no unit normalization is applied.

With rank-1, a single alpha controls the refusal/compliance axis. With multi-rank, each direction captures a different aspect of refusal behavior — v_1 is the primary refusal axis, v_2 might distinguish hard refusal from safety hedging — and each gets its own independently tunable alpha.

The hooks are registered before generation and removed after — the model weights remain untouched throughout. This makes dynamic steering fully reversible and allows rapid experimentation with different layers, alpha values, and steering vectors without reloading the model.

**Component targeting:** Hooks can be attached to specific transformer submodules — `attn` (self-attention output), `mlp` (MLP output), or `layer` (full layer output). For Qwen3.5-9B, MLP produces higher refusal correlations at 10/13 harm categories compared to attention. The attention submodule name is auto-discovered across architectures (`self_attn` for Llama/Qwen2, `linear_attn` for Qwen3.5 GatedDeltaNet, etc.).

**Multi-layer steering:** The toolkit supports distributing perturbation across multiple layers with per-layer alpha scaling. The recommended approach is `alpha = -perturbation_target / vector_norm` for each layer, which equalizes the perturbation magnitude regardless of the vector's natural scale. This is critical because vector norms vary 20x across layers (L16: ~1.5, L30: ~20) — without normalization, deep layers dominate and cause immediate degeneration.

**Dual-component steering** (`attn+mlp`): `SteeringHookGroup` manages independent hooks on both the attention and MLP submodules simultaneously, allowing per-component alpha values.

For deployment scenarios where runtime hooks are impractical, the toolkit supports **static steering** by permanently merging `alpha * v` into bias terms — `down_proj.bias` for MLP, `o_proj.bias` for attention, or both for dual-component. See the **[full usage guide](docs/USAGE.md#5-merge-steering-into-model-weights)** for details.

### Architecture & Data Flow

The project uses judge scores from LLM-Refusal-Evaluation to train steering vectors:

1. **LLM-Refusal-Evaluation/** generates judge scores for model responses
2. **extract_activations.py** uses judge scores to label activations as refusal/compliant
3. **compute_wrmd.py** computes steering vectors weighted by judge confidence scores (rank-1 or multi-rank via PCA)
4. **find_best_layers.py** correlates activation projections with judge scores to identify effective steering layers
5. **optimize_alpha.py** finds optimal alpha(s) via grid search or Bayesian optimization (Optuna)
6. **eval_capability.py** verifies steering doesn't degrade model capability (MCQ, perplexity, KL divergence)
7. **validate_category_steering.py** runs per-category Bayesian alpha optimization and compares against global steering
8. **test_steering.py** dynamically steers the model at runtime via forward hooks with optimal alpha(s)

### Steering Vector Methods

**Mean Difference (MD):** `v = mean(refusal) - mean(compliant)`

**Ridge Mean Difference (RMD):** `v = (C + lambda*I)^(-1) * (mean_refusal - mean_compliant)`

**Weighted Ridge Mean Difference (WRMD):** `v = (C_weighted + lambda*I)^(-1) * (weighted_mean_refusal - weighted_mean_compliant)`

WRMD is theoretically strongest as it leverages judge confidence scores to weight examples by certainty, but see [Why MD is Default](#why-md-is-default) for when to prefer MD.

### Judge Score Interpretation

Judge scores range from -1 to 1:
- **score > 0.1**: Model refused (higher = stronger refusal)
- **score < -0.1**: Model complied (more negative = stronger compliance)
- **-0.1 <= score <= 0.1**: Uncertain/mixed behavior

Scores are used to:
1. **Filter samples** during activation extraction (only clear behavior)
2. **Weight samples** during WRMD computation (higher confidence = higher weight)
3. **Evaluate steering** during alpha optimization (measure effectiveness)

---

## Dependencies

### Core
- PyTorch >= 2.0.0 (with bfloat16 support)
- Transformers >= 4.30.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- tqdm >= 4.65.0

### Optional
- vLLM (for LLM-Refusal-Evaluation batch inference)
- `datasets` (for MMLU/ARC benchmarks: `pip install datasets`)
- `optuna` (for Bayesian alpha optimization: `pip install -e ".[bayesian]"`)
- `bitsandbytes` (for 4-bit/8-bit quantization and FP8 checkpoint support: `pip install bitsandbytes`)

---

## License

MIT License

---

## Related Projects

- [LLM-Refusal-Evaluation](https://github.com/CompactifAI/LLM-Refusal-Evaluation): Inference-time evaluation framework for measuring refusal behavior

---

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

---

## Contact

For questions or issues, open an issue on GitHub.
