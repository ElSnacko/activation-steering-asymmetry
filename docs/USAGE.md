# Usage Guide

Full reference for all CLI scripts, arguments, file formats, and library API. For an overview of the project and how it works, see the [README](../README.md).

## Table of Contents

- [Pipeline Steps](#pipeline-steps)
  - [1. Extract Activations](#1-extract-activations-from-model)
  - [2. Compute Steering Vectors](#2-compute-steering-vectors)
  - [3. Find Best Layers](#3-find-best-layers-for-steering)
  - [4. Test Steering](#4-test-steering-interactively)
  - [4a. Optimize Alpha](#4a-optimize-alpha-parameter)
  - [4b. Evaluate Capability](#4b-evaluate-capability-preservation)
  - [5. Merge Into Weights](#5-merge-steering-into-model-weights)
  - [6. Export to GGUF](#6-export-to-gguf-format)
  - [7. Generate Judge Scores](#7-run-llm-refusal-evaluation)
- [Category-Specific Steering](#category-specific-steering)
  - [How It Works](#how-it-works)
  - [Workflow](#workflow)
  - [Angular Distance Analysis](#angular-distance-analysis)
  - [Output Structure](#output-structure)
  - [Supported Datasets](#supported-datasets)
  - [Library Usage](#library-usage)
  - [Validating Category-Specific Vectors](#validating-category-specific-vectors)
  - [Edge Cases](#edge-cases)
  - [Category Router](#category-router)
- [Using as a Library](#using-as-a-library)
- [File Format Details](#file-format-details)
- [Implementation Details](#implementation-details)
- [Troubleshooting](#troubleshooting)

---

## Pipeline Steps

### 1. Extract Activations from Model

Extract internal activations labeled by judge scores:

```bash
python scripts/extract_activations.py \
    --model Qwen/Qwen3.5-9B \
    --results-dir LLM-Refusal-Evaluation/results/qwen3.5-9b_baseline \
    --output activations_qwen3.5-9b_judged.pt \
    --refusal-threshold 0.1 \
    --compliance-threshold -0.1

# With category metadata for per-category steering vectors
python scripts/extract_activations.py \
    --model Qwen/Qwen3.5-9B \
    --results-dir LLM-Refusal-Evaluation/results/beavertails \
    --dataset PKU-Alignment/BeaverTails-Evaluation \
    --output activations_beavertails.pt
```

**What it does:**
- Loads judge scores from LLM-Refusal-Evaluation results
- Extracts model activations at the last token position for all layers
- Filters samples: only includes clear refusal (score > 0.1) or compliance (score < -0.1)
- Saves metadata with judge scores for downstream use
- When `--dataset` is provided, cross-references prompts with the HuggingFace dataset to inject category labels into metadata (for per-category steering)

**Arguments:**
- `--model`: HuggingFace model name or path
- `--results-dir`: Path to LLM-Refusal-Evaluation baseline results
- `--components`: Which submodule activations to extract: `attn`, `mlp`, `layer`, or multiple (default: `attn`). Cross-architecture: auto-discovers attention submodule name (`self_attn`, `linear_attn`, etc.)
- `--refusal-threshold`: Minimum score to label as refusal (default: 0.1)
- `--compliance-threshold`: Maximum score to label as compliance (default: -0.1)
- `--dataset`: HuggingFace dataset for category metadata (e.g., `PKU-Alignment/BeaverTails-Evaluation`)
- `--prompt-column`: Prompt column name in HF dataset (default: `prompt`)
- `--category-column`: Category column name in HF dataset (default: `category`)
- `--dataset-split`: Dataset split to use (default: auto-detect)
- `--dataset-categories`: Only include prompts from these categories
- `--output-dir`: Custom output directory (optional)
- `--run-id`: Custom run ID (optional)
- `--max-samples`: Limit samples for testing (optional)

---

### 2. Compute Steering Vectors

Compute steering vectors (MD is the default — see [Why MD is Default](../README.md#why-md-is-default)):

```bash
# Single method (MD default — stable, no covariance inversion)
python scripts/compute_wrmd.py \
    --activations outputs/qwen3.5-9b/20231227-035148/extract_activations/activations_qwen3.5-9b_judged.pt

# Use WRMD if you have thousands of samples
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_*.pt \
    --method wrmd --lambda-ridge 0.1

# Multi-rank: compute additional steering directions via PCA of residuals
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_qwen3.5-9b_judged.pt \
    --rank 2

# Category-specific vectors (requires --dataset at extraction time)
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_beavertails.pt \
    --category "Violence, Aiding and Abetting, Incitement"

# All categories at once (produces per-category + global vector files)
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_beavertails.pt \
    --all-categories

# List available categories in an activation file
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_beavertails.pt \
    --list-categories

# Compare all methods (MD, RMD, WRMD)
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_qwen3.5-9b_judged.pt \
    --compare-methods
```

**Methods:**
- **MD (Mean Difference)**: Simple difference of means
- **RMD (Ridge Mean Difference)**: Adds ridge regularization
- **WRMD (Weighted Ridge Mean Difference)**: Weights samples by judge confidence scores (recommended)

**Multi-rank decomposition** (`--rank k`): The primary steering vector (v_1) captures the mean refusal/compliance axis. With `--rank 2` or higher, additional directions are computed via PCA on refusal activations after projecting out v_1. These capture structured variance in *how* the model refuses (e.g., hard refusal vs. safety hedging), enabling independent control via per-direction alpha values. Practical limit is rank 2-3 for typical dataset sizes.

**Category-specific steering** (`--category` / `--all-categories`): When activations include category metadata (extracted with `--dataset`), you can compute steering vectors using only refusal samples from specific harm categories. The compliant baseline always uses all compliant samples (category-agnostic) — only the refusal cohort is filtered. This captures category-specific refusal directions (e.g., violence vs. privacy violation may activate different internal circuits). See [Category-Specific Steering](#category-specific-steering) for details.

**Rank association analysis** (`--analyze-ranks`): After multi-rank decomposition, projects each prompt's activations onto each PCA component to determine which rank dominates. This reveals what each direction captures semantically — for example, v_1 might capture the primary refusal axis while v_2 captures a specific refusal style. Reports per-prompt associations, refusal/compliant breakdown, and example prompts per rank. Saves `rank_associations.json` and a visualization to the output directory. Use `--rank-layers` to limit analysis to specific layers (e.g., your best steering layers).

**Arguments:**
- `--activations`: Path to activations .pt file
- `--method`: Choose 'md', 'rmd', or 'wrmd' (default: md)
- `--component`: Which component's activations to use: `attn`, `mlp`, or `layer` (default: `attn`)
- `--lambda-ridge`: Ridge regularization parameter (default: 0.1)
- `--rank`: Number of steering directions per layer (default: 1)
- `--normalize`: Normalize vectors to unit length
- `--no-score-weighting`: Disable judge score weighting (uniform weights)
- `--category` / `--categories`: Compute vectors using only these refusal categories
- `--all-categories`: Compute separate vector file per category plus a global file
- `--list-categories`: Print available categories and exit
- `--compare-methods`: Generate comparison plots for all methods
- `--analyze-ranks`: Analyze which PCA rank each prompt is most associated with (requires `--rank > 1`)
- `--rank-layers`: Layer indices to use for rank analysis (default: all layers)
- `--analyze-categories`: Compute pairwise angular distances between per-category steering vectors
- `--analyze-intra-category`: Measure intra-category angular spread (coherence analysis)
- `--compare-vectors FILE [FILE ...]`: Compare pre-computed `.pt` vector files directly (standalone, no `--activations` needed)
- `--category-layers`: Layer indices for category angular distance analysis (default: all)
- `--min-category-samples`: Minimum refusal samples per category for analysis (default: 10)
- `--bootstrap-stability`: Compute bootstrap stability analysis for category steering vectors
- `--bootstrap-samples`: Number of bootstrap iterations (default: 20)
- `--bootstrap-ratio`: Fraction of refusal samples to resample per iteration (default: 0.8)
- `--bootstrap-convergence`: Run bootstrap convergence analysis (angular spread vs pool size learning curve)
- `--convergence-min-pool`: Minimum pool size for convergence analysis (default: 50)
- `--convergence-pool-step`: Pool size step for convergence analysis (default: 50)
- `--convergence-targets N [N ...]`: Target sample sizes for extrapolation (default: 500 1000)
- `--convergence-instability-ceiling`: Floor threshold in degrees above which a category is flagged as unstable (default: 30.0)
- `--output-dir`: Custom output directory (optional)
- `--run-id`: Custom run ID (optional)

---

### 3. Find Best Layers for Steering

Identify which layers are most effective:

```bash
python scripts/find_best_layers.py \
    --activations outputs/qwen3.5-9b/20231227-035148/extract_activations/activations_qwen3.5-9b_judged.pt \
    --steering-vectors outputs/qwen3.5-9b/20231227-035148/compute_wrmd/steering_vectors_wrmd.pt \
    --top-k 5
```

**What it does:**
- Computes Pearson correlation between activation projections and judge scores for each layer
- Higher absolute correlation = better steering effectiveness
- Generates visualizations showing layer effectiveness

**Arguments:**
- `--activations`: Path to activations .pt file
- `--steering-vectors`: Path to steering vectors .pt file
- `--component`: Which component to analyze: `attn`, `mlp`, or `layer` (default: `attn`)
- `--top-k`: Number of best layers to identify (default: 5)
- `--output-dir`: Custom output directory (optional)
- `--run-id`: Custom run ID (optional)

---

### 4. Test Steering Interactively

Apply dynamic steering and compare outputs:

```bash
# Test top-3 layers individually
python scripts/test_steering.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors steering_vectors_wrmd.pt \
    --correlations layer_correlations.json \
    --top-k 3 \
    --alpha -2.0 \
    --num-prompts 3

# Test specific layers together
python scripts/test_steering.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors steering_vectors_wrmd.pt \
    --layers 9 10 11 \
    --alpha -2.0
```

**Alpha parameter controls steering strength:**
- **Negative alpha** (e.g., -2.0): Reduce refusal behavior
- **Positive alpha** (e.g., +2.0): Increase refusal behavior

**Arguments:**
- `--model`: HuggingFace model name or path
- `--steering-vectors`: Path to steering vectors .pt file
- `--correlations`: Path to layer correlations JSON (for `--top-k`)
- `--results-dir`: Path to baseline evaluation results (for loading refusal prompts)
- `--layers`: Specific layers to test (alternative to `--top-k`)
- `--top-k`: Use top K layers from correlations
- `--alpha`: Steering strength (negative=reduce refusal, default: -2.0)
- `--component`: Which submodule to steer: `attn`, `mlp`, `layer`, or `attn+mlp` (default: `attn`)
- `--num-prompts`: Number of prompts to test (default: 3)
- `--min-refusal-score`: Only test prompts with baseline refusal score above this threshold (default: 0.5)
- `--max-tokens`: Max tokens to generate (default: 500)
- `--router`: Path to `calibration.json` for automatic category routing (see [Category Router](#category-router))

---

### 4a. Optimize Alpha Parameter

Automatically find the optimal alpha value using judge scores. Two optimization strategies are available:

#### Grid Search (default)

Linear sweep with early stopping — good for rank-1 vectors:

```bash
python scripts/optimize_alpha.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/qwen3.5-9b_baseline \
    --top-k 3 \
    --alpha-start -2.0 \
    --alpha-min -5.0 \
    --alpha-max 0.0 \
    --alpha-step 0.5 \
    --num-prompts 100

# Fine-grained search around specific alpha
python scripts/optimize_alpha.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors steering_vectors_wrmd.pt \
    --baseline-results LLM-Refusal-Evaluation/results/qwen3.5-9b_baseline \
    --layers 10 11 12 \
    --alpha-start -2.0 \
    --alpha-min -2.5 \
    --alpha-max -1.5 \
    --alpha-step 0.1 \
    --num-prompts 200
```

#### Bayesian Optimization (recommended for multi-rank)

Uses Optuna's TPE sampler for sample-efficient search — essential for multi-rank vectors where the parameter space is multi-dimensional:

```bash
# Bayesian optimization with rank-1 vectors
python scripts/optimize_alpha.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/qwen3.5-9b_baseline \
    --top-k 3 \
    --optimizer bayesian \
    --bayesian-trials 20 \
    --num-prompts 100

# Bayesian optimization with multi-rank vectors (optimizes per-rank alphas)
python scripts/optimize_alpha.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors outputs/.../steering_vectors_wrmd_rank2.pt \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/qwen3.5-9b_baseline \
    --top-k 3 \
    --optimizer bayesian \
    --bayesian-trials 30 \
    --num-prompts 100
```

Bayesian optimization requires `pip install optuna` (or `pip install -e ".[bayesian]"`). For multi-rank vectors, it optimizes one alpha per rank direction independently (e.g., alpha_1 for the primary refusal axis, alpha_2 for the secondary refusal-style axis).

**Key Features:**
- Loads baseline (alpha=0) metrics from existing evaluation results (no re-run needed)
- **Grid search**: Tests alphas starting from `--alpha-start`, sweeping outward with early stopping
- **Bayesian**: Uses Optuna TPE sampler for efficient exploration of the alpha space
- Optional KL divergence measurement on harmless prompts (`--kl-divergence`)
- Optional perplexity measurement (`--perplexity`)
- Optional capability evaluation (`--capability-eval`)
- Generates visualizations and summary with optimal alpha recommendation

**Early Stopping (grid search only):**
- Tests alphas in order: -2.0 -> -2.5 -> -3.0 -> ... (away from zero)
- Stops if performance degrades beyond tolerance from best prior result
- If far alphas fail, continues testing closer to zero
- Skips alpha=0.0 (already have baseline metrics)

**Outputs:**
- `optimization_summary.json`: Full results with optimal alpha(s)
- `alpha_optimization.png`: Multi-panel visualization (scores, rates, KL, perplexity, etc.)
- Per-alpha or per-trial response files

**Arguments:**

*Required:*
- `--model`: HuggingFace model name or path
- `--steering-vectors`: Path to steering vectors .pt file (required unless `--stable-categories` is used)
- `--baseline-results`: Path to baseline evaluation results

*Layer selection:*
- `--correlations`: Path to layer correlations JSON (for `--top-k`)
- `--layers`: Specific layers to test (alternative to `--top-k`)
- `--top-k`: Use top K layers from correlations (default: 3)

*Optimization strategy:*
- `--optimizer`: `grid` (default) or `bayesian`
- `--bayesian-trials`: Number of Optuna trials (default: 20)
- `--objective`: Optimization objective: `minimize_refusal`, `maximize_compliance_rate`, `balanced`, `kl_weighted` (default: `minimize_refusal`)
- `--resume`: Resume Bayesian optimization from trial checkpoint

*Alpha range:*
- `--alpha-start`: Starting alpha for grid search (default: -2.0)
- `--alpha-min`: Minimum alpha to test (default: -5.0)
- `--alpha-max`: Maximum alpha to test (default: 0.0)
- `--alpha-step`: Alpha step size (default: 0.5 for grid, continuous for Bayesian)

*Early stopping (grid only):*
- `--early-stopping` / `--no-early-stopping`: Toggle early stopping (default: enabled)
- `--stopping-metric`: Metric for comparison: `mean_score`, `compliance_rate`, `refusal_rate` (default: `mean_score`)
- `--stopping-tolerance`: Tolerance before stopping (default: 0.1)

*Test prompts:*
- `--num-prompts`: Number of test prompts (default: 50)
- `--splits`: Which splits to use from baseline results (default: `test`)

*KL divergence:*
- `--kl-divergence`: Measure KL divergence on harmless prompts at each alpha
- `--kl-method`: `teacher_forced` (multi-token, default) or `first_token` (fastest)
- `--kl-tokens`: Tokens for teacher-forced KL (default: 32)
- `--kl-prompts`: Path to custom harmless prompts JSON
- `--num-kl-prompts`: Number of KL prompts (default: 20)

*Perplexity:*
- `--perplexity`: Measure perplexity on a diverse text corpus at each alpha
- `--perplexity-corpus`: Path to custom text passages JSON

*Capability preservation:*
- `--capability-eval`: Run MCQ evaluation at each alpha
- `--capability-questions`: Path to custom questions JSON
- `--num-capability-questions`: Limit number of questions
- `--capability-threshold`: Max accuracy drop before flagging degradation (default: 0.05)

*Generation:*
- `--max-new-tokens`: Max tokens to generate (default: 2048)
- `--generate-timeout`: Timeout per prompt in seconds (default: 120)
- `--temperature`: Sampling temperature (default: 0.6)
- `--top-p`: Nucleus sampling threshold (default: 0.95)
- `--sample-top-k`: Top-k sampling (default: 20)
- `--thinking-string`: Delimiter for thinking/reasoning models (e.g., `</think>`)

*Judge:*
- `--judge-model`: Judge model for scoring (default: `unsloth/gpt-oss-20b`)
- `--enforce-eager`: Pass `enforce_eager=True` to vLLM judge
- `--judge-max-model-len`: Max sequence length for vLLM judge

*Steering:*
- `--component`: Which submodule to steer: `attn`, `mlp`, `layer`, or `attn+mlp` (default: `attn`)
- `--rank`: Override rank from multi-rank vectors (default: auto-detect)
- `--load-in-4bit`: Load model with BitsAndBytes 4-bit quantization (NF4)
- `--load-in-8bit`: Load model with BitsAndBytes 8-bit quantization (auto-triggered for FP8 checkpoints)

*Stable categories mode:*
- `--stable-categories DIR`: Directory with per-category .pt files + `category_summary.json` (from `compute_wrmd --all-categories`). Runs optimization for each qualifying category independently.
- `--bootstrap-stability PATH`: Path to `bootstrap_stability.json` for filtering by stability label
- `--bootstrap-convergence PATH`: Path to `bootstrap_convergence.json` for filtering by convergence action
- `--stability-filter LABEL [...]`: Stability labels to EXCLUDE (default: `unreliable`)
- `--convergence-filter ACTION [...]`: Convergence actions to EXCLUDE (default: `unstable`)
- `--min-category-samples N`: Minimum refusal prompts per category (default: 10)
- `--categories CAT [...]`: Explicit list of category names to include (overrides discovery)

*Output:*
- `--output-dir`: Custom output directory
- `--run-id`: Custom run ID

#### Stable Categories Mode

When you have per-category steering vectors (from `compute_wrmd --all-categories`) and want to optimize alpha for each category independently, use `--stable-categories` instead of `--steering-vectors`:

```bash
# Optimize all stable categories (grid search)
python scripts/optimize_alpha.py \
    --model Qwen/Qwen3.5-9B \
    --stable-categories outputs/.../compute_wrmd/ \
    --bootstrap-stability outputs/.../compute_wrmd/bootstrap_stability.json \
    --bootstrap-convergence outputs/.../compute_wrmd/bootstrap_convergence.json \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/baseline \
    --top-k 3 --num-prompts 50

# Bayesian optimization for all stable categories
python scripts/optimize_alpha.py \
    --model Qwen/Qwen3.5-9B \
    --stable-categories outputs/.../compute_wrmd/ \
    --bootstrap-stability outputs/.../compute_wrmd/bootstrap_stability.json \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/baseline \
    --top-k 3 --num-prompts 50 \
    --optimizer bayesian --bayesian-trials 20

# Only specific categories
python scripts/optimize_alpha.py \
    --model Qwen/Qwen3.5-9B \
    --stable-categories outputs/.../compute_wrmd/ \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/baseline \
    --categories violence financial_crime \
    --top-k 3 --num-prompts 50
```

**How filtering works:**

Categories must pass all active filters to be included:
- **Bootstrap stability** (`--bootstrap-stability`): Excludes categories with labels in `--stability-filter` (default: `unreliable`). Categories labeled `stable` or `moderate` pass.
- **Bootstrap convergence** (`--bootstrap-convergence`): Excludes categories with actions in `--convergence-filter` (default: `unstable`). Categories whose convergence floor >= 30 deg are excluded.
- **Minimum samples** (`--min-category-samples`): Skips categories with fewer refusal prompts.
- **Explicit list** (`--categories`): When set, only these categories are considered.

**Output structure:**

```
optimize_alpha/
    stable_categories_summary.json      # Combined summary across all categories
    category_checkpoint.json            # Resume checkpoint
    violence/
        optimization_summary.json       # Per-category results
        alpha_optimization.png          # Visualization (grid)
        alpha_neg2p00/                  # Per-alpha responses
            responses.json
    financial_crime/
        ...
```

Use `--resume` to resume an interrupted multi-category run from the last checkpoint.

---

### 4b. Evaluate Capability Preservation

Verify that steering doesn't degrade model coherence on non-sensitive tasks. Runs multiple-choice questions (MCQ) and compares accuracy before and after steering. No judge model required — scoring is deterministic.

```bash
# Quick smoke test with built-in questions (50 questions, 10 categories)
python scripts/eval_capability.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --correlations outputs/.../layer_correlations.json \
    --top-k 3 --alpha -2.5

# Use MMLU benchmark (requires: pip install datasets)
python scripts/eval_capability.py \
    --model Qwen/Qwen3.5-9B \
    --benchmark mmlu --max-questions 200 \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --correlations outputs/.../layer_correlations.json \
    --top-k 3 --alpha -2.5

# MMLU with specific subjects
python scripts/eval_capability.py \
    --model Qwen/Qwen3.5-9B \
    --benchmark mmlu \
    --subjects abstract_algebra anatomy computer_security \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --layers 10 11 12 --alpha -2.5

# ARC-Challenge benchmark
python scripts/eval_capability.py \
    --model Qwen/Qwen3.5-9B \
    --benchmark arc_challenge \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --correlations outputs/.../layer_correlations.json \
    --top-k 3 --alpha -2.5

# Baseline only (no steering, establish reference accuracy)
python scripts/eval_capability.py \
    --model Qwen/Qwen3.5-9B \
    --benchmark mmlu --max-questions 500

# List available MMLU subjects
python scripts/eval_capability.py \
    --model Qwen/Qwen3.5-9B \
    --benchmark mmlu --subjects ""
```

**What it does:**
- Runs MCQ questions through baseline model, then through steered model
- Parses answer letters from model output (no judge needed)
- Reports per-category accuracy and overall delta
- Flags degradation (>5% drop) or severe degradation (>15% drop)

**Question sources** (mutually exclusive):
- Built-in 50-question set (`data/capability_questions.json`) — quick smoke test, no dependencies
- `--benchmark mmlu` — Full MMLU (14K questions, 57 subjects) via HuggingFace `datasets`
- `--benchmark arc_easy` — ARC-Easy (2.4K questions)
- `--benchmark arc_challenge` — ARC-Challenge (1.2K questions)
- `--questions file.json` — Custom question file (same format as built-in)

**Integration with alpha optimization:**

Add `--capability-eval` to `optimize_alpha.py` to track capability at each alpha alongside refusal metrics:

```bash
python scripts/optimize_alpha.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/baseline \
    --top-k 3 --num-prompts 100 \
    --capability-eval
```

This adds capability accuracy to the optimization summary and visualization, so you can see where refusal reduction begins to trade off against coherence.

**Integration with main pipeline:**

```bash
python main.py \
    --model Qwen/Qwen3.5-9B \
    --results-dir LLM-Refusal-Evaluation/results/baseline \
    --capability-eval \
    --capability-benchmark mmlu \
    --capability-max-questions 200
```

**Arguments:**
- `--model`: HuggingFace model name or path
- `--steering-vectors`: Path to steering vectors .pt file (omit for baseline-only)
- `--correlations`: Path to layer correlations JSON (for `--top-k`)
- `--layers`: Specific layers to steer (alternative to `--top-k`)
- `--top-k`: Use top K layers from correlations (default: 3)
- `--alpha`: Steering strength (default: -2.0)
- `--benchmark`: HuggingFace benchmark (`mmlu`, `arc_easy`, `arc_challenge`)
- `--subjects`: MMLU subjects to include (default: all; pass empty string to list available)
- `--questions`: Path to custom questions JSON
- `--max-questions`: Limit number of questions
- `--max-new-tokens`: Max tokens per answer (default: 32)
- `--few-shot`: Number of few-shot examples (default: auto-detect — 5 for base models, 0 for instruct)
- `--perplexity`: Measure perplexity on a diverse text corpus
- `--perplexity-corpus`: Path to custom text passages JSON
- `--generation-kl`: Measure KL divergence on open-ended generation prompts
- `--generation-kl-prompts`: Path to custom generation prompts JSON
- `--generation-kl-tokens`: Tokens to generate per prompt for KL measurement (default: 64)
- `--output-dir`: Custom output directory
- `--run-id`: Custom run ID

---

### 5. Merge Steering Into Model Weights

Permanently merge steering vectors into model weights to create a standalone steered model:

```bash
# Merge using best layers from correlation analysis (default: MLP bias)
python scripts/merge_steering.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors steering_vectors_wrmd.pt \
    --correlations layer_correlations.json \
    --top-k 3 \
    --alpha -2.0 \
    --output-dir Qwen3.5-9B-Steered

# Merge into attention output projection bias (o_proj)
python scripts/merge_steering.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors steering_vectors_wrmd.pt \
    --correlations layer_correlations.json \
    --top-k 3 \
    --alpha -2.0 \
    --component attn \
    --output-dir Qwen3.5-9B-Steered

# Merge dual-component (attn+mlp) with per-component alphas
python scripts/merge_steering.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors steering_vectors_attn_mlp.pt \
    --correlations layer_correlations.json \
    --top-k 3 \
    --alpha-attn -3.0 \
    --alpha-mlp -0.5 \
    --component attn+mlp \
    --output-dir Qwen3.5-9B-Steered

# Merge and export to GGUF format in one step
python scripts/merge_steering.py \
    --model Qwen/Qwen3.5-9B \
    --steering-vectors steering_vectors_wrmd.pt \
    --correlations layer_correlations.json \
    --top-k 3 \
    --alpha -2.0 \
    --output-dir Qwen3.5-9B-Steered \
    --export-gguf \
    --gguf-quantization q4_0

# Verify the merged model
python scripts/merge_steering.py \
    --verify \
    --merged-model Qwen3.5-9B-Steered \
    --original-model Qwen/Qwen3.5-9B \
    --layers 10 11 12
```

**What it does:**
- Merges steering vectors into model bias terms as a permanent offset
- **MLP component** (default): Merges `alpha * v` into `down_proj.bias` (MLP output projection)
- **Attn component**: Merges `alpha * v` into `o_proj.bias` (attention output projection)
- **Attn+MLP component**: Merges into both `o_proj.bias` and `down_proj.bias` with independent alphas
- Creates a standalone model with built-in steering behavior — no runtime hooks required
- Adds zero biases to all non-steered layers and patches `config.json` for framework compatibility
- Saves a sidecar `steering_biases.safetensors` as fallback for frameworks that don't support all bias flags
- Optionally exports to GGUF format for llama.cpp

**Framework compatibility:**
- **Attention-merged models** load natively in HuggingFace and vLLM — the config is patched with `attention_bias: true`
- **MLP-merged models** require `load_merged_model()` — the `mlp_bias` config flag is not yet recognized by HF/vLLM, so MLP biases are injected from the sidecar file
- **Dual-component models**: attention loads natively, MLP uses sidecar fallback

**WARNING:** Static merging permanently modifies the model weights — unlike dynamic steering (which uses runtime hooks and leaves weights untouched), merged changes cannot be reversed. Always merge from the original base model, not an already-merged copy.

**Arguments:**
- `--model`: HuggingFace model name or path
- `--steering-vectors`: Path to steering vectors .pt file
- `--correlations`: Path to layer correlations JSON (for --top-k)
- `--layers`: Specific layers to merge (alternative to --top-k)
- `--top-k`: Merge top K layers from correlations
- `--alpha`: Steering strength to merge (shared across components)
- `--alpha-attn`: Override alpha for attention component (dual-component mode)
- `--alpha-mlp`: Override alpha for MLP component (dual-component mode)
- `--component`: Target component: `mlp` (default), `attn`, or `attn+mlp`
- `--output-dir`: Directory to save merged model
- `--export-gguf`: Also export to GGUF format
- `--gguf-quantization`: Quantization type (f16, q4_0, q8_0, etc.)
- `--verify`: Verify merged model against original
- `--merged-model`: Path to merged model (for verification)
- `--original-model`: Path to original model (for verification)

---

### 6. Export to GGUF Format

Export any HuggingFace model (including merged models) to GGUF format for llama.cpp:

```bash
# Export with default FP16 quantization
python scripts/export_to_gguf.py \
    --model-dir Qwen3.5-9B-Steered \
    --quantization f16

# Export with 4-bit quantization for smaller size
python scripts/export_to_gguf.py \
    --model-dir Qwen3.5-9B-Steered \
    --quantization q4_0 \
    --output model-q4_0.gguf
```

**Quantization Options:**
- `f32`: Full 32-bit precision (largest, highest quality)
- `f16`: Half precision (recommended default)
- `q8_0`: 8-bit quantization (good quality, smaller)
- `q5_0/q5_1`: 5-bit quantization (balanced)
- `q4_0/q4_1`: 4-bit quantization (smallest, lowest quality)

**Requirements:**
- llama.cpp with `convert-hf-to-gguf.py` script available in PATH or common locations
- Or install: `pip install llama-cpp-python`

**Arguments:**
- `--model-dir`: Directory containing HuggingFace model
- `--output`: Output path for GGUF file (optional)
- `--quantization`: Quantization type (default: f16)
- `--verbose`: Print detailed conversion output

---

### 7. Run LLM-Refusal-Evaluation

Generate baseline judge scores needed for activation extraction:

```bash
cd LLM-Refusal-Evaluation
uv run python -m src.compute_refusal_score --config configs/Qwen3-4B-Instruct-2507.yaml
```

Or with conda (e.g., `activation-steer` env):

```bash
cd LLM-Refusal-Evaluation
PYTHONPATH=. python src/compute_refusal_score.py --config configs/my-model.yaml
```

**Custom datasets with category support:**

```bash
# BeaverTails with auto-detected categories, balanced sampling, truncated generation
PYTHONPATH=. python src/compute_refusal_score.py \
  --config configs/my-model.yaml \
  --custom-dataset PKU-Alignment/BeaverTails-Evaluation \
  --dataset-split test \
  --samples-per-category 20 \
  --max-new-tokens 512 \
  --seed 42
```

When `category_column` is set to `"auto"` (or a known dataset adapter applies it), the pipeline auto-detects boolean category columns and propagates category labels through the entire output. The parent project's `extract_activations.py` reads these categories directly from `censor_scores.json`, eliminating the need to re-load the HuggingFace dataset for category injection.

**Combining multiple datasets in one config:**

```yaml
dataset_splits:
  - name: "beavertails"
    dataset_id: "PKU-Alignment/BeaverTails-Evaluation"
    split: "test"
    category_column: "auto"
  - name: "general_prompts"
    dataset_id: "Iker/refusal-evaluation"
    split: "general_prompts"
```

**Merging results from multiple runs:**

```bash
cd LLM-Refusal-Evaluation
PYTHONPATH=. python merge_results.py \
  --input-dirs results/run1 results/run2 \
  --output-dir results/merged
```

**Output fields:** Each entry in `censor_scores.json` now includes `category`, `source_dataset`, `source_split`, `source_row_index`, `prompt_hash`, `classification_method`, and (for compliant samples) `compliance_quality`.

See `LLM-Refusal-Evaluation/README.md` for full configuration and CLI reference.

---

## Category-Specific Steering

Models refuse different harm categories using different internal circuits — violence, privacy violations, and child abuse may each activate distinct refusal pathways. Category-specific steering computes separate steering vectors per harm category, enabling fine-grained control over which types of refusal behavior are modified.

### How It Works

The standard pipeline treats "refusal" as a single monolithic class. Category-specific steering splits the refusal cohort by harm category:

1. **Extract activations once** with category metadata from a labeled dataset (e.g., BeaverTails)
2. **Compute per-category vectors** by filtering the refusal cohort — the compliant baseline is always global (all compliant samples, category-agnostic)

The WRMD formula `ridge_inv @ (weighted_refusal_mean - weighted_compliant_mean)` uses:
- **Refusal activations + weights**: only from the target category's judge scores
- **Compliant activations + weights**: from all compliant samples (shared baseline)
- **Covariance matrix**: computed from the global compliant distribution

This captures the direction each category's refusals differ from the shared compliant baseline.

### Workflow

```bash
# 1. Run judge evaluation on a categorized prompt set (e.g., BeaverTails)
#    With category_column: "auto", categories are embedded in the output
cd LLM-Refusal-Evaluation
PYTHONPATH=. python src/compute_refusal_score.py \
    --config configs/my-model.yaml \
    --custom-dataset PKU-Alignment/BeaverTails-Evaluation \
    --dataset-split test \
    --samples-per-category 50 \
    --seed 42
cd ..

# 2. Extract activations with category metadata
#    Categories are read directly from censor_scores.json (no --dataset needed
#    when the submodule output already contains category labels)
python scripts/extract_activations.py \
    --model Qwen/Qwen3.5-9B \
    --results-dir LLM-Refusal-Evaluation/results/beavertails

# Or with explicit dataset for backward compatibility with old results
python scripts/extract_activations.py \
    --model Qwen/Qwen3.5-9B \
    --results-dir LLM-Refusal-Evaluation/results/beavertails \
    --dataset PKU-Alignment/BeaverTails-Evaluation

# 3a. List available categories
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_*.pt \
    --list-categories

# 3b. Compute vectors for a specific category
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_*.pt \
    --category "Violence, Aiding and Abetting, Incitement" \
    --rank 2

# 3c. Compute vectors for ALL categories at once
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_*.pt \
    --all-categories --rank 1

# 3d. Compute all categories + analyze angular distances between them
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_*.pt \
    --all-categories --analyze-categories

# 3e. Full analysis: inter-category + intra-category + bootstrap stability
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_*.pt \
    --all-categories --analyze-categories --analyze-intra-category \
    --bootstrap-stability --bootstrap-samples 20 \
    --min-category-samples 10

# 3f. Bootstrap convergence analysis (how many samples does each category need?)
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_*.pt \
    --all-categories --bootstrap-convergence \
    --convergence-min-pool 50 --convergence-pool-step 50 \
    --convergence-targets 500 1000 \
    --convergence-instability-ceiling 30

# 3g. Compare pre-computed vector files directly (no activations needed)
python scripts/compute_wrmd.py \
    --compare-vectors outputs/.../steering_vectors_wrmd_violence*.pt \
                      outputs/.../steering_vectors_wrmd_child_abuse.pt \
    --output-dir outputs/comparison/
```

### Angular Distance Analysis

**Inter-category** (`--analyze-categories`): Measures how similar/different per-category refusal directions are by computing pairwise angular distances (`arccos(|cos_sim|)` in degrees). This reveals which harm categories the model treats similarly (shared refusal circuit) vs. differently (independent circuits), informing whether per-category steering adds value for a given pair.

**Intra-category** (`--analyze-intra-category`): Measures how coherent/stable each category's refusal circuit is by computing pairwise angular distances between individual refusal activation vectors within each category. A low mean angle indicates a coherent refusal direction; a high mean angle indicates a fragmented one where the model uses different internal representations for prompts in the same harm category.

**Bootstrap stability** (`--bootstrap-stability`): Measures whether a steering vector would change substantially with a slightly different dataset. Resamples 80% of refusal samples with replacement, recomputes the steering vector 20 times, then measures the spread. Categorizes each vector as "stable" (<5 deg), "moderate" (5-15 deg), or "unreliable" (>15 deg). When combined with `--analyze-categories`, annotates inter-category distances with significance z-scores: if both categories have bootstrap std of 3 deg but are 30 deg apart, that difference is highly significant (z=7.1). The compliant covariance inverse is cached across iterations for efficiency.

**Bootstrap convergence** (`--bootstrap-convergence`): Goes beyond stability to answer *"would collecting more data help?"*. Runs bootstrap stability at progressively larger pool sizes (50, 100, 150, ..., N) and fits a convergence curve `angular_spread(n) = k / sqrt(n) + floor`. The fitted `floor` parameter captures intrinsic geometric instability that no amount of data will resolve — it's the best stability achievable for this category on this model. The `k` parameter captures sensitivity to sample size.

The stability target is set to the global bootstrap floor (the best the full dataset can achieve). For each category, the analysis computes how many samples would be needed to reach that target: `samples_needed = ceil((k / (global_floor - category_floor))^2)`. Categories are classified into four actions:

| Action | Meaning |
|--------|---------|
| `collect more data` | Floor is below global spread and current spread has room to improve (>15% above floor) |
| `near floor` | Current spread is within 15% of floor — diminishing returns from more data |
| `geometric limit` | Floor is above global spread but below the instability ceiling — category is inherently harder but still usable |
| `unstable` | Floor >= instability ceiling (default 30 deg) — the refusal representation is too diffuse for reliable linear steering |

Example output:
```
[BOOTSTRAP CONVERGENCE] (n_bootstrap=20, ratio=0.8, method=md)
   Stability target: 10.1° (global floor)
   Instability ceiling: 30.0°
   Category                       Samples   Current     Floor   Needed   Pred@500   Pred@1000     R²               Action
   ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   [global]                           820     12.3°     10.1°        ─      11.2°       10.8°   0.98          (reference)
   ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   violence                           332     18.0°     14.8°        ─      16.2°       15.4°   0.97           near floor
   financial_crime                    139     22.4°     13.2°      482      17.1°       15.8°   0.95     collect more data
   sexually_explicit                   54     45.0°     35.1°        ∞      38.2°       36.9°   0.89             unstable
```

Use `--convergence-min-pool` to set the smallest pool size (default: 50), `--convergence-pool-step` for the step between pool sizes (default: 50), `--convergence-targets` for extrapolation targets (default: 500 1000), and `--convergence-instability-ceiling` for the floor threshold above which a category is flagged unstable (default: 30 deg).

Use `--min-category-samples N` to skip categories with fewer than N refusal samples (default: 10). Use `--category-layers` to restrict analysis to specific layers.

Outputs:
- `category_angular_distances.json` + `.png` — inter-category pairwise matrix, heatmap + dendrogram
- `intra_category_coherence.json` + `.png` — per-category coherence ranking, bar chart
- `bootstrap_stability.json` + `.png` — per-category bootstrap spread + stability labels
- `bootstrap_convergence.json` + `.png` — convergence curves, fitted floor/k, samples needed predictions (if `--bootstrap-convergence`)
- `category_distance_significance.json` — z-scores for inter-category distances (if both `--analyze-categories` and `--bootstrap-stability`)

### Output Structure

`--all-categories` produces a separate vector file per category plus a global file and summary:

```
compute_wrmd/
    steering_vectors_wrmd.pt                          # global (all categories)
    steering_vectors_wrmd_animal_abuse.pt              # per-category
    steering_vectors_wrmd_child_abuse.pt
    steering_vectors_wrmd_violence_aiding_and_abetting_incitement.pt
    ...
    category_summary.json                             # maps category -> file, sample counts
    category_angular_distances.json                   # if --analyze-categories
    category_angular_distances.png                    # if --analyze-categories
    intra_category_coherence.json                     # if --analyze-intra-category
    intra_category_coherence.png                      # if --analyze-intra-category
    bootstrap_stability.json                          # if --bootstrap-stability
    bootstrap_stability.png                           # if --bootstrap-stability
    bootstrap_convergence.json                        # if --bootstrap-convergence
    bootstrap_convergence.png                         # if --bootstrap-convergence
    category_distance_significance.json               # if --analyze-categories + --bootstrap-stability
```

### Supported Datasets

Any HuggingFace dataset with a prompt column and a category column works. The primary tested dataset is **PKU-Alignment/BeaverTails-Evaluation** (700 prompts, 14 harm categories, 50 each).

The submodule has built-in adapters that auto-detect column mappings for known datasets:

| Dataset | prompt_column | category_column |
|---------|---------------|-----------------|
| `PKU-Alignment/BeaverTails*` | `prompt` | `auto` (boolean columns) |
| `allenai/wildjailbreak` | `vanilla` | `risk_category` |
| `sorry-bench/*` | `prompt` | `category` |

For other datasets, use `--prompt-column` and `--category-column` in the submodule CLI, or set them in the YAML config.

### Library Usage

```python
from activation_steering import (
    WRMDCalculator,
    load_prompts_from_judge_scores_with_categories,
    load_prompts_from_dataset,
    category_to_slug,
)

# Load prompts with category metadata
# If the submodule output already contains categories (category_column configured),
# this reads them directly without re-loading the HuggingFace dataset:
prompts, labels, metadata = load_prompts_from_judge_scores_with_categories(
    "results/beavertails",
)

# For old results without embedded categories, fall back to dataset re-load:
prompts, labels, metadata = load_prompts_from_judge_scores_with_categories(
    "results/beavertails",
    dataset_name="PKU-Alignment/BeaverTails-Evaluation",
)

# After extraction, compute per-category vectors
calculator = WRMDCalculator("activations.pt")
print(calculator.get_available_categories())

# Category-filtered MD (default)
vectors = calculator.compute_steering_vectors(
    categories=["Violence, Aiding and Abetting, Incitement"],
)
calculator.save_vectors(
    vectors, f"steering_vectors_md_{category_to_slug('Violence')}.pt",
    categories=["Violence, Aiding and Abetting, Incitement"],
)
```

#### Angular Distance Library Usage

```python
from activation_steering import (
    WRMDCalculator,
    compute_category_angular_distances,
    load_category_vectors_from_files,
    plot_category_angular_distances,
)

# Option 1: From activations (on-the-fly computation)
calculator = WRMDCalculator("activations.pt")
results = calculator.analyze_category_distances(
    target_layers=[10, 15, 20], output_dir="outputs/"
)

# Option 2: From pre-computed .pt files
category_vectors = load_category_vectors_from_files([
    "steering_vectors_md_violence.pt",
    "steering_vectors_md_child_abuse.pt",
    "steering_vectors_md_fraud.pt",
])
results = compute_category_angular_distances(category_vectors, target_layers=[10, 15, 20])
plot_category_angular_distances(results, output_dir="outputs/")

# Access results
print(results["overall"]["mean_pairwise_angle_deg"])
print(results["overall"]["most_similar"])   # {"cat_a": ..., "cat_b": ..., "angle_deg": ...}
print(results["overall"]["most_different"])
for pair in results["pairwise"]:  # sorted by angle
    print(f"{pair['cat_a']} <-> {pair['cat_b']}: {pair['mean_angle_deg']:.1f} deg")
```

### Validating Category-Specific Vectors

After computing per-category vectors and analyzing their stability, validate whether they actually steer *better* than the global vector. `validate_category_steering.py` runs Bayesian alpha optimization with both vectors on each category's prompts, then compares optimal alpha, KL divergence, and refusal reduction.

```bash
# Full validation with judge scoring
python scripts/validate_category_steering.py \
    --model Qwen/Qwen3.5-9B \
    --global-vectors outputs/.../compute_wrmd/steering_vectors_md.pt \
    --category-vectors-dir outputs/.../compute_wrmd/ \
    --baseline-results LLM-Refusal-Evaluation/results/beavertails \
    --dataset PKU-Alignment/BeaverTails-Evaluation \
    --correlations outputs/.../find_best_layers/layer_correlations.json \
    --bootstrap-stability outputs/.../compute_wrmd/bootstrap_stability.json \
    --bayesian-trials 10

# Quick KL-only validation (no judge, faster)
python scripts/validate_category_steering.py \
    --model Qwen/Qwen3.5-9B \
    --global-vectors outputs/.../steering_vectors_md.pt \
    --category-vectors-dir outputs/.../compute_wrmd/ \
    --baseline-results LLM-Refusal-Evaluation/results/beavertails \
    --dataset PKU-Alignment/BeaverTails-Evaluation \
    --layers 10 15 20 \
    --no-judge --bayesian-trials 5

# Single category test
python scripts/validate_category_steering.py \
    --model Qwen/Qwen3.5-9B \
    --global-vectors outputs/.../steering_vectors_md.pt \
    --category-vectors-dir outputs/.../compute_wrmd/ \
    --baseline-results LLM-Refusal-Evaluation/results/beavertails \
    --dataset PKU-Alignment/BeaverTails-Evaluation \
    --layers 10 15 20 \
    --categories "terrorism" --no-judge --bayesian-trials 2
```

**Vector normalization**: Global and category-specific vectors can have very different magnitudes (e.g. a category WRMD vector may have 6x the norm of the global MD vector). Without normalization, the same alpha produces wildly different perturbation strengths, making KL divergence and optimal alpha incomparable across vector types. Use `--normalize-vectors` to L2-normalize all vectors per-layer before optimization, so alpha represents comparable perturbation strength:

```bash
python scripts/validate_category_steering.py \
    --model Qwen/Qwen3.5-9B \
    --global-vectors outputs/.../steering_vectors_md.pt \
    --category-vectors-dir outputs/.../compute_wrmd/ \
    --baseline-results LLM-Refusal-Evaluation/results/beavertails \
    --dataset PKU-Alignment/BeaverTails-Evaluation \
    --layers 10 15 20 \
    --normalize-vectors --bayesian-trials 10
```

The script filters categories by bootstrap stability (stable/moderate only) and minimum sample count, then for each category runs two Optuna studies: one with the global vector, one with the category-specific vector. Outputs include a comparison table, `category_validation.json`, and `category_validation_summary.png` (grouped bar chart).

**Arguments:**

*Required:*
- `--model`: HuggingFace model name or path
- `--global-vectors`: Path to global steering vectors .pt file
- `--category-vectors-dir`: Directory with per-category .pt files + `category_summary.json`

*Prompt source (one required):*
- `--activations`: Path to activations .pt file (with category metadata)
- `--baseline-results`: Path to baseline LLM-Refusal-Evaluation results (requires `--dataset`)
- `--dataset`: HuggingFace dataset for category metadata (e.g., `PKU-Alignment/BeaverTails-Evaluation`)
- `--prompt-column`: Prompt column name in HF dataset (default: `prompt`)
- `--category-column`: Category column name in HF dataset (default: `category`)
- `--dataset-split`: Dataset split to use (default: auto-detect)

*Layer selection:*
- `--correlations`: Path to layer correlations JSON (for `--top-k`)
- `--layers`: Explicit layer indices
- `--top-k`: Use top K layers from correlations (default: 5)

*Alpha optimization:*
- `--alpha-min`: Minimum alpha to test (default: -5.0)
- `--alpha-max`: Maximum alpha to test (default: 0.0)
- `--alpha-step`: Discrete step for Bayesian search (default: continuous)
- `--bayesian-trials`: Trials per Optuna study (default: 10)
- `--objective`: Optimization objective: `minimize_refusal`, `maximize_compliance_rate`, `balanced`, `kl_weighted` (default: `minimize_refusal`)

*Category filtering:*
- `--bootstrap-stability`: Path to `bootstrap_stability.json` (filters unstable categories)
- `--min-category-samples`: Minimum refusal samples per category (default: 10)
- `--categories`: Explicit category list (default: auto-discover from vectors)
- `--normalize-vectors`: L2-normalize vectors per-layer before optimization

*KL divergence:*
- `--kl-divergence` / `--no-kl-divergence`: Enable/disable KL divergence measurement (default: enabled)
- `--kl-method`: `teacher_forced` (multi-token, default) or `first_token` (fastest)
- `--kl-tokens`: Tokens to generate for teacher-forced KL (default: 32)
- `--kl-prompts`: Path to custom harmless prompts JSON
- `--num-kl-prompts`: Number of KL prompts (default: 20)

*Judge scoring:*
- `--no-judge`: Skip judge scoring (KL-only mode, much faster)
- `--judge-model`: Judge model for scoring (default: `openai/gpt-oss-20b`)
- `--enforce-eager`: Pass `enforce_eager=True` to vLLM judge
- `--judge-max-model-len`: Max sequence length for vLLM judge
- `--max-judge-failures`: Abort study after N consecutive judge failures (default: 3)

*Generation:*
- `--max-new-tokens`: Max tokens to generate (default: 512)
- `--generate-timeout`: Timeout per prompt in seconds (default: 120)
- `--temperature`: Sampling temperature (default: 0.6)
- `--top-p`: Nucleus sampling threshold (default: 0.95)
- `--sample-top-k`: Top-k sampling (default: 20)
- `--thinking-string`: Delimiter for thinking/reasoning models (e.g., `</think>`)

*Output:*
- `--output-dir`: Custom output directory
- `--run-id`: Custom run ID
- `--component`: Steering component (default: `attn`)
- `--log-file`: Log file path — `auto` (default) writes to `experiment.log` in output dir, `none` disables
- `--resume`: Resume from checkpoint — skip categories already in `category_checkpoint.json`

### Edge Cases

- **Empty cohort**: If a category has 0 refusal samples after judging (model complied with all), `--all-categories` skips it with a warning; `--category` raises an error
- **Small cohorts**: Ridge regularization handles small N, but minimum 2 refusal samples required
- **Prompt matching**: Cross-referencing uses `.strip()` normalization (exact match, no fuzzy matching)
- **Multi-component**: `--all-categories` requires a single component (`attn` or `mlp`, not `attn+mlp`)

### Category Router

The category router eliminates external classifiers by using the steering vectors themselves for classification. It works in two stages:

1. **Refusal detection** (Stage 1): Project activations onto the global steering vector. If the projection exceeds a calibrated threshold, the prompt is a refusal candidate.
2. **Category routing** (Stage 2): Project activations onto residual category vectors (global component removed), z-score normalize per category, and select the best match.

#### Calibrating the Router

```bash
python scripts/calibrate_router.py \
    --activations outputs/.../activations_*.pt \
    --global-vectors outputs/.../steering_vectors_md.pt \
    --category-vectors outputs/.../steering_vectors_md_*.pt \
    --correlations outputs/.../layer_correlations.json \
    --bootstrap-stability outputs/.../bootstrap_stability.json \
    --stability-filter unreliable \
    --output-dir outputs/.../calibrate_router/
```

This produces `calibration.json` with:
- **Stage 1 threshold** from ROC analysis (Youden's J statistic) and AUC
- **Per-category residual projection stats** (own-sample mean/std) for z-score normalization
- **Excluded categories** filtered by bootstrap stability
- **Per-category precision/recall/F1** reported for three configurations: raw vectors, residual vectors, and residual + z-score

The `--stability-filter` flag controls which categories are excluded from routing (default: `unreliable`). Categories without bootstrap stability data are also excluded. Excluded categories still participate in classification (so false positive rates are accurate) but predictions landing on them trigger global fallback.

#### Why Residual Vectors + Z-Score

Raw category vectors share most of the global refusal signal. Hate speech (mean=9.24) and discrimination (mean=8.62) project high for *all* refusal samples because their vectors align with the generic refusal direction. A vector good for *steering* (shifts activations away from refusal) is not necessarily good for *classification* (separates its category from others).

**Residual vectors** project out the shared global component: `v_residual = v_cat - (v_cat . v_global / v_global . v_global) * v_global`. What remains is the category-specific signal orthogonal to the shared refusal direction.

**Z-score normalization** handles the residual norm scaling problem. After removing the global component, each category's residual has a different norm (discrimination retains 79% of its original norm, terrorism retains 42%). Raw argmax over residuals is biased toward categories with larger residual norms. Z-scoring asks "how many standard deviations above this category's own baseline?" instead of "which category has the largest absolute projection?" This brought terrorism routing from F1=0.000 to F1=0.867 in testing.

#### Using the Router

```bash
# Interactive testing with router
python scripts/test_steering.py \
    --model Qwen/Qwen3.5-9B \
    --calibration outputs/.../calibrate_router/calibration.json \
    --router --alpha -3.0
```

The router is internal to `CategoryRouter.classify()`. At inference:
- Stage 1 uses the global vector (unchanged)
- Stage 2 uses residual vectors with z-score normalization
- Predictions landing on excluded categories fall back to global steering
- Actual steering always uses the raw category vectors (residuals are for classification only)

```python
from activation_steering import CategoryRouter

router = CategoryRouter.from_calibration_file("calibration.json")
decision = router.classify_prompt(model, tokenizer, "How do I make a bomb?")
# decision.selected_category = "terrorism,organized_crime"
# decision.use_global_fallback = False

hook = router.create_steering_hooks(model, decision, alpha=-3.0)
```

---

## Using as a Library

The package can be imported directly in Python code:

### Basic Usage

```python
from activation_steering import (
    ActivationExtractor,
    WRMDCalculator,
    SteeringHook,
    load_prompts_from_judge_scores,
    load_prompts_from_judge_scores_with_categories,
    category_to_slug,
)

# Extract activations (default: attn component)
extractor = ActivationExtractor("Qwen/Qwen3.5-9B", components=["attn"])
prompts, labels, metadata = load_prompts_from_judge_scores("results/baseline")
extractor.extract_dataset(prompts, labels, "activations.pt", metadata)

# Compute steering vectors (rank-1, attn component, MD default)
calculator = WRMDCalculator("activations.pt")
vectors = calculator.compute_steering_vectors(component='attn')
calculator.save_vectors(vectors, "steering_vectors.pt", component='attn')

# Or compute multi-rank vectors
vectors_mr = calculator.compute_steering_vectors(rank=2)
calculator.save_vectors(vectors_mr, "steering_vectors_rank2.pt", rank=2)

# Category-specific vectors (requires category metadata in activations)
# Extract with categories: load_prompts_from_judge_scores_with_categories(...)
calculator = WRMDCalculator("activations_with_categories.pt")
print(calculator.get_available_categories())
vectors_cat = calculator.compute_steering_vectors(categories=["Child Abuse"])
calculator.save_vectors(vectors_cat, "steering_vectors_child_abuse.pt", categories=["Child Abuse"])

# Apply steering at runtime (hooks on attn submodule)
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", ...)
steerer = SteeringHook(model, vectors, target_layers=[10, 11], alpha=-2.0, component="attn")
steerer.register_hooks()
# ... run generation ...
steerer.remove_hooks()

# Apply multi-rank steering with per-direction alphas
steerer = SteeringHook(model, vectors_mr, target_layers=[10, 11], alpha=[-2.0, -1.0], component="attn")
steerer.register_hooks()
# ... run generation ...
steerer.remove_hooks()
```

### Dual-Component Steering (attn+mlp)

```python
from activation_steering import SteeringHook
from activation_steering.steering import SteeringHookGroup
import torch

# Load or create a combined steering file with both component vectors
attn_data = torch.load("steering_vectors_attn.pt")
mlp_data = torch.load("steering_vectors_mlp.pt")
combined = {
    "steering_vectors_attn": attn_data["steering_vectors"],
    "steering_vectors_mlp": mlp_data["steering_vectors"],
    "steering_vectors": attn_data["steering_vectors"],  # fallback
    "num_layers": attn_data["num_layers"],
    "hidden_size": attn_data["hidden_size"],
    "component": "attn+mlp",
}

# Create dual-component hook group with per-component alphas
group = SteeringHookGroup.from_steering_data(
    model=model,
    steering_data=combined,
    target_layers=[10, 11, 12],
    alpha=-2.0,           # shared default
    alpha_attn=-3.0,      # override for attention (optional)
    alpha_mlp=-0.5,       # override for MLP (optional)
    components=("attn", "mlp"),
)
group.register_hooks()
# ... run generation ...
group.remove_hooks()
```

### Capability Preservation Check

```python
from activation_steering import (
    SteeringHook,
    evaluate_capability,
    compare_capability,
    load_questions,
    load_hf_benchmark,
)

# Load questions -- built-in, custom file, or HuggingFace benchmark
questions = load_questions()                                    # built-in 50 questions
questions = load_questions("my_questions.json")                 # custom file
questions = load_hf_benchmark("mmlu", max_questions=200)        # MMLU subset (requires `datasets`)
questions = load_hf_benchmark("arc_challenge")                  # ARC-Challenge

# Baseline eval (no steering)
baseline = evaluate_capability(model, tokenizer, questions)
print(f"Baseline: {baseline.accuracy:.1%}")

# Steered eval
steerer = SteeringHook(model, vectors, target_layers=[10, 11], alpha=-2.5)
steerer.register_hooks()
steered = evaluate_capability(model, tokenizer, questions)
steerer.remove_hooks()

# Compare
comparison = compare_capability(baseline, steered)
print(f"Delta: {comparison['accuracy_delta']:+.1%}")
print(f"Degraded: {comparison['degraded']}")  # True if >5% drop
```

### Permanent Steering via Weight Merging

```python
from activation_steering import merge_steering_into_model

# Merge into MLP bias (default)
metadata = merge_steering_into_model(
    base_model_path="Qwen/Qwen3.5-9B",
    steering_vectors_file="steering_vectors.pt",
    target_layers=[10, 11],
    alpha=-2.0,
    output_dir="Qwen3.5-9B-Steered"
)

# Merge into attention output projection bias
metadata = merge_steering_into_model(
    base_model_path="Qwen/Qwen3.5-9B",
    steering_vectors_file="steering_vectors.pt",
    target_layers=[10, 11],
    alpha=-2.0,
    component="attn",
    output_dir="Qwen3.5-9B-Steered"
)

# Merge dual-component with per-component alphas
metadata = merge_steering_into_model(
    base_model_path="Qwen/Qwen3.5-9B",
    steering_vectors_file="steering_vectors_attn_mlp.pt",
    target_layers=[10, 11],
    alpha_attn=-3.0,
    alpha_mlp=-0.5,
    component="attn+mlp",
    output_dir="Qwen3.5-9B-Steered"
)

# Load attention-merged models with any framework (natively compatible)
from transformers import AutoModelForCausalLM
steered_model = AutoModelForCausalLM.from_pretrained("Qwen3.5-9B-Steered")
# No hooks needed - steering is built-in!

# Load MLP-merged or dual-component models (sidecar fallback for MLP biases)
from activation_steering import load_merged_model
model, tokenizer = load_merged_model("Qwen3.5-9B-Steered")

# load_merged_model handles both native and sidecar loading:
# - Attention biases: loaded natively via attention_bias config flag
# - MLP biases: injected from steering_biases.safetensors sidecar
```

---

## File Format Details

### Activation Files (.pt)
```python
{
    'activations_attn': Tensor[N, num_layers, hidden_size],  # if component=attn
    'activations_mlp': Tensor[N, num_layers, hidden_size],   # if component=mlp
    'activations': Tensor[N, num_layers, hidden_size],       # if component=layer (or legacy)
    'labels': Tensor[N],  # 0=compliant, 1=refusal
    'prompts': List[str],
    'num_layers': int,
    'hidden_size': int,
    'components': List[str],  # e.g. ['attn'], ['mlp'], ['attn', 'mlp']
    'metadata': List[dict]  # Contains judge scores and split info
}
```

The activation key depends on the extracted component: `activations_attn` for attention, `activations_mlp` for MLP, `activations` for full layer output. Multi-component extractions include multiple keys. Legacy files with only `'activations'` are loaded as `"layer"` for backward compatibility.

### Steering Vector Files (.pt)
```python
{
    'steering_vectors_attn': Tensor[num_layers, hidden_size],     # if component=attn
    'steering_vectors_mlp': Tensor[num_layers, hidden_size],      # if component=mlp
    'steering_vectors': Tensor[num_layers, hidden_size],          # if component=layer (rank-1)
    #                or Tensor[num_layers, rank, hidden_size],    # multi-rank
    'num_layers': int,
    'hidden_size': int,
    'method': str,  # 'md', 'rmd', or 'wrmd'
    'component': str,  # 'attn', 'mlp', 'layer', or 'attn+mlp'
    'lambda_ridge': float,
    'use_score_weighting': bool,
    'num_refusal_samples': int,
    'num_compliant_samples': int,
    'rank': int  # 1 for standard, >1 for multi-rank
}
```

For dual-component (`attn+mlp`) steering, the file should contain both `steering_vectors_attn` and `steering_vectors_mlp` keys. These can be created by combining vectors from separate single-component runs, or by extracting both components simultaneously and computing vectors for each.

### Layer Correlation Files (.json)
```python
{
    'best_layers': [int, ...],  # Top K layers by absolute correlation
    'all_correlations': [
        {
            'layer': int,
            'correlation': float,
            'abs_correlation': float,
            'p_value': float,
            'projection_mean': float,
            'projection_std': float
        },
        ...
    ]
}
```

---

## Implementation Details

### Activation Extraction (`src/activation_steering/extraction.py`)
- Extracts activations at the **last token position** for all layers
- Supports component-specific extraction: `attn` (attention output), `mlp` (MLP output), `layer` (full layer output), or multiple simultaneously
- Auto-discovers attention submodule names across architectures via `_get_attn_submodule()` (`self_attn`, `linear_attn`, `attention`, `attn`)
- Uses judge scores from LLM-Refusal-Evaluation metadata to filter samples
- Only includes samples with clear behavior (score > refusal_threshold OR score < compliance_threshold)
- Skips uncertain samples (scores near zero)
- Saves metadata alongside activations for downstream use

### WRMD Computation (`src/activation_steering/computation.py`)
- Converts judge scores to weights: higher magnitude = higher confidence = higher weight
- For refusals (label=1): weight = score (already positive)
- For compliances (label=0): weight = -score (convert negative to positive)
- Computes weighted covariance matrix from compliant distribution
- Uses float32 for matrix inversion, then casts back to original dtype (bfloat16)
- Ridge regularization (lambda) stabilizes inversion of covariance matrix
- **Multi-rank** (`--rank k`): v_1 is the standard WRMD vector. Additional directions are found by projecting v_1 out of refusal activations and computing PCA (SVD) on the residuals. Each additional direction captures structured variance in refusal behavior orthogonal to v_1. Reports variance explained by each component.
- **Rank association analysis** (`--analyze-ranks`): Projects each prompt's activations onto each rank direction across the target layers and determines the dominant rank per prompt. Reports distribution statistics broken down by refusal/compliant labels and shows example prompts for each rank, helping interpret what each PCA component captures semantically.

### Layer Correlation Analysis (`src/activation_steering/analysis.py`)
- Normalizes steering vectors to unit length before projection
- Computes projection = activations @ steering_vector for each layer
- Correlates projections with judge scores using Pearson correlation
- High positive/negative correlation = layer is effective for steering
- Top layers by absolute correlation are recommended for steering

### Dynamic Steering (`src/activation_steering/steering.py`)
- `SteeringHook`: Registers PyTorch forward hooks on a single target submodule (`attn`, `mlp`, or `layer`)
- Each hook shifts the **last token** hidden states by `alpha * v` (rank-1) or `sum(alpha_i * v_i)` (multi-rank), using raw (unnormalized) vectors
- Hooks are added and removed on demand — model weights stay unchanged
- Can steer a single layer or multiple layers simultaneously
- Supports per-rank alpha values for multi-rank vectors (pass `alpha=[a1, a2, ...]`)
- Auto-detects rank from tensor shape: 2D = rank-1, 3D = multi-rank
- Uses `_get_attn_submodule()` for cross-architecture attention hook registration
- Loads actual refusal prompts from baseline evaluation results for side-by-side comparison
- `SteeringHookGroup`: Manages multiple `SteeringHook` instances for dual-component (`attn+mlp`) steering. Created via `SteeringHookGroup.from_steering_data()`, which reads `steering_vectors_attn` and `steering_vectors_mlp` keys from a combined steering file. Supports per-component alpha overrides (`alpha_attn`, `alpha_mlp`) and shared alpha as fallback

### Static Merge (`src/activation_steering/merge_steering_into_weights.py`)
- `merge_steering_into_model()`: Permanently merges `alpha * v` into output projection bias terms (`down_proj.bias` for MLP, `o_proj.bias` for attention)
- Multi-rank vectors are auto-detected (3D tensor shape) and rank 0 is selected for static merge
- After merging steered layers, zero biases are added to ALL non-steered layers for checkpoint consistency
- For attention: zero biases are also added to q/k/v projections (required when `attention_bias: true`)
- `config.json` is patched with `attention_bias: true` and/or `mlp_bias: true` so loaders create matching architecture
- A `steering_biases.safetensors` sidecar is saved containing only the steered biases for backward compatibility
- `load_merged_model()`: Loads a merged model using `from_pretrained` then injects any biases from the sidecar that weren't loaded natively (e.g. MLP biases where `mlp_bias` isn't recognized by the architecture)
- `verify_merged_model()`: Compares merged model against original to confirm bias modifications were applied correctly

### Model Loading Pattern

All scripts use this pattern for efficient loading:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Reduces memory usage
    device_map="auto"             # Automatic multi-GPU distribution
)
```

For statically-merged models, use `load_merged_model()` which handles both native and sidecar bias loading:

```python
from activation_steering import load_merged_model

# Automatically loads native biases + injects sidecar biases where needed
model, tokenizer = load_merged_model("Qwen3.5-9B-Steered")
```

Attention-merged models also load natively with `from_pretrained()` or `vllm.LLM()` since `attention_bias: true` is set in config.

### Utility Functions (`src/activation_steering/utils.py`)

- `extract_model_name(model_string)`: Converts "Qwen/Qwen3.5-9B" -> "qwen3.5-9b"
- `generate_run_id()`: Creates timestamp-based ID (YYYYMMDD-HHMMSS)
- `setup_model_run_dirs(model_name, run_id)`: Creates full output directory structure
- `ensure_dir(path)`: Creates directory if it doesn't exist

All scripts use these utilities for consistent output organization.

---

## Troubleshooting

### Common Issues

**GPU Memory Errors:**
- Use smaller batch sizes or models
- Enable `device_map="auto"` for multi-GPU distribution
- Use `torch_dtype=torch.bfloat16` for memory efficiency
- **Alpha optimization OOM during judge scoring:** The optimizer offloads the steering model to CPU before spawning the judge subprocess, then restores it to GPU afterward. If you still get OOM, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:512` before running — this ensures `torch.cuda.empty_cache()` actually releases reserved memory back to the OS so the judge subprocess (vllm) can use it. The script sets this automatically, but external processes inheriting a CUDA context may need it set explicitly.

**Judge Score Files Not Found:**
- Ensure LLM-Refusal-Evaluation ran successfully
- Check that `--results-dir` points to the correct baseline results
- Verify `judge_scores.json` or `aggregated_results.json` exists

**Import Errors:**
- Install package in editable mode: `pip install -e .`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**Correlation Analysis Shows Weak Correlations:**
- Try different steering vector methods (MD default, or WRMD for large datasets)
- Adjust ridge regularization parameter `--lambda-ridge`
- Ensure sufficient samples with clear refusal/compliance behavior

### FP8 Checkpoints and Quantized Loading

FP8 checkpoints (e.g., `Qwen3.5-27B-FP8`) store weights in 8-bit floating point, but the `transformers` library does not natively support this format — it loads the weights as bfloat16 while ignoring the FP8 scale factors, causing the model to overflow GPU memory and produce incorrect activations.

All scripts auto-detect FP8 checkpoints and load them with BitsAndBytes 8-bit quantization instead. Detection checks `config.json`, safetensors metadata, and the model name. No extra flags needed:

```bash
# FP8 checkpoint — auto-detected, loaded as BnB 8-bit
python scripts/extract_activations.py --model Qwen/Qwen3.5-27B-FP8 ...
python scripts/test_steering.py --model Qwen/Qwen3.5-27B-FP8 ...
```

You can also force quantization explicitly for any model:

```bash
# Force 4-bit quantization (NF4) — ~0.5 bytes/param
python scripts/test_steering.py --model Qwen/Qwen3.5-9B --load-in-4bit ...

# Force 8-bit quantization — ~1 byte/param
python scripts/extract_activations.py --model Qwen/Qwen3.5-9B --load-in-8bit ...
```

**Important notes:**

- **Consistency**: If you extract activations with quantization, use the same quantization when steering. Quantized activations differ from unquantized ones, so steering vectors may need different alpha values.
- **`optimize_alpha.py` with quantized models**: The judge subprocess shares the GPU with the quantized model (instead of offloading). This works when there is at least ~8 GiB free after model loading. On a 45 GiB A40 with a 27B 8-bit model (~27 GiB), there is enough room for the judge.
- **`validate_category_steering.py`**: Does not support quantized models due to its explicit device placement for judge offloading. Use `optimize_alpha.py --stable-categories` as an alternative.
- **`--load-in-4bit` / `--load-in-8bit`** are available on: `extract_activations.py`, `optimize_alpha.py`, `test_steering.py`, `eval_capability.py`.
- Requires `pip install bitsandbytes`.

### Qwen3.5 MoE Models

Qwen3.5 MoE models (e.g. `Qwen/Qwen3.5-MoE-A3B`) have additional requirements:

1. **Nightly vllm required** — the stable release does not support Qwen3.5 MoE architecture. Install from source:
   ```bash
   pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
   ```

2. **Transformers from source** — the released transformers package may lack Qwen3.5 MoE support:
   ```bash
   pip install git+https://github.com/huggingface/transformers.git
   ```

3. **`enforce_eager=True` on GPUs with < ~80 GB VRAM** — CUDA graph capture can exceed memory on smaller GPUs. Pass the `--enforce-eager` flag when running `optimize_alpha.py`:
   ```bash
   python scripts/optimize_alpha.py \
       --model Qwen/Qwen3.5-MoE-A3B \
       --enforce-eager \
       ...
   ```

4. **Use absolute model paths** — relative paths can trigger `huggingface_hub` validation errors. The scripts now resolve local paths automatically, but if you hit path issues, pass the full absolute path to `--model`.
