#!/bin/bash
# Full pipeline: Qwen3.5-9B and Mistral-7B with BeaverTails baseline (train/holdout split)
# Run: bash run_pipeline.sh [--resume] 2>&1 | tee outputs/pipeline.log
set -euo pipefail

# Source env file if present (picks up API keys for nohup/background runs)
[[ -f /workspace/.env ]] && set -a && source /workspace/.env && set +a

RESUME=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME=true; shift;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

CONDA_ENV="activation-steer"
RUN_ID="20260526-beavertails-mlp"

# Use conda env if available, otherwise fall back to system python
if command -v conda &>/dev/null && conda env list | grep -q "^${CONDA_ENV} "; then
    PY() { conda run -n "${CONDA_ENV}" python "$@"; }
else
    PY() { python "$@"; }
fi

# Override these with local paths if models are cached (e.g. on local machine)
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-9B}"
MISTRAL_MODEL="${MISTRAL_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
JUDGE_MODEL="deepseek-v4-flash"
JUDGE_API_BASE="https://api.deepseek.com/v1"

QWEN_BASELINE="baselines/qwen3-5-9b-beavertails"
MISTRAL_BASELINE="baselines/mistral-7b-beavertails"

QWEN_BASE="outputs/qwen3-5-9b/${RUN_ID}"
MISTRAL_BASE="outputs/mistral-7b-instruct-v0.2/${RUN_ID}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "ERROR: $*"; exit 1; }

_reconstruct_checkpoint() {
    # Reconstruct bayesian_trial_checkpoint.jsonl from existing trial_metrics.json files.
    # Usage: _reconstruct_checkpoint <optimize_alpha_dir> <objective>
    local dir="$1"
    local objective="$2"
    local checkpoint="${dir}/bayesian_trial_checkpoint.jsonl"
    [[ -f "${checkpoint}" ]] && return 0
    PY -c "
import json, pathlib, sys
d = pathlib.Path('${dir}')
objective = '${objective}'
entries = []
for td in sorted(d.glob('trial_*')):
    mf = td / 'trial_metrics.json'
    if not mf.exists():
        continue
    m = json.loads(mf.read_text())
    kl  = m.get('kl_divergence', {}).get('mean_kl', 0.0)
    degen = m.get('metrics', {}).get('degenerate_rate', 0.0)
    alpha = m['alpha'] if isinstance(m['alpha'], (int, float)) else m['alpha'][0]
    if objective == 'kl_weighted':
        obj_val = m['metrics']['mean_score'] + 0.1 * kl + 5.0 * degen
    elif objective == 'refusal_kl_weighted':
        obj_val = -m['metrics']['mean_score'] + 0.1 * kl + 5.0 * degen
    else:
        obj_val = m['metrics']['mean_score']
    entries.append({'trial_number': m['trial'], 'objective_value': obj_val,
                    'params': {'alpha': alpha}, 'result': m})
if entries:
    out = d / 'bayesian_trial_checkpoint.jsonl'
    with open(out, 'w') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')
    print(f'Reconstructed checkpoint: {len(entries)} trials -> {out}')
" || true
}

run_pipeline() {
    local model="$1"
    local baseline="$2"
    local base_dir="$3"
    local label="$4"
    local output_name="$5"

    log "========================================================"
    log "${label}: STARTING PIPELINE"
    log "========================================================"

    # ── Step 0: Generate BeaverTails + general_prompts baseline ──
    local train_marker="${baseline}/beavertails_evaluation/censor_scores.json"
    local holdout_marker="${baseline}-holdout/beavertails_evaluation/censor_scores.json"
    local general_marker="${baseline}/general_prompts/censor_scores.json"
    if [[ "${RESUME}" == "true" && -f "${train_marker}" && -f "${holdout_marker}" && -f "${general_marker}" ]]; then
        log "${label}: [SKIP] gen_beavertails_baseline (all splits exist)"
    else
        log "${label}: gen_beavertails_baseline (beavertails_evaluation + general_prompts)"
        PY scripts/gen_beavertails_baseline.py \
            --model "${model}" \
            --output-name "${output_name}" \
            --judge-model "${JUDGE_MODEL}" \
            --judge-api-base "${JUDGE_API_BASE}" \
            --judge-api-workers 16 \
            --batch-size 4 \
            --max-new-tokens 512
    fi
    [[ -f "${train_marker}" ]] || die "BeaverTails train baseline not found: ${train_marker}"
    [[ -f "${holdout_marker}" ]] || die "BeaverTails holdout baseline not found: ${holdout_marker}"
    [[ -f "${general_marker}" ]] || die "General prompts baseline not found: ${general_marker}"

    # ── Step 1: Extract activations ──────────────────────────────
    local activations="${base_dir}/extract_activations/activations.pt"
    if [[ "${RESUME}" == "true" && -f "${activations}" ]]; then
        log "${label}: [SKIP] extract_activations (exists)"
    else
        log "${label}: extract_activations"
        PY scripts/extract_activations.py \
            --model "${model}" \
            --results-dir "${baseline}" \
            --components mlp \
            --run-id "${RUN_ID}"
    fi
    [[ -f "${activations}" ]] || die "Activations not found: ${activations}"

    # ── Step 2: Compute global steering vectors (MD, MLP) ────────
    local steering_vectors="${base_dir}/compute_wrmd/steering_vectors_md_mlp.pt"
    if [[ "${RESUME}" == "true" && -f "${steering_vectors}" ]]; then
        log "${label}: [SKIP] compute_wrmd (exists)"
    else
        log "${label}: compute_wrmd"
        PY scripts/compute_wrmd.py \
            --activations "${activations}" \
            --component mlp \
            --method md
    fi
    [[ -f "${steering_vectors}" ]] || die "Steering vectors not found: ${steering_vectors}"

    # ── Step 2.5: Compute per-category steering vectors ───────────
    # Produces one vector per BeaverTails harm category (min 15 refused samples).
    # category_summary.json is used downstream to drive per-category optimization
    # and to determine which categories to exclude from the leftover run.
    local cat_vectors_dir="${base_dir}/category_vectors"
    local cat_summary="${cat_vectors_dir}/category_summary.json"
    if [[ "${RESUME}" == "true" && -f "${cat_summary}" ]]; then
        log "${label}: [SKIP] compute_category_vectors (exists)"
    else
        log "${label}: compute_category_vectors"
        PY scripts/compute_category_vectors.py \
            --activations "${activations}" \
            --output-dir "${cat_vectors_dir}" \
            --min-refused 15 \
            --min-complied 3 \
            --component mlp
    fi
    [[ -f "${cat_summary}" ]] || die "Category summary not found: ${cat_summary}"

    # ── Step 3: Find best layers ──────────────────────────────────
    local correlations="${base_dir}/find_best_layers/layer_correlations_mlp.json"
    [[ -f "${correlations}" ]] || correlations="${base_dir}/find_best_layers/layer_correlations.json"
    if [[ "${RESUME}" == "true" && -f "${correlations}" ]]; then
        log "${label}: [SKIP] find_best_layers (exists)"
    else
        log "${label}: find_best_layers"
        PY scripts/find_best_layers.py \
            --activations "${activations}" \
            --steering-vectors "${steering_vectors}" \
            --component mlp \
            --top-k 4
        correlations="${base_dir}/find_best_layers/layer_correlations_mlp.json"
        [[ -f "${correlations}" ]] || correlations="${base_dir}/find_best_layers/layer_correlations.json"
    fi
    [[ -f "${correlations}" ]] || die "Correlations not found (tried _mlp and plain variants)"

    # Parse top-4 layers for random direction control
    local layers
    layers=$(PY -c "
import json
d = json.load(open('${correlations}'))
print(' '.join(str(l) for l in d['best_layers'][:4]))
")
    log "${label}: top-4 layers = ${layers}"

    # Parse per-category names from summary (used to exclude from leftover run)
    local per_cat_names
    per_cat_names=$(PY -c "
import json
d = json.load(open('${cat_summary}'))
print(' '.join(d.keys()))
")
    log "${label}: per-category names = ${per_cat_names}"

    # ── Step 4a: Optimize alpha — per-category compliance ─────────
    # Each BeaverTails category with ≥15 refused samples gets its own Bayesian
    # search using a category-specific steering vector. This enables the alpha
    # efficiency comparison that certifies Finding 2 (geometry → behavior link).
    local cat_comply_dir="${base_dir}/optimize_alpha_category_comply"
    local cat_comply_marker="${cat_comply_dir}/category_checkpoint.json"
    if [[ "${RESUME}" == "true" && -f "${cat_comply_marker}" ]]; then
        log "${label}: [SKIP] optimize_alpha_category_comply (checkpoint exists)"
    else
        log "${label}: optimize_alpha (per-category comply, stable-categories mode)"
        PY scripts/optimize_alpha.py \
            --model "${model}" \
            --baseline-results "${baseline}" \
            --stable-categories "${cat_vectors_dir}" \
            --correlations "${correlations}" \
            --top-k 4 \
            --alpha-min -5.0 --alpha-max 0.0 \
            --optimizer bayesian --bayesian-trials 7 \
            --objective kl_weighted \
            --kl-divergence --kl-method first_token \
            --component mlp \
            --judge-model "${JUDGE_MODEL}" \
            --judge-api-base "${JUDGE_API_BASE}" \
            --judge-api-workers 16 \
            --max-new-tokens 512 \
            --batch-size 2 \
            --output-dir "${cat_comply_dir}"
    fi

    # ── Step 4b: Optimize alpha — leftover compliance ─────────────
    # Global steering vector over all refused prompts NOT covered by Step 4a.
    # Includes remaining BeaverTails categories (below min-refused threshold or
    # skipped) plus all non-BeaverTails refused prompts (JBB benign, XSTest safe,
    # general prompts). Prompts with no category label are always included.
    local leftover_comply_dir="${base_dir}/optimize_alpha_leftover_comply"
    local leftover_comply_summary="${leftover_comply_dir}/optimization_summary.json"
    if [[ "${RESUME}" == "true" && -f "${leftover_comply_summary}" ]]; then
        log "${label}: [SKIP] optimize_alpha_leftover_comply (exists)"
    else
        local leftover_resume_flag=""
        if [[ "${RESUME}" == "true" && -d "${leftover_comply_dir}" ]]; then
            _reconstruct_checkpoint "${leftover_comply_dir}" "kl_weighted"
            leftover_resume_flag="--resume"
        fi
        log "${label}: optimize_alpha (leftover comply, global vector)"
        PY scripts/optimize_alpha.py \
            --model "${model}" \
            --baseline-results "${baseline}" \
            --steering-vectors "${steering_vectors}" \
            --correlations "${correlations}" \
            --top-k 4 \
            --alpha-min -5.0 --alpha-max 0.0 \
            --optimizer bayesian --bayesian-trials 7 \
            --objective kl_weighted \
            --kl-divergence --kl-method first_token \
            --component mlp \
            --exclude-categories ${per_cat_names} \
            --max-per-category 10 \
            --num-prompts 500 \
            --judge-model "${JUDGE_MODEL}" \
            --judge-api-base "${JUDGE_API_BASE}" \
            --judge-api-workers 16 \
            --max-new-tokens 512 \
            --batch-size 2 \
            --output-dir "${leftover_comply_dir}" \
            ${leftover_resume_flag}
    fi

    # ── Step 4c: Global validation — balanced category sample ─────
    # Runs global steering vector on 15 prompts per BeaverTails category.
    # Provides the alpha efficiency baseline for Finding 2 certification:
    # alpha_eff(global) / alpha_eff(per-category) should correlate with
    # angular distance from global vector per category.
    local global_val_dir="${base_dir}/optimize_alpha_global_validation"
    local global_val_summary="${global_val_dir}/optimization_summary.json"
    if [[ "${RESUME}" == "true" && -f "${global_val_summary}" ]]; then
        log "${label}: [SKIP] optimize_alpha_global_validation (exists)"
    else
        local global_val_resume_flag=""
        if [[ "${RESUME}" == "true" && -d "${global_val_dir}" ]]; then
            _reconstruct_checkpoint "${global_val_dir}" "kl_weighted"
            global_val_resume_flag="--resume"
        fi
        log "${label}: optimize_alpha (global validation, balanced 15/category)"
        PY scripts/optimize_alpha.py \
            --model "${model}" \
            --baseline-results "${baseline}" \
            --steering-vectors "${steering_vectors}" \
            --correlations "${correlations}" \
            --top-k 4 \
            --alpha-min -5.0 --alpha-max 0.0 \
            --optimizer bayesian --bayesian-trials 7 \
            --objective kl_weighted \
            --kl-divergence --kl-method first_token \
            --component mlp \
            --categories ${per_cat_names} \
            --max-per-category 10 \
            --num-prompts 500 \
            --judge-model "${JUDGE_MODEL}" \
            --judge-api-base "${JUDGE_API_BASE}" \
            --judge-api-workers 16 \
            --max-new-tokens 512 \
            --batch-size 2 \
            --output-dir "${global_val_dir}" \
            ${global_val_resume_flag}
    fi

    # ── Step 5: Optimize alpha — refusal direction (alpha > 0) ────
    # Global vector only. Uses full complied pool (BeaverTails + JBB benign +
    # XSTest safe + general prompts) to characterize the refusal–compliance
    # spectrum. KL/PPL on capability probe bounds the alpha ceiling.
    # XSTest over-refusal at optimal alpha is reported as a behavioral
    # illustration of the asymmetry (not a failure mode to avoid).
    local refuse_summary="${base_dir}/optimize_alpha_refuse/optimization_summary.json"
    if [[ "${RESUME}" == "true" && -f "${refuse_summary}" ]]; then
        log "${label}: [SKIP] optimize_alpha_refuse (exists)"
    else
        local refuse_resume_flag=""
        if [[ "${RESUME}" == "true" && -d "${base_dir}/optimize_alpha_refuse" ]]; then
            _reconstruct_checkpoint "${base_dir}/optimize_alpha_refuse" "refusal_kl_weighted"
            refuse_resume_flag="--resume"
        fi
        log "${label}: optimize_alpha (refuse, alpha=0..5, refusal_kl_weighted)"
        PY scripts/optimize_alpha.py \
            --model "${model}" \
            --baseline-results "${baseline}" \
            --steering-vectors "${steering_vectors}" \
            --correlations "${correlations}" \
            --top-k 4 \
            --alpha-min 0.0 --alpha-max 5.0 \
            --optimizer bayesian --bayesian-trials 7 \
            --objective refusal_kl_weighted \
            --kl-divergence --kl-method first_token \
            --component mlp \
            --judge-model "${JUDGE_MODEL}" \
            --judge-api-base "${JUDGE_API_BASE}" \
            --judge-api-workers 16 \
            --max-new-tokens 512 \
            --batch-size 2 \
            --output-dir "${base_dir}/optimize_alpha_refuse" \
            ${refuse_resume_flag}
    fi

    # ── Step 6: Random-direction KL control ───────────────────────
    # 50 random unit vectors per layer, scaled to same perturbation magnitude
    # as steering vectors. Tests whether refusal direction is geometrically
    # privileged (low KL) and compliance direction unusually disruptive (high KL)
    # relative to the ambient distribution — required evidence for Finding 1.
    local rdc_out="${base_dir}/random_direction_control/random_direction_control.json"
    if [[ "${RESUME}" == "true" && -f "${rdc_out}" ]]; then
        log "${label}: [SKIP] random_direction_control (exists)"
    else
        log "${label}: random_direction_control"
        PY scripts/random_direction_control.py \
            --model "${model}" \
            --steering-vectors "${steering_vectors}" \
            --layers ${layers} \
            --component mlp \
            --alphas -2.0 2.0 \
            --n-random 50 \
            --output-dir "${base_dir}/random_direction_control"
    fi

    log "${label}: PIPELINE COMPLETE"
}

mkdir -p outputs

run_pipeline "${QWEN_MODEL}"    "${QWEN_BASELINE}"    "${QWEN_BASE}"    "Qwen3.5-9B"  "qwen3-5-9b-beavertails"
run_pipeline "${MISTRAL_MODEL}" "${MISTRAL_BASELINE}" "${MISTRAL_BASE}" "Mistral-7B"  "mistral-7b-beavertails"

log "All pipelines complete!"
