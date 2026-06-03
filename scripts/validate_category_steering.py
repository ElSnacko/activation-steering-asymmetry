#!/usr/bin/env python3
"""
Validate domain-specific steering vectors against the global vector.

For each stable category, runs Bayesian alpha optimization with both the global
and category-specific steering vectors, then compares optimal alpha, KL divergence,
and refusal reduction to determine whether category-specific vectors steer better.
"""

import os

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False,max_split_size_mb:512"

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).parent.parent))

from activation_steering import (
    SanitizeLogitsProcessor,
    SteeringHook,
    category_to_slug,
    collect_first_token_logits,
    collect_logits,
    collect_teacher_forced_logits,
    compute_kl_divergence,
    ensure_dir,
    extract_model_name,
    load_harmless_prompts,
    load_prompts_from_judge_scores_with_categories,
    resolve_model_path,
)
from activation_steering.steering import SteeringHookGroup

# Reuse helpers from optimize_alpha — import via importlib to avoid requiring
# scripts/ to be a package (no __init__.py)
_optimize_alpha_path = Path(__file__).parent / "optimize_alpha.py"
_spec = importlib.util.spec_from_file_location("optimize_alpha", _optimize_alpha_path)
_optimize_alpha = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_optimize_alpha)

_check_cuda_health = _optimize_alpha._check_cuda_health
_create_steerer = _optimize_alpha._create_steerer
_generate_with_timeout = _optimize_alpha._generate_with_timeout
_GenerateTimeout = _optimize_alpha._GenerateTimeout
_kill_orphaned_vllm_engines = _optimize_alpha._kill_orphaned_vllm_engines
_model_to_cuda_with_timeout = _optimize_alpha._model_to_cuda_with_timeout
compute_alpha_metrics = _optimize_alpha.compute_alpha_metrics
find_optimal_alpha = _optimize_alpha.find_optimal_alpha
load_baseline_metrics = _optimize_alpha.load_baseline_metrics
score_with_judge = _optimize_alpha.score_with_judge
strip_thinking = _optimize_alpha.strip_thinking
_save_trial_checkpoint = _optimize_alpha._save_trial_checkpoint
_load_trial_checkpoint = _optimize_alpha._load_trial_checkpoint
_replay_trials_into_study = _optimize_alpha._replay_trials_into_study


def _normalize_vectors_per_layer(vectors: torch.Tensor) -> torch.Tensor:
    """L2-normalize steering vectors per layer so alpha controls perturbation on a unit scale.

    Args:
        vectors: shape [num_layers, hidden_size] or [num_layers, rank, hidden_size]

    Returns:
        Unit-normalized vectors (same shape).
    """
    if vectors.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D steering vectors, got shape {vectors.shape}")
    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return vectors / norms


def _normalize_steering_data(steering_data: dict) -> dict:
    """Return a copy of steering_data with all vector tensors L2-normalized per layer.

    Handles keys like 'steering_vectors', 'steering_vectors_attn', 'steering_vectors_mlp'.
    """
    out = dict(steering_data)
    for key in list(out.keys()):
        if key.startswith("steering_vectors") and isinstance(out[key], torch.Tensor):
            out[key] = _normalize_vectors_per_layer(out[key])
    return out


def load_prompts_from_activations(activations_path: Path) -> Dict[str, List[dict]]:
    """Load prompts with category labels from an activations .pt file.

    The activations file produced by extract_activations.py stores prompts,
    labels (0=compliant, 1=refusal), and metadata (with optional "category" key).

    Returns:
        Dict mapping category name -> list of {"prompt": str, "label": int} dicts
    """
    data = torch.load(activations_path, map_location="cpu", weights_only=True)
    prompts = data["prompts"]
    labels = data["labels"].tolist() if hasattr(data["labels"], "tolist") else data["labels"]
    metadata = data.get("metadata", [{}] * len(prompts))

    prompts_by_category: Dict[str, List[dict]] = {}
    for prompt, label, meta in zip(prompts, labels, metadata):
        raw_cat = meta.get("category") if isinstance(meta, dict) else None
        if raw_cat is None:
            continue
        cat_names = raw_cat if isinstance(raw_cat, list) else [str(raw_cat)]
        for cat in cat_names:
            if cat not in prompts_by_category:
                prompts_by_category[cat] = []
            prompts_by_category[cat].append({"prompt": prompt, "label": label})

    print(f"[LOAD] Loaded prompts from activations: {activations_path}")
    for cat, cat_prompts in sorted(prompts_by_category.items()):
        refusal = sum(1 for p in cat_prompts if p["label"] == 1)
        compliant = sum(1 for p in cat_prompts if p["label"] == 0)
        print(f"  {cat}: {refusal} refusal, {compliant} compliant")

    return prompts_by_category


def discover_categories(
    category_vectors_dir: Path,
    prompts_by_category: Dict[str, List[dict]],
    bootstrap_file: Optional[Path] = None,
    min_samples: int = 10,
    explicit_categories: Optional[List[str]] = None,
) -> List[Tuple[str, Path, List[dict]]]:
    """Discover categories with valid vectors and sufficient prompts.

    Args:
        category_vectors_dir: Directory containing per-category .pt files + category_summary.json
        prompts_by_category: Dict mapping category name -> list of prompt dicts
        bootstrap_file: Optional path to bootstrap_stability.json for filtering
        min_samples: Minimum refusal prompts required per category
        explicit_categories: If set, only include these categories

    Returns:
        List of (category_name, vector_path, prompts_list) tuples
    """
    summary_file = category_vectors_dir / "category_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"category_summary.json not found in {category_vectors_dir}")

    with open(summary_file) as f:
        summary = json.load(f)

    # Load bootstrap stability if provided
    stability_labels = {}
    if bootstrap_file is not None:
        with open(bootstrap_file) as f:
            bootstrap_data = json.load(f)
        # Handle both formats: {"per_category": {name: {...}}} and {"categories": [{...}]}
        per_cat = bootstrap_data.get("per_category", {})
        if isinstance(per_cat, dict):
            for name, entry in per_cat.items():
                stability_labels[name] = entry.get("stability_label", "unknown")
        else:
            for entry in bootstrap_data.get("categories", []):
                stability_labels[entry["category"]] = entry.get("stability_label", "unknown")

    # Handle both formats: flat dict {name: {file, ...}} and list [{category, file, ...}]
    cat_entries = []
    if isinstance(summary, dict):
        for name, entry in summary.items():
            if name.startswith("_"):  # Skip _global
                continue
            if isinstance(entry, dict):
                cat_entries.append({"category": name, **entry})
    elif isinstance(summary, list):
        cat_entries = summary

    categories = []
    for cat_entry in cat_entries:
        name = cat_entry["category"]
        vector_file = cat_entry.get("file")

        if not vector_file:
            continue

        vector_path = category_vectors_dir / vector_file
        if not vector_path.exists():
            print(f"  [SKIP] {name}: vector file not found ({vector_file})")
            continue

        # Check explicit filter
        if explicit_categories and name not in explicit_categories:
            continue

        # Check bootstrap stability
        if bootstrap_file is not None:
            label = stability_labels.get(name, "unknown")
            if label not in ("stable", "moderate"):
                print(f"  [SKIP] {name}: stability={label} (need stable/moderate)")
                continue

        # Check prompt availability
        cat_prompts = prompts_by_category.get(name, [])
        refusal_prompts = [p for p in cat_prompts if p["label"] == 1]
        if len(refusal_prompts) < min_samples:
            print(
                f"  [SKIP] {name}: only {len(refusal_prompts)} refusal prompts "
                f"(need {min_samples})"
            )
            continue

        categories.append((name, vector_path, refusal_prompts))

    return categories


def run_category_optimization(
    model,
    tokenizer,
    prompts: List[dict],
    steering_vectors,
    layers: List[int],
    args,
    output_dir: Path,
    baseline_logits: Optional[List[torch.Tensor]],
    baseline_sequences: Optional[List[torch.Tensor]],
    kl_prompts_formatted: Optional[List[str]],
    category_name: str,
    vector_label: str,
    steering_data: Optional[dict] = None,
    resume: bool = False,
) -> Tuple[List[Dict], float]:
    """Run Bayesian optimization for a single (category, vector) pair.

    Returns:
        Tuple of (all_results, best_alpha)
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna required. Install with: pip install optuna")

    class _JudgeAbort(BaseException):
        """Raised to abort the Optuna study (not caught by catch=(Exception,))."""

        pass

    class _CudaFatalError(BaseException):
        """Raised when CUDA hits an unrecoverable error (device-side assert)."""

        pass

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    prompt_texts = [p["prompt"] for p in prompts]
    all_results = []
    trial_count = [0]
    consecutive_judge_failures = [0]

    # Always write trial-level checkpoints so progress survives crashes.
    # Only load/replay them when --resume is passed.
    ensure_dir(output_dir)
    checkpoint_path = output_dir / "trial_checkpoint.jsonl"
    prior_entries = []
    if resume:
        prior_entries = _load_trial_checkpoint(checkpoint_path)
        if prior_entries:
            all_results = [e["result"] for e in prior_entries]
            trial_count[0] = len(prior_entries)
            print(f"  [RESUME] Loaded {len(prior_entries)} prior trials from checkpoint")
    else:
        # Fresh run — clear any stale checkpoint from a previous run
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def objective(trial):
        nonlocal model, tokenizer, steering_vectors

        # Bail early if CUDA context is poisoned from a previous trial
        try:
            _check_cuda_health()
        except RuntimeError as e:
            raise _CudaFatalError(str(e)) from e

        step_kwargs = {"step": args.alpha_step / 2} if args.alpha_step is not None else {}
        alpha_param = trial.suggest_float("alpha", args.alpha_min, args.alpha_max, **step_kwargs)
        trial_count[0] += 1
        print(
            f"\n  [{vector_label}] Trial {trial_count[0]}/{args.bayesian_trials} "
            f"(alpha={alpha_param:+.2f})"
        )

        # Generate steered outputs
        steerer = _create_steerer(
            model,
            steering_vectors,
            layers,
            alpha_param,
            args.component,
            steering_data if args.component == "attn+mlp" else None,
        )
        steerer.register_hooks()

        try:
            do_sample = args.temperature > 0
            outputs = []
            for prompt_text in prompt_texts:
                messages = [{"role": "user", "content": prompt_text}]
                try:
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    formatted = prompt_text
                inputs = tokenizer(formatted, return_tensors="pt").to(
                    next(model.parameters()).device
                )
                generate_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([SanitizeLogitsProcessor()]),
                )
                if do_sample:
                    generate_kwargs.update(
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.sample_top_k,
                    )
                else:
                    generate_kwargs["do_sample"] = False
                try:
                    output_ids = _generate_with_timeout(
                        model,
                        timeout_seconds=args.generate_timeout,
                        **inputs,
                        **generate_kwargs,
                    )
                except _GenerateTimeout as e:
                    raise _CudaFatalError(str(e)) from e
                except torch.AcceleratorError as e:
                    if "device-side assert" in str(e):
                        raise _CudaFatalError(
                            f"CUDA device-side assert at alpha={alpha_param:.2f}. "
                            "CUDA context is unrecoverable — aborting study."
                        ) from e
                    raise
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                response = output_text[
                    len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :
                ].strip()
                response = strip_thinking(response, args.thinking_string)
                outputs.append(response)

            # KL divergence (while model is on GPU)
            kl_result = None
            if args.kl_divergence and baseline_logits is not None:
                if args.kl_method == "teacher_forced":
                    steered_logits = collect_teacher_forced_logits(
                        model,
                        tokenizer,
                        kl_prompts_formatted,
                        baseline_sequences,
                        show_progress=False,
                    )
                else:
                    steered_logits = collect_first_token_logits(
                        model,
                        tokenizer,
                        kl_prompts_formatted,
                        show_progress=False,
                    )
                kl_result = compute_kl_divergence(baseline_logits, steered_logits)
                print(f"    KL: mean={kl_result.mean_kl:.4f}")
                # Guard: KL beyond theoretical max (ln(vocab_size)) means
                # numerical garbage — skip the rest of this trial to avoid
                # poisoning the CUDA context with subsequent NaN/Inf ops.
                vocab_size = model.config.vocab_size
                kl_max_theoretical = math.log(vocab_size) if vocab_size else 15.0
                if (
                    not math.isfinite(kl_result.mean_kl)
                    or kl_result.mean_kl > kl_max_theoretical * 2
                ):
                    print(
                        f"    [WARN] KL={kl_result.mean_kl:.1f} exceeds "
                        f"2x theoretical max ({kl_max_theoretical:.1f}) — "
                        f"alpha={alpha_param:+.2f} is numerically unstable, "
                        f"pruning trial"
                    )
                    raise optuna.TrialPruned()
        finally:
            steerer.remove_hooks()
            del steerer  # Release model reference held by hook

        if args.no_judge:
            # KL-only mode: build result without judge scores
            result_entry = {
                "alpha": alpha_param,
                "alphas": [alpha_param],
                "direction": "bayesian",
                "metrics": {
                    "mean_score": 0.0,
                    "compliance_rate": 0.0,
                    "refusal_rate": 0.0,
                    "uncertain_rate": 0.0,
                },
                "trial": trial_count[0],
            }
            if kl_result is not None:
                result_entry["kl_divergence"] = {
                    "mean_kl": kl_result.mean_kl,
                    "max_kl": kl_result.max_kl,
                    "min_kl": kl_result.min_kl,
                    "std_kl": kl_result.std_kl,
                    "num_prompts": kl_result.num_prompts,
                    "num_tokens_avg": kl_result.num_tokens_avg,
                }
            all_results.append(result_entry)
            # In KL-only mode, minimize KL directly
            obj_val = kl_result.mean_kl if kl_result else 0.0
            _save_trial_checkpoint(
                checkpoint_path,
                trial_count[0],
                obj_val,
                {"alpha": alpha_param},
                result_entry,
            )
            return obj_val

        # Offload model to CPU to free GPU memory for judge subprocess.
        # Safe because we load with explicit .to("cuda") instead of device_map="auto".
        print("    Offloading model to CPU for judge...")
        model.cpu()
        steering_vectors = steering_vectors.cpu()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)

        free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info()
        gpu_util = max(0.3, (free_mem_bytes / total_mem_bytes) - 0.05)
        print(
            f"    [DEBUG] GPU after offload: {free_mem_bytes / 1e9:.1f} / "
            f"{total_mem_bytes / 1e9:.1f} GB free, gpu_util={gpu_util:.2f}"
        )

        try:
            judge_scores = score_with_judge(
                prompt_texts,
                outputs,
                args.judge_model,
                gpu_memory_utilization=gpu_util,
                enforce_eager=args.enforce_eager,
                judge_max_model_len=args.judge_max_model_len,
            )
            consecutive_judge_failures[0] = 0  # Reset on success
        except Exception as e:
            # Ensure model returns to GPU even if judge fails, otherwise the
            # next Optuna trial would run generation on CPU (hanging).
            print("    [ERROR] Judge failed, moving model back to GPU...")
            _kill_orphaned_vllm_engines()
            # Wait for orphaned processes to release GPU memory before moving
            # the model back. Without this, model.cuda() can OOM (the killed
            # subprocess hasn't released VRAM yet), leaving the model in a
            # split CPU/GPU state that poisons all subsequent trials.
            time.sleep(5)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Verify CUDA isn't poisoned before attempting model.cuda()
            try:
                _check_cuda_health()
            except RuntimeError as cuda_err:
                raise _CudaFatalError(
                    f"CUDA context dead after judge failure: {cuda_err}"
                ) from cuda_err
            try:
                _model_to_cuda_with_timeout(model)
            except torch.cuda.OutOfMemoryError:
                # Partial move — some params on GPU, some on CPU.
                # Force everything back to CPU, wait longer, then retry.
                print("    [WARN] OOM on model.cuda(), forcing CPU and retrying...")
                model.cpu()
                _kill_orphaned_vllm_engines()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(10)
                try:
                    _model_to_cuda_with_timeout(model)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_err:
                    # Second OOM — model is in a split state, force CPU and abort
                    model.cpu()
                    raise _CudaFatalError(
                        f"model.cuda() failed twice after judge failure: {retry_err}"
                    ) from retry_err
            except RuntimeError as cuda_err:
                raise _CudaFatalError(
                    f"model.cuda() hung after judge failure: {cuda_err}"
                ) from cuda_err
            steering_vectors = steering_vectors.cuda()
            consecutive_judge_failures[0] += 1
            if consecutive_judge_failures[0] >= args.max_judge_failures:
                raise _JudgeAbort(
                    f"Aborting: {consecutive_judge_failures[0]} consecutive judge failures. "
                    f"Last error: {e}"
                )
            raise

        # Move model back — both health check and model.cuda() can hang
        # if the CUDA context is poisoned, so both have timeouts.
        print("    Moving model back to GPU...")
        try:
            _check_cuda_health()
            _model_to_cuda_with_timeout(model)
        except RuntimeError as cuda_err:
            # Force model fully back to CPU to avoid split state
            model.cpu()
            raise _CudaFatalError(f"CUDA context dead after judge: {cuda_err}") from cuda_err
        steering_vectors = steering_vectors.cuda()

        # Compute metrics
        judge_results = [
            {"prompt": p, "response": r, "judge_score": s}
            for p, r, s in zip(prompt_texts, outputs, judge_scores)
        ]
        metrics = compute_alpha_metrics(judge_results)

        result_entry = {
            "alpha": alpha_param,
            "alphas": [alpha_param],
            "direction": "bayesian",
            "metrics": metrics,
            "trial": trial_count[0],
        }
        if kl_result is not None:
            result_entry["kl_divergence"] = {
                "mean_kl": kl_result.mean_kl,
                "max_kl": kl_result.max_kl,
                "min_kl": kl_result.min_kl,
                "std_kl": kl_result.std_kl,
                "num_prompts": kl_result.num_prompts,
                "num_tokens_avg": kl_result.num_tokens_avg,
            }
        all_results.append(result_entry)

        print(
            f"    Metrics: mean_score={metrics['mean_score']:.4f}, "
            f"compliance={metrics['compliance_rate']:.1%}"
        )

        # Compute objective value
        obj_val = metrics["mean_score"]
        if args.objective == "kl_weighted" and kl_result is not None:
            obj_val += 0.5 * min(kl_result.mean_kl, 2.0)
        elif args.objective == "maximize_compliance_rate":
            obj_val = -metrics["compliance_rate"]
        elif args.objective == "balanced":
            obj_val = metrics["mean_score"] + 0.5 * metrics["uncertain_rate"]

        _save_trial_checkpoint(
            checkpoint_path,
            trial_count[0],
            obj_val,
            {"alpha": alpha_param},
            result_entry,
        )
        return obj_val

    # Use a seed offset by the number of completed trials so the sampler
    # explores new regions on resume instead of re-proposing the same
    # candidates (frozen trials don't advance the sampler's internal RNG).
    sampler_seed = 42 + len(prior_entries)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )
    if prior_entries:
        _replay_trials_into_study(
            study,
            prior_entries,
            args.alpha_min,
            args.alpha_max,
            args.alpha_step,
        )
    remaining_trials = max(0, args.bayesian_trials - len(prior_entries))
    try:
        study.optimize(objective, n_trials=remaining_trials, n_jobs=1, catch=(Exception,))
    except _JudgeAbort as e:
        print(f"    [ABORT] {e}")
    except _CudaFatalError as e:
        print(f"    [FATAL] {e}")
        raise SystemExit(1)

    # Kill any orphaned vLLM engines left over from failed trials
    _kill_orphaned_vllm_engines()

    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None

    if best_trial is not None:
        best_alpha = best_trial.params["alpha"]
    elif all_results:
        # All trials failed in Optuna but we collected some results (KL-only)
        best_alpha = min(
            all_results, key=lambda r: r.get("kl_divergence", {}).get("mean_kl", float("inf"))
        )["alpha"]
    else:
        raise RuntimeError(
            f"All {args.bayesian_trials} trials failed for {category_name} ({vector_label}). "
            "Check judge subprocess logs."
        )
    return all_results, best_alpha


def compare_results(
    global_results: List[Dict],
    specific_results: List[Dict],
    global_best_alpha: float,
    specific_best_alpha: float,
    objective: str,
    no_judge: bool = False,
) -> Dict:
    """Compare global vs specific optimization results."""
    # Find best result entries
    global_best = min(global_results, key=lambda x: abs(x["alpha"] - global_best_alpha))
    specific_best = min(specific_results, key=lambda x: abs(x["alpha"] - specific_best_alpha))

    comparison = {
        "alpha_reduction": abs(global_best_alpha) - abs(specific_best_alpha),
        "global_optimal_alpha": global_best_alpha,
        "specific_optimal_alpha": specific_best_alpha,
    }

    # Judge-based comparison
    g_metrics = global_best.get("metrics", {})
    s_metrics = specific_best.get("metrics", {})
    if (
        g_metrics.get("compliance_rate") is not None
        and s_metrics.get("compliance_rate") is not None
    ):
        comparison["compliance_delta"] = s_metrics["compliance_rate"] - g_metrics["compliance_rate"]
        comparison["mean_score_delta"] = s_metrics.get("mean_score", 0) - g_metrics.get(
            "mean_score", 0
        )

    # KL comparison
    g_kl = global_best.get("kl_divergence", {}).get("mean_kl")
    s_kl = specific_best.get("kl_divergence", {}).get("mean_kl")
    if g_kl is not None and s_kl is not None and g_kl > 0:
        comparison["kl_reduction_pct"] = (g_kl - s_kl) / g_kl * 100
        comparison["global_kl"] = g_kl
        comparison["specific_kl"] = s_kl

    # Determine winner
    if no_judge and g_kl is not None and s_kl is not None:
        # In no-judge mode, compare purely on KL divergence
        comparison["specific_better"] = s_kl < g_kl
    elif objective == "kl_weighted" and g_kl is not None and s_kl is not None:
        g_obj = g_metrics.get("mean_score", 0) + 0.5 * min(g_kl, 2.0)
        s_obj = s_metrics.get("mean_score", 0) + 0.5 * min(s_kl, 2.0)
        comparison["specific_better"] = s_obj < g_obj
    elif g_metrics.get("mean_score") is not None and s_metrics.get("mean_score") is not None:
        comparison["specific_better"] = s_metrics["mean_score"] < g_metrics["mean_score"]
    elif s_kl is not None:
        comparison["specific_better"] = s_kl < (g_kl or float("inf"))
    else:
        comparison["specific_better"] = False

    return comparison


def print_comparison_table(category_results: Dict[str, Dict], no_judge: bool = False):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("[CATEGORY VALIDATION RESULTS]")
    print("=" * 90)

    if no_judge:
        header = f"{'Category':<30} {'Vector':<10} {'alpha*':>8} {'Mean KL':>10}"
        print(header)
        print("-" * 60)
    else:
        header = (
            f"{'Category':<30} {'Vector':<10} {'alpha*':>8} "
            f"{'Mean Score':>12} {'Compliance%':>13} {'KL Div':>10}"
        )
        print(header)
        print("-" * 90)

    for cat_name, data in sorted(category_results.items()):
        g = data["global"]
        s = data["specific"]
        comp = data["comparison"]

        g_best = min(g["all_results"], key=lambda x: abs(x["alpha"] - g["optimal_alpha"]))
        s_best = min(s["all_results"], key=lambda x: abs(x["alpha"] - s["optimal_alpha"]))

        if no_judge:
            g_kl = g_best.get("kl_divergence", {}).get("mean_kl", float("nan"))
            s_kl = s_best.get("kl_divergence", {}).get("mean_kl", float("nan"))
            print(f"{cat_name:<30} {'global':<10} {g['optimal_alpha']:>+8.2f} {g_kl:>10.4f}")
            print(f"{'':<30} {'specific':<10} {s['optimal_alpha']:>+8.2f} {s_kl:>10.4f}")
        else:
            g_m = g_best.get("metrics", {})
            s_m = s_best.get("metrics", {})
            g_kl = g_best.get("kl_divergence", {}).get("mean_kl", float("nan"))
            s_kl = s_best.get("kl_divergence", {}).get("mean_kl", float("nan"))

            print(
                f"{cat_name:<30} {'global':<10} {g['optimal_alpha']:>+8.2f} "
                f"{g_m.get('mean_score', 0):>12.4f} "
                f"{g_m.get('compliance_rate', 0)*100:>12.1f}% "
                f"{g_kl:>10.4f}"
            )
            print(
                f"{'':<30} {'specific':<10} {s['optimal_alpha']:>+8.2f} "
                f"{s_m.get('mean_score', 0):>12.4f} "
                f"{s_m.get('compliance_rate', 0)*100:>12.1f}% "
                f"{s_kl:>10.4f}"
            )

        # Winner annotation
        if comp.get("specific_better"):
            parts = []
            if "alpha_reduction" in comp:
                parts.append(f"alpha reduced by {comp['alpha_reduction']:.2f}")
            if "kl_reduction_pct" in comp:
                parts.append(f"KL reduced by {comp['kl_reduction_pct']:.1f}%")
            print(f"  -> specific better: {', '.join(parts)}")
        else:
            print(f"  -> global better or equal")
        print()


def plot_comparison(category_results: Dict[str, Dict], output_path: Path, no_judge: bool = False):
    """Create a grouped bar chart comparing global vs specific KL at optimal alpha."""
    categories = sorted(category_results.keys())
    if not categories:
        return

    global_kls = []
    specific_kls = []
    for cat in categories:
        data = category_results[cat]
        g_best = min(
            data["global"]["all_results"],
            key=lambda r: abs(r["alpha"] - data["global"]["optimal_alpha"]),
        )
        s_best = min(
            data["specific"]["all_results"],
            key=lambda r: abs(r["alpha"] - data["specific"]["optimal_alpha"]),
        )
        global_kls.append(g_best.get("kl_divergence", {}).get("mean_kl", 0))
        specific_kls.append(s_best.get("kl_divergence", {}).get("mean_kl", 0))

    x = np.arange(len(categories))
    width = 0.35

    fig, axes = plt.subplots(1, 2 if not no_judge else 1, figsize=(14, 6), squeeze=False)

    # KL divergence bars
    ax = axes[0, 0]
    ax.bar(x - width / 2, global_kls, width, label="Global vector", color="#4C72B0")
    ax.bar(x + width / 2, specific_kls, width, label="Specific vector", color="#DD8452")
    ax.set_xlabel("Category")
    ax.set_ylabel("Mean KL Divergence (nats)")
    ax.set_title("KL Divergence at Optimal Alpha: Global vs Specific")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if not no_judge:
        # Compliance rate bars
        ax2 = axes[0, 1]
        global_comp = []
        specific_comp = []
        for cat in categories:
            data = category_results[cat]
            g_best = min(
                data["global"]["all_results"],
                key=lambda r: abs(r["alpha"] - data["global"]["optimal_alpha"]),
            )
            s_best = min(
                data["specific"]["all_results"],
                key=lambda r: abs(r["alpha"] - data["specific"]["optimal_alpha"]),
            )
            global_comp.append(g_best.get("metrics", {}).get("compliance_rate", 0) * 100)
            specific_comp.append(s_best.get("metrics", {}).get("compliance_rate", 0) * 100)

        ax2.bar(x - width / 2, global_comp, width, label="Global vector", color="#4C72B0")
        ax2.bar(x + width / 2, specific_comp, width, label="Specific vector", color="#DD8452")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("Compliance Rate (%)")
        ax2.set_title("Compliance at Optimal Alpha: Global vs Specific")
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate category-specific vs global steering vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--global-vectors", required=True, help="Path to global steering vectors .pt file"
    )
    parser.add_argument(
        "--category-vectors-dir",
        required=True,
        help="Directory with per-category .pt files + category_summary.json",
    )

    # Prompt sourcing (one of these two approaches)
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument(
        "--activations",
        type=str,
        help="Path to activations .pt file (with category metadata from extract_activations.py)",
    )
    prompt_source.add_argument(
        "--baseline-results",
        type=str,
        help="Path to baseline LLM-Refusal-Evaluation results (requires --dataset)",
    )
    parser.add_argument("--dataset", type=str, help="HF dataset for category metadata")

    # Layer selection
    parser.add_argument("--layers", type=int, nargs="+", help="Explicit layer indices")
    parser.add_argument("--correlations", type=str, help="Path to layer_correlations.json")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K layers from correlations")

    # Alpha range
    parser.add_argument("--alpha-min", type=float, default=-5.0)
    parser.add_argument("--alpha-max", type=float, default=0.0)
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=None,
        help="Discrete step for Bayesian search (default: continuous)",
    )
    parser.add_argument("--bayesian-trials", type=int, default=10, help="Trials per Optuna study")
    parser.add_argument(
        "--objective",
        choices=["minimize_refusal", "maximize_compliance_rate", "balanced", "kl_weighted"],
        default="minimize_refusal",
    )

    # Category filtering
    parser.add_argument("--bootstrap-stability", type=str, help="Path to bootstrap_stability.json")
    parser.add_argument("--min-category-samples", type=int, default=10)
    parser.add_argument("--categories", nargs="+", help="Explicit category list")

    # Vector normalization
    parser.add_argument(
        "--normalize-vectors",
        action="store_true",
        default=False,
        help="L2-normalize steering vectors per-layer before optimization so that "
        "alpha represents comparable perturbation strength across vectors with "
        "different magnitudes (global vs category-specific)",
    )

    # KL divergence
    parser.add_argument(
        "--kl-divergence", action="store_true", default=True, help="Compute KL divergence"
    )
    parser.add_argument("--no-kl-divergence", action="store_false", dest="kl_divergence")
    parser.add_argument(
        "--kl-method",
        choices=["teacher_forced", "first_token"],
        default="teacher_forced",
        help="KL measurement method: teacher_forced (multi-token, default) "
        "or first_token (single forward pass, fastest)",
    )
    parser.add_argument(
        "--kl-tokens",
        type=int,
        default=32,
        help="Tokens to generate for teacher-forced KL (ignored for first_token) (default: 32)",
    )
    parser.add_argument("--num-kl-prompts", type=int, default=20)
    parser.add_argument("--kl-prompts", type=str, help="Path to custom KL prompts JSON")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--generate-timeout",
        type=int,
        default=120,
        help="Timeout in seconds per prompt for model.generate() (default: 120). "
        "Aborts the study if generation hangs (e.g. poisoned CUDA context).",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--sample-top-k", type=int, default=20)
    parser.add_argument("--thinking-string", type=str, default=None)

    # Judge
    parser.add_argument("--judge-model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="Skip judge scoring (KL-only mode)")
    parser.add_argument(
        "--judge-max-model-len",
        type=int,
        default=None,
        help="Max sequence length for vLLM judge. Default: 4096.",
    )
    parser.add_argument(
        "--max-judge-failures",
        type=int,
        default=3,
        help="Abort Optuna study after this many consecutive judge failures (default: 3)",
    )

    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--run-id", type=str, help="Run ID (default: timestamp)")
    parser.add_argument("--component", type=str, default="attn", help="Steering component")
    parser.add_argument(
        "--log-file",
        type=str,
        default="auto",
        help="Log file path. 'auto' = output_dir/experiment.log, 'none' = disable (default: auto)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint — skip categories already in category_checkpoint.json",
    )

    # Dataset columns
    parser.add_argument("--prompt-column", type=str, default="prompt")
    parser.add_argument("--category-column", type=str, default="category")
    parser.add_argument("--dataset-split", type=str, default=None)

    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    model_name = extract_model_name(args.model)
    if not args.no_judge:
        args.judge_model = resolve_model_path(args.judge_model)

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_id = args.run_id or __import__("activation_steering").generate_run_id()
        output_dir = Path("outputs") / model_name / run_id / "validate_category"
    ensure_dir(output_dir)

    # Set up experiment logging — tee stdout+stderr to a log file
    if args.log_file != "none":
        log_path = Path(args.log_file) if args.log_file != "auto" else output_dir / "experiment.log"
        ensure_dir(log_path.parent)

        class _Tee:
            """Write to both a file and the original stream."""

            def __init__(self, stream, log_file):
                self._stream = stream
                self._log = log_file

            def write(self, data):
                self._stream.write(data)
                self._log.write(data)
                self._log.flush()

            def flush(self):
                self._stream.flush()
                self._log.flush()

            def fileno(self):
                return self._stream.fileno()

            def isatty(self):
                return self._stream.isatty()

        _log_fh = open(log_path, "w")  # overwrite previous log
        sys.stdout = _Tee(sys.__stdout__, _log_fh)
        sys.stderr = _Tee(sys.__stderr__, _log_fh)

        # Ensure file handle is closed on exit
        def _cleanup_log():
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            _log_fh.close()

        import atexit

        atexit.register(_cleanup_log)
        print(f"[LOG] Logging to {log_path}")

    print(f"[CONFIG] Model: {args.model}")
    print(f"[CONFIG] Output: {output_dir}")
    print(f"[CONFIG] Bayesian trials per study: {args.bayesian_trials}")
    print(f"[CONFIG] Objective: {args.objective}")
    print(f"[CONFIG] Judge: {'disabled' if args.no_judge else args.judge_model}")
    print(f"[CONFIG] Normalize vectors: {args.normalize_vectors}")
    print(f"[CONFIG] KL divergence: {args.kl_divergence}")

    # Load layers
    if args.layers:
        layers = args.layers
    elif args.correlations:
        from activation_steering import load_best_layers_from_correlations

        layers = load_best_layers_from_correlations(args.correlations, top_k=args.top_k)
    else:
        parser.error("Must specify --layers or --correlations")

    print(f"[CONFIG] Layers: {layers}")

    # Load prompts with categories
    print("\n[STEP 1] Loading prompts with category metadata...")
    if args.activations:
        # Load directly from activations file (has category metadata from extraction)
        prompts_by_category = load_prompts_from_activations(Path(args.activations))
    else:
        # Cross-reference judge results with HF dataset
        if not args.dataset:
            parser.error("--dataset is required when using --baseline-results")
        prompts, labels, metadata = load_prompts_from_judge_scores_with_categories(
            results_dir=args.baseline_results,
            dataset_name=args.dataset,
            prompt_column=args.prompt_column,
            category_column=args.category_column,
            dataset_split=args.dataset_split,
        )
        prompts_by_category: Dict[str, List[dict]] = {}
        for i, (prompt, label) in enumerate(zip(prompts, labels)):
            raw_cat = metadata[i].get("category")
            if raw_cat is None:
                continue
            cat_names = raw_cat if isinstance(raw_cat, list) else [str(raw_cat)]
            for cat in cat_names:
                if cat not in prompts_by_category:
                    prompts_by_category[cat] = []
                prompts_by_category[cat].append({"prompt": prompt, "label": label})

    # Discover categories
    print("\n[STEP 2] Discovering categories...")
    category_vectors_dir = Path(args.category_vectors_dir)
    bootstrap_file = Path(args.bootstrap_stability) if args.bootstrap_stability else None

    discovered = discover_categories(
        category_vectors_dir,
        prompts_by_category,
        bootstrap_file=bootstrap_file,
        min_samples=args.min_category_samples,
        explicit_categories=args.categories,
    )

    if not discovered:
        print("[ERROR] No valid categories found. Check --category-vectors-dir and prompt data.")
        sys.exit(1)

    print(f"\n[OK] {len(discovered)} categories to validate:")
    for name, vpath, cat_prompts in discovered:
        print(f"  {name}: {len(cat_prompts)} refusal prompts, vector={vpath.name}")

    # Kill any stale vLLM engines from previous crashed runs before loading
    # the model — they hold GPU memory as zombies and cause OOM.
    if _kill_orphaned_vllm_engines():
        # Wait for killed processes to actually release GPU memory.  A zombie's
        # GPU memory is only freed when the kernel reaps it, which requires the
        # parent to be dead too (so init inherits the zombie and reaps it).
        # Poll nvidia-smi to confirm memory is freed rather than relying on a
        # fixed sleep which may not be long enough.
        import subprocess as sp

        print("  Waiting for GPU memory to be released...")
        for attempt in range(12):  # up to ~30s
            time.sleep(2.5)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                result = sp.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                free_mb = int(result.stdout.strip().split("\n")[0])
                # Qwen3.5-9B in bfloat16 needs ~18GB; ensure enough headroom
                if free_mb > 20000:
                    print(f"  GPU memory freed: {free_mb} MiB available")
                    break
                print(f"  Still waiting... ({free_mb} MiB free, need ~20000)")
            except Exception:
                pass
        else:
            print("  [WARN] GPU memory may not be fully released — proceeding anyway")

    # Load model
    print(f"\n[STEP 3] Loading model: {model_path}")
    # Use explicit device placement instead of device_map="auto" so that
    # model.cpu()/model.cuda() work correctly for judge offloading.
    # device_map="auto" creates an accelerate dispatch table that corrupts
    # on repeated cpu/cuda round-trips.
    #
    # Check for FP8 checkpoint — these cannot be loaded correctly by transformers
    # (scale factors are ignored), and BnB quantization is incompatible with the
    # model.cpu()/cuda() judge offloading pattern used here.
    from activation_steering.utils import _detect_fp8_checkpoint

    if _detect_fp8_checkpoint(model_path):
        print(
            "\n[ERROR] FP8 checkpoint detected. validate_category_steering.py requires "
            "model.cpu()/cuda() for judge offloading, which is incompatible with "
            "BitsAndBytes quantization.\n"
            "Options:\n"
            "  1. Use a non-FP8 checkpoint (BF16/FP16)\n"
            "  2. Use optimize_alpha.py --stable-categories instead (supports quantized models)\n"
        )
        sys.exit(1)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load global vectors
    print(f"\n[STEP 4] Loading global steering vectors: {args.global_vectors}")
    global_steering_data = torch.load(args.global_vectors, map_location="cpu", weights_only=True)
    # Extract vectors for the target component
    if args.component == "attn" and "steering_vectors_attn" in global_steering_data:
        global_vectors = global_steering_data["steering_vectors_attn"]
    elif args.component == "mlp" and "steering_vectors_mlp" in global_steering_data:
        global_vectors = global_steering_data["steering_vectors_mlp"]
    elif "steering_vectors" in global_steering_data:
        global_vectors = global_steering_data["steering_vectors"]
    else:
        # Try the first tensor we find
        for key in global_steering_data:
            if isinstance(global_steering_data[key], torch.Tensor):
                global_vectors = global_steering_data[key]
                break
        else:
            raise ValueError(f"No steering vectors found in {args.global_vectors}")
    # Use first parameter's device since model.device is unreliable with device_map="auto"
    _model_device = next(model.parameters()).device
    if args.normalize_vectors:
        global_norms = global_vectors.norm(dim=-1)
        print(
            f"  Global vector norms: mean={global_norms.mean():.4f}, "
            f"range=[{global_norms.min():.4f}, {global_norms.max():.4f}]"
        )
        global_vectors = _normalize_vectors_per_layer(global_vectors)
        global_steering_data = _normalize_steering_data(global_steering_data)
        print(f"  Normalized global vectors to unit length per layer")
    global_vectors = global_vectors.to(_model_device)
    print(f"  Global vectors shape: {global_vectors.shape}")

    # Collect KL baseline logits (no steering)
    baseline_logits = None
    baseline_sequences = None
    kl_prompts_formatted = None
    if args.kl_divergence:
        print(f"\n[STEP 5] Collecting baseline KL logits (method: {args.kl_method})...")
        kl_prompts_formatted = load_harmless_prompts(
            path=args.kl_prompts,
            tokenizer=tokenizer,
            max_prompts=args.num_kl_prompts,
        )
        if args.kl_method == "teacher_forced":
            print(f"  Generating baseline sequences ({args.kl_tokens} tokens, no steering)...")
            _, baseline_sequences = collect_logits(
                model,
                tokenizer,
                kl_prompts_formatted,
                max_new_tokens=args.kl_tokens,
            )
            # Collect baseline logits via teacher-forcing so both baseline and
            # steered logits come from the same forward-pass path.
            print(f"  Teacher-forcing baseline logits on {len(baseline_sequences)} prompts...")
            baseline_logits = collect_teacher_forced_logits(
                model,
                tokenizer,
                kl_prompts_formatted,
                baseline_sequences,
            )
            print(
                f"  Collected baseline logits for {len(baseline_logits)} prompts "
                f"({args.kl_tokens} tokens each)"
            )
        else:
            baseline_logits = collect_first_token_logits(
                model,
                tokenizer,
                kl_prompts_formatted,
            )
            print(f"  Collected baseline first-token logits for {len(baseline_logits)} prompts")

    # Main loop: optimize each category with global and specific vectors
    print(f"\n[STEP 6] Running category validation...")
    category_results = {}
    checkpoint_path = output_dir / "category_checkpoint.json"

    # Resume from checkpoint if available
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            category_results = json.load(f)
        print(f"[RESUME] Loaded {len(category_results)} completed categories from checkpoint")

    for cat_idx, (cat_name, cat_vector_path, cat_prompts) in enumerate(discovered):
        if cat_name in category_results:
            print(f"\n[SKIP] {cat_name}: already in checkpoint")
            continue

        print(f"\n{'='*70}")
        print(f"[{cat_idx+1}/{len(discovered)}] Category: {cat_name} ({len(cat_prompts)} prompts)")
        print(f"{'='*70}")

        # Load category-specific vectors
        cat_steering_data = torch.load(cat_vector_path, map_location="cpu", weights_only=True)
        if args.component == "attn" and "steering_vectors_attn" in cat_steering_data:
            cat_vectors = cat_steering_data["steering_vectors_attn"]
        elif args.component == "mlp" and "steering_vectors_mlp" in cat_steering_data:
            cat_vectors = cat_steering_data["steering_vectors_mlp"]
        elif "steering_vectors" in cat_steering_data:
            cat_vectors = cat_steering_data["steering_vectors"]
        else:
            for key in cat_steering_data:
                if isinstance(cat_steering_data[key], torch.Tensor):
                    cat_vectors = cat_steering_data[key]
                    break
            else:
                print(f"  [ERROR] No vectors in {cat_vector_path}, skipping")
                continue
        if args.normalize_vectors:
            cat_norms = cat_vectors.norm(dim=-1)
            print(
                f"  Specific vector norms: mean={cat_norms.mean():.4f}, "
                f"range=[{cat_norms.min():.4f}, {cat_norms.max():.4f}]"
            )
            cat_vectors = _normalize_vectors_per_layer(cat_vectors)
            cat_steering_data = _normalize_steering_data(cat_steering_data)
            print(f"  Normalized specific vectors to unit length per layer")
        # Keep category vectors on CPU during global optimization to save VRAM
        cat_vectors_cpu = cat_vectors

        # Check for saved global results from a prior crashed run.
        # This avoids re-running the global optimization if it completed
        # before the script crashed during the specific optimization.
        cat_output_dir = output_dir / category_to_slug(cat_name)
        global_results_file = cat_output_dir / "global" / "optimization_result.json"
        if args.resume and global_results_file.exists():
            print(f"\n  --- Loading GLOBAL results from checkpoint ---")
            with open(global_results_file) as f:
                saved_global = json.load(f)
            global_results = saved_global["all_results"]
            global_best_alpha = saved_global["optimal_alpha"]
            print(
                f"  [RESUME] Loaded global results: {len(global_results)} trials, "
                f"best alpha={global_best_alpha:+.2f}"
            )
        else:
            # Optimize with GLOBAL vector on this category's prompts
            print(f"\n  --- Optimizing with GLOBAL vector ---")
            global_results, global_best_alpha = run_category_optimization(
                model=model,
                tokenizer=tokenizer,
                prompts=cat_prompts,
                steering_vectors=global_vectors,
                layers=layers,
                args=args,
                output_dir=cat_output_dir / "global",
                baseline_logits=baseline_logits,
                baseline_sequences=baseline_sequences,
                kl_prompts_formatted=kl_prompts_formatted,
                category_name=cat_name,
                vector_label="global",
                steering_data=global_steering_data if args.component == "attn+mlp" else None,
                resume=args.resume,
            )
            # Save global results immediately so they survive a crash
            # before the specific optimization completes.
            ensure_dir(cat_output_dir / "global")
            with open(global_results_file, "w") as f:
                json.dump(
                    {"optimal_alpha": global_best_alpha, "all_results": global_results},
                    f,
                    indent=2,
                )

        # Move category vectors to GPU for specific optimization
        cat_vectors = cat_vectors_cpu.to(next(model.parameters()).device)

        # Optimize with CATEGORY-SPECIFIC vector
        print(f"\n  --- Optimizing with SPECIFIC vector ---")
        specific_results, specific_best_alpha = run_category_optimization(
            model=model,
            tokenizer=tokenizer,
            prompts=cat_prompts,
            steering_vectors=cat_vectors,
            layers=layers,
            args=args,
            output_dir=output_dir / category_to_slug(cat_name) / "specific",
            baseline_logits=baseline_logits,
            baseline_sequences=baseline_sequences,
            kl_prompts_formatted=kl_prompts_formatted,
            category_name=cat_name,
            vector_label="specific",
            steering_data=cat_steering_data if args.component == "attn+mlp" else None,
            resume=args.resume,
        )

        # Compare
        comparison = compare_results(
            global_results,
            specific_results,
            global_best_alpha,
            specific_best_alpha,
            args.objective,
            no_judge=args.no_judge,
        )

        category_results[cat_name] = {
            "num_prompts": len(cat_prompts),
            "global": {
                "optimal_alpha": global_best_alpha,
                "all_results": global_results,
            },
            "specific": {
                "optimal_alpha": specific_best_alpha,
                "all_results": specific_results,
            },
            "comparison": comparison,
        }

        # Checkpoint after each category so progress survives crashes
        with open(checkpoint_path, "w") as f:
            json.dump(category_results, f, indent=2)
        print(f"  [CHECKPOINT] Saved {len(category_results)} categories to {checkpoint_path}")

    # Free GPU resources before reporting results — without this, PyTorch CUDA
    # cleanup during interpreter shutdown can hang indefinitely.
    del model, tokenizer, global_vectors, baseline_logits, baseline_sequences
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Print comparison table
    print_comparison_table(category_results, no_judge=args.no_judge)

    # Summary statistics
    num_tested = len(category_results)
    num_specific_better = sum(
        1 for d in category_results.values() if d["comparison"].get("specific_better")
    )

    if num_specific_better > num_tested / 2:
        recommendation = (
            f"Category-specific vectors are better for {num_specific_better}/{num_tested} "
            f"categories. Recommend using per-category steering where available."
        )
    elif num_specific_better > 0:
        recommendation = (
            f"Mixed results: specific vectors better for {num_specific_better}/{num_tested} "
            f"categories. Consider per-category steering for those categories only."
        )
    else:
        recommendation = (
            f"Global vector is better or equal for all {num_tested} categories. "
            f"Per-category steering provides no benefit."
        )

    print(f"\n[SUMMARY] {recommendation}")

    # Save JSON
    output_json = {
        "config": {
            "model": args.model,
            "global_vectors": args.global_vectors,
            "category_vectors_dir": args.category_vectors_dir,
            "layers": layers,
            "alpha_range": [args.alpha_min, args.alpha_max, args.alpha_step],
            "bayesian_trials": args.bayesian_trials,
            "objective": args.objective,
            "component": args.component,
            "kl_divergence": args.kl_divergence,
            "normalize_vectors": args.normalize_vectors,
            "no_judge": args.no_judge,
        },
        "categories": {},
        "summary": {
            "categories_tested": num_tested,
            "specific_better_count": num_specific_better,
            "recommendation": recommendation,
        },
    }

    # Serialize category results (make tensors JSON-safe)
    for cat_name, data in category_results.items():
        cat_json = {
            "num_prompts": data["num_prompts"],
            "global": {
                "optimal_alpha": data["global"]["optimal_alpha"],
                "all_results": data["global"]["all_results"],
            },
            "specific": {
                "optimal_alpha": data["specific"]["optimal_alpha"],
                "all_results": data["specific"]["all_results"],
            },
            "comparison": data["comparison"],
        }
        output_json["categories"][cat_name] = cat_json

    json_path = output_dir / "category_validation.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"[SAVE] Results saved to {json_path}")

    # Plot
    plot_path = output_dir / "category_validation_summary.png"
    plot_comparison(category_results, plot_path, no_judge=args.no_judge)

    print(f"\n[DONE] Category validation complete.")
    print(f"  Categories tested: {num_tested}")
    print(f"  Specific better: {num_specific_better}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
