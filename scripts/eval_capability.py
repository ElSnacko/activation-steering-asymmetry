#!/usr/bin/env python3
"""
Evaluate capability preservation of steered models.

Runs multiple-choice questions on baseline and steered models, comparing
accuracy to detect coherence degradation from steering.

Question sources (mutually exclusive):
    --questions file.json          Custom question file
    --benchmark mmlu               HuggingFace MMLU (requires `datasets`)
    --benchmark arc_easy           ARC-Easy
    --benchmark arc_challenge      ARC-Challenge
    (default)                      Built-in 50-question smoke test

Usage:
    # Quick smoke test with built-in questions
    python scripts/eval_capability.py --model Qwen/Qwen3.5-9B

    # MMLU subset (200 random questions)
    python scripts/eval_capability.py \
        --model Qwen/Qwen3.5-9B \
        --benchmark mmlu --max-questions 200

    # MMLU specific subjects
    python scripts/eval_capability.py \
        --model Qwen/Qwen3.5-9B \
        --benchmark mmlu --subjects abstract_algebra anatomy computer_security

    # List available MMLU subjects
    python scripts/eval_capability.py \
        --model Qwen/Qwen3.5-9B \
        --benchmark mmlu --subjects ""

    # Compare baseline vs steered
    python scripts/eval_capability.py \
        --model Qwen/Qwen3.5-9B \
        --benchmark mmlu --max-questions 200 \
        --steering-vectors outputs/.../steering_vectors_wrmd.pt \
        --correlations outputs/.../layer_correlations.json \
        --top-k 3 --alpha -2.5

    # ARC-Challenge
    python scripts/eval_capability.py \
        --model Qwen/Qwen3.5-9B \
        --benchmark arc_challenge \
        --steering-vectors outputs/.../steering_vectors_wrmd.pt \
        --layers 10 11 12 --alpha -2.5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from activation_steering import SteeringHook
from activation_steering.capability import (
    SUPPORTED_BENCHMARKS,
    CapabilityResult,
    compare_capability,
    compare_perplexity,
    evaluate_capability,
    evaluate_perplexity,
    load_hf_benchmark,
    load_perplexity_corpus,
    load_questions,
)
from activation_steering.kl_divergence import (
    collect_logits,
    collect_teacher_forced_logits,
    compute_kl_divergence,
    load_harmless_prompts,
)
from activation_steering.utils import (
    check_gpu_memory,
    ensure_dir,
    extract_model_name,
    generate_run_id,
    infer_run_from_path,
    resolve_model_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate capability preservation of steered models"
    )

    parser.add_argument("--model", required=True, help="Model name or path")

    # Steering (optional -- omit for baseline-only evaluation)
    parser.add_argument("--steering-vectors", help="Path to steering vectors .pt file")
    parser.add_argument("--correlations", help="Path to layer correlations .json")
    parser.add_argument("--layers", type=int, nargs="+", help="Specific layers to steer")
    parser.add_argument("--top-k", type=int, default=3, help="Use top K layers from correlations")
    parser.add_argument("--alpha", type=float, default=-2.0, help="Steering alpha")

    # Question source (mutually exclusive: --questions, --benchmark, or default built-in)
    parser.add_argument(
        "--questions",
        help="Path to custom questions JSON (default: data/capability_questions.json)",
    )
    parser.add_argument(
        "--benchmark",
        choices=list(SUPPORTED_BENCHMARKS.keys()),
        help="Load from HuggingFace benchmark (requires `datasets` package). "
        "Options: "
        + ", ".join(f"{k} ({v['description']})" for k, v in SUPPORTED_BENCHMARKS.items()),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="For MMLU: specific subjects to include (default: all). "
        "Pass empty string to list available subjects.",
    )
    parser.add_argument("--max-questions", type=int, help="Limit number of questions")

    # Generation
    parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="Max tokens per answer (MCQ needs few)"
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: auto-detect, 5 for base models, 0 for instruct)",
    )

    # Perplexity evaluation
    parser.add_argument(
        "--perplexity",
        action="store_true",
        help="Measure perplexity on a diverse text corpus (sensitive to fluency degradation)",
    )
    parser.add_argument(
        "--perplexity-corpus",
        help="Path to JSON file with text passages for perplexity (default: built-in 10 passages)",
    )

    # Generation KL divergence
    parser.add_argument(
        "--generation-kl",
        action="store_true",
        help="Measure KL divergence on open-ended generation prompts",
    )
    parser.add_argument(
        "--generation-kl-prompts",
        help="Path to JSON file with generation prompts (default: built-in 15 prompts)",
    )
    parser.add_argument(
        "--generation-kl-tokens",
        type=int,
        default=64,
        help="Tokens to generate per prompt for KL measurement (default: 64)",
    )

    # Output
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--run-id", help="Run ID")

    # Quantization
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model with BitsAndBytes 4-bit quantization",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model with BitsAndBytes 8-bit quantization "
        "(auto-triggered for FP8 checkpoints)",
    )

    args = parser.parse_args()
    args.model = resolve_model_path(args.model)
    check_gpu_memory()

    # Setup output — infer run ID from steering vectors path if available
    sv_path = getattr(args, "steering_vectors", None)
    inferred_model, inferred_run_id = infer_run_from_path(sv_path) if sv_path else (None, None)
    model_name = inferred_model or extract_model_name(args.model)
    run_id = args.run_id or inferred_run_id or generate_run_id()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs") / model_name / run_id / "eval_capability"
    ensure_dir(output_dir)

    # Load questions from one of three sources
    if args.benchmark:
        if args.questions:
            parser.error("--benchmark and --questions are mutually exclusive")
        subjects = args.subjects if args.subjects else None
        # Handle --subjects "" (empty string to list subjects)
        if subjects is not None and len(subjects) == 1 and subjects[0] == "":
            subjects = []
        print(
            f"Loading benchmark: {args.benchmark} ({SUPPORTED_BENCHMARKS[args.benchmark]['description']})"
        )
        questions = load_hf_benchmark(
            args.benchmark,
            subjects=subjects,
            max_questions=args.max_questions,
        )
        if not questions:
            return  # Subject listing mode
        print(f"Loaded {len(questions)} questions from {args.benchmark}")
    else:
        questions = load_questions(args.questions)
        if args.max_questions:
            questions = questions[: args.max_questions]
        source = args.questions or "built-in"
        print(f"Loaded {len(questions)} capability questions ({source})")

    # Load model
    from activation_steering import load_model

    quantize = None
    if getattr(args, "load_in_4bit", False):
        quantize = "4bit"
    elif getattr(args, "load_in_8bit", False):
        quantize = "8bit"

    model, tokenizer = load_model(args.model, quantize=quantize)

    # ---- Baseline evaluations ----
    print("\n--- Baseline MCQ Evaluation ---")
    baseline_result = evaluate_capability(
        model,
        tokenizer,
        questions,
        max_new_tokens=args.max_new_tokens,
        few_shot=args.few_shot,
    )
    print(
        f"Baseline accuracy: {baseline_result.accuracy:.1%} ({baseline_result.num_correct}/{baseline_result.num_total})"
    )
    for cat, stats in baseline_result.per_category.items():
        print(f"  {cat}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    # Baseline perplexity
    baseline_ppl = None
    if args.perplexity:
        print("\n--- Baseline Perplexity ---")
        corpus = load_perplexity_corpus(args.perplexity_corpus)
        baseline_ppl = evaluate_perplexity(model, tokenizer, corpus)
        print(
            f"Baseline perplexity: {baseline_ppl.perplexity:.2f} "
            f"(mean loss: {baseline_ppl.mean_loss:.4f}, {baseline_ppl.num_tokens} tokens)"
        )

    # Baseline generation KL logits
    baseline_gen_logits = None
    baseline_gen_sequences = None
    gen_kl_prompts = None
    if args.generation_kl:
        print("\n--- Baseline Generation Logits ---")
        gen_kl_prompts = load_harmless_prompts(
            path=args.generation_kl_prompts,
            tokenizer=tokenizer,
            max_prompts=15,
            prompt_set="generation",
        )
        print(
            f"Generating baseline sequences on {len(gen_kl_prompts)} generation prompts "
            f"({args.generation_kl_tokens} tokens each)..."
        )
        _, baseline_gen_sequences = collect_logits(
            model,
            tokenizer,
            gen_kl_prompts,
            max_new_tokens=args.generation_kl_tokens,
        )
        # Collect baseline logits via teacher-forcing so both baseline and
        # steered logits come from the same forward-pass path.
        print("Teacher-forcing baseline logits...")
        baseline_gen_logits = collect_teacher_forced_logits(
            model,
            tokenizer,
            gen_kl_prompts,
            baseline_gen_sequences,
        )

    # Save baseline results
    baseline_data = {"mcq": baseline_result.to_dict()}
    if baseline_ppl is not None:
        baseline_data["perplexity"] = baseline_ppl.to_dict()
    baseline_file = output_dir / "capability_baseline.json"
    with open(baseline_file, "w") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"\nBaseline saved to: {baseline_file}")

    # ---- Steered evaluation ----
    if args.steering_vectors:
        print(f"\n--- Steered Evaluation (alpha={args.alpha:+.2f}) ---")

        # Load steering vectors
        steering_data = torch.load(args.steering_vectors, weights_only=True)
        if isinstance(steering_data, dict):
            steering_vectors = steering_data["steering_vectors"]
        else:
            steering_vectors = steering_data

        # Determine layers
        if args.layers:
            layers = args.layers
        elif args.correlations:
            with open(args.correlations) as f:
                corr_data = json.load(f)
            layers = corr_data["best_layers"][: args.top_k]
        else:
            raise ValueError(
                "Must specify --layers or --correlations when using --steering-vectors"
            )

        print(f"Steering layers: {layers}, alpha: {args.alpha}")

        # Register hooks
        steerer = SteeringHook(model, steering_vectors, target_layers=layers, alpha=args.alpha)
        steerer.register_hooks()

        # Steered MCQ
        steered_result = evaluate_capability(
            model,
            tokenizer,
            questions,
            max_new_tokens=args.max_new_tokens,
            few_shot=args.few_shot,
        )
        print(
            f"Steered accuracy: {steered_result.accuracy:.1%} ({steered_result.num_correct}/{steered_result.num_total})"
        )
        for cat, stats in steered_result.per_category.items():
            print(f"  {cat}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

        # Steered perplexity
        steered_ppl = None
        ppl_comparison = None
        if args.perplexity and baseline_ppl is not None:
            print("\n--- Steered Perplexity ---")
            steered_ppl = evaluate_perplexity(model, tokenizer, corpus)
            ppl_comparison = compare_perplexity(baseline_ppl, steered_ppl)
            print(
                f"Steered perplexity: {steered_ppl.perplexity:.2f} "
                f"(baseline: {baseline_ppl.perplexity:.2f}, "
                f"ratio: {ppl_comparison['perplexity_ratio']:.3f})"
            )
            if ppl_comparison["severely_degraded"]:
                print("WARNING: Severe perplexity degradation (>50% increase)")
            elif ppl_comparison["degraded"]:
                print("WARNING: Perplexity degradation (>10% increase)")

        # Steered generation KL (teacher-forced on baseline sequences)
        gen_kl_result = None
        if args.generation_kl and baseline_gen_logits is not None:
            print("\n--- Generation KL Divergence (teacher-forced) ---")
            steered_gen_logits = collect_teacher_forced_logits(
                model,
                tokenizer,
                gen_kl_prompts,
                baseline_gen_sequences,
                show_progress=True,
            )
            gen_kl_result = compute_kl_divergence(baseline_gen_logits, steered_gen_logits)
            print(
                f"Generation KL: mean={gen_kl_result.mean_kl:.4f}, "
                f"max={gen_kl_result.max_kl:.4f}"
            )

        steerer.remove_hooks()

        # Compare MCQ
        comparison = compare_capability(baseline_result, steered_result)

        delta = comparison["accuracy_delta"]
        sign = "+" if delta >= 0 else ""
        print(f"\n{'='*60}")
        print(f"MCQ accuracy delta: {sign}{delta:.1%}")
        if comparison["severely_degraded"]:
            print("  MCQ: SEVERE degradation (>15% accuracy drop)")
        elif comparison["degraded"]:
            print("  MCQ: degradation detected (>5% accuracy drop)")
        else:
            print("  MCQ: preserved (accuracy drop <= 5%)")

        if ppl_comparison is not None:
            ratio = ppl_comparison["perplexity_ratio"]
            print(
                f"  Perplexity ratio: {ratio:.3f}x "
                f"({'DEGRADED' if ppl_comparison['degraded'] else 'OK'})"
            )

        if gen_kl_result is not None:
            print(f"  Generation KL: {gen_kl_result.mean_kl:.4f} nats")

        print(f"{'='*60}")

        print("\nPer-category MCQ deltas:")
        for cat, d in comparison["per_category"].items():
            sign = "+" if d["delta"] >= 0 else ""
            print(f"  {cat}: {d['baseline']:.1%} -> {d['steered']:.1%} ({sign}{d['delta']:.1%})")

        # Save steered results and comparison
        steered_data = {"mcq": steered_result.to_dict()}
        if steered_ppl is not None:
            steered_data["perplexity"] = steered_ppl.to_dict()
        steered_file = output_dir / "capability_steered.json"
        with open(steered_file, "w") as f:
            json.dump(steered_data, f, indent=2)

        comparison_data = {
            "alpha": args.alpha,
            "layers": layers,
            "steering_vectors": args.steering_vectors,
            "mcq": comparison,
        }
        if ppl_comparison is not None:
            comparison_data["perplexity"] = ppl_comparison
        if gen_kl_result is not None:
            comparison_data["generation_kl"] = {
                "mean_kl": gen_kl_result.mean_kl,
                "max_kl": gen_kl_result.max_kl,
                "min_kl": gen_kl_result.min_kl,
                "std_kl": gen_kl_result.std_kl,
                "num_prompts": gen_kl_result.num_prompts,
            }

        comparison_file = output_dir / "capability_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
