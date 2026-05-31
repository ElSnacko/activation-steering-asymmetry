#!/usr/bin/env python3
"""
CLI script for extracting activations from language models.

This script provides a command-line interface to the ActivationExtractor class.
"""

import argparse
import os
import random
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from activation_steering import (
    ActivationExtractor,
    analyze_dataset_quality,
    check_gpu_memory,
    ensure_dir,
    extract_model_name,
    generate_run_id,
    load_prompts_from_judge_scores,
    resolve_model_path,
    setup_model_run_dirs,
)
from activation_steering.extraction import load_prompts_from_judge_scores_with_categories


def main():
    parser = argparse.ArgumentParser(description="Extract activations using judge scores")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to evaluation results (e.g., LLM-Refusal-Evaluation/results/qwen3_8b_baseline)",
    )
    parser.add_argument("--output", default="activations.pt")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory (default: outputs/{model_name}/{run_id}/extract_activations/)",
    )
    parser.add_argument(
        "--run-id", default=None, help="Custom run ID (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--refusal-threshold", type=float, default=0.1, help="Judge score above this = refusal"
    )
    parser.add_argument(
        "--compliance-threshold",
        type=float,
        default=-0.1,
        help="Judge score below this = compliant",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling samples (default: 42). Set to -1 for no shuffle.",
    )
    parser.add_argument(
        "--components",
        default="attn",
        help="Which submodule activations to extract: layer, attn, mlp, attn+mlp, or all "
        "(default: attn)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset to cross-reference for category metadata "
        "(e.g., PKU-Alignment/BeaverTails-Evaluation)",
    )
    parser.add_argument(
        "--prompt-column",
        default="prompt",
        help="Column name for prompt text in HF dataset (default: prompt)",
    )
    parser.add_argument(
        "--category-column",
        default="category",
        help="Column name for category in HF dataset (default: category)",
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="Dataset split to use (default: auto-detect)",
    )
    parser.add_argument(
        "--dataset-categories",
        nargs="+",
        default=None,
        help="Only include prompts from these categories",
    )
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts per forward pass (default: 1, serial). "
        "Set > 1 for higher GPU utilization with similar-length prompts.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode in chat template (default: off). Only relevant for "
        "reasoning models like Qwen3; ignored by models without thinking support.",
    )
    args = parser.parse_args()

    # Parse components string into list
    if args.components == "all":
        args.components = ["layer", "attn", "mlp"]
    else:
        args.components = [c.strip() for c in args.components.split("+")]

    # Resolve model path to absolute (avoids huggingface_hub validation errors)
    args.model = resolve_model_path(args.model)

    # Check GPU memory before loading model
    check_gpu_memory()

    # Set up output directory
    if args.output_dir is None:
        model_name = extract_model_name(args.model)
        run_id = args.run_id or generate_run_id()
        output_dirs = setup_model_run_dirs(model_name=model_name, run_id=run_id)
        output_dir = output_dirs["extract_activations"]
        print(f"📁 Using output directory: {output_dir}")
        print(f"   Model: {model_name}")
        print(f"   Run ID: {run_id}")
    else:
        output_dir = ensure_dir(args.output_dir)

    # Load prompts based on actual judge scores
    if args.dataset:
        prompts, labels, metadata = load_prompts_from_judge_scores_with_categories(
            args.results_dir,
            dataset_name=args.dataset,
            prompt_column=args.prompt_column,
            category_column=args.category_column,
            dataset_split=args.dataset_split,
            refusal_threshold=args.refusal_threshold,
            compliance_threshold=args.compliance_threshold,
        )
        # Apply --dataset-categories filter if specified
        if args.dataset_categories:
            combined = [
                (p, l, m)
                for p, l, m in zip(prompts, labels, metadata)
                if m.get("category") in args.dataset_categories
            ]
            if not combined:
                print(
                    f"[ERROR] --dataset-categories filter removed all samples. "
                    f"Check category names."
                )
                sys.exit(1)
            prompts, labels, metadata = map(list, zip(*combined))
            print(f"[INFO] After --dataset-categories filter: {len(prompts)} samples remain")
    else:
        prompts, labels, metadata = load_prompts_from_judge_scores(
            args.results_dir, args.refusal_threshold, args.compliance_threshold
        )

    # Analyze dataset quality
    analyze_dataset_quality(metadata)

    # Shuffle and optionally limit samples.
    # Shuffling ensures random selection per category rather than taking the first
    # N samples in dataset order (which is split-by-split, potentially ordered by category).
    if args.seed is not None and args.seed >= 0:
        print(f"\n[INFO] Shuffling samples with seed={args.seed}")
        rng = random.Random(args.seed)
        combined = list(zip(prompts, labels, metadata))
        rng.shuffle(combined)
        prompts, labels, metadata = map(list, zip(*combined))

    if args.max_samples:
        print(f"[WARN]  Limiting to {args.max_samples} samples (randomly selected)")
        prompts = prompts[: args.max_samples]
        labels = labels[: args.max_samples]
        metadata = metadata[: args.max_samples]

    # Extract activations
    quantize = None
    if getattr(args, "load_in_4bit", False):
        quantize = "4bit"
    elif getattr(args, "load_in_8bit", False):
        quantize = "8bit"
    extractor = ActivationExtractor(
        args.model,
        components=args.components,
        quantize=quantize,
        enable_thinking=args.enable_thinking,
    )
    extractor.extract_dataset(
        prompts, labels, args.output, metadata, output_dir, batch_size=args.batch_size
    )

    # Determine the actual output path
    if not os.path.isabs(args.output):
        actual_output_path = os.path.join(output_dir, args.output)
    else:
        actual_output_path = args.output

    print(f"\n[OK] Done! Activations saved to {actual_output_path}")


if __name__ == "__main__":
    main()
