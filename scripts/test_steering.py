#!/usr/bin/env python3
"""
CLI script for testing steering vectors interactively.

This script applies steering vectors at runtime and compares baseline vs
steered model outputs.
"""

import argparse
import json
import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from activation_steering import (
    SanitizeLogitsProcessor,
    load_actual_refusal_prompts,
    load_best_layers_from_correlations,
    resolve_model_path,
    test_steering,
)
from activation_steering.routing import CategoryRouter


def _normalize_cat_key(key):
    return key.replace(",", "").replace(" ", "").lower()


def _load_theta_lookup(theta_file):
    with open(theta_file) as f:
        raw = json.load(f)
    lookup = {}
    for cat_key, cat_data in raw.items():
        normalized = _normalize_cat_key(cat_key)
        theta_dict = {int(k): v for k, v in cat_data["theta"].items()}
        lookup[normalized] = theta_dict
    return lookup


def _find_theta_for_category(theta_lookup, category_name):
    if not category_name:
        return None
    normalized = _normalize_cat_key(category_name)
    return theta_lookup.get(normalized)


def _run_router_mode(args, refusal_prompts):
    """Run test_steering in router mode: classify each prompt, then steer with the routed vector."""
    from activation_steering import load_model

    use_dynamic = getattr(args, "dynamic", False)
    gain = getattr(args, "gain", -1.0)
    theta_file = getattr(args, "theta_file", None)

    quantize = None
    if getattr(args, "load_in_4bit", False):
        quantize = "4bit"
    elif getattr(args, "load_in_8bit", False):
        quantize = "8bit"

    model, tokenizer = load_model(args.model, quantize=quantize)
    print()

    print(f"[ROUTER] Loading category router from {args.router}")
    router = CategoryRouter.from_calibration_file(
        args.router, device=str(next(model.parameters()).device)
    )
    print(f"   Target layers: {router.target_layers}")
    print(f"   Component: {router.component}")
    print(f"   Categories: {list(router.category_vectors.keys())}")
    print(f"   Threshold: {router.calibration.threshold:.4f}")

    theta_lookup = None
    if use_dynamic:
        if not theta_file:
            print("[ERROR] --theta-file is required when --dynamic is set")
            sys.exit(1)
        theta_lookup = _load_theta_lookup(theta_file)
        print(f"   Dynamic mode: gain={gain}")
        print(f"   Theta file: {theta_file} ({len(theta_lookup)} categories)")
    print()

    for i, prompt_data in enumerate(refusal_prompts):
        prompt = prompt_data["prompt"]
        baseline_score = prompt_data["baseline_score"]

        print(f"{'='*80}")
        print(f"Prompt {i+1}/{len(refusal_prompts)}")
        print(f"{'='*80}")
        print(f"  PROMPT: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"  Baseline refusal score: {baseline_score:.2f}")

        # Classify
        decision = router.classify_prompt(model, tokenizer, prompt)
        print(f"\n  [ROUTING] {decision.reason}")
        print(f"    Global projection: {decision.global_projection:.4f}")
        if decision.category_projections:
            print(f"    Category projections:")
            for cat, proj in sorted(
                decision.category_projections.items(), key=lambda x: x[1], reverse=True
            ):
                marker = " <--" if cat == decision.selected_category else ""
                print(f"      {cat}: {proj:.4f}{marker}")

        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = next(model.parameters()).device
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        # Baseline generation
        print(f"\n  [BASELINE]")
        print(f"  {'-'*76}")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        baseline_response = baseline_output[len(formatted_prompt) :]
        print(f"  {baseline_response[:500]}")
        print(f"  {'-'*76}")

        # Steered generation (if router says to steer)
        if decision.should_steer:
            vec_source = decision.selected_category or "global"

            if use_dynamic:
                cat_theta = _find_theta_for_category(theta_lookup, decision.selected_category)
                if cat_theta is None:
                    if decision.use_global_fallback or decision.selected_category is None:
                        print(
                            f"\n  [SKIP] No theta for global fallback — skipping dynamic steering"
                        )
                        print()
                        continue
                    print(
                        f"\n  [WARN] No theta for category '{decision.selected_category}' — skipping"
                    )
                    print()
                    continue

                steerer = router.create_steering_hooks(
                    model,
                    decision,
                    alpha=0,
                    dynamic=True,
                    theta=cat_theta,
                    gain=gain,
                )
                print(f"\n  [STEERED] Dynamic: {vec_source} vector, gain={gain}")
                if cat_theta:
                    theta_str = ", ".join(f"L{l}:{t:.2f}" for l, t in sorted(cat_theta.items()))
                    print(f"    Theta: {theta_str}")
            else:
                steerer = router.create_steering_hooks(model, decision, alpha=args.alpha)
                print(f"\n  [STEERED] Using {vec_source} vector (alpha={args.alpha})")
            print(f"  {'-'*76}")

            steerer.register_hooks()
            try:
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        logits_processor=LogitsProcessorList([SanitizeLogitsProcessor()]),
                    )
                steered_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                steered_response = steered_output[len(formatted_prompt) :]
                print(f"  {steered_response[:800]}")
                if len(steered_response) > 800:
                    print(f"  ... ({len(steered_response)} chars total)")
                print(f"  {'-'*76}")

                if use_dynamic and hasattr(steerer, "last_scale") and steerer.last_scale:
                    for lid, sval in sorted(steerer.last_scale.items()):
                        sv = sval.item() if sval.numel() == 1 else sval.tolist()
                        label = (
                            f"{lid[0]} layer {lid[1]}" if isinstance(lid, tuple) else f"layer {lid}"
                        )
                        print(f"    Dynamic scale ({label}): {sv:.4f}")
            finally:
                steerer.remove_hooks()
        else:
            print(f"\n  [SKIP] Router decided not to steer (benign prompt)")

        print()

    print(f"{'='*80}")
    print("[OK] Router testing complete!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Test steering with correlation-informed layer selection"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--steering-vectors", required=True, help="Path to steering vectors .pt file"
    )
    parser.add_argument("--correlations", default=None, help="Path to layer_correlations.json")
    parser.add_argument("--results-dir", default=None, help="Path to baseline evaluation results")

    # Layer selection modes
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--top-k", type=int, default=None, help="Test top K layers individually"
    )
    layer_group.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Manually specify layers to test together",
    )

    parser.add_argument(
        "--alpha", type=float, default=-2.0, help="Steering coefficient (negative = reduce refusal)"
    )
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--min-refusal-score", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens to generate")
    parser.add_argument(
        "--component",
        default="attn",
        help="Which activation component to steer: layer, attn, mlp, or attn+mlp (default: attn)",
    )
    parser.add_argument(
        "--router",
        default=None,
        help="Path to calibration.json for automatic category routing",
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
        "--dynamic",
        action="store_true",
        help="Use dynamic SiLU-gated steering (requires --router and --theta-file)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=-1.0,
        help="Gain for dynamic steering mode (negative = reduce refusal, default: -1.0)",
    )
    parser.add_argument(
        "--theta-file",
        default=None,
        help="Path to per_category_theta.json (required with --dynamic)",
    )

    args = parser.parse_args()

    # Resolve model path to absolute (avoids huggingface_hub validation errors)
    args.model = resolve_model_path(args.model)

    # Load prompts
    print("[LOAD] Loading prompts that refused in baseline evaluation...")
    refusal_prompts = load_actual_refusal_prompts(
        args.results_dir, min_score=args.min_refusal_score, max_prompts=args.num_prompts
    )

    if not refusal_prompts:
        print(f"❌ No prompts found with refusal score > {args.min_refusal_score}")
        sys.exit(1)

    print(f"[OK] Found {len(refusal_prompts)} high-refusal prompts\n")

    # === Router mode ===
    if args.router:
        _run_router_mode(args, refusal_prompts)
        return

    # Determine which layers to test
    layer_configs = []

    if args.layers is not None:
        layer_configs = [args.layers]
        print(f"[TARGET] Manual mode: Testing layers {args.layers} together\n")

    elif args.top_k is not None:
        best_layers = load_best_layers_from_correlations(args.correlations, args.top_k)
        if best_layers is None:
            sys.exit(1)
        layer_configs = [[layer] for layer in best_layers]
        print(
            f"\n[TARGET] Top-{args.top_k} mode: Testing {len(layer_configs)} layers individually\n"
        )

    else:
        best_layers = load_best_layers_from_correlations(args.correlations, top_k=1)
        if best_layers is None:
            sys.exit(1)
        layer_configs = [[best_layers[0]]]
        print(f"\n[TARGET] Default mode: Testing only best layer ({best_layers[0]})\n")

    # Run tests
    quantize = None
    if getattr(args, "load_in_4bit", False):
        quantize = "4bit"
    elif getattr(args, "load_in_8bit", False):
        quantize = "8bit"

    test_steering(
        model_name=args.model,
        steering_file=args.steering_vectors,
        layer_configs=layer_configs,
        alpha=args.alpha,
        test_prompts=refusal_prompts,
        max_new_tokens=args.max_tokens,
        component=args.component,
        quantize=quantize,
    )

    print("\n" + "=" * 80)
    print("[OK] All tests complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
