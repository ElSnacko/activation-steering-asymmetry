#!/usr/bin/env python3
"""
Main pipeline script for activation steering.

This script runs the complete pipeline:
1. Extract activations from a model based on judge scores
2. Compute steering vectors using WRMD
3. Find best layers for steering
4. Test steering on sample prompts

Usage:
    python main.py --config config.yaml

Or with command-line arguments:
    python main.py \
        --model /path/to/Qwen3.5-9B \
        --results-dir LLM-Refusal-Evaluation/results/Qwen3.5-9B \
        --output-dir outputs/my-run
"""

import argparse
import gc
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import json

import torch

from activation_steering import (
    ActivationExtractor,
    SteeringHook,
    WRMDCalculator,
    analyze_dataset_quality,
    check_gpu_memory,
    compare_capability,
    compute_dynamic_params,
    compute_layer_correlations,
    evaluate_capability,
    extract_model_name,
    find_best_layers,
    find_best_layers_dynamic,
    generate_run_id,
    load_actual_refusal_prompts,
    load_prompts_from_judge_scores,
    load_questions,
    merge_steering_into_model,
    plot_correlations,
    resolve_model_path,
    save_dynamic_steered_model,
    setup_model_run_dirs,
    test_steering,
    visualize_layer_projections,
)


def run_pipeline(
    model_name,
    results_dir,
    output_dir=None,
    run_id=None,
    refusal_threshold=0.1,
    compliance_threshold=-0.1,
    max_samples=None,
    method="wrmd",
    lambda_ridge=0.1,
    normalize=False,
    top_k_layers=5,
    test_alpha=-2.0,
    test_num_prompts=3,
    test_min_score=0.5,
    test_max_score=None,
    test_max_tokens=500,
    dynamic=False,
    dynamic_theta=None,
    dynamic_gain=None,
    merge_output=None,
    capability_eval=False,
    capability_questions=None,
    capability_benchmark=None,
    capability_subjects=None,
    capability_max_questions=None,
    skip_extraction=False,
    skip_computation=False,
    skip_analysis=False,
    skip_testing=False,
    skip_merge=False,
):
    """
    Run the complete activation steering pipeline.

    Args:
        model_name: HuggingFace model identifier
        results_dir: Path to LLM-Refusal-Evaluation results
        output_dir: Custom output directory (optional)
        run_id: Custom run ID (optional)
        refusal_threshold: Judge score threshold for refusals
        compliance_threshold: Judge score threshold for compliance
        max_samples: Limit number of samples (optional)
        method: Steering vector method ('md', 'rmd', 'wrmd')
        lambda_ridge: Ridge regularization parameter
        normalize: Whether to normalize steering vectors
        top_k_layers: Number of top layers to identify
        test_alpha: Steering coefficient for testing
        test_num_prompts: Number of prompts to test
        test_min_score: Minimum refusal score for test prompts
        test_max_score: Maximum refusal score for test prompts (for borderline selection)
        test_max_tokens: Max tokens to generate during testing
        dynamic: If True, use dynamic steering mode instead of fixed alpha
        dynamic_theta: Manual override for theta (auto-computed if None)
        dynamic_gain: Gain multiplier for dynamic mode (negative = reduce refusal, default: -1.0)
        merge_output: Output directory for merged/saved model (None = skip merge step)
        capability_eval: If True, run capability preservation eval (MCQ accuracy)
        capability_questions: Path to custom capability questions JSON
        capability_benchmark: HuggingFace benchmark name (mmlu, arc_easy, arc_challenge)
        capability_subjects: MMLU subjects to include (None = all)
        capability_max_questions: Limit number of capability questions
        skip_extraction: Skip activation extraction step
        skip_computation: Skip steering vector computation step
        skip_analysis: Skip layer analysis step
        skip_testing: Skip steering testing step
        skip_merge: Skip model merge/save step even if merge_output is set

    Returns:
        Dictionary with paths to generated files
    """

    # Resolve model path to absolute (avoids huggingface_hub validation errors)
    model_name = resolve_model_path(model_name)

    # Check GPU memory before loading model
    check_gpu_memory()

    # Setup output directories
    if output_dir is None:
        model_base_name = extract_model_name(model_name)
        run_id = run_id or generate_run_id()
        output_dirs = setup_model_run_dirs(model_name=model_base_name, run_id=run_id)
    else:
        output_dirs = {
            "extract_activations": os.path.join(output_dir, "extract_activations"),
            "compute_wrmd": os.path.join(output_dir, "compute_wrmd"),
            "find_best_layers": os.path.join(output_dir, "find_best_layers"),
        }
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    print("=" * 80)
    print("ACTIVATION STEERING PIPELINE")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {str(output_dirs['extract_activations']).rsplit('/', 2)[0]}")
    print("=" * 80)

    results = {}

    # Step 1: Extract Activations
    if not skip_extraction:
        print("\n" + "=" * 80)
        print("STEP 1: EXTRACTING ACTIVATIONS")
        print("=" * 80)

        # Load prompts from judge scores
        prompts, labels, metadata = load_prompts_from_judge_scores(
            results_dir, refusal_threshold, compliance_threshold
        )

        analyze_dataset_quality(metadata)

        # Limit samples if requested
        if max_samples:
            print(f"\n[WARN] Limiting to {max_samples} samples for testing")
            prompts = prompts[:max_samples]
            labels = labels[:max_samples]
            metadata = metadata[:max_samples]

        # Extract activations
        extractor = ActivationExtractor(model_name)
        activation_file = "activations.pt"
        extractor.extract_dataset(
            prompts,
            labels,
            activation_file,
            metadata,
            output_dir=output_dirs["extract_activations"],
        )

        activation_path = os.path.join(output_dirs["extract_activations"], activation_file)
        results["activations"] = activation_path
        print(f"\n[OK] Activations saved to: {activation_path}")

        # Free GPU memory from extraction model before later steps
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
        print("[OK] Freed extraction model from GPU memory")

    else:
        # Look for existing activation file (try multiple patterns)
        activation_dir = output_dirs["extract_activations"]
        activation_path = None
        for pattern in ["activations.pt", "activations_*.pt"]:
            import glob

            matches = glob.glob(os.path.join(str(activation_dir), pattern))
            if matches:
                activation_path = matches[0]
                break
        if activation_path is None or not os.path.exists(activation_path):
            raise FileNotFoundError(f"No activation files found in: {activation_dir}")
        results["activations"] = activation_path
        print(f"\n[INFO] Using existing activations: {activation_path}")

    # Step 2: Compute Steering Vectors
    if not skip_computation:
        print("\n" + "=" * 80)
        print("STEP 2: COMPUTING STEERING VECTORS")
        print("=" * 80)

        calculator = WRMDCalculator(results["activations"])

        vectors = calculator.compute_steering_vectors(
            method=method, lambda_ridge=lambda_ridge, use_score_weighting=True, normalize=normalize
        )

        calculator.analyze_vectors(vectors, output_dir=output_dirs["compute_wrmd"])

        vector_file = f"steering_vectors_{method}.pt"
        calculator.save_vectors(
            vectors,
            vector_file,
            method=method,
            lambda_ridge=lambda_ridge,
            use_score_weighting=True,
            output_dir=output_dirs["compute_wrmd"],
        )

        vector_path = os.path.join(output_dirs["compute_wrmd"], vector_file)
        results["steering_vectors"] = vector_path
        print(f"\n[OK] Steering vectors saved to: {vector_path}")

    else:
        # Look for existing steering vector file (try multiple patterns)
        vector_dir = output_dirs["compute_wrmd"]
        vector_path = None
        for pattern in [f"steering_vectors_{method}.pt", "steering_vectors_*.pt"]:
            import glob

            matches = glob.glob(os.path.join(str(vector_dir), pattern))
            if matches:
                vector_path = matches[0]
                break
        if vector_path is None or not os.path.exists(vector_path):
            raise FileNotFoundError(f"No steering vector files found in: {vector_dir}")
        results["steering_vectors"] = vector_path
        print(f"\n[INFO] Using existing steering vectors: {vector_path}")

    # Step 3: Find Best Layers
    if not skip_analysis:
        print("\n" + "=" * 80)
        print("STEP 3: ANALYZING BEST LAYERS")
        print("=" * 80)

        correlations, projections_all, judge_scores = compute_layer_correlations(
            results["activations"], results["steering_vectors"]
        )

        plot_correlations(
            correlations,
            output_dir=output_dirs["find_best_layers"],
            output_file="layer_correlations.png",
        )

        best_layers = find_best_layers(correlations, top_k=top_k_layers)

        # For dynamic mode, re-rank using SiLU coverage and consistency
        if dynamic:
            best_layers = find_best_layers_dynamic(
                results["activations"], results["steering_vectors"], top_k=top_k_layers
            )

        # Generate scatter plots for top 3 layers
        print("\n[INFO] Generating scatter plots for top 3 layers...")
        sorted_corrs = sorted(correlations, key=lambda x: x["abs_correlation"], reverse=True)
        for c in sorted_corrs[:3]:
            layer = c["layer"]
            visualize_layer_projections(
                layer,
                projections_all[layer],
                judge_scores,
                c["correlation"],
                output_dir=output_dirs["find_best_layers"],
                output_file=f"layer_{layer}_projection_scatter.png",
            )

        # Save results
        analysis_output = {"best_layers": best_layers, "all_correlations": correlations}

        json_path = os.path.join(output_dirs["find_best_layers"], "layer_correlations.json")
        with open(json_path, "w") as f:
            json.dump(analysis_output, f, indent=2)

        results["layer_analysis"] = json_path
        results["best_layers"] = best_layers
        print(f"\n[OK] Layer analysis saved to: {json_path}")
        print(f"[OK] Best layers for steering: {best_layers}")

    else:
        # Load existing analysis
        json_path = os.path.join(output_dirs["find_best_layers"], "layer_correlations.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Layer analysis not found: {json_path}")
        with open(json_path) as f:
            analysis_output = json.load(f)
        results["layer_analysis"] = json_path

        if dynamic:
            # Re-rank layers using SiLU coverage and consistency
            results["best_layers"] = find_best_layers_dynamic(
                results["activations"], results["steering_vectors"], top_k=top_k_layers
            )
        else:
            results["best_layers"] = analysis_output["best_layers"][:top_k_layers]

        print(f"\n[INFO] Using existing layer analysis: {json_path}")
        print(f"[INFO] Best layers: {results['best_layers']}")

    # Compute dynamic steering parameters if needed
    theta = dynamic_theta
    gain = dynamic_gain
    if dynamic and theta is None:
        print("\n" + "=" * 80)
        print("COMPUTING DYNAMIC STEERING PARAMETERS")
        print("=" * 80)

        target_layers = results["best_layers"][:top_k_layers]
        theta = compute_dynamic_params(
            results["activations"], results["steering_vectors"], target_layers
        )

        results["dynamic_params"] = {
            "theta": theta,
            "gain": gain,
            "target_layers": target_layers,
        }

    # Step 4: Test Steering
    if not skip_testing:
        print("\n" + "=" * 80)
        print("STEP 4: TESTING STEERING")
        print("=" * 80)

        # Load test prompts
        print("[LOAD] Loading prompts that refused in baseline evaluation...")
        refusal_prompts = load_actual_refusal_prompts(
            results_dir,
            min_score=test_min_score,
            max_score=test_max_score,
            max_prompts=test_num_prompts,
        )

        if not refusal_prompts:
            print(f"[WARN] No prompts found with refusal score > {test_min_score}")
            print("[WARN] Skipping testing step")
        else:
            print(f"[OK] Found {len(refusal_prompts)} high-refusal prompts\n")

            # Test top-k layers together
            layer_configs = [results["best_layers"][:top_k_layers]]

            test_steering(
                model_name=model_name,
                steering_file=results["steering_vectors"],
                layer_configs=layer_configs,
                alpha=test_alpha,
                test_prompts=refusal_prompts,
                max_new_tokens=test_max_tokens,
                dynamic=dynamic,
                theta=theta,
                gain=gain,
            )

            print("\n[OK] Steering tests complete!")

    # Step 4b: Capability Preservation Evaluation (optional)
    if capability_eval and not skip_testing:
        print("\n" + "=" * 80)
        print("STEP 4b: CAPABILITY PRESERVATION EVALUATION")
        print("=" * 80)

        # Load questions from benchmark or file
        from activation_steering.capability import load_hf_benchmark

        if capability_benchmark:
            print(f"Loading benchmark: {capability_benchmark}")
            cap_questions = load_hf_benchmark(
                capability_benchmark,
                subjects=capability_subjects,
                max_questions=capability_max_questions,
            )
        else:
            cap_questions = load_questions(capability_questions)
            if capability_max_questions:
                cap_questions = cap_questions[:capability_max_questions]

        print(f"Evaluating on {len(cap_questions)} capability questions")

        # Need model loaded -- load if not already available
        from activation_steering import load_model

        cap_model, cap_tokenizer = load_model(model_name)

        # Baseline capability
        print("\n[BASELINE] Running capability evaluation...")
        cap_baseline = evaluate_capability(
            cap_model, cap_tokenizer, cap_questions, max_new_tokens=32
        )
        print(
            f"Baseline capability: {cap_baseline.accuracy:.1%} "
            f"({cap_baseline.num_correct}/{cap_baseline.num_total})"
        )
        for cat, stats in cap_baseline.per_category.items():
            print(f"  {cat}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

        # Steered capability
        steering_data = torch.load(results["steering_vectors"])
        if isinstance(steering_data, dict):
            sv = steering_data["steering_vectors"]
        else:
            sv = steering_data

        target_layers = results["best_layers"][:top_k_layers]
        alpha = test_alpha

        if dynamic:
            steerer = SteeringHook(
                cap_model, sv, target_layers=target_layers, dynamic=True, theta=theta, gain=gain
            )
        else:
            steerer = SteeringHook(cap_model, sv, target_layers=target_layers, alpha=alpha)
        steerer.register_hooks()

        print(f"\n[STEERED] Running capability evaluation (alpha={alpha:+.2f})...")
        cap_steered = evaluate_capability(
            cap_model, cap_tokenizer, cap_questions, max_new_tokens=32
        )
        steerer.remove_hooks()

        print(
            f"Steered capability: {cap_steered.accuracy:.1%} "
            f"({cap_steered.num_correct}/{cap_steered.num_total})"
        )

        # Compare
        cap_comparison = compare_capability(cap_baseline, cap_steered)
        delta = cap_comparison["accuracy_delta"]
        sign = "+" if delta >= 0 else ""
        print(f"\nCapability delta: {sign}{delta:.1%}")
        if cap_comparison["severely_degraded"]:
            print("[WARNING] SEVERE capability degradation detected (>15% accuracy drop)")
        elif cap_comparison["degraded"]:
            print("[WARNING] Capability degradation detected (>5% accuracy drop)")
        else:
            print("[OK] Capability preserved (accuracy drop <= 5%)")

        # Save results
        cap_output_dir = output_dirs.get("find_best_layers", output_dir or "outputs")
        cap_results_path = os.path.join(cap_output_dir, "capability_eval.json")
        with open(cap_results_path, "w") as f:
            json.dump(
                {
                    "baseline": cap_baseline.to_dict(),
                    "steered": cap_steered.to_dict(),
                    "comparison": cap_comparison,
                    "alpha": alpha,
                    "layers": target_layers,
                    "benchmark": capability_benchmark,
                    "questions_file": capability_questions,
                },
                f,
                indent=2,
            )
        results["capability_eval"] = cap_results_path
        print(f"\n[OK] Capability evaluation saved to: {cap_results_path}")

        # Clean up
        del cap_model, cap_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Step 5: Save/Merge Model (optional)
    if merge_output and not skip_merge:
        print("\n" + "=" * 80)
        print("STEP 5: SAVING STEERED MODEL")
        print("=" * 80)

        # Free GPU memory from previous steps
        gc.collect()
        torch.cuda.empty_cache()

        target_layers = results["best_layers"][:top_k_layers]

        if dynamic:
            # Dynamic mode: save with SiLU-gated steering
            if theta is None:
                print("[ERROR] Dynamic theta not available for merge step")
            else:
                metadata = save_dynamic_steered_model(
                    base_model_path=model_name,
                    steering_vectors_file=results["steering_vectors"],
                    target_layers=target_layers,
                    theta=theta,
                    gain=gain if gain is not None else -1.0,
                    output_dir=merge_output,
                )
                results["merged_model"] = merge_output
                print(f"\n[OK] Dynamic steered model saved to: {merge_output}")
        else:
            # Fixed-alpha mode: permanently merge into MLP biases
            metadata = merge_steering_into_model(
                base_model_path=model_name,
                steering_vectors_file=results["steering_vectors"],
                target_layers=target_layers,
                alpha=test_alpha,
                output_dir=merge_output,
            )
            results["merged_model"] = merge_output
            print(f"\n[OK] Merged model saved to: {merge_output}")

    # Final Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nGenerated Files:")
    for key, path in results.items():
        if key != "best_layers":
            print(f"  {key}: {path}")
    print("\nBest Layers:", results.get("best_layers", "N/A"))
    if "dynamic_params" in results:
        dp = results["dynamic_params"]
        gain_str = dp.get("gain") or "-1.0 (default)"
        if isinstance(dp["theta"], dict):
            print(f"Dynamic Params (gain={gain_str}):")
            for lid in sorted(dp["theta"].keys()):
                print(f"   Layer {lid}: theta={dp['theta'][lid]:.4f}")
        else:
            print(f"Dynamic Params: theta={dp['theta']:.4f}, gain={gain_str}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run complete activation steering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        "--model-path",
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen3.5-9B) " "or local path (e.g., /path/to/model)",
    )
    parser.add_argument(
        "--results-dir", required=True, help="Path to LLM-Refusal-Evaluation results directory"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory (default: outputs/{model_name}/{run_id})",
    )
    parser.add_argument(
        "--run-id", default=None, help="Custom run ID (default: auto-generated timestamp)"
    )

    # Extraction parameters
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=0.1,
        help="Judge score above this = refusal (default: 0.1)",
    )
    parser.add_argument(
        "--compliance-threshold",
        type=float,
        default=-0.1,
        help="Judge score below this = compliant (default: -0.1)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit number of samples for testing"
    )

    # Computation parameters
    parser.add_argument(
        "--method",
        choices=["md", "rmd", "wrmd"],
        default="md",
        help="Steering vector computation method (default: md)",
    )
    parser.add_argument(
        "--lambda-ridge",
        type=float,
        default=0.1,
        help="Ridge regularization parameter (default: 0.1)",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize steering vectors to unit length"
    )

    # Analysis parameters
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top layers to identify (default: 5)"
    )

    # Testing parameters
    parser.add_argument(
        "--test-alpha",
        type=float,
        default=-2.0,
        help="Steering coefficient for testing (default: -2.0)",
    )
    parser.add_argument(
        "--test-num-prompts", type=int, default=3, help="Number of prompts to test (default: 3)"
    )
    parser.add_argument(
        "--test-min-score",
        type=float,
        default=0.5,
        help="Minimum refusal score for test prompts (default: 0.5)",
    )
    parser.add_argument(
        "--test-max-score",
        type=float,
        default=None,
        help="Maximum refusal score for test prompts (for borderline selection)",
    )
    parser.add_argument(
        "--test-max-tokens",
        type=int,
        default=500,
        help="Max tokens to generate during testing (default: 500)",
    )

    # Dynamic steering parameters
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic steering (SiLU-based scaling) instead of fixed alpha",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Manual theta for dynamic mode (auto-computed from activations if omitted)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Gain for dynamic mode (negative=reduce refusal, default: -1.0). "
        "Works like alpha but SiLU-modulated per-sample.",
    )

    # Capability preservation evaluation
    parser.add_argument(
        "--capability-eval",
        action="store_true",
        help="Run capability preservation evaluation after steering test",
    )
    parser.add_argument("--capability-questions", help="Path to custom capability questions JSON")
    parser.add_argument(
        "--capability-benchmark",
        choices=["mmlu", "arc_easy", "arc_challenge"],
        help="Load from HuggingFace benchmark (requires `datasets` package)",
    )
    parser.add_argument(
        "--capability-subjects",
        nargs="*",
        help="For MMLU: specific subjects to include (default: all)",
    )
    parser.add_argument(
        "--capability-max-questions", type=int, help="Limit number of capability questions"
    )

    # Merge/save options
    parser.add_argument(
        "--merge-output", help="Output directory for saved steered model (enables merge step)"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip model merge/save step even if --merge-output is set",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-extraction", action="store_true", help="Skip activation extraction (use existing)"
    )
    parser.add_argument(
        "--skip-computation",
        action="store_true",
        help="Skip steering vector computation (use existing)",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true", help="Skip layer analysis (use existing)"
    )
    parser.add_argument("--skip-testing", action="store_true", help="Skip steering testing")

    args = parser.parse_args()

    try:
        results = run_pipeline(
            model_name=args.model,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            run_id=args.run_id,
            refusal_threshold=args.refusal_threshold,
            compliance_threshold=args.compliance_threshold,
            max_samples=args.max_samples,
            method=args.method,
            lambda_ridge=args.lambda_ridge,
            normalize=args.normalize,
            top_k_layers=args.top_k,
            test_alpha=args.test_alpha,
            test_num_prompts=args.test_num_prompts,
            test_min_score=args.test_min_score,
            test_max_score=args.test_max_score,
            test_max_tokens=args.test_max_tokens,
            dynamic=args.dynamic,
            dynamic_theta=args.theta,
            dynamic_gain=args.gain,
            merge_output=args.merge_output,
            capability_eval=args.capability_eval,
            capability_questions=args.capability_questions,
            capability_benchmark=args.capability_benchmark,
            capability_subjects=args.capability_subjects,
            capability_max_questions=args.capability_max_questions,
            skip_extraction=args.skip_extraction,
            skip_computation=args.skip_computation,
            skip_analysis=args.skip_analysis,
            skip_testing=args.skip_testing,
            skip_merge=args.skip_merge,
        )

        return 0

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
