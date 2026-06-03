#!/usr/bin/env python3
"""
CLI script for finding the best layers for steering.

This script analyzes layer correlations to identify which layers are most
effective for steering interventions.
"""

import argparse
import json
import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from activation_steering import (
    compute_layer_correlations,
    ensure_dir,
    find_best_layers,
    generate_run_id,
    infer_run_from_path,
    plot_correlations,
    setup_model_run_dirs,
    visualize_layer_projections,
)
from activation_steering.analysis import (
    compute_per_category_layer_correlations,
    plot_per_category_layer_correlations,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--steering-vectors", required=True)
    parser.add_argument("--judge-scores", default=None, help="Optional separate judge scores file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory (default: outputs/{model_name}/{run_id}/find_best_layers/)",
    )
    parser.add_argument(
        "--run-id", default=None, help="Custom run ID (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--component",
        default="attn",
        help="Which activation component to analyze: layer, attn, or mlp (default: attn)",
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="Also compute per-category layer correlations using per-category steering vectors",
    )
    parser.add_argument(
        "--category-vectors",
        default=None,
        help="Directory containing per-category steering vector files (required for --per-category). "
        "Defaults to same directory as --steering-vectors.",
    )
    parser.add_argument(
        "--min-category-samples",
        type=int,
        default=10,
        help="Minimum refusal samples per category for per-category analysis (default: 10)",
    )
    parser.add_argument(
        "--layer-selection-method",
        default="auc",
        choices=["auc", "spearman", "pearson"],
        help="Metric for ranking layers: auc (default), spearman, or pearson. "
        "auc treats layer selection as binary comply/refuse classification (no ordinal assumption). "
        "spearman uses rank correlation (appropriate for ordinal judge scores). "
        "pearson kept for backward compatibility with old binary-judge results.",
    )
    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        # Infer model name and run ID from input path
        model_name, inferred_run_id = infer_run_from_path(args.activations)
        if model_name is None:
            model_name = "unknown_model"
        run_id = args.run_id or inferred_run_id or generate_run_id()
        output_dirs = setup_model_run_dirs(model_name=model_name, run_id=run_id)
        output_dir = output_dirs["find_best_layers"]
        print(f"📁 Using output directory: {output_dir}")
        print(f"   Model: {model_name}")
        print(f"   Run ID: {run_id}")
    else:
        output_dir = ensure_dir(args.output_dir)

    # Compute correlations
    correlations, projections_all, judge_scores = compute_layer_correlations(
        args.activations,
        args.steering_vectors,
        args.judge_scores,
        component=args.component,
        method=args.layer_selection_method,
    )

    # Plot overall correlations
    plot_correlations(correlations, output_dir=output_dir, output_file="layer_correlations.png")

    # Find best layers
    best_layers = find_best_layers(correlations, args.top_k)

    # Visualize top 3 layers
    print("\n[INFO] Generating scatter plots for top 3 layers...")
    sorted_corrs = sorted(correlations, key=lambda x: x["abs_correlation"], reverse=True)
    for i, c in enumerate(sorted_corrs[:3]):
        layer = c["layer"]
        visualize_layer_projections(
            layer,
            projections_all[layer],
            judge_scores,
            c["correlation"],
            output_dir=output_dir,
            output_file=f"layer_{layer}_projection_scatter.png",
        )

    # Save results (include component in filename to avoid overwriting)
    suffix = f"_{args.component}" if args.component != "attn" else ""
    output = {"best_layers": best_layers, "all_correlations": correlations}

    json_output_path = os.path.join(output_dir, f"layer_correlations{suffix}.json")
    with open(json_output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to {json_output_path}")
    print(f"[OK] Best layers for steering: {best_layers}")

    # Per-category layer correlation analysis
    if args.per_category:
        cat_vec_dir = args.category_vectors
        if cat_vec_dir is None:
            cat_vec_dir = os.path.dirname(os.path.abspath(args.steering_vectors))

        print(f"\n{'='*70}")
        print("Per-Category Layer Correlations")
        print(f"{'='*70}")
        print(f"   Category vectors dir: {cat_vec_dir}")
        print(f"   Component: {args.component}")
        print(f"   Min samples: {args.min_category_samples}")
        print()

        per_cat_results = compute_per_category_layer_correlations(
            args.activations,
            cat_vec_dir,
            component=args.component,
            top_k=args.top_k,
            min_samples=args.min_category_samples,
        )

        if per_cat_results:
            # Save per-category results (include component in filename to avoid overwriting)
            suffix = f"_{args.component}" if args.component != "attn" else ""
            per_cat_path = os.path.join(output_dir, f"per_category_layer_correlations{suffix}.json")
            # Convert correlations to serializable format
            serializable = {}
            for cat_key, data in per_cat_results.items():
                serializable[cat_key] = {
                    "best_layers": data["best_layers"],
                    "num_refusal": data["num_refusal"],
                    "num_compliant": data["num_compliant"],
                    "correlations": data["correlations"],
                }
            with open(per_cat_path, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"\n[SAVE] Per-category results saved to {per_cat_path}")

            # Summary table
            print(f"\n{'Category':<50} {'Best Layer':<12} {'Top-3 Layers'}")
            print("-" * 90)
            for cat_key in sorted(per_cat_results.keys()):
                data = per_cat_results[cat_key]
                best = data["best_layers"][0]
                top3 = data["best_layers"][:3]
                top3_str = ", ".join(f"L{l}" for l in top3)
                print(f"  {cat_key:<48} L{best:<10} {top3_str}")

            # Check: do best layers differ across categories?
            best_set = set(per_cat_results[cat]["best_layers"][0] for cat in per_cat_results)
            if len(best_set) > 1:
                print(f"\n[INSIGHT] Best layers differ across categories: {sorted(best_set)}")
                print("   → Per-category layer selection may improve steering effectiveness")
            else:
                print(f"\n[INSIGHT] All categories converge on layer {best_set}")
                print("   → Global layer selection is sufficient")

            # Generate plots
            plot_per_category_layer_correlations(per_cat_results, output_dir=output_dir)
        else:
            print("[WARN] No per-category results (insufficient samples or missing vector files)")


if __name__ == "__main__":
    main()
