#!/usr/bin/env python3
"""
CLI script for calibrating the category router.

Computes the optimal refusal detection threshold and per-category projection
statistics from labeled activations and steering vectors.

Outputs:
  - calibration.json: Router parameters for use with --router in test_steering.py
  - router_roc.png: ROC curve visualization
"""

import argparse
import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from activation_steering import load_best_layers_from_correlations
from activation_steering.routing import calibrate_router


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate category router from activations and steering vectors"
    )
    parser.add_argument(
        "--activations",
        required=True,
        help="Path to activations .pt file with labels",
    )
    parser.add_argument(
        "--global-vectors",
        required=True,
        help="Path to global steering vectors .pt file",
    )
    parser.add_argument(
        "--category-vectors",
        required=True,
        nargs="+",
        help="Paths to category steering vector .pt files",
    )
    parser.add_argument(
        "--bootstrap-stability",
        default=None,
        help="Path to bootstrap stability JSON file",
    )
    parser.add_argument(
        "--correlations",
        default=None,
        help="Path to layer_correlations.json for automatic layer selection",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top layers to use (default: 3)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Manually specify target layers (overrides --correlations)",
    )
    parser.add_argument(
        "--component",
        default="attn",
        choices=["attn", "mlp", "layer"],
        help="Activation component (default: attn)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as activations file)",
    )
    parser.add_argument(
        "--stability-filter",
        default="unreliable",
        choices=["stable", "moderate", "unreliable"],
        help="Exclude categories at or above this instability level (default: unreliable)",
    )

    args = parser.parse_args()

    # Determine target layers
    if args.layers is not None:
        target_layers = args.layers
        print(f"[INFO] Using manually specified layers: {target_layers}")
    elif args.correlations is not None:
        target_layers = load_best_layers_from_correlations(args.correlations, args.top_k)
        if target_layers is None:
            print("[ERROR] Could not load layers from correlations file")
            sys.exit(1)
    else:
        print("[ERROR] Must provide either --layers or --correlations")
        sys.exit(1)

    # Default output dir
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.activations))

    print(f"\n[INFO] Calibrating router:")
    print(f"   Activations: {args.activations}")
    print(f"   Global vectors: {args.global_vectors}")
    print(f"   Category vectors: {len(args.category_vectors)} files")
    print(f"   Target layers: {target_layers}")
    print(f"   Component: {args.component}")
    print(f"   Output: {args.output_dir}")
    print()

    calibration = calibrate_router(
        activations_path=args.activations,
        global_vectors_path=args.global_vectors,
        category_vector_paths=args.category_vectors,
        target_layers=target_layers,
        component=args.component,
        bootstrap_stability_path=args.bootstrap_stability,
        output_dir=args.output_dir,
        stability_filter=args.stability_filter,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Calibration Summary")
    print(f"{'='*60}")
    print(f"   AUC: {calibration.auc:.4f}")
    print(f"   Threshold: {calibration.threshold:.4f}")
    print(f"   Min category threshold: {calibration.min_category_threshold:.4f}")
    print(f"   Categories: {len(calibration.category_vectors_paths)}")
    if calibration.bootstrap_stability:
        print(f"   Bootstrap stability:")
        for cat, stab in sorted(calibration.bootstrap_stability.items()):
            excluded = " (EXCLUDED)" if cat in calibration.excluded_categories else ""
            print(f"      {cat}: {stab}{excluded}")
    if calibration.excluded_categories:
        print(f"   Excluded from routing: {len(calibration.excluded_categories)} categories")
        print(
            f"   Routable categories: {len(calibration.category_vectors_paths) - len(calibration.excluded_categories)}"
        )
    print(f"\n[OK] Calibration complete!")


if __name__ == "__main__":
    main()
