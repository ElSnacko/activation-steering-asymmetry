#!/usr/bin/env python3
"""
CLI script for computing steering vectors using WRMD and related methods.

This script provides a command-line interface to the WRMDCalculator class.
"""

import argparse
import json
import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from activation_steering import (
    WRMDCalculator,
    analyze_rank_associations,
    compare_methods,
    compute_category_angular_distances,
    ensure_dir,
    generate_run_id,
    infer_run_from_path,
    load_category_vectors_from_files,
    plot_category_angular_distances,
    setup_model_run_dirs,
)
from activation_steering.utils import category_to_slug


def main():
    parser = argparse.ArgumentParser(description="Compute WRMD steering vectors")
    parser.add_argument("--activations", default=None)
    parser.add_argument("--output", default="steering_vectors_md.pt")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory (default: outputs/{model_name}/{run_id}/compute_wrmd/)",
    )
    parser.add_argument(
        "--run-id", default=None, help="Custom run ID (default: auto-generated timestamp)"
    )
    parser.add_argument("--method", choices=["md", "rmd", "wrmd"], default="md")
    parser.add_argument("--lambda-ridge", type=float, default=0.1)
    parser.add_argument(
        "--no-score-weighting",
        action="store_true",
        help="Don't weight by judge scores (uniform weights)",
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Number of steering directions per layer (default: 1). "
        "rank > 1 uses PCA of residuals after projecting out v_1.",
    )
    parser.add_argument("--compare-methods", action="store_true")
    parser.add_argument(
        "--analyze-ranks",
        action="store_true",
        help="Analyze which PCA rank each prompt is most associated with (requires rank > 1)",
    )
    parser.add_argument(
        "--rank-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to use for rank analysis (default: all layers)",
    )
    parser.add_argument(
        "--component",
        default="attn",
        help="Which activation component to compute vectors for: layer, attn, mlp, or attn+mlp "
        "(default: attn)",
    )
    parser.add_argument(
        "--category",
        "--categories",
        nargs="+",
        default=None,
        dest="categories",
        help="Compute vectors using only these refusal categories "
        "(compliant baseline is always global)",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Compute separate vector file per category plus a global file",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print available categories from the activation file and exit",
    )
    parser.add_argument(
        "--analyze-categories",
        action="store_true",
        help="Compute pairwise angular distances between category steering vectors",
    )
    parser.add_argument(
        "--compare-vectors",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Compare pre-computed .pt vector files directly (standalone, no --activations needed)",
    )
    parser.add_argument(
        "--category-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices for category angular distance analysis (default: all)",
    )
    parser.add_argument(
        "--min-category-samples",
        type=int,
        default=10,
        help="Minimum refusal samples per category for angular distance analysis (default: 10)",
    )
    parser.add_argument(
        "--analyze-intra-category",
        action="store_true",
        help="Measure intra-category angular spread of refusal activations (coherence analysis)",
    )
    parser.add_argument(
        "--bootstrap-stability",
        action="store_true",
        help="Compute bootstrap stability analysis for category steering vectors",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20,
        help="Number of bootstrap iterations (default: 20)",
    )
    parser.add_argument(
        "--bootstrap-ratio",
        type=float,
        default=0.8,
        help="Fraction of refusal samples to resample per bootstrap iteration (default: 0.8)",
    )
    parser.add_argument(
        "--bootstrap-convergence",
        action="store_true",
        help="Run bootstrap convergence analysis (angular spread vs pool size)",
    )
    parser.add_argument(
        "--convergence-min-pool",
        type=int,
        default=50,
        help="Minimum pool size for convergence analysis (default: 50)",
    )
    parser.add_argument(
        "--convergence-pool-step",
        type=int,
        default=50,
        help="Pool size step for convergence analysis (default: 50)",
    )
    parser.add_argument(
        "--convergence-targets",
        type=int,
        nargs="+",
        default=[500, 1000],
        help="Target sample sizes for extrapolation (default: 500 1000)",
    )
    parser.add_argument(
        "--convergence-instability-ceiling",
        type=float,
        default=30.0,
        help="Floor threshold (degrees) above which a category is unstable (default: 30.0)",
    )
    args = parser.parse_args()

    if args.bootstrap_stability and args.bootstrap_samples < 2:
        parser.error("--bootstrap-samples must be at least 2")

    # Path C: --compare-vectors (standalone mode, no activations needed)
    if args.compare_vectors:
        if args.output_dir:
            output_dir = ensure_dir(args.output_dir)
        else:
            output_dir = os.getcwd()

        category_vectors = load_category_vectors_from_files(
            args.compare_vectors,
            component=args.component if "+" not in args.component else "attn",
            min_samples=args.min_category_samples,
        )
        if len(category_vectors) < 2:
            print("[ERROR] Need at least 2 category vector files for comparison.")
            sys.exit(1)

        results = compute_category_angular_distances(
            category_vectors, target_layers=args.category_layers
        )
        plot_category_angular_distances(results, output_dir=output_dir)

        json_path = os.path.join(output_dir, "category_angular_distances.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVE] Saved category angular distances to {json_path}")
        return

    # All other modes require --activations
    if args.activations is None:
        parser.error("--activations is required (unless using --compare-vectors)")

    # Set up output directory
    if args.output_dir is None:
        # Infer model name and run ID from input activations path
        model_name, inferred_run_id = infer_run_from_path(args.activations)
        if model_name is None:
            model_name = "unknown_model"
        run_id = args.run_id or inferred_run_id or generate_run_id()
        output_dirs = setup_model_run_dirs(model_name=model_name, run_id=run_id)
        output_dir = output_dirs["compute_wrmd"]
        print(f"📁 Using output directory: {output_dir}")
        print(f"   Model: {model_name}")
        print(f"   Run ID: {run_id}")
    else:
        output_dir = ensure_dir(args.output_dir)

    calculator = WRMDCalculator(args.activations)

    # Handle --list-categories
    if args.list_categories:
        available = calculator.get_available_categories()
        if not available:
            print("\n[INFO] No category metadata found in activation file.")
            print("   Extract activations with --dataset to include category information.")
        else:
            print(f"\n[INFO] Available categories ({len(available)}):")
            for cat in available:
                print(f"   - {cat}")
        return

    output_file = args.output

    # Parse component(s)
    if "+" in args.component:
        components = [c.strip() for c in args.component.split("+")]
    else:
        components = [args.component]

    # Handle --all-categories mode
    if args.all_categories and len(components) > 1:
        print(
            "[ERROR] --all-categories does not support multi-component (attn+mlp). "
            "Use a single component or run per-category manually."
        )
        sys.exit(1)

    if args.all_categories:
        available = calculator.get_available_categories()
        if not available:
            print("[ERROR] No category metadata found. Use --dataset when extracting activations.")
            sys.exit(1)

        category_summary = {}
        category_vectors_cache = {}  # Retain tensors for --analyze-categories

        # First compute global vectors (all categories)
        print(f"\n{'='*60}")
        print("Computing GLOBAL steering vectors (all categories)")
        print(f"{'='*60}")
        global_vectors = calculator.compute_steering_vectors(
            args.method,
            args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            normalize=args.normalize,
            rank=args.rank,
            component=components[0],
        )
        global_filename = f"steering_vectors_{args.method}.pt"
        # Avoid overwriting attn vectors when component is different
        if components[0] != "attn" and global_filename == f"steering_vectors_{args.method}.pt":
            global_filename = f"steering_vectors_{args.method}_{components[0]}.pt"
        calculator.save_vectors(
            global_vectors,
            global_filename,
            method=args.method,
            lambda_ridge=args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            output_dir=output_dir,
            rank=args.rank,
            component=components[0],
        )
        category_summary["_global"] = {
            "file": global_filename,
            "num_refusal_samples": (calculator.labels == 1).sum().item(),
            "num_compliant_samples": (calculator.labels == 0).sum().item(),
        }

        # Then compute per-category vectors
        for cat in available:
            print(f"\n{'='*60}")
            print(f"Computing vectors for category: {cat}")
            print(f"{'='*60}")

            slug = category_to_slug(cat)
            cat_filename = f"steering_vectors_{args.method}_{slug}.pt"
            # Avoid overwriting attn vectors when component is different
            if components[0] != "attn":
                cat_filename = f"steering_vectors_{args.method}_{components[0]}_{slug}.pt"

            # Check if this category has enough refusal samples
            try:
                cat_vectors = calculator.compute_steering_vectors(
                    args.method,
                    args.lambda_ridge,
                    use_score_weighting=not args.no_score_weighting,
                    normalize=args.normalize,
                    rank=args.rank,
                    component=components[0],
                    categories=[cat],
                )
            except ValueError as e:
                print(f"[WARN] Skipping category '{cat}': {e}")
                category_summary[cat] = {"file": None, "skipped": str(e)}
                continue

            calculator.save_vectors(
                cat_vectors,
                cat_filename,
                method=args.method,
                lambda_ridge=args.lambda_ridge,
                use_score_weighting=not args.no_score_weighting,
                output_dir=output_dir,
                rank=args.rank,
                component=components[0],
                categories=[cat],
            )

            # Retain tensor for angular distance analysis
            category_vectors_cache[cat] = cat_vectors

            # Count filtered refusal samples for summary
            cat_mask = calculator._build_category_mask([cat])
            n_refusal = ((calculator.labels == 1) & cat_mask).sum().item()
            category_summary[cat] = {
                "file": cat_filename,
                "num_refusal_samples": n_refusal,
            }

        # Save category summary
        summary_path = os.path.join(output_dir, "category_summary.json")
        with open(summary_path, "w") as f:
            json.dump(category_summary, f, indent=2)
        print(f"\n[SAVE] Category summary saved to {summary_path}")

        # Path A: --all-categories --analyze-categories — use retained vectors
        inter_category_results = None
        if args.analyze_categories:
            # Filter by min_category_samples
            filtered_cache = {
                cat: vecs
                for cat, vecs in category_vectors_cache.items()
                if category_summary.get(cat, {}).get("num_refusal_samples", 0)
                >= args.min_category_samples
            }
            skipped = set(category_vectors_cache.keys()) - set(filtered_cache.keys())
            if skipped:
                print(
                    f"[FILTER] Skipped {len(skipped)} categories with "
                    f"< {args.min_category_samples} refusal samples: "
                    f"{', '.join(sorted(skipped))}"
                )

            if len(filtered_cache) >= 2:
                inter_category_results = compute_category_angular_distances(
                    filtered_cache, target_layers=args.category_layers
                )
                plot_category_angular_distances(inter_category_results, output_dir=output_dir)
                json_path = os.path.join(output_dir, "category_angular_distances.json")
                with open(json_path, "w") as f:
                    json.dump(inter_category_results, f, indent=2)
                print(f"[SAVE] Saved category angular distances to {json_path}")
            else:
                print(
                    "[WARN] Not enough categories with valid vectors for angular distance analysis"
                )

        # Intra-category coherence analysis (works with --all-categories)
        if args.analyze_intra_category:
            calculator.analyze_intra_category_distances(
                target_layers=args.category_layers,
                component=components[0],
                min_samples=args.min_category_samples,
                output_dir=output_dir,
            )

        # Bootstrap stability analysis (works with --all-categories)
        if args.bootstrap_stability:
            calculator.analyze_bootstrap_stability(
                method=args.method,
                lambda_ridge=args.lambda_ridge,
                use_score_weighting=not args.no_score_weighting,
                component=components[0],
                target_layers=args.category_layers,
                n_bootstrap=args.bootstrap_samples,
                sample_ratio=args.bootstrap_ratio,
                min_samples=args.min_category_samples,
                output_dir=output_dir,
                distance_results=inter_category_results,
            )

        # Bootstrap convergence analysis (works with --all-categories)
        if args.bootstrap_convergence:
            calculator.analyze_bootstrap_convergence(
                method=args.method,
                lambda_ridge=args.lambda_ridge,
                use_score_weighting=not args.no_score_weighting,
                component=components[0],
                target_layers=args.category_layers,
                n_bootstrap=args.bootstrap_samples,
                sample_ratio=args.bootstrap_ratio,
                min_pool_size=args.convergence_min_pool,
                pool_step=args.convergence_pool_step,
                min_samples=args.min_category_samples,
                target_sizes=tuple(args.convergence_targets),
                instability_ceiling=args.convergence_instability_ceiling,
                output_dir=output_dir,
            )

        print(f"\n[OK] Done! Computed vectors for {len(available)} categories + global")
        return

    if args.compare_methods:
        all_vectors = compare_methods(
            calculator,
            args.lambda_ridge,
            output_dir,
            component=components[0],
            rank=args.rank,
        )

        # Save the best one (WRMD with score weighting)
        calculator.save_vectors(
            all_vectors["WRMD (weighted)"],
            args.output,
            method="wrmd",
            lambda_ridge=args.lambda_ridge,
            use_score_weighting=True,
            output_dir=output_dir,
            component=components[0],
            rank=args.rank,
        )
    elif len(components) > 1:
        # Multi-component: compute vectors for each component
        all_comp_vectors = {}
        for comp in components:
            print(f"\n{'='*60}")
            print(f"Computing vectors for component: {comp}")
            print(f"{'='*60}")
            vectors = calculator.compute_steering_vectors(
                args.method,
                args.lambda_ridge,
                use_score_weighting=not args.no_score_weighting,
                normalize=args.normalize,
                rank=args.rank,
                component=comp,
            )
            all_comp_vectors[comp] = vectors

        calculator.save_vectors(
            all_comp_vectors,
            args.output,
            method=args.method,
            lambda_ridge=args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            output_dir=output_dir,
            rank=args.rank,
            component=args.component,
        )
    else:
        # Determine output filename — adjust if category filter or non-default component
        output_file = args.output
        is_default_name = output_file == "steering_vectors_md.pt"
        # Add category slug if filtering by category
        if args.categories and is_default_name:
            slug = "_".join(category_to_slug(c) for c in args.categories)
            output_file = f"steering_vectors_{args.method}_{slug}.pt"
        # Add component suffix for non-default components (attn is default)
        # Must check AFTER category slug is added, so both suffixes are present
        if components[0] != "attn":
            base_without_ext = output_file.replace(".pt", "")
            # Avoid double-adding component suffix
            if not base_without_ext.endswith(f"_{components[0]}"):
                output_file = f"{base_without_ext}_{components[0]}.pt"

        vectors = calculator.compute_steering_vectors(
            args.method,
            args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            normalize=args.normalize,
            rank=args.rank,
            component=components[0],
            categories=args.categories,
        )

        if args.rank == 1:
            calculator.analyze_vectors(vectors, output_dir)
        else:
            # For multi-rank, analyze v_1 norms only
            calculator.analyze_vectors(vectors[:, 0, :], output_dir)

        calculator.save_vectors(
            vectors,
            output_file,
            method=args.method,
            lambda_ridge=args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            output_dir=output_dir,
            rank=args.rank,
            component=components[0],
            categories=args.categories,
        )

    # Rank association analysis (for multi-rank vectors)
    if args.analyze_ranks:
        if args.rank <= 1:
            print("[WARN] --analyze-ranks requires --rank > 1")
        else:
            # Get the vectors to analyze (handle all code paths)
            if args.compare_methods:
                rank_vecs = all_vectors["WRMD (weighted)"]
            elif len(components) > 1:
                rank_vecs = list(all_comp_vectors.values())[0]
            else:
                rank_vecs = vectors
            calculator.analyze_rank_associations(
                rank_vecs,
                target_layers=args.rank_layers,
                component=components[0],
                output_dir=output_dir,
            )

    # Path B: --analyze-categories without --all-categories (on-the-fly computation)
    if args.analyze_categories and not args.all_categories:
        calculator.analyze_category_distances(
            method=args.method,
            lambda_ridge=args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            normalize=args.normalize,
            rank=args.rank,
            component=components[0],
            target_layers=args.category_layers,
            output_dir=output_dir,
            min_samples=args.min_category_samples,
        )

    # Intra-category coherence analysis (standalone, without --all-categories)
    if args.analyze_intra_category and not args.all_categories:
        calculator.analyze_intra_category_distances(
            target_layers=args.category_layers,
            component=components[0],
            min_samples=args.min_category_samples,
            output_dir=output_dir,
        )

    # Bootstrap stability analysis (standalone, without --all-categories)
    if args.bootstrap_stability and not args.all_categories:
        calculator.analyze_bootstrap_stability(
            method=args.method,
            lambda_ridge=args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            component=components[0],
            target_layers=args.category_layers,
            n_bootstrap=args.bootstrap_samples,
            sample_ratio=args.bootstrap_ratio,
            min_samples=args.min_category_samples,
            output_dir=output_dir,
        )

    # Bootstrap convergence analysis (standalone, without --all-categories)
    if args.bootstrap_convergence and not args.all_categories:
        calculator.analyze_bootstrap_convergence(
            method=args.method,
            lambda_ridge=args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            component=components[0],
            target_layers=args.category_layers,
            n_bootstrap=args.bootstrap_samples,
            sample_ratio=args.bootstrap_ratio,
            min_pool_size=args.convergence_min_pool,
            pool_step=args.convergence_pool_step,
            min_samples=args.min_category_samples,
            target_sizes=tuple(args.convergence_targets),
            instability_ceiling=args.convergence_instability_ceiling,
            output_dir=output_dir,
        )

    # Determine the actual output path
    if not os.path.isabs(output_file):
        actual_output_path = os.path.join(output_dir, output_file)
    else:
        actual_output_path = output_file

    print(f"\n[OK] Done! Steering vectors saved to {actual_output_path}")


if __name__ == "__main__":
    main()
