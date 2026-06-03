"""
Axis rotation analysis: cosine similarity between consecutive-layer WRMD vectors.

Tests whether the refusal/comply steering axis is stable or rotates between
transformer layers in the unsteered model. The WRMD vectors are computed from
unsteered refused vs. complied prompts, so this is a property of the model's
natural representation, not an artifact of steering.

Finding: consecutive-layer WRMD vectors are ~90° apart in both Mistral-7B and
Qwen3.5-9B. The refusal/comply axis is layer-specific — steering must be applied
at each layer independently because adjacent layers encode the same information
in nearly orthogonal subspaces.

Usage:
    python scripts/analyze_axis_rotation.py \\
      --steering-vectors outputs/.../steering_vectors_md_mlp.pt \\
      --correlations outputs/.../layer_correlations_mlp.json \\
      --output-dir outputs/.../axis_rotation

    # Or compare two models:
    python scripts/analyze_axis_rotation.py \\
      --steering-vectors model_a.pt model_b.pt \\
      --labels "Mistral-7B" "Qwen3.5-9B" \\
      --output-dir outputs/axis_rotation_comparison
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--steering-vectors", nargs="+", required=True, help="One or more steering vector .pt files"
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each steering vector file (defaults to filename)",
    )
    p.add_argument(
        "--correlations",
        nargs="+",
        default=None,
        help="layer_correlations JSON files (one per steering vector, optional)",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def load_vectors(path):
    sv = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sv, dict) and "steering_vectors" in sv:
        return sv["steering_vectors"].float()
    elif isinstance(sv, torch.Tensor):
        return sv.float()
    raise ValueError(f"Unrecognised steering vector format in {path}")


def load_best_layers(corr_path):
    if corr_path is None:
        return []
    with open(corr_path) as f:
        return json.load(f).get("best_layers", [])[:4]


def compute_rotation(vecs, best_layers):
    """
    For each consecutive pair (l, l+1) compute:
      cos_sim, angle_deg, and whether either layer is a best layer.

    vecs: [num_layers, hidden_dim] float tensor
    Returns list of dicts.
    """
    vecs_n = F.normalize(vecs, dim=-1)
    results = []
    for l in range(len(vecs) - 1):
        c = (vecs_n[l] * vecs_n[l + 1]).sum().item()
        c = max(-1.0, min(1.0, c))
        deg = math.degrees(math.acos(c))
        results.append(
            {
                "layer_a": l,
                "layer_b": l + 1,
                "cos_sim": round(c, 5),
                "angle_deg": round(deg, 2),
                "is_best_pair": l in best_layers or (l + 1) in best_layers,
            }
        )
    return results


def summarise(results):
    angles = [r["angle_deg"] for r in results]
    best_angles = [r["angle_deg"] for r in results if r["is_best_pair"]]
    return {
        "mean_angle_deg": round(sum(angles) / len(angles), 2),
        "min_angle_deg": round(min(angles), 2),
        "max_angle_deg": round(max(angles), 2),
        "mean_cos_sim": round(sum(r["cos_sim"] for r in results) / len(results), 5),
        "mean_best_layer_angle_deg": (
            round(sum(best_angles) / len(best_angles), 2) if best_angles else None
        ),
        "n_pairs": len(results),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n = len(args.steering_vectors)
    labels = args.labels if args.labels else [os.path.basename(p) for p in args.steering_vectors]
    corr_paths = args.correlations if args.correlations else [None] * n

    if len(corr_paths) < n:
        corr_paths = corr_paths + [None] * (n - len(corr_paths))

    all_results = {}

    for sv_path, label, corr_path in zip(args.steering_vectors, labels, corr_paths):
        print(f"\n=== {label} ===")
        vecs = load_vectors(sv_path)
        best_layers = load_best_layers(corr_path)
        print(f"  {len(vecs)} layers, hidden_dim={vecs.shape[1]}, best_layers={best_layers}")

        results = compute_rotation(vecs, best_layers)
        summary = summarise(results)

        print(
            f"  Mean angle:  {summary['mean_angle_deg']:.1f}°  "
            f"(range {summary['min_angle_deg']:.1f}°–{summary['max_angle_deg']:.1f}°)"
        )
        print(f"  Mean cos_sim: {summary['mean_cos_sim']:+.4f}")
        if summary["mean_best_layer_angle_deg"]:
            print(f"  Best-layer pairs mean angle: {summary['mean_best_layer_angle_deg']:.1f}°")

        print(f"\n  {'Pair':>8}  {'cos_sim':>9}  {'angle':>8}  {'note':>6}")
        for r in results:
            note = "BEST" if r["is_best_pair"] else ""
            print(
                f"  {r['layer_a']:>3}→{r['layer_b']:<3}  {r['cos_sim']:>+9.4f}  {r['angle_deg']:>7.1f}°  {note:>6}"
            )

        all_results[label] = {"summary": summary, "per_pair": results}

    # Save
    out_path = os.path.join(args.output_dir, "axis_rotation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Cross-model comparison if multiple
    if n > 1:
        print("\n=== Cross-model summary ===")
        for label in labels:
            s = all_results[label]["summary"]
            print(
                f"  {label}: mean={s['mean_angle_deg']:.1f}°  "
                f"range={s['min_angle_deg']:.1f}°–{s['max_angle_deg']:.1f}°  "
                f"best-layer-pairs={s['mean_best_layer_angle_deg']}°"
            )


if __name__ == "__main__":
    main()
