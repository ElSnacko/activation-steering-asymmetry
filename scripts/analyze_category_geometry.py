"""
Analyze per-category activation geometry in two complementary spaces:

1. WRMD refusal-direction space: pairwise angles between category steering vectors
   (output: angular_distance_analysis.json)

2. Compliance activation space: pairwise angles between per-category mean activations
   of complied-only examples, plus within-category comply↔refuse gap
   (output: comply_angular_distance_analysis.json)

The contrast between the two spaces is the core geometric finding: compliance
activations converge (low std, compressed geometry) while refusal directions
diverge (high std, category-specific). This supports the global steering vector
being adequate for most categories while explaining per-category outliers.

Usage:
    python scripts/analyze_category_geometry.py \\
        --activations outputs/qwen3-5-9b/.../extract_activations/activations.pt \\
        --category-vectors-dir outputs/qwen3-5-9b/.../category_vectors \\
        --global-vector outputs/qwen3-5-9b/.../compute_wrmd/steering_vectors_md_mlp.pt \\
        --layers 22 29 23 25 \\
        --component mlp \\
        --output-dir outputs/qwen3-5-9b/.../category_vectors \\
        --min-complied 3
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--activations", required=True, help="Path to activations .pt from extract_activations.py"
    )
    p.add_argument(
        "--category-vectors-dir",
        required=True,
        help="Directory containing per-category .pt steering vectors",
    )
    p.add_argument(
        "--global-vector", required=True, help="Path to global WRMD .pt file (from compute_wrmd.py)"
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Target layers to average over (e.g., 22 29 23 25)",
    )
    p.add_argument(
        "--component",
        default="mlp",
        choices=["mlp", "attn", "residual"],
        help="Activation component used for WRMD (default: mlp)",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for JSON files")
    p.add_argument(
        "--min-complied",
        type=int,
        default=3,
        help="Min complied examples per category for geometry analysis (default: 3)",
    )
    p.add_argument(
        "--min-refused",
        type=int,
        default=3,
        help="Min refused examples per category for within-gap analysis (default: 3)",
    )
    return p.parse_args()


def angular_distance_deg(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Angle in degrees between two vectors (arccos of cosine similarity)."""
    v1n = F.normalize(v1.float(), dim=0)
    v2n = F.normalize(v2.float(), dim=0)
    cos_sim = float((v1n * v2n).sum().clamp(-1.0, 1.0))
    return math.degrees(math.acos(cos_sim))


def mean_angular_distance_deg(vecs1: list, vecs2: list) -> float:
    """Average angular distance across paired layer vectors."""
    angles = [angular_distance_deg(v1, v2) for v1, v2 in zip(vecs1, vecs2)]
    return sum(angles) / len(angles)


def load_steering_tensor(path: str, component: str) -> torch.Tensor:
    """Load a steering vector .pt file, return tensor of shape (n_layers, hidden)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        return data.float()
    if isinstance(data, dict):
        for key in [f"steering_vectors_{component}", "steering_vectors"]:
            if key in data and isinstance(data[key], torch.Tensor):
                return data[key].float()
        # fall through: dict of {int: tensor} layer maps
        max_layer = max(int(k) for k in data if str(k).lstrip("-").isdigit())
        tensor = torch.stack([data[i].float() for i in range(max_layer + 1)])
        return tensor
    raise ValueError(f"Unexpected steering vector format at {path}: {type(data)}")


def category_key(meta: dict) -> str | None:
    """Return the grouping key for a sample: category field or split fallback."""
    return meta.get("category") or meta.get("split") or None


def analyze_wrmd_geometry(
    cat_vecs_dir: Path,
    global_vec_path: str,
    component: str,
    target_layers: list[int],
) -> dict:
    """
    Compute pairwise angular distances between per-category WRMD vectors and
    each category's angle from the global WRMD vector.

    Returns a dict ready for JSON serialization.
    """
    print("\n=== WRMD refusal-direction angular distance ===")

    global_tensor = load_steering_tensor(global_vec_path, component)

    # Load all category .pt files
    cat_layer_vecs: dict[str, list[torch.Tensor]] = {}
    for pt_file in sorted(cat_vecs_dir.glob("*.pt")):
        cat_name = pt_file.stem
        try:
            tensor = load_steering_tensor(str(pt_file), component)
        except Exception as e:
            print(f"  [WARN] could not load {pt_file.name}: {e}")
            continue
        # Extract vectors for target layers
        vecs = []
        for layer in target_layers:
            if layer < tensor.shape[0]:
                vecs.append(tensor[layer])
            else:
                print(
                    f"  [WARN] layer {layer} out of range for {cat_name} ({tensor.shape[0]} layers)"
                )
        if len(vecs) == len(target_layers):
            cat_layer_vecs[cat_name] = vecs

    cats = sorted(cat_layer_vecs.keys())
    print(f"  Loaded {len(cats)} category vectors: {cats}")

    # Extract global vector at target layers
    global_layer_vecs = [global_tensor[layer] for layer in target_layers]

    # Angle from global vector
    angle_from_global: dict[str, float] = {}
    for cat in cats:
        angle_from_global[cat] = mean_angular_distance_deg(cat_layer_vecs[cat], global_layer_vecs)

    # Pairwise angles
    pairwise: dict[str, float] = {}
    for i, cat_a in enumerate(cats):
        for cat_b in cats[i + 1 :]:
            key = f"{cat_a}__vs__{cat_b}"
            pairwise[key] = mean_angular_distance_deg(cat_layer_vecs[cat_a], cat_layer_vecs[cat_b])

    # Summary statistics
    pairwise_vals = list(pairwise.values())
    stats = {
        "n_pairs": len(pairwise_vals),
        "min_deg": round(min(pairwise_vals), 2),
        "mean_deg": round(sum(pairwise_vals) / len(pairwise_vals), 2),
        "max_deg": round(max(pairwise_vals), 2),
        "std_deg": round(
            math.sqrt(
                sum((v - sum(pairwise_vals) / len(pairwise_vals)) ** 2 for v in pairwise_vals)
                / len(pairwise_vals)
            ),
            2,
        ),
    }
    print(
        f"  Pairwise: min={stats['min_deg']}° mean={stats['mean_deg']}° "
        f"max={stats['max_deg']}° std={stats['std_deg']}°"
    )
    print(f"  Angle from global (outliers):")
    for cat, ang in sorted(angle_from_global.items(), key=lambda x: -x[1])[:5]:
        print(f"    {cat}: {ang:.1f}°")

    # Sort pairwise by descending angle
    pairwise_sorted = dict(sorted(pairwise.items(), key=lambda x: -x[1]))

    return {
        "layers_analysed": target_layers,
        "angle_from_global_avg": {
            k: round(v, 4) for k, v in sorted(angle_from_global.items(), key=lambda x: -x[1])
        },
        "pairwise_avg": {k: round(v, 4) for k, v in pairwise_sorted.items()},
        "summary": stats,
    }


def analyze_comply_geometry(
    act_path: str,
    target_layers: list[int],
    component: str,
    min_complied: int,
    min_refused: int,
) -> dict:
    """
    Compute pairwise angular distances between per-category mean complied activations,
    and within-category comply↔refuse gap.

    Returns a dict ready for JSON serialization.
    """
    print("\n=== Compliance activation angular distance ===")

    act = torch.load(act_path, map_location="cpu", weights_only=False)
    act_key = f"activations_{component}"
    if act_key not in act:
        act_key = "activations"
    mlp = act[act_key].float()  # (N, n_layers, hidden)
    labels = act["labels"]  # 1=refused, 0=complied
    meta = act["metadata"]
    print(f"  {len(labels)} samples | {mlp.shape[1]} layers | {mlp.shape[2]} hidden")

    # Group by category key
    cat_comply: dict[str, list] = defaultdict(list)
    cat_refuse: dict[str, list] = defaultdict(list)
    for i, m in enumerate(meta):
        key = category_key(m)
        if key is None:
            continue
        if labels[i] == 0:
            cat_comply[key].append(mlp[i])  # (n_layers, hidden)
        else:
            cat_refuse[key].append(mlp[i])

    # Compute mean complied activation per category at target layers
    cat_mean_vecs: dict[str, list[torch.Tensor]] = {}
    n_complied_per_cat: dict[str, int] = {}
    n_refused_per_cat: dict[str, int] = {}

    all_cats = sorted(set(list(cat_comply.keys()) + list(cat_refuse.keys())))
    for cat in all_cats:
        n_c = len(cat_comply[cat])
        n_r = len(cat_refuse[cat])
        n_complied_per_cat[cat] = n_c
        n_refused_per_cat[cat] = n_r
        if n_c >= min_complied:
            stacked = torch.stack(cat_comply[cat])  # (n_c, n_layers, hidden)
            mean_act = stacked.mean(0)  # (n_layers, hidden)
            vecs = [mean_act[layer] for layer in target_layers]
            cat_mean_vecs[cat] = vecs

    eligible_cats = sorted(cat_mean_vecs.keys())
    print(f"  {len(eligible_cats)} categories with >= {min_complied} complied samples")

    # Pairwise angles between complied means
    pairwise: dict[str, float] = {}
    for i, cat_a in enumerate(eligible_cats):
        for cat_b in eligible_cats[i + 1 :]:
            key = f"{cat_a}__vs__{cat_b}"
            pairwise[key] = mean_angular_distance_deg(cat_mean_vecs[cat_a], cat_mean_vecs[cat_b])

    pairwise_vals = list(pairwise.values())
    stats = {
        "n_pairs": len(pairwise_vals),
        "min_deg": round(min(pairwise_vals), 2),
        "mean_deg": round(sum(pairwise_vals) / len(pairwise_vals), 2),
        "max_deg": round(max(pairwise_vals), 2),
        "std_deg": round(
            math.sqrt(
                sum((v - sum(pairwise_vals) / len(pairwise_vals)) ** 2 for v in pairwise_vals)
                / len(pairwise_vals)
            ),
            2,
        ),
    }
    print(
        f"  Compliance pairwise: min={stats['min_deg']}° mean={stats['mean_deg']}° "
        f"max={stats['max_deg']}° std={stats['std_deg']}°"
    )

    # Within-category comply↔refuse gap
    within_gap: dict[str, float] = {}
    for cat in all_cats:
        n_c = len(cat_comply[cat])
        n_r = len(cat_refuse[cat])
        if n_c < min_complied or n_r < min_refused:
            continue
        comply_mean = torch.stack(cat_comply[cat]).mean(0)  # (n_layers, hidden)
        refuse_mean = torch.stack(cat_refuse[cat]).mean(0)
        comply_vecs = [comply_mean[layer] for layer in target_layers]
        refuse_vecs = [refuse_mean[layer] for layer in target_layers]
        within_gap[cat] = mean_angular_distance_deg(comply_vecs, refuse_vecs)

    print(f"  Within-category gaps:")
    for cat, ang in sorted(within_gap.items(), key=lambda x: -x[1])[:5]:
        print(f"    {cat}: {ang:.1f}° (largest gap)")
    for cat, ang in sorted(within_gap.items(), key=lambda x: x[1])[:3]:
        print(f"    {cat}: {ang:.1f}° (smallest gap)")

    pairwise_sorted = dict(sorted(pairwise.items(), key=lambda x: -x[1]))
    within_sorted = dict(sorted(within_gap.items(), key=lambda x: -x[1]))

    return {
        "description": f"Angular distances on complied-only mean activations ({component}, top-{len(target_layers)} layers)",
        "layers": target_layers,
        "comply_pairwise_deg": {k: round(v, 4) for k, v in pairwise_sorted.items()},
        "within_category_comply_vs_refused_deg": {k: round(v, 4) for k, v in within_sorted.items()},
        "summary": stats,
        "n_complied_per_category": n_complied_per_cat,
        "n_refused_per_category": n_refused_per_cat,
    }


def print_contrast(wrmd_stats: dict, comply_stats: dict) -> None:
    """Print the key geometric contrast between the two spaces."""
    ws = wrmd_stats["summary"]
    cs = comply_stats["summary"]
    print("\n=== GEOMETRIC CONTRAST ===")
    print(f"{'Space':<45} {'Min°':>6} {'Mean°':>7} {'Max°':>7} {'Std°':>6}")
    print(
        f"{'Compliance activations (pairwise)':<45} {cs['min_deg']:>6.1f} {cs['mean_deg']:>7.1f} "
        f"{cs['max_deg']:>7.1f} {cs['std_deg']:>6.1f}"
    )
    print(
        f"{'WRMD steering vectors (refusal direction)':<45} {ws['min_deg']:>6.1f} {ws['mean_deg']:>7.1f} "
        f"{ws['max_deg']:>7.1f} {ws['std_deg']:>6.1f}"
    )
    print(
        f"\nCompliance activations are {'more' if cs['std_deg'] < ws['std_deg'] else 'less'} "
        f"compressed (std={cs['std_deg']:.1f}° vs {ws['std_deg']:.1f}°)"
    )


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrmd_results = analyze_wrmd_geometry(
        cat_vecs_dir=Path(args.category_vectors_dir),
        global_vec_path=args.global_vector,
        component=args.component,
        target_layers=args.layers,
    )

    comply_results = analyze_comply_geometry(
        act_path=args.activations,
        target_layers=args.layers,
        component=args.component,
        min_complied=args.min_complied,
        min_refused=args.min_refused,
    )

    wrmd_path = out_dir / "angular_distance_analysis.json"
    comply_path = out_dir / "comply_angular_distance_analysis.json"

    with open(wrmd_path, "w") as f:
        json.dump(wrmd_results, f, indent=2)
    print(f"\nWRMD analysis saved to {wrmd_path}")

    with open(comply_path, "w") as f:
        json.dump(comply_results, f, indent=2)
    print(f"Compliance analysis saved to {comply_path}")

    print_contrast(wrmd_results, comply_results)


if __name__ == "__main__":
    main()
