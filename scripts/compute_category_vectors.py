"""
Compute per-category MD steering vectors from saved activations.

For categories with insufficient complied samples (<MIN_COMPLIED), falls back
to the global complied mean as the "complied" anchor. This is appropriate for
categories where the model almost never complies at baseline — we steer toward
general compliance rather than category-specific compliance.

Output:
    <output-dir>/
        <category_slug>.pt          # steering vector in same format as compute_wrmd.py
        category_summary.json       # index for optimize_alpha.py --stable-categories

Usage:
    python scripts/compute_category_vectors.py \
        --activations outputs/qwen3-5-9b/.../extract_activations/activations.pt \
        --output-dir outputs/qwen3-5-9b/.../category_vectors \
        --min-refused 15 \
        --min-complied 3
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def compute_md_vector(refused_acts: torch.Tensor, complied_acts: torch.Tensor) -> torch.Tensor:
    """Mean difference: mean(refused) - mean(complied), per layer."""
    return refused_acts.mean(0) - complied_acts.mean(0)  # (layers, hidden)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activations", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--min-refused", type=int, default=15, help="Min refused samples per category (default 15)"
    )
    p.add_argument(
        "--min-complied",
        type=int,
        default=3,
        help="Min complied samples for category-specific complied mean; "
        "falls back to global complied mean if below threshold (default 3)",
    )
    p.add_argument("--component", default="mlp")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading activations: {args.activations}")
    act = torch.load(args.activations, map_location="cpu")
    mlp = act["activations_mlp"].float()  # (N, layers, hidden)
    labels = act["labels"]  # 1=refused, 0=complied
    meta = act["metadata"]
    num_layers = act["num_layers"]
    hidden_size = act["hidden_size"]
    print(f"  {len(labels)} samples | {num_layers} layers | {hidden_size} hidden")

    # Global complied mean (fallback anchor)
    complied_mask = labels == 0
    global_complied = mlp[complied_mask].mean(0)  # (layers, hidden)
    print(f"  Global complied pool: {complied_mask.sum().item()} samples")

    # Group by category (BeaverTails only)
    from collections import defaultdict

    cat_refused = defaultdict(list)
    cat_complied = defaultdict(list)
    for i, m in enumerate(meta):
        if m.get("source_dataset") != "PKU-Alignment/BeaverTails-Evaluation":
            continue
        cat = m.get("category")
        if not cat:
            continue
        if labels[i] == 1:
            cat_refused[cat].append(mlp[i])
        else:
            cat_complied[cat].append(mlp[i])

    summary = {}
    skipped = []
    computed = []

    for cat in sorted(cat_refused.keys()):
        n_r = len(cat_refused[cat])
        n_c = len(cat_complied[cat])

        if n_r < args.min_refused:
            skipped.append(f"  [SKIP] {cat}: only {n_r} refused (need {args.min_refused})")
            continue

        refused_acts = torch.stack(cat_refused[cat])  # (n_r, layers, hidden)

        if n_c >= args.min_complied:
            complied_acts = torch.stack(cat_complied[cat])  # (n_c, layers, hidden)
            fallback = False
        else:
            complied_acts = global_complied.unsqueeze(0)  # (1, layers, hidden)
            fallback = True

        vec = compute_md_vector(refused_acts, complied_acts)  # (layers, hidden)

        slug = slugify(cat)
        fname = f"{slug}.pt"
        fpath = out_dir / fname

        torch.save(
            {
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "method": "md",
                "component": args.component,
                "category": cat,
                "num_refusal_samples": n_r,
                "num_compliant_samples": n_c,
                "complied_fallback": fallback,
                "steering_vectors": vec,  # (layers, hidden) — same key as global vectors
            },
            fpath,
        )

        summary[cat] = {
            "file": fname,
            "slug": slug,
            "n_refused": n_r,
            "n_complied": n_c,
            "complied_fallback": fallback,
        }
        tag = " [fallback complied]" if fallback else ""
        computed.append(f"  {cat}: {n_r}R {n_c}C{tag} → {fname}")

    for s in skipped:
        print(s)
    for c in computed:
        print(c)

    summary_path = out_dir / "category_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {len(computed)} category vectors → {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
