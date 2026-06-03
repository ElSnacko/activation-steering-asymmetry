#!/usr/bin/env python3
"""
Bootstrap confidence intervals for the per-layer axis-rotation finding.

CONTEXT
-------
analyze_axis_rotation.py reports a single point estimate: the mean angle between
consecutive-layer WRMD vectors (~93.4 deg Qwen, ~88.8 deg Mistral). That number
has no uncertainty attached. This script adds two complementary CIs, because
they answer different questions and a reviewer will ask which one you mean:

  (A) DISPERSION across depth  [descriptive]
      The 31 consecutive-layer pairs are a census of one fixed model, not a
      random sample. A bootstrap over the 31 per-pair angles describes how much
      the angle varies across depth. It is NOT an inferential CI on a true
      parameter -- it is a dispersion summary. Reported as percentile bands over
      the per-pair angles. Use it to say "the ~90 deg holds tightly across all
      depths" not "the true mean is 93.4 +/- x".

  (B) PROMPT-LEVEL bootstrap   [inferential -- the one that matters]
      Each layer's WRMD vector is estimated from a finite pool of refused /
      complied prompt activations. Resample that pool with replacement B times,
      recompute every layer's WRMD vector, recompute the consecutive-layer
      angles, and take the distribution of the mean angle. This answers the real
      objection: "would you still get ~90 deg with different prompts?" If the CI
      is tight and far from 0, the rotation is not a sampling artifact.

(A) needs only the steering-vector .pt (runs anywhere). (B) needs the per-sample
activations + labels (runs on the box that has activations_*.pt).

USAGE
-----
  # (A) dispersion CI from saved vectors -- no activations needed:
  python scripts/bootstrap_axis_rotation.py dispersion \
      --steering-vectors model_a.pt model_b.pt --labels Mistral Qwen \
      --out outputs/axis_rotation_comparison

  # (B) prompt-level CI -- needs activations + the same WRMD recipe:
  python scripts/bootstrap_axis_rotation.py prompt \
      --activations outputs/.../extract_activations/activations_beavertails_sanitized.pt \
      --component mlp --label Qwen3.5-9B \
      --n-boot 1000 --seed 42 \
      --out outputs/qwen3-5-9b/experiments/axis_rotation_ci
"""

import argparse
import json
import math
import os

import numpy as np

# ----------------------------------------------------------------------------
# Shared: angle computation (matches analyze_axis_rotation.py conventions)
# ----------------------------------------------------------------------------


def consecutive_angles(vecs: np.ndarray) -> np.ndarray:
    """vecs: [L, H]. Returns [L-1] angles (deg) between consecutive rows.

    Mirrors analyze_axis_rotation.compute_rotation: L2-normalize, dot, clamp,
    arccos in degrees.
    """
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    vn = vecs / np.clip(n, 1e-12, None)
    cs = np.clip(np.sum(vn[:-1] * vn[1:], axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cs))


def pct_ci(samples: np.ndarray, lo=2.5, hi=97.5):
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


# ----------------------------------------------------------------------------
# (A) dispersion CI over the per-pair angles (vectors only)
# ----------------------------------------------------------------------------


def load_vectors(path: str) -> np.ndarray:
    import torch

    sv = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sv, dict):
        for k in ("steering_vectors", "steering_vectors_mlp", "steering_vectors_attn"):
            if k in sv:
                sv = sv[k]
                break
        else:
            sv = next(iter(sv.values()))
    return sv.float().cpu().numpy()


def run_dispersion(args):
    labels = args.labels or [os.path.basename(p) for p in args.steering_vectors]
    rng = np.random.default_rng(args.seed)
    out = {}
    for path, label in zip(args.steering_vectors, labels):
        vecs = load_vectors(path)
        angles = consecutive_angles(vecs)
        n = len(angles)
        # bootstrap the MEAN of the per-pair angles (dispersion of depth census)
        means = np.array([angles[rng.integers(0, n, n)].mean() for _ in range(args.n_boot)])
        lo, hi = pct_ci(means)
        out[label] = {
            "n_pairs": int(n),
            "mean_angle_deg": round(float(angles.mean()), 2),
            "median_angle_deg": round(float(np.median(angles)), 2),
            "std_angle_deg": round(float(angles.std(ddof=1)), 2),
            "min_angle_deg": round(float(angles.min()), 2),
            "max_angle_deg": round(float(angles.max()), 2),
            "dispersion_bootstrap_ci95": [round(lo, 2), round(hi, 2)],
            "note": "DESCRIPTIVE dispersion across depth (31 pairs are a census, not a sample).",
        }
        print(
            f"{label}: mean {out[label]['mean_angle_deg']} deg  "
            f"std {out[label]['std_angle_deg']}  "
            f"range {out[label]['min_angle_deg']}-{out[label]['max_angle_deg']}  "
            f"dispersion-CI95 {out[label]['dispersion_bootstrap_ci95']}"
        )
    _save(args.out, "axis_rotation_dispersion_ci.json", out)


# ----------------------------------------------------------------------------
# (B) prompt-level bootstrap (activations + labels)
# ----------------------------------------------------------------------------


def md_vector_per_layer(acts: np.ndarray, refused: np.ndarray) -> np.ndarray:
    """acts: [N, L, H]; refused: bool [N]. Mean-difference per layer:
    mean(refused) - mean(complied). Returns [L, H]."""
    r = acts[refused].mean(axis=0)
    c = acts[~refused].mean(axis=0)
    return r - c


def run_prompt(args):
    import torch

    blob = torch.load(args.activations, map_location="cpu", weights_only=False)
    # Activations file layout from extract_activations.py:
    #   key activations_mlp / activations_attn: [N, L, H]
    #   plus a per-sample score or label field.
    key = f"activations_{args.component}"
    if isinstance(blob, dict) and key in blob:
        acts = blob[key]
    elif isinstance(blob, dict) and "activations" in blob:
        acts = blob["activations"]
    else:
        raise SystemExit(
            f"Could not find '{key}' or 'activations' in {args.activations}. "
            f"Top keys: {list(blob.keys()) if isinstance(blob, dict) else type(blob)}"
        )
    acts = np.asarray(acts.float().cpu().numpy() if hasattr(acts, "float") else acts)

    # Derive refused mask from saved judge scores. Refusal threshold matches
    # the pipeline default: score > 0.1 => refused, < -0.1 => complied.
    scores = None
    for k in ("scores", "judge_scores", "answer_censor_score", "labels", "sample_scores"):
        if isinstance(blob, dict) and k in blob:
            scores = np.asarray(blob[k]).astype(float).ravel()
            break
    if scores is None:
        raise SystemExit(
            "No per-sample score field found in activations file "
            "(looked for scores/judge_scores/answer_censor_score/labels/sample_scores). "
            "Cannot derive refused/complied mask for the prompt-level bootstrap."
        )
    # Binary {0, 1} encoding (extract_activations.py default: 1=refused, 0=complied).
    # Remap to {+1, -1} so the standard threshold logic applies unchanged.
    unique_vals = set(np.unique(scores).tolist())
    if unique_vals <= {0.0, 1.0}:
        scores = scores * 2.0 - 1.0  # 0 -> -1 (complied), 1 -> +1 (refused)
    keep = np.abs(scores) > 0.1
    acts, scores = acts[keep], scores[keep]
    refused = scores > 0.1
    n_ref, n_comp = int(refused.sum()), int((~refused).sum())
    print(
        f"{args.label}: N={len(acts)} usable ({n_ref} refused / {n_comp} complied), "
        f"L={acts.shape[1]}, H={acts.shape[2]}"
    )
    if n_ref < 5 or n_comp < 5:
        raise SystemExit("Too few refused or complied samples for a stable bootstrap.")

    # point estimate
    point = consecutive_angles(md_vector_per_layer(acts, refused)).mean()

    rng = np.random.default_rng(args.seed)
    ref_idx = np.where(refused)[0]
    comp_idx = np.where(~refused)[0]
    boot_means = np.empty(args.n_boot)
    for b in range(args.n_boot):
        rs = rng.choice(ref_idx, size=len(ref_idx), replace=True)
        cs = rng.choice(comp_idx, size=len(comp_idx), replace=True)
        sub = np.concatenate([rs, cs])
        mask = np.zeros(len(acts), dtype=bool)
        mask[rs] = True  # refused-resampled rows
        # rebuild MD from the resampled pools directly (cleaner than mask):
        r = acts[rs].mean(axis=0)
        c = acts[cs].mean(axis=0)
        boot_means[b] = consecutive_angles(r - c).mean()
        if (b + 1) % max(1, args.n_boot // 10) == 0:
            print(f"  bootstrap {b+1}/{args.n_boot}")
    lo, hi = pct_ci(boot_means)
    out = {
        args.label: {
            "n_refused": n_ref,
            "n_complied": n_comp,
            "n_boot": args.n_boot,
            "point_mean_angle_deg": round(float(point), 2),
            "bootstrap_mean_angle_deg": round(float(boot_means.mean()), 2),
            "bootstrap_ci95_deg": [round(lo, 2), round(hi, 2)],
            "bootstrap_se_deg": round(float(boot_means.std(ddof=1)), 3),
            "note": "INFERENTIAL: resamples the prompt pool; answers 'would different prompts change ~90 deg?'",
        }
    }
    print(
        f"\n{args.label}: point {out[args.label]['point_mean_angle_deg']} deg, "
        f"bootstrap mean {out[args.label]['bootstrap_mean_angle_deg']} deg, "
        f"95% CI {out[args.label]['bootstrap_ci95_deg']}, "
        f"SE {out[args.label]['bootstrap_se_deg']}"
    )
    _save(args.out, f"axis_rotation_prompt_ci_{args.label.replace('/', '_')}.json", out)


def _save(out_dir, name, obj):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved -> {p}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("dispersion", help="descriptive CI from saved vectors (no activations)")
    d.add_argument("--steering-vectors", nargs="+", required=True)
    d.add_argument("--labels", nargs="+", default=None)
    d.add_argument("--n-boot", type=int, default=2000)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--out", required=True)
    d.set_defaults(func=run_dispersion)

    p = sub.add_parser("prompt", help="inferential CI from activations (resamples prompts)")
    p.add_argument("--activations", required=True)
    p.add_argument("--component", default="mlp", choices=["mlp", "attn", "layer"])
    p.add_argument("--label", default="model")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_prompt)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
