#!/usr/bin/env python3
"""Perpendicular-specific null test for the hedge off-line bow (§9.4.7 #1 remainder).

The §9.4.4 bootstrap left "is bow > 0 real?" inconclusive because the noise floor
used the FULL centroid-displacement wobble (all directions), which over-penalises.
This test builds the correct null: how much *perpendicular* (off-line) bow does a
cluster that genuinely lies ON the refuse→comply line show from finite-sample
noise alone? refuse and comply ARE the line endpoints, so bootstrap-resampling
them and measuring their perpendicular distance from the FIXED full-data line
gives exactly that on-line null. Hedge bow is significant iff it sits clearly
above that null.

CPU only; reads the cached cells from hedge_bootstrap_ci_mistral.py.

    python scripts/hedge_bow_nulltest_mistral.py [--B 2000]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

OUT_DIR = "outputs/mistral-7b-instruct-v0-2/experiments/hedge_regime_geometry"
CELLS_PT = os.path.join(OUT_DIR, "hedge_cells.pt")


def centroid(cells, idx_multiset, label):
    acc = [
        c["acts"]
        for i in idx_multiset
        for c in cells
        if c["prompt_idx"] == i and c["label"] == label
    ]
    if not acc:
        return None
    return np.concatenate(acc, axis=0).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    blob = torch.load(CELLS_PT, weights_only=False)
    cells = blob["cells"]
    prompts = sorted({c["prompt_idx"] for c in cells})
    n = max(prompts) + 1

    # full-data line
    a = centroid(cells, range(n), "refuse")
    b = centroid(cells, range(n), "comply")
    h = centroid(cells, range(n), "hedge")
    seg = b - a
    seg_n = float(np.linalg.norm(seg))

    def bow_of(cen):
        t = float(((cen - a) @ seg) / (seg_n**2))
        return float(np.linalg.norm((cen - a) - t * seg)) / seg_n

    obs = {k: bow_of(c) for k, c in [("refuse", a), ("hedge", h), ("comply", b)]}
    # refuse/comply bow are ~0 by construction (they define the line)

    rng = np.random.default_rng(args.seed)
    dist = {"refuse": [], "comply": [], "hedge": []}
    for _ in range(args.B):
        idx = rng.integers(0, n, size=n).tolist()
        for lab in dist:
            c = centroid(cells, idx, lab)
            if c is not None:
                dist[lab].append(bow_of(c))

    def pct(arr, p):
        return float(np.percentile(np.array(arr), p))

    print("=" * 72)
    print(f"PERPENDICULAR-SPECIFIC NULL TEST  (B={args.B})")
    print("=" * 72)
    print(f"  ‖refuse→comply‖ = {seg_n:.3f}")
    print(f"\n  {'cluster':>8}  {'bow p50':>8}  {'bow p95':>8}  {'bow p97.5':>9}  role")
    for lab, role in [("refuse", "on-line NULL"), ("comply", "on-line NULL"), ("hedge", "TEST")]:
        d = dist[lab]
        print(f"  {lab:>8}  {pct(d,50):>8.3f}  {pct(d,95):>8.3f}  {pct(d,97.5):>9.3f}  {role}")

    null_ceiling = max(pct(dist["refuse"], 97.5), pct(dist["comply"], 97.5))
    hedge_lo = pct(dist["hedge"], 2.5)
    # one-sided p: fraction of on-line null (pooled refuse+comply) >= observed hedge bow
    null_pool = np.array(dist["refuse"] + dist["comply"])
    p_val = float((null_pool >= obs["hedge"]).mean())

    print(f"\n  observed hedge bow            = {obs['hedge']:.3f}")
    print(f"  hedge bow 2.5th pctile        = {hedge_lo:.3f}")
    print(f"  on-line null ceiling (97.5%)  = {null_ceiling:.3f}  (max of refuse/comply)")
    print(f"  one-sided p (null bow ≥ obs)  = {p_val:.4f}")
    verdict = (
        "SIGNIFICANT: hedge sits off the line beyond on-line sampling noise"
        if hedge_lo > null_ceiling and p_val < 0.05
        else "NOT significant: hedge bow within on-line noise"
    )
    print(f"\n  VERDICT: {verdict}")

    res = {
        "seg_norm": round(seg_n, 4),
        "observed_hedge_bow": round(obs["hedge"], 4),
        "hedge_bow_lo2.5": round(hedge_lo, 4),
        "null_ceiling_97.5": round(null_ceiling, 4),
        "null_refuse_p97.5": round(pct(dist["refuse"], 97.5), 4),
        "null_comply_p97.5": round(pct(dist["comply"], 97.5), 4),
        "one_sided_p": round(p_val, 5),
        "B": args.B,
        "verdict": verdict,
    }
    with open(os.path.join(OUT_DIR, "hedge_bow_nulltest.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  Saved {os.path.join(OUT_DIR, 'hedge_bow_nulltest.json')}")


if __name__ == "__main__":
    main()
