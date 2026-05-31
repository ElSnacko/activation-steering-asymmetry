"""
Layer correlation analysis to identify effective steering layers.

Computes correlations between activation projections and judge scores to determine
which layers are most effective for steering interventions.  Also provides tools for
comparing per-category steering vectors via pairwise angular distances.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


def compute_layer_correlations(
    activations_file,
    steering_vectors_file,
    judge_scores_file=None,
    component="attn",
    method="auc",
):
    """
    Compute layer-wise separability between activation projections and judge scores.

    Args:
        activations_file: Path to activation .pt file
        steering_vectors_file: Path to steering vector .pt file
        judge_scores_file: Optional separate judge scores JSON file
        component: Which activation component to use ("layer", "attn", "mlp").
        method: Separability metric — "auc" (default), "spearman", or "pearson".
            "auc"      — AUC of binary comply/refuse classification via projection.
                         No ordinal assumption; best choice with 4-class judge scores.
            "spearman" — Rank correlation. Appropriate for ordinal judge scores.
            "pearson"  — Pearson r. Assumes interval-level scores; kept for backward
                         compatibility with results computed under the old binary judge.

    Returns:
        Tuple of (correlations, projections_all_layers, judge_scores)
    """
    print("[LOAD] Loading data...")

    # Load activations
    act_data = torch.load(activations_file, weights_only=True)
    labels = act_data["labels"]

    # Select activations for the requested component
    act_key = "activations" if component == "layer" else f"activations_{component}"
    if act_key in act_data:
        activations = act_data[act_key]
    elif "activations" in act_data:
        activations = act_data["activations"]
    else:
        raise ValueError(f"No activations found for component '{component}'")

    # Load steering vectors
    vec_data = torch.load(steering_vectors_file, weights_only=True)
    sv_key = f"steering_vectors_{component}"
    if sv_key in vec_data:
        steering_vectors = vec_data[sv_key]
    else:
        steering_vectors = vec_data["steering_vectors"]  # [num_layers, hidden_size]

    print(f"   Component: {component}")

    # Multi-rank vectors: use v_1 (primary direction) for correlation analysis
    if steering_vectors.dim() == 3:
        rank = steering_vectors.shape[1]
        print(f"   Multi-rank vectors detected (rank={rank}), using v_1 for correlations")
        steering_vectors = steering_vectors[:, 0, :]  # [num_layers, hidden_size]

    print("🔧 Normalizing steering vectors to unit length...")
    steering_vectors = steering_vectors / steering_vectors.norm(dim=-1, keepdim=True)

    # Get judge scores from metadata or separate file
    if "metadata" in act_data:
        judge_scores = torch.tensor([m["score"] for m in act_data["metadata"]])
        print("[OK] Using judge scores from activations metadata")
    elif judge_scores_file:
        with open(judge_scores_file) as f:
            data = json.load(f)
        judge_scores = torch.tensor([item["answer_censor_score"] for item in data])
        print("[OK] Using judge scores from separate file")
    else:
        raise ValueError("No judge scores found!")

    print(f"   Samples: {len(activations)}")
    print(f"   Layers: {activations.shape[1]}")
    print(f"   Judge score range: [{judge_scores.min():.2f}, {judge_scores.max():.2f}]")
    print(f"   Layer selection method: {method}")

    # Binary labels for AUC: comply (score <= 0) = 0, refuse (score > 0) = 1.
    # Uncertain / exactly-zero scores are treated as comply for AUC purposes.
    binary_labels = (judge_scores.numpy() > 0).astype(int)
    n_refuse = binary_labels.sum()
    n_comply = len(binary_labels) - n_refuse
    if method == "auc":
        print(f"   AUC binary split: {n_refuse} refuse / {n_comply} comply")

    # Compute layer-wise separability
    print("\n[COMPUTE] Computing layer-wise separability...")

    correlations = []
    projections_all_layers = []

    for layer in range(activations.shape[1]):
        layer_acts = activations[:, layer, :]  # [N, hidden_size]
        steering_vec = steering_vectors[layer]  # [hidden_size]
        projections = (layer_acts @ steering_vec).float().numpy()  # [N]
        projections_all_layers.append(projections)

        scores_np = judge_scores.numpy()

        if method == "auc":
            # AUC: does the projection separate comply from refuse?
            # Both classes must be present for AUC to be defined.
            if n_refuse == 0 or n_comply == 0:
                auc = 0.5
            else:
                auc = float(roc_auc_score(binary_labels, projections))
            # Store as correlation-compatible value: (AUC - 0.5) * 2 maps [0.5,1] -> [0,1]
            # abs_correlation stores raw AUC for ranking; correlation stores signed shift.
            corr = (auc - 0.5) * 2  # signed: positive if projection predicts refuse
            abs_corr = auc
            p_value = float("nan")
            extra = {"auc": auc}
            print(f"  Layer {layer:2d}: AUC={auc:.4f}")
        elif method == "spearman":
            corr, p_value = spearmanr(projections, scores_np)
            corr, p_value = float(corr), float(p_value)
            abs_corr = abs(corr)
            extra = {}
            print(f"  Layer {layer:2d}: rho={corr:+.4f} (p={p_value:.2e})")
        else:  # pearson
            corr, p_value = pearsonr(projections, scores_np)
            corr, p_value = float(corr), float(p_value)
            abs_corr = abs(corr)
            extra = {}
            print(f"  Layer {layer:2d}: r={corr:+.4f} (p={p_value:.2e})")

        entry = {
            "layer": int(layer),
            "correlation": corr,
            "abs_correlation": abs_corr,
            "p_value": p_value,
            "method": method,
            "projection_mean": float(projections.mean()),
            "projection_std": float(projections.std()),
        }
        entry.update(extra)
        correlations.append(entry)

    return correlations, projections_all_layers, judge_scores.numpy()


def plot_correlations(correlations, output_dir=None, output_file="layer_correlations.png"):
    """
    Plot correlation by layer.

    Args:
        correlations: List of correlation dicts from compute_layer_correlations
        output_dir: Directory to save plot
        output_file: Filename for plot
    """
    from .utils import get_output_path

    if output_dir is None:
        output_dir = get_output_path(script_name="find_best_layers")

    layers = [c["layer"] for c in correlations]
    corrs = [c["correlation"] for c in correlations]
    abs_corrs = [c["abs_correlation"] for c in correlations]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Signed correlation
    ax1.plot(layers, corrs, marker="o", linewidth=2, markersize=6)
    ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("Correlation (r)", fontsize=12)
    ax1.set_title("Layer-wise Correlation: Activation Projection vs Judge Score", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Highlight top layers
    top_5_idx = sorted(range(len(abs_corrs)), key=lambda i: abs_corrs[i], reverse=True)[:5]
    for idx in top_5_idx:
        ax1.axvline(x=layers[idx], color="r", linestyle="--", alpha=0.3)
        ax1.text(layers[idx], corrs[idx], f" L{layers[idx]}", fontsize=9, color="red")

    # Plot 2: Absolute correlation
    ax2.bar(layers, abs_corrs, alpha=0.7)
    ax2.set_xlabel("Layer Index", fontsize=12)
    ax2.set_ylabel("|Correlation|", fontsize=12)
    ax2.set_title("Absolute Correlation by Layer", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    # Highlight top 5
    for idx in top_5_idx:
        ax2.bar(layers[idx], abs_corrs[idx], color="red", alpha=0.7)

    plt.tight_layout()

    # Save plot
    if not os.path.isabs(output_file):
        output_path = os.path.join(output_dir, output_file)
    else:
        output_path = output_file

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[PLOT] Saved plot: {output_path}")


def find_best_layers(correlations, top_k=5):
    """
    Find top K layers by absolute correlation.

    Args:
        correlations: List of correlation dicts
        top_k: Number of top layers to return

    Returns:
        List of best layer indices
    """
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations, key=lambda x: x["abs_correlation"], reverse=True)

    print(f"\n[TOP] Top {top_k} Layers by Correlation:")
    print(f"{'Layer':<8} {'r':<10} {'|r|':<10} {'p-value':<12}")
    print("-" * 45)

    best_layers = []
    for i, c in enumerate(sorted_corrs[:top_k]):
        print(
            f"{c['layer']:<8} {c['correlation']:+.4f}    {c['abs_correlation']:.4f}    {c['p_value']:.2e}"
        )
        best_layers.append(c["layer"])

    return best_layers


def compute_per_category_layer_correlations(
    activations_file,
    category_vector_dir,
    component="attn",
    top_k=5,
    min_samples=10,
):
    """Compute per-category layer correlations using category-specific steering vectors.

    For each category, filters to that category's refusal samples + all compliant
    samples, projects onto the category's own steering vector, and computes
    per-layer correlations. This reveals whether different refusal categories
    concentrate at different layer depths.

    Args:
        activations_file: Path to activations .pt file with category metadata.
        category_vector_dir: Directory containing per-category steering vector
            files (e.g. steering_vectors_md_animal_abuse.pt). Also looks for a
            category_summary.json to discover available categories.
        component: Which activation component to analyze ("attn", "mlp", "layer").
        top_k: Number of top layers to report per category.
        min_samples: Minimum refusal samples for a category to be included.

    Returns:
        Dict mapping category_name -> {
            "best_layers": [int],
            "num_refusal": int,
            "num_compliant": int,
            "correlations": [{"layer", "correlation", "abs_correlation", "p_value"}, ...],
        }
    """
    # Load activations + metadata
    print("[LOAD] Loading activations...")
    act_data = torch.load(activations_file, weights_only=True)
    labels = act_data["labels"]
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    act_key = "activations" if component == "layer" else f"activations_{component}"
    if act_key in act_data:
        activations = act_data[act_key]
    elif "activations" in act_data:
        activations = act_data["activations"]
    else:
        raise ValueError(f"No activations found for component '{component}'")

    metadata = act_data.get("metadata")
    if metadata is None:
        raise ValueError("No metadata in activations file — need category labels")

    judge_scores = torch.tensor([m["score"] for m in metadata])

    # Build per-sample category membership
    # Each sample can belong to multiple categories (multi-label from BeaverTails)
    sample_categories = []
    for m in metadata:
        raw = m.get("category")
        if raw is None:
            sample_categories.append(set())
        elif isinstance(raw, list):
            sample_categories.append(set(str(c) for c in raw))
        else:
            sample_categories.append({str(raw)})

    # Discover category vector files
    summary_path = os.path.join(category_vector_dir, "category_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            category_summary = json.load(f)
    else:
        # Fallback: glob for vector files
        import glob as globmod

        category_summary = {}
        for pt_file in sorted(globmod.glob(os.path.join(category_vector_dir, "*.pt"))):
            name = os.path.basename(pt_file)
            if name.startswith("steering_vectors_") and name != "steering_vectors_md.pt":
                cat_key = name.replace("steering_vectors_md_", "").replace(".pt", "")
                category_summary[cat_key] = {"file": name}

    # Exclude _global entry
    categories = {k: v for k, v in category_summary.items() if k != "_global"}

    # Also build a mapping from subcategory names to category vector keys
    # (e.g., "violence" -> "violence,aiding_and_abetting,incitement")
    subcat_to_key = {}
    for cat_key in categories:
        for sub in cat_key.split(","):
            subcat_to_key[sub] = cat_key

    refusal_mask = labels == 1
    compliant_mask = labels == 0
    n_compliant = compliant_mask.sum().item()

    print(f"   Component: {component}")
    print(
        f"   Total samples: {len(labels)} (refusal={refusal_mask.sum()}, compliant={n_compliant})"
    )
    print(f"   Categories in vector dir: {len(categories)}")

    results = {}

    for cat_key, cat_info in sorted(categories.items()):
        vec_file = cat_info.get("file")
        if not vec_file:
            continue

        vec_path = os.path.join(category_vector_dir, vec_file)
        if not os.path.exists(vec_path):
            print(f"[SKIP] {cat_key}: vector file not found ({vec_path})")
            continue

        # Find samples belonging to this category.
        # Category values are compound strings like "violence,aiding_and_abetting,incitement"
        # stored as list elements — match by exact string membership, not sub-string.
        cat_sample_mask = torch.zeros(len(labels), dtype=torch.bool)
        for i in range(len(labels)):
            if cat_key in sample_categories[i]:
                cat_sample_mask[i] = True

        cat_refusal_mask = refusal_mask & cat_sample_mask
        n_refusal = cat_refusal_mask.sum().item()

        if n_refusal < min_samples:
            print(f"[SKIP] {cat_key}: only {n_refusal} refusal samples (min={min_samples})")
            continue

        # Load category steering vector
        vec_data = torch.load(vec_path, map_location="cpu", weights_only=True)
        sv_key = f"steering_vectors_{component}"
        if sv_key in vec_data:
            steering_vectors = vec_data[sv_key]
        elif "steering_vectors" in vec_data:
            steering_vectors = vec_data["steering_vectors"]
        else:
            print(f"[SKIP] {cat_key}: no steering vectors for component '{component}'")
            continue

        if steering_vectors.dim() == 3:
            steering_vectors = steering_vectors[:, 0, :]

        # Normalize to unit length
        sv_norms = steering_vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        steering_vectors = steering_vectors / sv_norms

        # Subselect: this category's refusal samples + all compliant samples
        eval_mask = cat_refusal_mask | compliant_mask
        eval_acts = activations[eval_mask]
        eval_scores = judge_scores[eval_mask]

        if len(eval_scores) < 20:
            print(f"[SKIP] {cat_key}: only {len(eval_scores)} eval samples after filtering")
            continue

        # Compute per-layer correlations
        correlations = []
        for layer in range(eval_acts.shape[1]):
            layer_acts = eval_acts[:, layer, :]
            sv = steering_vectors[layer]
            projections = (layer_acts @ sv).float().numpy()
            scores_np = eval_scores.float().numpy()

            # Check if projections have zero variance (pearsonr would fail)
            if projections.std() < 1e-12:
                correlations.append(
                    {
                        "layer": int(layer),
                        "correlation": 0.0,
                        "abs_correlation": 0.0,
                        "p_value": 1.0,
                    }
                )
                continue

            corr, p_value = pearsonr(projections, scores_np)
            correlations.append(
                {
                    "layer": int(layer),
                    "correlation": float(corr),
                    "abs_correlation": float(abs(corr)),
                    "p_value": float(p_value),
                }
            )

        # Find best layers
        sorted_corrs = sorted(correlations, key=lambda x: x["abs_correlation"], reverse=True)
        best = [c["layer"] for c in sorted_corrs[:top_k]]

        results[cat_key] = {
            "best_layers": best,
            "num_refusal": n_refusal,
            "num_compliant": n_compliant,
            "correlations": correlations,
        }

        # Print summary
        top_str = ", ".join(f"L{c['layer']}({c['abs_correlation']:.3f})" for c in sorted_corrs[:3])
        print(f"  {cat_key:<50} n={n_refusal:>4}  best: {top_str}")

    return results


def plot_per_category_layer_correlations(per_category_results, output_dir=None):
    """Plot per-category best layer comparison.

    Generates:
    1. A heatmap of |correlation| across layers (rows) x categories (columns)
    2. A bar chart of best layer per category

    Args:
        per_category_results: Dict from compute_per_category_layer_correlations()
        output_dir: Directory to save plots
    """
    if not per_category_results:
        return

    if output_dir is None:
        from .utils import get_output_path

        output_dir = get_output_path(script_name="find_best_layers")

    categories = sorted(per_category_results.keys())

    # Collect correlation matrices
    all_layers = set()
    cat_layer_corrs = {}
    for cat in categories:
        corrs = {
            c["layer"]: c["abs_correlation"] for c in per_category_results[cat]["correlations"]
        }
        cat_layer_corrs[cat] = corrs
        all_layers.update(corrs.keys())

    layers = sorted(all_layers)
    n_cats = len(categories)
    n_layers = len(layers)

    # Heatmap
    heatmap = np.zeros((n_layers, n_cats))
    for j, cat in enumerate(categories):
        for i, layer in enumerate(layers):
            heatmap[i, j] = cat_layer_corrs[cat].get(layer, 0.0)

    fig, ax = plt.subplots(figsize=max((n_cats * 0.8, 8), (14, 8)))
    im = ax.imshow(heatmap.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(n_cats))
    # Truncate long category names for display
    short_cats = [c[:35] + "..." if len(c) > 35 else c for c in categories]
    ax.set_yticklabels(short_cats, fontsize=8)
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xticklabels([str(layers[i]) for i in range(0, n_layers, 2)], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_title("Per-Category |Correlation| by Layer", fontsize=14)
    plt.colorbar(im, ax=ax, label="|Correlation|")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "per_category_layer_heatmap.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Bar chart: best layer per category
    fig, ax = plt.subplots(figsize=(max(n_cats * 0.5, 8), 6))
    best_layers_list = [per_category_results[cat]["best_layers"][0] for cat in categories]
    best_corrs = [
        next(
            c["abs_correlation"]
            for c in per_category_results[cat]["correlations"]
            if c["layer"] == per_category_results[cat]["best_layers"][0]
        )
        for cat in categories
    ]

    bars = ax.barh(range(n_cats), best_layers_list, color="steelblue", alpha=0.8)
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(short_cats, fontsize=8)
    ax.set_xlabel("Best Layer Index")
    ax.set_title("Best Layer per Category", fontsize=14)
    ax.invert_yaxis()

    # Annotate with correlation value
    for i, (layer, corr) in enumerate(zip(best_layers_list, best_corrs)):
        ax.text(layer + 0.3, i, f"L{layer} (r={corr:.3f})", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "per_category_best_layers.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"[PLOT] Saved per-category layer analysis to {output_dir}")


def find_best_layers_dynamic(
    activations_file, steering_vectors_file, top_k=5, min_sv_norm=1.0, component="attn"
):
    """
    Find best layers for dynamic (SiLU-based) steering mode.

    Unlike find_best_layers() which ranks by correlation alone, this considers:
    - Correlation strength (refusal signal quality)
    - SiLU coverage: fraction of refused samples with positive projections
      (negative projections get clamped to zero by the SiLU gate)
    - Selectivity: fraction of compliant samples with negative projections.
      Layers where both refused AND compliant project positively (MIXED)
      cannot distinguish the two classes via the SiLU gate.

    The hook normalizes steering vectors to unit length, so sv_norm does not
    affect perturbation magnitude — only directional quality matters.

    Composite score = correlation * coverage * selectivity.

    Args:
        activations_file: Path to activations .pt file
        steering_vectors_file: Path to steering vectors .pt file
        top_k: Number of top layers to return
        min_sv_norm: Minimum steering vector L2 norm (default: 1.0).
            Filters near-zero vectors that are numerically unstable.
        component: Which activation component to use (default: "attn")

    Returns:
        List of best layer indices for dynamic steering
    """
    act_data = torch.load(activations_file, weights_only=True)
    labels = act_data["labels"]

    # Select activations for the requested component
    act_key = "activations" if component == "layer" else f"activations_{component}"
    if act_key in act_data:
        activations = act_data[act_key]
    elif "activations" in act_data:
        activations = act_data["activations"]
    else:
        raise ValueError(f"No activations found for component '{component}'")

    vec_data = torch.load(steering_vectors_file, weights_only=True)
    sv_key = f"steering_vectors_{component}"
    if sv_key in vec_data:
        steering_vectors = vec_data[sv_key]
    else:
        steering_vectors = vec_data["steering_vectors"]  # [num_layers, hidden]

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    refused_mask = labels == 1
    compliant_mask = labels == 0

    num_layers = activations.shape[1]
    layer_scores = []

    print(f"\n[DYNAMIC] Ranking layers for dynamic steering mode...")
    print(
        f"   Refused: {refused_mask.sum().item()}, Compliant: {compliant_mask.sum().item()}, Total: {len(labels)}"
    )
    print(f"   Steering vectors normalized to unit length for projections")
    print(
        f"\n{'Layer':>5} | {'Corr':>6} | {'Ref>0%':>7} | {'Comp<0%':>7} | {'sv_norm':>8} | {'Score':>7}"
    )
    print("-" * 65)

    # Get judge scores for correlation
    if "metadata" in act_data:
        judge_scores = np.array([m["score"] for m in act_data["metadata"]])
    else:
        judge_scores = labels.float().numpy()

    for layer_idx in range(num_layers):
        sv = steering_vectors[layer_idx].float()
        sv_norm = sv.norm().item()

        # First check for exact zero norm (cannot normalize at all)
        if sv_norm < 1e-8:
            layer_scores.append(
                {
                    "layer": layer_idx,
                    "correlation": 0.0,
                    "coverage": 0.0,
                    "selectivity": 0.0,
                    "composite": 0.0,
                    "sv_norm": sv_norm,
                    "theta": 0.0,
                }
            )
            print(
                f"{layer_idx:>5} | {'N/A':>6} | {'N/A':>7} | {'N/A':>7} | {sv_norm:>8.2e} | {'0.0000':>7} [zero-norm]"
            )
            continue

        # Filter near-zero vectors below min_sv_norm threshold
        if sv_norm < min_sv_norm:
            layer_scores.append(
                {
                    "layer": layer_idx,
                    "correlation": 0.0,
                    "coverage": 0.0,
                    "selectivity": 0.0,
                    "composite": 0.0,
                    "sv_norm": sv_norm,
                    "theta": 0.0,
                }
            )
            print(
                f"{layer_idx:>5} | {'N/A':>6} | {'N/A':>7} | {'N/A':>7} | {sv_norm:>8.2f} | {'0.0000':>7} [near-zero]"
            )
            continue

        # Normalize to unit vector (matching the hook)
        sv_unit = sv / sv_norm

        ref_acts = activations[refused_mask, layer_idx, :].float()
        comp_acts = activations[compliant_mask, layer_idx, :].float()

        ref_projs = ref_acts @ sv_unit
        comp_projs = comp_acts @ sv_unit

        # Coverage: fraction of refused samples with positive projections
        coverage = (ref_projs > 0).float().mean().item()

        # Selectivity: fraction of compliant samples with negative projections
        # High selectivity = SiLU gate correctly does NOT activate for compliant prompts
        selectivity = (comp_projs < 0).float().mean().item()

        # Pearson correlation using all samples (with normalized vector)
        all_acts = activations[:, layer_idx, :].float()
        all_projs = (all_acts @ sv_unit).numpy()
        corr, _ = pearsonr(all_projs, judge_scores)
        abs_corr = abs(corr)

        # Composite: all three factors must be present
        composite = abs_corr * coverage * selectivity

        # Theta computed on normalized projections (matching the hook)
        theta = -ref_projs.median().item()

        layer_scores.append(
            {
                "layer": layer_idx,
                "correlation": abs_corr,
                "coverage": coverage,
                "selectivity": selectivity,
                "composite": composite,
                "sv_norm": sv_norm,
                "theta": theta,
            }
        )

        dir_label = ""
        if coverage > 0.8 and selectivity < 0.2:
            dir_label = " [MIXED]"
        elif selectivity > 0.8 and coverage > 0.8:
            dir_label = " [OK]"

        print(
            f"{layer_idx:>5} | {abs_corr:>.4f} | {coverage*100:>6.1f}% | {selectivity*100:>6.1f}% | {sv_norm:>8.2f} | {composite:>.4f}{dir_label}"
        )

    # Sort by composite score
    layer_scores.sort(key=lambda x: x["composite"], reverse=True)

    print(f"\n[TOP] Top {top_k} Layers for Dynamic Steering:")
    print(
        f"{'Layer':>5} | {'Corr':>6} | {'Ref>0%':>7} | {'Comp<0%':>7} | {'sv_norm':>8} | {'Score':>7} | {'theta':>10}"
    )
    print("-" * 80)

    best_layers = []
    for entry in layer_scores[:top_k]:
        if entry["composite"] == 0:
            break
        print(
            f"{entry['layer']:>5} | {entry['correlation']:>.4f} | "
            f"{entry['coverage']*100:>6.1f}% | "
            f"{entry['selectivity']*100:>6.1f}% | "
            f"{entry['sv_norm']:>8.2f} | "
            f"{entry['composite']:>.4f} | {entry['theta']:>10.4f}"
        )
        best_layers.append(entry["layer"])

    return best_layers


def visualize_layer_projections(
    layer, projections, judge_scores, correlation, output_dir=None, output_file=None
):
    """
    Scatter plot: projection vs judge score for a specific layer.

    Args:
        layer: Layer index
        projections: Projection values for this layer
        judge_scores: Judge scores
        correlation: Correlation coefficient
        output_dir: Directory to save plot
        output_file: Filename for plot
    """
    from .utils import get_output_path

    if output_dir is None:
        output_dir = get_output_path(script_name="find_best_layers")

    plt.figure(figsize=(10, 6))
    plt.scatter(projections, judge_scores, alpha=0.5, s=30)
    plt.xlabel("Activation Projection onto Steering Vector", fontsize=12)
    plt.ylabel("Judge Refusal Score", fontsize=12)
    plt.title(f"Layer {layer}: Projection vs Judge Score (r={correlation:+.4f})", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add regression line
    z = np.polyfit(projections, judge_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(projections.min(), projections.max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    if output_file:
        if not os.path.isabs(output_file):
            output_path = os.path.join(output_dir, output_file)
        else:
            output_path = output_file

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved scatter plot: {output_path}")
    else:
        plt.show()


def load_category_vectors_from_files(vector_files, component="attn", min_samples=None):
    """
    Load per-category steering vectors from multiple .pt files.

    For standalone mode: loads files produced by --all-categories and extracts
    the category name from the saved metadata or filename.

    Args:
        vector_files: List of paths to steering vector .pt files
        component: Which component's vectors to load ("attn", "mlp", "layer")
        min_samples: If set, skip files where num_refusal_samples < min_samples

    Returns:
        dict[str, Tensor] mapping category name -> [num_layers, hidden_size]
    """
    category_vectors = {}

    for fpath in vector_files:
        data = torch.load(fpath, weights_only=True)

        # Skip global file (no "categories" key)
        if "categories" not in data:
            print(f"[SKIP] {os.path.basename(fpath)}: no category metadata (global file)")
            continue

        # Extract category name from metadata
        cat_name = "+".join(data["categories"])

        # Skip categories with too few samples
        if min_samples is not None:
            n = data.get("num_refusal_samples", 0)
            if n < min_samples:
                print(f"[SKIP] {cat_name}: only {n} refusal samples (minimum: {min_samples})")
                continue

        # Load vectors for the requested component
        sv_key = f"steering_vectors_{component}"
        if sv_key in data:
            vectors = data[sv_key]
        elif "steering_vectors" in data:
            vectors = data["steering_vectors"]
        else:
            print(f"[WARN] {os.path.basename(fpath)}: no steering vectors found, skipping")
            continue

        category_vectors[cat_name] = vectors
        print(f"[LOAD] {cat_name}: {vectors.shape} from {os.path.basename(fpath)}")

    return category_vectors


def compute_category_angular_distances(category_vectors, target_layers=None):
    """
    Compute pairwise angular distances between per-category steering vectors.

    Angular distance = arccos(|cos_sim|) * 180/pi.  Absolute cosine is used
    because the sign of a steering vector is arbitrary (flipped by alpha sign).

    Args:
        category_vectors: dict[str, Tensor] — each tensor is
            [num_layers, hidden_size] or [num_layers, rank, hidden_size]
            (uses v_1 only if 3D)
        target_layers: Optional list of layer indices to analyze.
            Default: all layers.

    Returns:
        dict with keys: categories, num_categories, target_layers,
        per_layer, summary, pairwise, overall
    """
    categories = sorted(category_vectors.keys())
    n = len(categories)

    if n < 2:
        raise ValueError(f"Need at least 2 categories for comparison, got {n}")

    # Stack vectors: [n_categories, num_layers, hidden_size]
    vecs_list = []
    for cat in categories:
        v = category_vectors[cat]
        # If multi-rank, use v_1 only
        if v.dim() == 3:
            v = v[:, 0, :]
        vecs_list.append(v)

    all_vecs = torch.stack(vecs_list).double()  # [N, num_layers, hidden_size]
    num_layers = all_vecs.shape[1]

    if target_layers is None:
        target_layers = list(range(num_layers))

    # Normalize to unit length per layer
    norms = all_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    all_vecs = all_vecs / norms

    # Compute per-layer pairwise cosine similarity and angular distance
    per_layer = {}
    # Accumulators for summary stats across target layers
    angle_stack = []  # list of [N, N] tensors

    for layer_idx in target_layers:
        layer_vecs = all_vecs[:, layer_idx, :]  # [N, hidden_size]
        # Pairwise cosine similarity: [N, N]
        cos_sim = layer_vecs @ layer_vecs.T
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        abs_cos = cos_sim.abs().clamp(min=-1.0, max=1.0)
        # Angular distance in degrees
        angular_dist = torch.acos(abs_cos) * (180.0 / torch.pi)
        # Zero the diagonal (self-comparison) to avoid floating point noise
        angular_dist.fill_diagonal_(0.0)

        per_layer[str(layer_idx)] = {
            "cosine_similarity": cos_sim.tolist(),
            "angular_distance_deg": angular_dist.tolist(),
        }
        angle_stack.append(angular_dist)

    # Summary across target layers: mean/min/max per pair
    angle_tensor = torch.stack(angle_stack)  # [num_target_layers, N, N]
    mean_angles = angle_tensor.mean(dim=0)  # [N, N]
    min_angles = angle_tensor.amin(dim=0)
    max_angles = angle_tensor.amax(dim=0)

    summary = {
        "mean_angular_distance_deg": mean_angles.tolist(),
        "min_angular_distance_deg": min_angles.tolist(),
        "max_angular_distance_deg": max_angles.tolist(),
    }

    # Flat pairwise list, sorted by mean angle
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(
                {
                    "cat_a": categories[i],
                    "cat_b": categories[j],
                    "mean_angle_deg": float(mean_angles[i, j]),
                    "min_angle_deg": float(min_angles[i, j]),
                    "max_angle_deg": float(max_angles[i, j]),
                }
            )
    pairwise.sort(key=lambda x: x["mean_angle_deg"])

    # Overall statistics
    # Collect all unique pair angles
    pair_angles = [p["mean_angle_deg"] for p in pairwise]
    overall_mean = float(np.mean(pair_angles)) if pair_angles else 0.0
    most_similar = pairwise[0] if pairwise else None
    most_different = pairwise[-1] if pairwise else None

    overall = {
        "mean_pairwise_angle_deg": overall_mean,
        "most_similar": (
            {
                "cat_a": most_similar["cat_a"],
                "cat_b": most_similar["cat_b"],
                "angle_deg": most_similar["mean_angle_deg"],
            }
            if most_similar
            else None
        ),
        "most_different": (
            {
                "cat_a": most_different["cat_a"],
                "cat_b": most_different["cat_b"],
                "angle_deg": most_different["mean_angle_deg"],
            }
            if most_different
            else None
        ),
    }

    results = {
        "categories": categories,
        "num_categories": n,
        "target_layers": target_layers,
        "per_layer": per_layer,
        "summary": summary,
        "pairwise": pairwise,
        "overall": overall,
    }

    # Console output
    print(f"\n[CATEGORY ANGULAR DISTANCES]")
    print(f"   Categories ({n}): {', '.join(categories)}")
    print(f"   Target layers: {len(target_layers)} layers")

    # Print matrix
    max_cat_len = max(len(c) for c in categories)
    header = " " * (max_cat_len + 2)
    for cat in categories:
        header += f"{cat[:8]:>9}"
    print(f"\n   Mean angular distance (degrees):")
    print(f"   {header}")
    for i, cat_a in enumerate(categories):
        row = f"   {cat_a:<{max_cat_len + 2}}"
        for j in range(n):
            row += f"{mean_angles[i, j].item():>9.1f}"
        print(row)

    print(f"\n   Overall mean pairwise angle: {overall_mean:.1f} deg")
    if most_similar:
        print(
            f"   Most similar:  {most_similar['cat_a']} <-> {most_similar['cat_b']} "
            f"({most_similar['mean_angle_deg']:.1f} deg)"
        )
    if most_different:
        print(
            f"   Most different: {most_different['cat_a']} <-> {most_different['cat_b']} "
            f"({most_different['mean_angle_deg']:.1f} deg)"
        )

    return results


def plot_category_angular_distances(
    results, output_dir=None, output_file="category_angular_distances.png"
):
    """
    Visualize pairwise angular distances between category steering vectors.

    Creates a 1x2 figure with:
      - Left: heatmap of mean angular distance matrix
      - Right: dendrogram from hierarchical clustering (skipped if <= 2 categories)

    Args:
        results: Output dict from compute_category_angular_distances()
        output_dir: Directory to save the plot
        output_file: Filename for the plot
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    from .utils import ensure_dir, get_output_path

    categories = results["categories"]
    n = results["num_categories"]
    mean_angles = np.array(results["summary"]["mean_angular_distance_deg"])

    show_dendrogram = n > 2

    if show_dendrogram:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))

    # Left: heatmap
    im = ax1.imshow(mean_angles, cmap="RdYlBu_r", aspect="auto")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.set_title("Mean Angular Distance (degrees)", fontsize=12)

    # Annotate cells (skip diagonal, use off-diagonal range for text color threshold)
    off_diag = mean_angles[~np.eye(n, dtype=bool)]
    color_threshold = (off_diag.min() + off_diag.max()) / 2 if off_diag.size > 0 else 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = mean_angles[i, j]
            color = "white" if val > color_threshold else "black"
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Right: dendrogram
    if show_dendrogram:
        # Symmetrize to eliminate float-precision asymmetry, then convert to condensed form
        sym_angles = (mean_angles + mean_angles.T) / 2
        np.fill_diagonal(sym_angles, 0.0)
        condensed = squareform(sym_angles)
        Z = linkage(condensed, method="average")
        dendrogram(Z, labels=categories, ax=ax2, leaf_rotation=45, leaf_font_size=9)
        ax2.set_title("Hierarchical Clustering (Average Linkage)", fontsize=12)
        ax2.set_ylabel("Angular Distance (degrees)")

    plt.tight_layout()

    # Save
    if output_dir is None:
        output_dir = ensure_dir(get_output_path(script_name="compute_wrmd"))

    if not os.path.isabs(output_file):
        output_path = os.path.join(output_dir, output_file)
    else:
        output_path = output_file

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved category angular distances plot: {output_path}")


def compute_intra_category_angular_distances(
    activations, labels, categories, target_layers=None, min_samples=10
):
    """
    Compute intra-category angular spread of refusal activations.

    For each category, measures how coherent the refusal direction is by
    computing pairwise angular distances between individual refusal activation
    vectors.  A low mean angle indicates a coherent/stable refusal circuit;
    a high mean angle indicates a fragmented one.

    Args:
        activations: [N, num_layers, hidden_size] tensor
        labels: [N] tensor (1=refusal, 0=compliant)
        categories: list of N category strings (or None entries)
        target_layers: list of layer indices (default: all)
        min_samples: minimum refusal samples per category (skip if fewer)

    Returns:
        dict with keys: categories, target_layers, per_category, summary
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    num_layers = activations.shape[1]
    if target_layers is None:
        target_layers = list(range(num_layers))

    unique_cats = sorted(
        {name for c in categories if c is not None for name in (c if isinstance(c, list) else [c])}
    )

    per_category = {}
    for cat in unique_cats:
        # Build mask: refusal AND this category (categories[i] may be a list of strings)
        cat_mask = torch.tensor(
            [
                (
                    categories[i] is not None
                    and (
                        cat in categories[i]
                        if isinstance(categories[i], list)
                        else categories[i] == cat
                    )
                    and labels[i].item() == 1
                )
                for i in range(len(labels))
            ]
        )
        # Ensure mask is on the same device as activations
        if activations.device != torch.device("cpu"):
            cat_mask = cat_mask.to(activations.device)

        n_cat = cat_mask.sum().item()

        if n_cat < min_samples:
            print(f"[SKIP] {cat}: {n_cat} refusal samples (min: {min_samples})")
            continue

        layer_results = {}
        all_means = []

        for layer_idx in target_layers:
            layer_acts = activations[cat_mask, layer_idx, :].double()  # [n_cat, hidden]

            # Normalize to unit length
            norms = layer_acts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            layer_acts_unit = layer_acts / norms

            # Pairwise cosine similarity
            cos_sim = layer_acts_unit @ layer_acts_unit.T
            cos_sim = cos_sim.clamp(-1.0, 1.0)
            abs_cos = cos_sim.abs().clamp(max=1.0)

            # Angular distance in degrees
            angular_dist = torch.acos(abs_cos) * (180.0 / torch.pi)
            angular_dist.fill_diagonal_(0.0)

            # Extract upper triangle (unique pairs)
            triu_mask = torch.triu(torch.ones(n_cat, n_cat, dtype=torch.bool), diagonal=1)
            pair_angles = angular_dist[triu_mask]

            mean_angle = pair_angles.mean().item()
            std_angle = pair_angles.std().item() if pair_angles.numel() > 1 else 0.0

            layer_results[str(layer_idx)] = {
                "mean_angle_deg": mean_angle,
                "std_angle_deg": std_angle,
                "num_pairs": int(triu_mask.sum().item()),
            }
            all_means.append(mean_angle)

        if not all_means:
            print(f"[WARN] {cat}: no target layers produced results, skipping")
            continue

        per_category[cat] = {
            "num_refusal_samples": n_cat,
            "per_layer": layer_results,
            "mean_across_layers_deg": float(np.mean(all_means)),
            "std_across_layers_deg": float(np.std(all_means)),
        }

    # Sort by coherence (lowest mean angle = most coherent)
    ranked = sorted(per_category.items(), key=lambda x: x[1]["mean_across_layers_deg"])

    summary = {
        "most_coherent": ranked[0][0] if ranked else None,
        "least_coherent": ranked[-1][0] if ranked else None,
        "ranking": [
            {
                "category": cat,
                "mean_angle_deg": data["mean_across_layers_deg"],
                "num_samples": data["num_refusal_samples"],
            }
            for cat, data in ranked
        ],
    }

    # Console output
    print(f"\n[INTRA-CATEGORY COHERENCE]")
    print(f"   {'Category':<40} {'Samples':>8} {'Mean Angle':>12} {'Std':>8}")
    print(f"   {'-' * 70}")
    for cat, data in ranked:
        print(
            f"   {cat:<40} {data['num_refusal_samples']:>8} "
            f"{data['mean_across_layers_deg']:>11.1f}° "
            f"{data['std_across_layers_deg']:>7.1f}°"
        )

    if ranked:
        print(
            f"\n   Most coherent:  {ranked[0][0]} ({ranked[0][1]['mean_across_layers_deg']:.1f}°)"
        )
        print(
            f"   Least coherent: {ranked[-1][0]} ({ranked[-1][1]['mean_across_layers_deg']:.1f}°)"
        )

    return {
        "categories": [cat for cat, _ in ranked],
        "target_layers": target_layers,
        "per_category": per_category,
        "summary": summary,
    }


def plot_intra_category_angular_distances(
    results, output_dir=None, output_file="intra_category_coherence.png"
):
    """
    Bar chart of mean intra-category angular distance per category.

    Args:
        results: Output from compute_intra_category_angular_distances()
        output_dir: Directory to save plot
        output_file: Filename for plot
    """
    from .utils import ensure_dir, get_output_path

    categories = results["categories"]
    if not categories:
        print("[WARN] No categories to plot for intra-category coherence")
        return

    means = [results["per_category"][c]["mean_across_layers_deg"] for c in categories]
    stds = [results["per_category"][c]["std_across_layers_deg"] for c in categories]

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.8), 6))

    # Color gradient: green (coherent/low) to red (fragmented/high)
    norm = plt.Normalize(vmin=min(means), vmax=max(means))
    cmap = plt.cm.RdYlGn_r
    colors = [cmap(norm(m)) for m in means]

    bars = ax.bar(range(len(categories)), means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Intra-Category Angle (degrees)", fontsize=11)
    ax.set_title("Intra-Category Coherence (lower = more coherent)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars with sample counts
    for i, cat in enumerate(categories):
        n = results["per_category"][cat]["num_refusal_samples"]
        ax.text(i, means[i] + stds[i] + 0.5, f"n={n}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()

    if output_dir is None:
        output_dir = ensure_dir(get_output_path(script_name="compute_wrmd"))

    if not os.path.isabs(output_file):
        output_path = os.path.join(output_dir, output_file)
    else:
        output_path = output_file

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved intra-category coherence plot: {output_path}")


def _build_compliant_cache(
    activations, labels, judge_scores, target_layers, method, lambda_ridge, use_score_weighting
):
    """
    Pre-compute compliant-side quantities per layer (constant across bootstrap iterations).

    Returns:
        compliant_cache: dict mapping layer_idx -> precomputed quantities
    """
    from .computation import compute_weighted_mean

    hidden_size = activations.shape[2]

    compliant_mask = labels == 0
    compliant_indices = compliant_mask.nonzero(as_tuple=True)[0]

    if judge_scores is not None and use_score_weighting and method == "wrmd":
        comp_scores = judge_scores[compliant_mask]
        comp_weights = torch.clamp(-comp_scores.clone(), min=0.01).to(activations.dtype)
    else:
        comp_weights = torch.ones(compliant_mask.sum().item(), dtype=activations.dtype)

    compliant_cache = {}

    for layer_idx in target_layers:
        comp_acts = activations[compliant_indices, layer_idx, :]

        if method == "md":
            comp_mean = comp_acts.mean(dim=0)
            compliant_cache[layer_idx] = {"comp_mean": comp_mean}
        elif method == "rmd":
            comp_mean = comp_acts.mean(dim=0)
            centered = comp_acts - comp_mean
            cov = (centered.T @ centered) / len(comp_acts)
            cov_f32 = cov.float()
            ridge_inv = torch.linalg.inv(
                cov_f32
                + lambda_ridge * torch.eye(hidden_size, dtype=torch.float32, device=cov.device)
            )
            ridge_inv = ridge_inv.to(cov.dtype)
            compliant_cache[layer_idx] = {
                "comp_mean": comp_mean,
                "ridge_inv": ridge_inv,
            }
        elif method == "wrmd":
            neutral_mean = compute_weighted_mean(comp_acts, comp_weights)
            comp_centered = comp_acts - neutral_mean
            comp_mean_centered = compute_weighted_mean(comp_centered, comp_weights)
            centered_from_mean = comp_centered - comp_mean_centered
            weighted_centered = centered_from_mean * comp_weights.unsqueeze(1)
            cov = (weighted_centered.T @ centered_from_mean) / comp_weights.sum()
            cov_f32 = cov.float()
            ridge_inv = torch.linalg.inv(
                cov_f32
                + lambda_ridge * torch.eye(hidden_size, dtype=torch.float32, device=cov.device)
            )
            ridge_inv = ridge_inv.to(cov.dtype)
            compliant_cache[layer_idx] = {
                "neutral_mean": neutral_mean,
                "comp_mean_centered": comp_mean_centered,
                "ridge_inv": ridge_inv,
            }

    return compliant_cache


def _run_bootstrap_for_pool(
    activations,
    refusal_idx,
    weights_full,
    compliant_cache,
    target_layers,
    method,
    n_bootstrap,
    sample_ratio,
    rng,
    label_name="",
):
    """
    Run bootstrap loop for a single refusal cohort and return results dict or None.

    This is a module-level helper shared by compute_bootstrap_stability and
    compute_bootstrap_convergence.

    Args:
        activations: [N, num_layers, hidden_size] tensor
        refusal_idx: list of indices into activations for the refusal cohort
        weights_full: tensor of weights for refusal samples, or None
        compliant_cache: dict from _build_compliant_cache()
        target_layers: list of layer indices
        method: 'md', 'rmd', or 'wrmd'
        n_bootstrap: number of bootstrap iterations
        sample_ratio: fraction of refusal samples to resample per iteration
        rng: numpy RandomState
        label_name: label for warning messages

    Returns:
        dict with bootstrap results, or None if no layers produced results
    """
    n_ref = len(refusal_idx)
    k = max(2, int(n_ref * sample_ratio))

    bootstrap_vecs = {li: [] for li in target_layers}

    for b in range(n_bootstrap):
        sampled_pos = rng.choice(n_ref, size=k, replace=True)
        sampled_idx = [refusal_idx[p] for p in sampled_pos]

        if weights_full is not None:
            sampled_weights = weights_full[sampled_pos]
        else:
            sampled_weights = None

        for layer_idx in target_layers:
            ref_acts = activations[sampled_idx, layer_idx, :]
            cache = compliant_cache[layer_idx]

            if method == "md":
                vec = ref_acts.mean(dim=0) - cache["comp_mean"]
            elif method == "rmd":
                mean_diff = ref_acts.mean(dim=0) - cache["comp_mean"]
                vec = cache["ridge_inv"] @ mean_diff
            elif method == "wrmd":
                neutral_mean = cache["neutral_mean"]
                ref_centered = ref_acts - neutral_mean
                if sampled_weights is not None:
                    w = sampled_weights.to(ref_acts.dtype)
                    w = w / w.sum()
                    ref_mean_centered = (ref_centered.T @ w).squeeze()
                else:
                    ref_mean_centered = ref_centered.mean(dim=0)
                mean_diff = ref_mean_centered - cache["comp_mean_centered"]
                vec = cache["ridge_inv"] @ mean_diff

            bootstrap_vecs[layer_idx].append(vec)

    # Compute pairwise angular distances between bootstrap vectors per layer
    layer_results = {}
    all_means = []

    for layer_idx in target_layers:
        vecs = torch.stack(bootstrap_vecs[layer_idx]).double()
        norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vecs_unit = vecs / norms

        cos_sim = vecs_unit @ vecs_unit.T
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        abs_cos = cos_sim.abs().clamp(max=1.0)
        angular_dist = torch.acos(abs_cos) * (180.0 / torch.pi)
        angular_dist.fill_diagonal_(0.0)

        triu_mask = torch.triu(torch.ones(n_bootstrap, n_bootstrap, dtype=torch.bool), diagonal=1)
        pair_angles = angular_dist[triu_mask]

        if pair_angles.numel() == 0:
            mean_angle = 0.0
            std_angle = 0.0
            min_angle = 0.0
            max_angle = 0.0
        else:
            mean_angle = pair_angles.mean().item()
            std_angle = pair_angles.std().item() if pair_angles.numel() > 1 else 0.0
            min_angle = pair_angles.min().item()
            max_angle = pair_angles.max().item()

        layer_results[str(layer_idx)] = {
            "mean_angle_deg": mean_angle,
            "std_angle_deg": std_angle,
            "min_angle_deg": min_angle,
            "max_angle_deg": max_angle,
            "num_pairs": int(triu_mask.sum().item()),
        }
        all_means.append(mean_angle)

    if not all_means:
        print(f"[WARN] {label_name}: no target layers produced results, skipping")
        return None

    mean_across = float(np.mean(all_means))

    return {
        "num_refusal_samples": n_ref,
        "num_bootstrap_samples_per_iter": k,
        "per_layer": layer_results,
        "mean_across_layers_deg": mean_across,
        "std_across_layers_deg": float(np.std(all_means)),
    }


def compute_bootstrap_stability(
    activations,
    labels,
    categories,
    judge_scores=None,
    target_layers=None,
    method="md",
    lambda_ridge=0.1,
    use_score_weighting=True,
    n_bootstrap=20,
    sample_ratio=0.8,
    min_samples=10,
    seed=42,
):
    """
    Compute bootstrap stability of per-category steering vectors.

    For each category, resamples the refusal cohort with replacement and
    recomputes the steering vector multiple times.  The spread of these
    bootstrap vectors (measured as pairwise angular distance) indicates
    how sensitive the vector is to the specific samples used.

    The compliant pool and its covariance inverse are constant across
    iterations and are pre-computed once per layer for efficiency.

    Args:
        activations: [N, num_layers, hidden_size] tensor
        labels: [N] tensor (1=refusal, 0=compliant)
        categories: list of N category strings (or None)
        judge_scores: [N] tensor of judge scores (for WRMD weighting)
        target_layers: list of layer indices (default: all)
        method: 'md', 'rmd', or 'wrmd'
        lambda_ridge: Ridge regularization parameter
        use_score_weighting: Whether to weight by judge scores (WRMD only)
        n_bootstrap: Number of bootstrap iterations
        sample_ratio: Fraction of refusal samples to resample per iteration
        min_samples: Minimum refusal samples per category
        seed: Random seed for reproducibility

    Returns:
        dict with keys: config, per_category, summary
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    num_layers = activations.shape[1]
    if target_layers is None:
        target_layers = list(range(num_layers))

    compliant_cache = _build_compliant_cache(
        activations, labels, judge_scores, target_layers, method, lambda_ridge, use_score_weighting
    )

    # --- Run bootstrap for global (all refusal samples) ---
    rng = np.random.RandomState(seed)

    all_refusal_idx = [i for i in range(len(labels)) if labels[i].item() == 1]
    n_all_refusal = len(all_refusal_idx)

    global_result = None
    if n_all_refusal >= min_samples:
        if judge_scores is not None and use_score_weighting and method == "wrmd":
            global_weights = torch.clamp(judge_scores[all_refusal_idx].clone(), min=0.01).to(
                activations.dtype
            )
        else:
            global_weights = None

        global_result = _run_bootstrap_for_pool(
            activations,
            all_refusal_idx,
            global_weights,
            compliant_cache,
            target_layers,
            method,
            n_bootstrap,
            sample_ratio,
            rng,
            "_global",
        )
    else:
        print(f"[SKIP] _global: {n_all_refusal} refusal samples (min: {min_samples})")

    # --- Run bootstrap for each category ---
    unique_cats = sorted(
        {name for c in categories if c is not None for name in (c if isinstance(c, list) else [c])}
    )
    per_category = {}

    for cat in unique_cats:
        cat_refusal_idx = [
            i
            for i in range(len(labels))
            if categories[i] is not None
            and (cat in categories[i] if isinstance(categories[i], list) else categories[i] == cat)
            and labels[i].item() == 1
        ]
        n_cat = len(cat_refusal_idx)

        if n_cat < min_samples:
            print(f"[SKIP] {cat}: {n_cat} refusal samples (min: {min_samples})")
            continue

        if judge_scores is not None and use_score_weighting and method == "wrmd":
            cat_weights = torch.clamp(judge_scores[cat_refusal_idx].clone(), min=0.01).to(
                activations.dtype
            )
        else:
            cat_weights = None

        result = _run_bootstrap_for_pool(
            activations,
            cat_refusal_idx,
            cat_weights,
            compliant_cache,
            target_layers,
            method,
            n_bootstrap,
            sample_ratio,
            rng,
            cat,
        )
        if result is not None:
            per_category[cat] = result

    # Classify stability relative to global bootstrap angle
    global_angle = global_result["mean_across_layers_deg"] if global_result else None

    def _classify_stability(mean_angle, reference_angle):
        """Classify stability as ratio of category angle to global angle."""
        if reference_angle is None or reference_angle < 1.0:
            # No meaningful global reference; fall back to absolute thresholds
            if mean_angle < 5.0:
                return "stable", None
            elif mean_angle < 15.0:
                return "moderate", None
            else:
                return "unreliable", None
        ratio = mean_angle / reference_angle
        if ratio <= 1.5:
            label = "stable"
        elif ratio <= 2.5:
            label = "moderate"
        else:
            label = "unreliable"
        return label, round(ratio, 2)

    if global_result is not None:
        global_result["stability_label"] = "reference"
        global_result["stability_ratio"] = 1.0

    if global_angle is None or global_angle < 1.0:
        print(
            "[WARN] No meaningful global bootstrap angle; "
            "falling back to absolute thresholds (5°/15°) for stability labels"
        )

    for cat, data in per_category.items():
        label, ratio = _classify_stability(data["mean_across_layers_deg"], global_angle)
        data["stability_label"] = label
        data["stability_ratio"] = ratio

    # Sort by stability (lowest mean angle = most stable)
    ranked = sorted(per_category.items(), key=lambda x: x[1]["mean_across_layers_deg"])

    summary = {
        "most_stable": ranked[0][0] if ranked else None,
        "least_stable": ranked[-1][0] if ranked else None,
        "global_reference_angle_deg": global_angle,
        "ranking": [
            {
                "category": cat,
                "mean_angle_deg": data["mean_across_layers_deg"],
                "num_samples": data["num_refusal_samples"],
                "stability_ratio": data["stability_ratio"],
                "label": data["stability_label"],
            }
            for cat, data in ranked
        ],
    }

    # Console output
    print(f"\n[BOOTSTRAP STABILITY] (n={n_bootstrap}, ratio={sample_ratio}, method={method})")
    print(
        f"   {'Category':<40} {'Samples':>8} {'k':>4} {'Mean Angle':>12} "
        f"{'Std':>8} {'Ratio':>7} {'Label':>12}"
    )
    print(f"   {'-' * 93}")

    # Print global first
    if global_result is not None:
        print(
            f"   {'_global (all refusal)':<40} {global_result['num_refusal_samples']:>8} "
            f"{global_result['num_bootstrap_samples_per_iter']:>4} "
            f"{global_result['mean_across_layers_deg']:>11.1f}° "
            f"{global_result['std_across_layers_deg']:>7.1f}° "
            f"{'1.00x':>7} "
            f"{'reference':>12}"
        )
        print(f"   {'-' * 93}")

    for cat, data in ranked:
        ratio_str = (
            f"{data['stability_ratio']:.2f}x" if data["stability_ratio"] is not None else "n/a"
        )
        print(
            f"   {cat:<40} {data['num_refusal_samples']:>8} "
            f"{data['num_bootstrap_samples_per_iter']:>4} "
            f"{data['mean_across_layers_deg']:>11.1f}° "
            f"{data['std_across_layers_deg']:>7.1f}° "
            f"{ratio_str:>7} "
            f"{data['stability_label']:>12}"
        )

    if ranked:
        print(
            f"\n   Most stable:   {ranked[0][0]} "
            f"({ranked[0][1]['mean_across_layers_deg']:.1f}°, "
            f"{ranked[0][1]['stability_label']})"
        )
        print(
            f"   Least stable:  {ranked[-1][0]} "
            f"({ranked[-1][1]['mean_across_layers_deg']:.1f}°, "
            f"{ranked[-1][1]['stability_label']})"
        )

    return {
        "config": {
            "n_bootstrap": n_bootstrap,
            "sample_ratio": sample_ratio,
            "method": method,
            "seed": seed,
        },
        "global": global_result,
        "per_category": per_category,
        "summary": summary,
    }


def annotate_distances_with_significance(distance_results, bootstrap_results, z_threshold=2.0):
    """
    Annotate inter-category angular distances with bootstrap significance.

    For each category pair, computes a z-score:
        z = inter_angle / sqrt(std_a^2 + std_b^2)
    where std_a/std_b are the bootstrap mean angular spreads for each category.

    Args:
        distance_results: Output from compute_category_angular_distances()
        bootstrap_results: Output from compute_bootstrap_stability()
        z_threshold: z-score threshold for significance (default: 2.0)

    Returns:
        dict with annotated pairs and summary counts
    """
    bootstrap_cats = bootstrap_results["per_category"]
    pairs = []

    for pair in distance_results["pairwise"]:
        cat_a, cat_b = pair["cat_a"], pair["cat_b"]
        inter_angle = pair["mean_angle_deg"]

        # Use mean bootstrap spread as sampling uncertainty for each category
        spread_a = bootstrap_cats.get(cat_a, {}).get("mean_across_layers_deg", None)
        spread_b = bootstrap_cats.get(cat_b, {}).get("mean_across_layers_deg", None)

        if spread_a is None or spread_b is None:
            pairs.append(
                {
                    "cat_a": cat_a,
                    "cat_b": cat_b,
                    "inter_angle_deg": inter_angle,
                    "significant": None,
                    "reason": "missing bootstrap data",
                }
            )
            continue

        combined_uncertainty = float(np.sqrt(spread_a**2 + spread_b**2))
        z_score = inter_angle / combined_uncertainty if combined_uncertainty > 0 else float("inf")
        significant = z_score > z_threshold

        pairs.append(
            {
                "cat_a": cat_a,
                "cat_b": cat_b,
                "inter_angle_deg": inter_angle,
                "cat_a_bootstrap_spread": spread_a,
                "cat_b_bootstrap_spread": spread_b,
                "combined_uncertainty_deg": combined_uncertainty,
                "z_score": float(z_score),
                "significant": significant,
            }
        )

    n_sig = sum(1 for p in pairs if p.get("significant") is True)
    n_nonsig = sum(1 for p in pairs if p.get("significant") is False)
    n_unknown = sum(1 for p in pairs if p.get("significant") is None)

    # Console output
    print(f"\n[SIGNIFICANCE] (z-threshold={z_threshold})")
    print(f"   Significant: {n_sig}, Not significant: {n_nonsig}, Unknown: {n_unknown}")
    print(f"\n   {'Cat A':<20} {'Cat B':<20} {'Angle':>8} {'Uncert':>8} " f"{'z':>6} {'Sig?':>6}")
    print(f"   {'-' * 70}")
    for p in sorted(pairs, key=lambda x: x.get("z_score", 0), reverse=True):
        if p.get("significant") is None:
            print(
                f"   {p['cat_a']:<20} {p['cat_b']:<20} "
                f"{p['inter_angle_deg']:>7.1f}° {'?':>8} {'?':>6} {'?':>6}"
            )
        else:
            sig_str = "YES" if p["significant"] else "no"
            print(
                f"   {p['cat_a']:<20} {p['cat_b']:<20} "
                f"{p['inter_angle_deg']:>7.1f}° "
                f"{p['combined_uncertainty_deg']:>7.1f}° "
                f"{p['z_score']:>6.2f} {sig_str:>6}"
            )

    return {
        "z_threshold": z_threshold,
        "pairs": pairs,
        "num_significant": n_sig,
        "num_not_significant": n_nonsig,
        "num_unknown": n_unknown,
    }


def plot_bootstrap_stability(results, output_dir=None, output_file="bootstrap_stability.png"):
    """
    Bar chart of bootstrap stability per category.

    Args:
        results: Output from compute_bootstrap_stability()
        output_dir: Directory to save plot
        output_file: Filename for plot
    """
    from .utils import ensure_dir, get_output_path

    ranking = results["summary"]["ranking"]
    global_result = results.get("global")

    if not ranking and global_result is None:
        print("[WARN] No categories to plot for bootstrap stability")
        return

    # Build bar data: global first, then per-category
    bar_names = []
    means = []
    stds = []
    labels_list = []
    sample_counts = []

    if global_result is not None:
        bar_names.append("_global")
        means.append(global_result["mean_across_layers_deg"])
        stds.append(global_result["std_across_layers_deg"])
        labels_list.append(global_result["stability_label"])
        sample_counts.append(global_result["num_refusal_samples"])

    for r in ranking:
        cat = r["category"]
        bar_names.append(cat)
        means.append(r["mean_angle_deg"])
        stds.append(results["per_category"][cat]["std_across_layers_deg"])
        labels_list.append(results["per_category"][cat]["stability_label"])
        sample_counts.append(r["num_samples"])

    fig, ax = plt.subplots(figsize=(max(8, len(bar_names) * 0.8), 6))

    # Color by stability label
    color_map = {
        "stable": "#4CAF50",
        "moderate": "#FF9800",
        "unreliable": "#F44336",
        "reference": "#2196F3",
    }
    colors = [color_map.get(l, "#9E9E9E") for l in labels_list]

    ax.bar(range(len(bar_names)), means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(range(len(bar_names)))
    ax.set_xticklabels(bar_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Bootstrap Angular Spread (degrees)", fontsize=11)
    ax.set_title(
        f"Bootstrap Stability (n={results['config']['n_bootstrap']}, "
        f"ratio={results['config']['sample_ratio']})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Reference lines for stability thresholds (relative to global angle)
    global_ref_angle = results["summary"].get("global_reference_angle_deg")
    if global_ref_angle is not None and global_ref_angle >= 1.0:
        ax.axhline(
            y=global_ref_angle * 1.5,
            color="#FF9800",
            linestyle="--",
            alpha=0.5,
            label=f"Moderate (1.5x global = {global_ref_angle * 1.5:.1f}°)",
        )
        ax.axhline(
            y=global_ref_angle * 2.5,
            color="#F44336",
            linestyle="--",
            alpha=0.5,
            label=f"Unreliable (2.5x global = {global_ref_angle * 2.5:.1f}°)",
        )
    else:
        # Fallback: absolute thresholds when no meaningful global reference
        ax.axhline(y=5.0, color="#FF9800", linestyle="--", alpha=0.5, label="Moderate (>5°)")
        ax.axhline(y=15.0, color="#F44336", linestyle="--", alpha=0.5, label="Unreliable (>15°)")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="upper left")

    # Annotate bars with sample counts and labels
    for i in range(len(bar_names)):
        ax.text(
            i,
            means[i] + stds[i] + 0.5,
            f"n={sample_counts[i]}\n{labels_list[i]}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()

    if output_dir is None:
        output_dir = ensure_dir(get_output_path(script_name="compute_wrmd"))

    if not os.path.isabs(output_file):
        output_path = os.path.join(output_dir, output_file)
    else:
        output_path = output_file

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved bootstrap stability plot: {output_path}")


def compute_cross_category_pca(
    category_vectors,
    global_vectors=None,
    target_layers=None,
    divergence_threshold=0.15,
):
    """Cross-category PCA to measure per-category divergence from shared refusal.

    Stacks all per-category steering vectors (at specified layers) and runs PCA
    on the centered matrix. PC1 captures the shared refusal direction; PC2 and
    beyond capture category-specific divergence. Categories with high PC2 loading
    diverge from the shared refusal subspace and may be better served by global
    vectors.

    .. warning::
        With few categories (N<30) in high-dimensional space, PCA components are
        unreliable. Use `compute_per_category_global_alignment` as the primary
        quality metric instead — cosine similarity with global directly measures
        whether a per-category vector captures the shared refusal direction.

    Args:
        category_vectors: dict[str, Tensor] mapping category name to
            [num_layers, hidden_size] or [num_layers, rank, hidden_size] vectors.
            Uses v_1 only if 3D.
        global_vectors: Optional [num_layers, hidden_size] tensor for the global
            steering vector. If provided, reports cosine similarity of PC1 with
            global direction.
        target_layers: List of layer indices to analyze. Default: all layers.
        divergence_threshold: Fraction of norm in non-PC1 components above which
            a category is flagged as "high divergence". Default: 0.15 (15%).

    Returns:
        dict with keys:
            categories: list of category names
            target_layers: list of layer indices used
            per_category: dict mapping category -> {pc1_frac, pc2_frac, pc2_angle_deg, ...}
            pc1_cosine_with_global: float or None
            flagged: list of categories with high divergence
    """
    from sklearn.decomposition import PCA

    categories = sorted(category_vectors.keys())
    n_cats = len(categories)

    if n_cats < 3:
        raise ValueError(f"Need at least 3 categories for PCA, got {n_cats}")

    # Extract vectors, flatten across target layers
    vecs_list = []
    for cat in categories:
        v = category_vectors[cat]
        if v.dim() == 3:
            v = v[:, 0, :]  # rank-1 only
        vecs_list.append(v)

    # Determine layers
    num_layers = vecs_list[0].shape[0]
    hidden_size = vecs_list[0].shape[1]
    if target_layers is None:
        target_layers = list(range(num_layers))

    # Stack and flatten: [n_cats, len(target_layers) * hidden_size]
    # Each row = one category's concatenated vector across target layers
    cat_matrix = (
        torch.stack([torch.cat([v[li] for li in target_layers]) for v in vecs_list]).float().numpy()
    )  # [n_cats, D]

    # Center the matrix (subtract mean across categories)
    mean_vec = cat_matrix.mean(axis=0, keepdims=True)
    centered = cat_matrix - mean_vec

    # PCA: n_cats is small (14), so we get at most n_cats-1 components
    n_components = min(n_cats - 1, 5)
    pca = PCA(n_components=n_components)
    pca.fit(centered)

    # Project each category onto principal components
    projections = pca.transform(centered)  # [n_cats, n_components]
    explained_var = pca.explained_variance_ratio_  # [n_components]

    # Per-category analysis
    per_category = {}
    for i, cat in enumerate(categories):
        cat_projs = projections[i]  # [n_components]
        total_energy = np.sum(cat_projs**2)

        pc1_frac = cat_projs[0] ** 2 / total_energy if total_energy > 0 else 0.0
        pc2_frac = (
            cat_projs[1] ** 2 / total_energy if total_energy > 0 and n_components > 1 else 0.0
        )
        non_pc1_frac = 1.0 - pc1_frac

        # Angle from PC1 direction (in the full PCA space)
        if total_energy > 0:
            cos_pc1 = cat_projs[0] / np.sqrt(total_energy)
            pc1_angle_deg = float(np.arccos(np.clip(abs(cos_pc1), 0, 1)) * 180 / np.pi)
        else:
            pc1_angle_deg = 90.0

        per_category[cat] = {
            "pc1_projection": float(cat_projs[0]),
            "pc2_projection": float(cat_projs[1]) if n_components > 1 else 0.0,
            "pc1_fraction": float(pc1_frac),
            "pc2_fraction": float(pc2_frac),
            "non_pc1_fraction": float(non_pc1_frac),
            "pc1_angle_deg": float(pc1_angle_deg),
        }

    # PC1 vs global alignment
    pc1_cosine_with_global = None
    if global_vectors is not None:
        if global_vectors.dim() == 3:
            global_vectors = global_vectors[:, 0, :]
        # Global vector flattened across target layers
        global_flat = torch.cat([global_vectors[li] for li in target_layers]).float().numpy()
        global_centered = global_flat - mean_vec[0]
        # Cosine similarity between global direction and PC1
        pc1_dir = pca.components_[0]
        cos_gl = np.dot(global_centered, pc1_dir) / (
            np.linalg.norm(global_centered) * np.linalg.norm(pc1_dir) + 1e-12
        )
        pc1_cosine_with_global = float(cos_gl)

    # Flag high-divergence categories
    flagged = [
        cat for cat in categories if per_category[cat]["non_pc1_fraction"] > divergence_threshold
    ]

    # Console output
    print(f"\n[CROSS-CATEGORY PCA]")
    print(f"   Categories: {n_cats}")
    print(f"   Target layers: {target_layers}")
    print(
        f"   Dimensionality: {len(target_layers)} × {hidden_size} = {len(target_layers) * hidden_size}"
    )
    print(f"   Divergence threshold: {divergence_threshold:.0%}")

    print(f"\n   Explained variance ratio:")
    for c in range(n_components):
        print(f"     PC{c+1}: {explained_var[c]:.1%}")

    if pc1_cosine_with_global is not None:
        print(f"\n   PC1 ↔ global cosine similarity: {pc1_cosine_with_global:.3f}")

    print(
        f"\n   {'Category':<50} {'PC1%':>6} {'PC2%':>6} {'Non-PC1':>8} {'PC1 angle':>10} {'Flag':>5}"
    )
    print(f"   {'-' * 90}")

    # Sort by non-PC1 fraction (most divergent first)
    sorted_cats = sorted(
        categories, key=lambda c: per_category[c]["non_pc1_fraction"], reverse=True
    )
    for cat in sorted_cats:
        d = per_category[cat]
        flag = "⚠️" if cat in flagged else " "
        print(
            f"   {cat:<50} {d['pc1_fraction']:>5.1%} {d['pc2_fraction']:>5.1%} "
            f"{d['non_pc1_fraction']:>7.1%} {d['pc1_angle_deg']:>9.1f}° {flag:>5}"
        )

    if flagged:
        print(
            f"\n   ⚠️  Flagged categories (>{divergence_threshold:.0%} non-PC1): {', '.join(flagged)}"
        )
        print(f"   → These should use GLOBAL vectors instead of per-category vectors")
    else:
        print(f"\n   ✓ All categories below divergence threshold")

    return {
        "categories": categories,
        "target_layers": target_layers,
        "explained_variance_ratio": explained_var.tolist(),
        "pc1_cosine_with_global": pc1_cosine_with_global,
        "per_category": per_category,
        "flagged": flagged,
        "divergence_threshold": divergence_threshold,
    }


def plot_cross_category_pca(results, output_dir=None, output_file="cross_category_pca.png"):
    """Visualize cross-category PCA results.

    Two-panel figure:
      Left: Stacked bar chart of PC1/PC2/other fractions per category
      Right: PC1 vs PC2 projection scatter

    Args:
        results: Output from compute_cross_category_pca()
        output_dir: Directory to save plot
        output_file: Filename for plot
    """
    from .utils import ensure_dir, get_output_path

    categories = results["categories"]
    per_category = results["per_category"]
    flagged = results["flagged"]
    threshold = results["divergence_threshold"]

    # Sort by non-PC1 fraction (most divergent first)
    sorted_cats = sorted(
        categories, key=lambda c: per_category[c]["non_pc1_fraction"], reverse=True
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Stacked bar chart
    pc1_fracs = [per_category[c]["pc1_fraction"] for c in sorted_cats]
    pc2_fracs = [per_category[c]["pc2_fraction"] for c in sorted_cats]
    other_fracs = [
        per_category[c]["non_pc1_fraction"] - per_category[c]["pc2_fraction"] for c in sorted_cats
    ]

    short_cats = [c[:35] + "..." if len(c) > 35 else c for c in sorted_cats]
    y_pos = range(len(sorted_cats))

    ax1.barh(y_pos, pc1_fracs, color="#4CAF50", alpha=0.85, label="PC1 (shared refusal)")
    ax1.barh(y_pos, pc2_fracs, left=pc1_fracs, color="#FF9800", alpha=0.85, label="PC2")
    ax1.barh(
        y_pos,
        other_fracs,
        left=[p1 + p2 for p1, p2 in zip(pc1_fracs, pc2_fracs)],
        color="#F44336",
        alpha=0.85,
        label="PC3+",
    )
    ax1.axvline(
        x=1.0 - threshold,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Threshold ({threshold:.0%})",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(short_cats, fontsize=8)
    ax1.set_xlabel("Fraction of norm")
    ax1.set_title("Per-Category PCA Decomposition")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.invert_yaxis()

    # Highlight flagged categories
    for i, cat in enumerate(sorted_cats):
        if cat in flagged:
            ax1.axhspan(i - 0.4, i + 0.4, color="red", alpha=0.1)

    # Right: PC1 vs PC2 scatter
    pc1_projs = [per_category[c]["pc1_projection"] for c in categories]
    pc2_projs = [per_category[c]["pc2_projection"] for c in categories]
    colors = ["#F44336" if c in flagged else "#4CAF50" for c in categories]

    ax2.scatter(pc1_projs, pc2_projs, c=colors, s=80, zorder=3)
    for i, cat in enumerate(categories):
        short = cat[:20] + "..." if len(cat) > 20 else cat
        ax2.annotate(
            short,
            (pc1_projs[i], pc2_projs[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
    ax2.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
    ax2.set_xlabel("PC1 projection")
    ax2.set_ylabel("PC2 projection")
    evr = results["explained_variance_ratio"]
    ax2.set_title(f"PC1 ({evr[0]:.0%}) vs PC2 ({evr[1]:.0%} variance)")

    # Add explained variance info
    info = f"PC1↔global cos_sim = {results.get('pc1_cosine_with_global', 'N/A')}"
    ax2.text(
        0.02,
        0.98,
        info,
        transform=ax2.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if output_dir is None:
        output_dir = ensure_dir(get_output_path(script_name="compute_wrmd"))

    if not os.path.isabs(output_file):
        output_path = os.path.join(output_dir, output_file)
    else:
        output_path = output_file

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved cross-category PCA plot: {output_path}")


def compute_per_category_global_alignment(
    category_vectors,
    global_vectors,
    target_layers=None,
    cosine_threshold=0.80,
):
    """Measure alignment between per-category and global steering vectors.

    For each category, computes the cosine similarity between its steering
    vector and the global vector at each target layer. Categories with low
    alignment (cos_sim < threshold) diverge from the shared refusal direction
    and may be better served by global vectors at steering time.

    This is the empirically validated quality metric: for Qwen3.5-9B,
    privacy_violation has cos_sim=0.677 and fails with per-category vectors
    (3% compliance) but succeeds with global vectors (57% compliance).
    All other categories have cos_sim > 0.85 and work well with per-cat vectors.

    Args:
        category_vectors: dict[str, Tensor] mapping category name to
            [num_layers, hidden_size] or [num_layers, rank, hidden_size] vectors.
            Uses v_1 only if 3D.
        global_vectors: [num_layers, hidden_size] tensor for the global
            steering vector.
        target_layers: List of layer indices to analyze. Default: all layers.
        cosine_threshold: Cosine similarity below which a category is flagged
            for global fallback. Default: 0.80.

    Returns:
        dict with keys:
            categories: list of category names
            target_layers: list of layer indices used
            threshold: float
            per_category: dict mapping category -> {mean_cos_sim, min_cos_sim,
                per_layer: {layer_idx: cos_sim}, recommended: "per_cat"|"global"}
            flagged: list of categories recommended for global fallback
    """
    import torch.nn.functional as F

    categories = sorted(category_vectors.keys())

    if global_vectors.dim() == 3:
        global_vectors = global_vectors[:, 0, :]

    num_layers = global_vectors.shape[0]
    if target_layers is None:
        target_layers = list(range(num_layers))

    per_category = {}
    for cat in categories:
        v = category_vectors[cat]
        if v.dim() == 3:
            v = v[:, 0, :]
        assert (
            v.shape == global_vectors.shape
        ), f"Shape mismatch: {cat} has {v.shape}, global has {global_vectors.shape}"

        per_layer = {}
        cos_values = []
        for li in target_layers:
            cos = F.cosine_similarity(
                v[li].float().unsqueeze(0),
                global_vectors[li].float().unsqueeze(0),
            ).item()
            per_layer[str(li)] = cos
            cos_values.append(cos)

        mean_cos = float(np.mean(cos_values))
        min_cos = float(np.min(cos_values))
        recommended = "global" if min_cos < cosine_threshold else "per_cat"

        per_category[cat] = {
            "mean_cosine_similarity": mean_cos,
            "min_cosine_similarity": min_cos,
            "per_layer": per_layer,
            "recommended": recommended,
        }

    flagged = [cat for cat in categories if per_category[cat]["recommended"] == "global"]

    # Console output
    print(f"\n[PER-CATEGORY GLOBAL ALIGNMENT]")
    print(f"   Categories: {len(categories)}")
    print(f"   Target layers: {target_layers}")
    print(f"   Cosine threshold: {cosine_threshold:.2f}")
    print(f"\n   {'Category':<50} {'Mean cos':>9} {'Min cos':>9} {'Recommended':>12}")
    print(f"   {'-' * 82}")

    sorted_cats = sorted(categories, key=lambda c: per_category[c]["min_cosine_similarity"])
    for cat in sorted_cats:
        d = per_category[cat]
        flag = " ⚠️ GLOBAL" if d["recommended"] == "global" else ""
        print(
            f"   {cat:<50} {d['mean_cosine_similarity']:>9.3f} "
            f"{d['min_cosine_similarity']:>9.3f} {d['recommended']:>12}{flag}"
        )

    if flagged:
        print(f"\n   ⚠️  Flagged for global fallback (cos_sim < {cosine_threshold}):")
        for cat in flagged:
            print(
                f"       - {cat} (min cos_sim = {per_category[cat]['min_cosine_similarity']:.3f})"
            )
    else:
        print(f"\n   ✓ All categories above cosine threshold ({cosine_threshold:.2f})")

    return {
        "categories": categories,
        "target_layers": target_layers,
        "threshold": cosine_threshold,
        "per_category": per_category,
        "flagged": flagged,
    }


def plot_per_category_global_alignment(
    results, output_dir=None, output_file="per_category_global_alignment.png"
):
    """Bar chart of per-category cosine similarity with global vector.

    Args:
        results: Output from compute_per_category_global_alignment()
        output_dir: Directory to save plot
        output_file: Filename for plot
    """
    from .utils import ensure_dir, get_output_path

    categories = results["categories"]
    per_category = results["per_category"]
    threshold = results["threshold"]
    flagged = results["flagged"]

    sorted_cats = sorted(categories, key=lambda c: per_category[c]["min_cosine_similarity"])
    min_cos = [per_category[c]["min_cosine_similarity"] for c in sorted_cats]
    mean_cos = [per_category[c]["mean_cosine_similarity"] for c in sorted_cats]
    short_cats = [c[:35] + "..." if len(c) > 35 else c for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.6), 6))

    x = range(len(sorted_cats))
    colors = ["#F44336" if c in flagged else "#4CAF50" for c in sorted_cats]

    ax.bar(x, min_cos, color=colors, alpha=0.85, label="Min cos_sim")
    ax.plot(x, mean_cos, "ko--", markersize=5, alpha=0.6, label="Mean cos_sim")
    ax.axhline(
        y=threshold, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({threshold:.2f})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cosine Similarity with Global")
    ax.set_title("Per-Category Alignment with Global Refusal Direction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_dir is None:
        output_dir = ensure_dir(get_output_path(script_name="compute_wrmd"))

    if not os.path.isabs(output_file):
        output_path = os.path.join(output_dir, output_file)
    else:
        output_path = output_file

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved per-category global alignment plot: {output_path}")


def compute_bootstrap_convergence(
    activations,
    labels,
    categories,
    judge_scores=None,
    target_layers=None,
    method="md",
    lambda_ridge=0.1,
    use_score_weighting=True,
    n_bootstrap=20,
    sample_ratio=0.8,
    min_pool_size=50,
    pool_step=50,
    min_samples=10,
    target_sizes=(500, 1000),
    instability_ceiling=30.0,
    seed=42,
):
    """
    Bootstrap convergence analysis: how steering vector stability changes with sample size.

    Runs bootstrap stability at progressively larger pool sizes drawn from existing data,
    fits angular_spread(n) = k / sqrt(n) + floor, and predicts how many samples are needed
    for each category to reach global-level stability.

    The fitted 'floor' parameter captures intrinsic geometric instability that no amount
    of data will resolve. Categories with floor >= instability_ceiling are flagged as
    fundamentally unstable.

    Args:
        activations: [N, num_layers, hidden_size] tensor
        labels: [N] tensor (1=refusal, 0=compliant)
        categories: list of N category strings (or None)
        judge_scores: [N] tensor of judge scores (for WRMD weighting)
        target_layers: list of layer indices (default: all)
        method: 'md', 'rmd', or 'wrmd'
        lambda_ridge: Ridge regularization parameter
        use_score_weighting: Whether to weight by judge scores (WRMD only)
        n_bootstrap: Number of bootstrap iterations per pool size
        sample_ratio: Bootstrap resample fraction (held constant)
        min_pool_size: Smallest pool size to test
        pool_step: Step between pool sizes
        min_samples: Minimum category samples to analyze at all
        target_sizes: Extrapolation target sample sizes
        instability_ceiling: Floor threshold (degrees) above which a category is unstable
        seed: Random seed for reproducibility

    Returns:
        dict with keys: config, global, per_category, summary
    """
    from scipy.optimize import curve_fit

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    num_layers = activations.shape[1]
    if target_layers is None:
        target_layers = list(range(num_layers))

    compliant_cache = _build_compliant_cache(
        activations, labels, judge_scores, target_layers, method, lambda_ridge, use_score_weighting
    )

    def _convergence_model(n, k, floor):
        return k / np.sqrt(n) + floor

    def _get_refusal_weights(refusal_idx):
        if judge_scores is not None and use_score_weighting and method == "wrmd":
            return torch.clamp(judge_scores[refusal_idx].clone(), min=0.01).to(activations.dtype)
        return None

    def _analyze_entity(refusal_idx, weights_full, entity_name):
        """Run convergence analysis for a single entity (global or category)."""
        import hashlib

        n_ref = len(refusal_idx)

        if n_ref < min_pool_size:
            return None

        # Stable hash for reproducibility across processes (hash() is randomised)
        name_hash = int(hashlib.md5(entity_name.encode()).hexdigest(), 16) % (2**31)

        # Build pool sizes
        pool_sizes = list(range(min_pool_size, n_ref + 1, pool_step))
        if pool_sizes[-1] != n_ref:
            pool_sizes.append(n_ref)

        pool_sizes_used = []
        spreads = []
        for n_pool in pool_sizes:
            # Independent seed per entity + pool size for independent draws
            pool_seed = (seed + name_hash + n_pool * 1000003) & 0x7FFFFFFF
            rng_pool = np.random.RandomState(pool_seed)
            pool_pos = rng_pool.choice(n_ref, size=n_pool, replace=False)
            pool_idx = [refusal_idx[p] for p in pool_pos]

            if weights_full is not None:
                pool_weights = weights_full[pool_pos]
            else:
                pool_weights = None

            # Independent RNG per pool size for bootstrap resampling
            bs_seed = (seed + name_hash + n_pool * 999983 + 1) & 0x7FFFFFFF
            rng_bootstrap = np.random.RandomState(bs_seed)
            result = _run_bootstrap_for_pool(
                activations,
                pool_idx,
                pool_weights,
                compliant_cache,
                target_layers,
                method,
                n_bootstrap,
                sample_ratio,
                rng_bootstrap,
                entity_name,
            )
            if result is None:
                print(
                    f"[WARN] {entity_name}: pool_size={n_pool} produced no results, skipping point"
                )
                continue
            spreads.append(result["mean_across_layers_deg"])
            pool_sizes_used.append(n_pool)

        if not spreads:
            return None

        current_spread = spreads[-1]  # spread at full sample size

        # Fit convergence curve
        fit_result = None
        if len(pool_sizes_used) >= 3:
            try:
                p0 = [spreads[0] * np.sqrt(pool_sizes_used[0]), 0.0]
                popt, pcov = curve_fit(
                    _convergence_model,
                    np.array(pool_sizes_used, dtype=float),
                    np.array(spreads, dtype=float),
                    p0=p0,
                    bounds=([0, 0], [np.inf, 90]),
                    maxfev=10000,
                )
                k_fit, floor_fit = float(popt[0]), float(popt[1])

                # R-squared
                predicted = _convergence_model(np.array(pool_sizes_used), k_fit, floor_fit)
                ss_res = np.sum((np.array(spreads) - predicted) ** 2)
                ss_tot = np.sum((np.array(spreads) - np.mean(spreads)) ** 2)
                r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                # Predictions at target sizes
                predictions = {}
                for ts in target_sizes:
                    predictions[str(ts)] = float(_convergence_model(ts, k_fit, floor_fit))

                fit_result = {
                    "k": k_fit,
                    "floor": floor_fit,
                    "r_squared": r_squared,
                    "predictions": predictions,
                }
            except (RuntimeError, ValueError):
                pass

        return {
            "num_samples": n_ref,
            "pool_sizes": pool_sizes_used,
            "angular_spreads": spreads,
            "current_spread": current_spread,
            "fit": fit_result,
        }

    # --- Run for global ---
    all_refusal_idx = [i for i in range(len(labels)) if labels[i].item() == 1]
    global_weights = _get_refusal_weights(all_refusal_idx)
    global_result = _analyze_entity(all_refusal_idx, global_weights, "_global")

    # Global spread is the stability target for categories
    global_spread = None
    if global_result is not None and global_result["fit"] is not None:
        global_spread = global_result["fit"]["floor"]
    elif global_result is not None:
        global_spread = global_result["current_spread"]

    # --- Run for each category ---
    unique_cats = sorted(
        {name for c in categories if c is not None for name in (c if isinstance(c, list) else [c])}
    )
    per_category = {}

    for cat in unique_cats:
        cat_refusal_idx = [
            i
            for i in range(len(labels))
            if categories[i] is not None
            and (cat in categories[i] if isinstance(categories[i], list) else categories[i] == cat)
            and labels[i].item() == 1
        ]

        if len(cat_refusal_idx) < min_samples:
            per_category[cat] = {
                "num_samples": len(cat_refusal_idx),
                "action": "too few samples",
                "skipped": True,
            }
            continue

        entity_result = _analyze_entity(cat_refusal_idx, _get_refusal_weights(cat_refusal_idx), cat)

        if entity_result is None:
            per_category[cat] = {
                "num_samples": len(cat_refusal_idx),
                "action": "analysis failed",
                "skipped": True,
            }
            continue

        # Classify action and compute samples_needed
        action = "insufficient data for fit"
        samples_needed = None
        improvement_ratio = None

        if entity_result["fit"] is not None:
            floor = entity_result["fit"]["floor"]
            current = entity_result["current_spread"]
            k_fit = entity_result["fit"]["k"]
            improvement_ratio = (current - floor) / current if current > 0 else 0.0

            if floor >= instability_ceiling:
                action = "unstable"
            elif global_spread is not None and floor >= global_spread:
                action = "geometric limit"
            elif improvement_ratio <= 0.15:
                action = "near floor"
            else:
                action = "collect more data"

            # Compute samples needed to reach global spread
            if global_spread is not None and floor < global_spread and k_fit > 0:
                gap = global_spread - floor
                if gap > 0:
                    samples_needed = int(np.ceil((k_fit / gap) ** 2))

        entity_result["action"] = action
        entity_result["samples_needed"] = samples_needed
        entity_result["improvement_ratio"] = improvement_ratio
        per_category[cat] = entity_result

    # Classify global result
    if global_result is not None:
        global_result["action"] = "reference"
        global_result["samples_needed"] = None
        global_result["improvement_ratio"] = None

    # Build summary
    ranked = sorted(
        [(cat, data) for cat, data in per_category.items() if not data.get("skipped")],
        key=lambda x: x[1]["current_spread"],
    )

    summary = {
        "global_reference_spread_deg": global_spread,
        "instability_ceiling_deg": instability_ceiling,
        "ranking": [
            {
                "category": cat,
                "current_spread": data["current_spread"],
                "floor": data["fit"]["floor"] if data.get("fit") else None,
                "samples_needed": data.get("samples_needed"),
                "action": data["action"],
            }
            for cat, data in ranked
        ],
    }

    # Console output
    print(
        f"\n[BOOTSTRAP CONVERGENCE] (n_bootstrap={n_bootstrap}, ratio={sample_ratio}, method={method})"
    )
    if global_spread is not None:
        print(f"   Stability target: {global_spread:.1f}° (global floor)")
        print(f"   Instability ceiling: {instability_ceiling:.1f}°")

    header = f"   {'Category':<30} {'Samples':>8} {'Current':>9} " f"{'Floor':>8} {'Needed':>8} "
    for ts in target_sizes:
        header += f"{'Pred@' + str(ts):>10} "
    header += f"{'R²':>6} {'Action':>20}"
    print(header)
    print(f"   {'─' * (len(header) - 3)}")

    # Print global first
    if global_result is not None:
        row = (
            f"   {'[global]':<30} {global_result['num_samples']:>8} "
            f"{global_result['current_spread']:>8.1f}° "
        )
        if global_result["fit"] is not None:
            row += f"{global_result['fit']['floor']:>7.1f}° "
            row += f"{'─':>8} "
            for ts in target_sizes:
                pred = global_result["fit"]["predictions"].get(str(ts))
                row += f"{pred:>9.1f}° " if pred is not None else f"{'─':>10} "
            row += f"{global_result['fit']['r_squared']:>6.2f} "
        else:
            row += f"{'─':>8} {'─':>8} "
            for ts in target_sizes:
                row += f"{'─':>10} "
            row += f"{'─':>6} "
        row += f"{'(reference)':>20}"
        print(row)
        print(f"   {'─' * (len(header) - 3)}")

    for cat, data in ranked:
        if data.get("skipped"):
            row = f"   {cat:<30} {data['num_samples']:>8} "
            row += f"{'─':>9} {'─':>8} {'─':>8} "
            for ts in target_sizes:
                row += f"{'─':>10} "
            row += f"{'─':>6} {data['action']:>20}"
            print(row)
            continue

        row = f"   {cat:<30} {data['num_samples']:>8} {data['current_spread']:>8.1f}° "

        if data["fit"] is not None:
            floor = data["fit"]["floor"]
            row += f"{floor:>7.1f}° "

            if data["samples_needed"] is not None:
                row += f"{data['samples_needed']:>8} "
            elif data["action"] == "unstable":
                row += f"{'∞':>8} "
            else:
                row += f"{'─':>8} "

            for ts in target_sizes:
                pred = data["fit"]["predictions"].get(str(ts))
                row += f"{pred:>9.1f}° " if pred is not None else f"{'─':>10} "
            row += f"{data['fit']['r_squared']:>6.2f} "
        else:
            row += f"{'─':>8} {'─':>8} "
            for ts in target_sizes:
                row += f"{'─':>10} "
            row += f"{'─':>6} "

        row += f"{data['action']:>20}"
        print(row)

    # Print skipped categories
    skipped = [(cat, data) for cat, data in per_category.items() if data.get("skipped")]
    for cat, data in sorted(skipped, key=lambda x: x[0]):
        row = f"   {cat:<30} {data['num_samples']:>8} "
        row += f"{'─':>9} {'─':>8} {'─':>8} "
        for ts in target_sizes:
            row += f"{'─':>10} "
        row += f"{'─':>6} {data['action']:>20}"
        print(row)

    results = {
        "config": {
            "n_bootstrap": n_bootstrap,
            "sample_ratio": sample_ratio,
            "method": method,
            "min_pool_size": min_pool_size,
            "pool_step": pool_step,
            "target_sizes": list(target_sizes),
            "instability_ceiling": instability_ceiling,
            "seed": seed,
        },
        "global": global_result,
        "per_category": per_category,
        "summary": summary,
    }

    return results


def plot_bootstrap_convergence(results, output_dir=None, output_file="bootstrap_convergence.png"):
    """
    Plot bootstrap convergence curves per category.

    Produces a multi-panel figure with one subplot per category showing:
    - Scatter points: (pool_size, angular_spread)
    - Fitted curve: k / sqrt(n) + floor
    - Horizontal dashed line at floor
    - Vertical markers at target sizes with predicted values

    Args:
        results: Output from compute_bootstrap_convergence()
        output_dir: Directory to save plot
        output_file: Filename for plot
    """
    from .utils import ensure_dir, get_output_path

    global_result = results.get("global")
    per_category = results.get("per_category", {})
    config = results.get("config", {})
    target_sizes = config.get("target_sizes", [500, 1000])
    instability_ceiling = config.get("instability_ceiling", 30.0)

    # Collect plottable entities (global + categories with data)
    entities = []
    if global_result is not None and not global_result.get("skipped"):
        entities.append(("[global]", global_result))
    for cat in sorted(per_category.keys()):
        data = per_category[cat]
        if not data.get("skipped"):
            entities.append((cat, data))

    if not entities:
        print("[WARN] No entities to plot for bootstrap convergence")
        return

    # Layout: up to 3 columns
    n_plots = len(entities)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    color_map = {
        "reference": "#2196F3",
        "collect more data": "#FF9800",
        "near floor": "#4CAF50",
        "geometric limit": "#9C27B0",
        "unstable": "#F44336",
        "insufficient data for fit": "#9E9E9E",
    }

    for idx, (name, data) in enumerate(entities):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        pool_sizes = data.get("pool_sizes")
        spreads = data.get("angular_spreads")
        action = data.get("action", "")
        color = color_map.get(action, "#9E9E9E")

        if not pool_sizes or not spreads:
            ax.set_title(f"{name}\n(no data)", fontsize=9)
            continue

        # Scatter actual data
        ax.scatter(pool_sizes, spreads, color=color, s=40, zorder=3, label="Measured")

        # Fit curve
        if data.get("fit") is not None:
            k_fit = data["fit"]["k"]
            floor = data["fit"]["floor"]
            r_sq = data["fit"]["r_squared"]

            # Smooth curve extending to max target size
            max_x = max(max(pool_sizes), max(target_sizes) if target_sizes else 0) * 1.1
            x_smooth = np.linspace(min(pool_sizes), max_x, 200)
            y_smooth = k_fit / np.sqrt(x_smooth) + floor

            ax.plot(x_smooth, y_smooth, color=color, alpha=0.7, linewidth=2, label="Fit")
            ax.axhline(y=floor, color=color, linestyle="--", alpha=0.4, label=f"Floor={floor:.1f}°")

            # Extrapolation markers at target sizes
            for ts in target_sizes:
                pred = data["fit"]["predictions"].get(str(ts))
                if pred is not None and ts > max(pool_sizes):
                    ax.plot(ts, pred, marker="v", color=color, markersize=8, alpha=0.7)
                    ax.annotate(
                        f"{pred:.1f}°",
                        (ts, pred),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=7,
                    )

            ax.set_title(f"{name}\n(R²={r_sq:.2f}, action={action})", fontsize=9)
        else:
            ax.set_title(f"{name}\n(no fit)", fontsize=9)

        # Instability ceiling reference
        ax.axhline(
            y=instability_ceiling,
            color="#F44336",
            linestyle=":",
            alpha=0.3,
            label=f"Ceiling={instability_ceiling:.0f}°",
        )

        ax.set_xlabel("Pool size (samples)", fontsize=9)
        ax.set_ylabel("Angular spread (°)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()

    if output_dir is None:
        output_dir = ensure_dir(get_output_path(script_name="compute_wrmd"))

    if not os.path.isabs(output_file):
        output_path = os.path.join(output_dir, output_file)
    else:
        output_path = output_file

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved bootstrap convergence plot: {output_path}")
