"""
Steering vector computation using various statistical methods.

Implements Mean Difference (MD), Ridge Mean Difference (RMD), and
Weighted Ridge Mean Difference (WRMD) for computing steering vectors.
Supports multi-rank decomposition via PCA of residuals after projecting
out the primary steering direction.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_md(refusal_acts, compliant_acts):
    """
    Simple Mean Difference.

    Args:
        refusal_acts: [N_refusal, hidden_size]
        compliant_acts: [N_compliant, hidden_size]

    Returns:
        Steering vector: [hidden_size]
    """
    return refusal_acts.mean(dim=0) - compliant_acts.mean(dim=0)


def compute_weighted_mean(activations, weights):
    """
    Compute weighted mean of activations.

    Args:
        activations: [N, hidden_size]
        weights: [N] - will be normalized to sum to 1

    Returns:
        weighted_mean: [hidden_size]
    """
    weights = weights.to(activations.dtype)
    weights = weights / weights.sum()
    return (activations.T @ weights).squeeze()


def compute_rmd(refusal_acts, compliant_acts, lambda_ridge=0.1):
    """
    Ridge Mean Difference.

    Args:
        refusal_acts: [N_refusal, hidden_size]
        compliant_acts: [N_compliant, hidden_size]
        lambda_ridge: Ridge regularization parameter

    Returns:
        Steering vector: [hidden_size]
    """
    compliant_mean = compliant_acts.mean(dim=0)
    centered = compliant_acts - compliant_mean
    cov = (centered.T @ centered) / len(compliant_acts)

    hidden_size = cov.shape[0]

    # Upcast to float32 for matrix inversion
    cov_f32 = cov.float()
    ridge_inv = torch.linalg.inv(
        cov_f32 + lambda_ridge * torch.eye(hidden_size, dtype=torch.float32, device=cov.device)
    )
    ridge_inv = ridge_inv.to(cov.dtype)

    mean_diff = refusal_acts.mean(dim=0) - compliant_mean

    return ridge_inv @ mean_diff


def compute_wrmd(
    refusal_acts,
    compliant_acts,
    refusal_weights=None,
    compliant_weights=None,
    neutral_acts=None,
    neutral_weights=None,
    lambda_ridge=0.1,
):
    """
    Weighted Ridge Mean Difference (WRMD).

    Weights samples by judge confidence scores and accounts for covariance structure.

    Args:
        refusal_acts: [N_refusal, hidden_size]
        compliant_acts: [N_compliant, hidden_size]
        refusal_weights: [N_refusal] - judge confidence scores
        compliant_weights: [N_compliant] - judge confidence scores
        neutral_acts: Optional neutral baseline activations
        neutral_weights: Optional weights for neutral samples
        lambda_ridge: Ridge regularization parameter

    Returns:
        Steering vector: [hidden_size]
    """
    # If no neutral provided, use compliant as neutral
    if neutral_acts is None:
        neutral_acts = compliant_acts
        neutral_weights = compliant_weights

    # Default to uniform weights if none provided
    if refusal_weights is None:
        refusal_weights = torch.ones(len(refusal_acts))
    if compliant_weights is None:
        compliant_weights = torch.ones(len(compliant_acts))
    if neutral_weights is None:
        neutral_weights = torch.ones(len(neutral_acts))

    # Convert weights to match activation dtype
    refusal_weights = refusal_weights.to(refusal_acts.dtype)
    compliant_weights = compliant_weights.to(compliant_acts.dtype)
    neutral_weights = neutral_weights.to(neutral_acts.dtype)

    # Compute weighted neutral mean
    neutral_mean = compute_weighted_mean(neutral_acts, neutral_weights)

    # Center activations by neutral
    refusal_centered = refusal_acts - neutral_mean
    compliant_centered = compliant_acts - neutral_mean

    # Compute weighted means of centered activations
    refusal_mean_centered = compute_weighted_mean(refusal_centered, refusal_weights)
    compliant_mean_centered = compute_weighted_mean(compliant_centered, compliant_weights)

    # Compute weighted covariance of compliant distribution
    centered_from_mean = compliant_centered - compliant_mean_centered
    weighted_centered = centered_from_mean * compliant_weights.unsqueeze(1)
    cov = (weighted_centered.T @ centered_from_mean) / compliant_weights.sum()

    # Ridge inverse (upcast to float32)
    hidden_size = cov.shape[0]
    cov_f32 = cov.float()
    ridge_inv = torch.linalg.inv(
        cov_f32 + lambda_ridge * torch.eye(hidden_size, dtype=torch.float32, device=cov.device)
    )
    ridge_inv = ridge_inv.to(cov.dtype)

    # Weighted mean difference
    mean_diff = refusal_mean_centered - compliant_mean_centered

    return ridge_inv @ mean_diff


def compute_multirank_vectors(
    refusal_acts,
    compliant_acts,
    refusal_weights=None,
    compliant_weights=None,
    neutral_acts=None,
    neutral_weights=None,
    lambda_ridge=0.1,
    rank=2,
    method="md",
):
    """
    Compute multi-rank steering vectors via PCA of residuals.

    v_1 is the standard WRMD/RMD/MD vector. For k > 1, project v_1 out of
    the refusal activations and compute PCA on the projected residuals to
    find v_2, v_3, etc. These additional directions capture structured
    variance in how the model refuses, not whether it refuses.

    Args:
        refusal_acts: [N_refusal, hidden_size]
        compliant_acts: [N_compliant, hidden_size]
        refusal_weights: [N_refusal] - judge confidence scores (WRMD only)
        compliant_weights: [N_compliant] - judge confidence scores (WRMD only)
        neutral_acts: Optional neutral baseline activations
        neutral_weights: Optional weights for neutral samples
        lambda_ridge: Ridge regularization parameter
        rank: Number of steering directions to compute (default: 2)
        method: 'md', 'rmd', or 'wrmd'

    Returns:
        Tensor [rank, hidden_size] — steering vectors ordered by importance
    """
    # Compute primary steering vector (v_1)
    if method == "md":
        v1 = compute_md(refusal_acts, compliant_acts)
    elif method == "rmd":
        v1 = compute_rmd(refusal_acts, compliant_acts, lambda_ridge)
    elif method == "wrmd":
        v1 = compute_wrmd(
            refusal_acts,
            compliant_acts,
            refusal_weights,
            compliant_weights,
            neutral_acts=neutral_acts,
            neutral_weights=neutral_weights,
            lambda_ridge=lambda_ridge,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if rank == 1:
        return v1.unsqueeze(0)

    # Normalize v1 for projection (guard against zero norm)
    v1_norm = v1.norm()
    if v1_norm < 1e-8:
        raise ValueError(
            "Cannot compute multi-rank vectors: primary steering vector has near-zero norm"
        )
    v1_unit = v1 / v1_norm

    # Work in float32 for numerical stability in SVD
    refusal_f32 = refusal_acts.float()

    # Center refusal activations
    refusal_mean = refusal_f32.mean(dim=0)
    refusal_centered = refusal_f32 - refusal_mean

    # Project out v1 from refusal activations
    v1_unit_f32 = v1_unit.float()
    projections = refusal_centered @ v1_unit_f32  # [N_refusal]
    refusal_projected = refusal_centered - projections.unsqueeze(1) * v1_unit_f32.unsqueeze(0)

    # PCA on projected residuals via SVD
    # U @ diag(S) @ V^T = refusal_projected
    # V columns are the principal directions
    U, S, Vt = torch.linalg.svd(refusal_projected, full_matrices=False)

    # Collect vectors: v1 + top-(rank-1) principal components from residuals
    vectors = [v1]
    max_available = Vt.shape[0]
    additional_needed = min(rank - 1, max_available)
    if additional_needed < rank - 1:
        print(
            f"      WARNING: only {additional_needed + 1} directions available "
            f"(requested rank={rank}, N_refusal={refusal_acts.shape[0]})"
        )

    # Report variance explained by additional components
    total_var = (S**2).sum().item()
    for i in range(additional_needed):
        vi = Vt[i].to(v1.dtype)
        var_explained = (S[i] ** 2).item() / total_var if total_var > 0 else 0.0
        print(
            f"      v_{i+2}: variance explained = {var_explained:.3f} "
            f"(singular value = {S[i].item():.2f})"
        )
        vectors.append(vi)

    return torch.stack(vectors)


class WRMDCalculator:
    """Calculate steering vectors using various methods."""

    def __init__(self, activation_file):
        """
        Initialize calculator from activation file.

        Args:
            activation_file: Path to .pt file with activations
        """
        print(f"[LOAD] Loading activations from: {activation_file}")
        data = torch.load(activation_file, weights_only=True)

        self.labels = data["labels"]  # [N]
        self.prompts = data["prompts"]
        self.num_layers = data["num_layers"]
        self.hidden_size = data["hidden_size"]

        # Detect available components
        self.available_components = data.get("components", ["layer"])

        # Load activations for each available component
        self._activations = {}
        if "activations_attn" in data:
            self._activations["attn"] = data["activations_attn"]
        if "activations_mlp" in data:
            self._activations["mlp"] = data["activations_mlp"]

        # Load residual stream ('activations' key) as 'layer' component.
        # Present when extraction ran with --components attn+mlp+layer.
        if "activations_layer" in data:
            self._activations["layer"] = data["activations_layer"]
        elif "activations" in data:
            self._activations["layer"] = data["activations"]

        # For backward compat: self.activations points to the default component
        default_comp = self.available_components[0] if self.available_components else "layer"
        self.activations = self._activations.get(
            default_comp, next(iter(self._activations.values()))
        )

        # Extract judge scores from metadata
        if "metadata" in data:
            self.metadata = data["metadata"]
            self.judge_scores = torch.tensor([m["score"] for m in self.metadata])
            print(f"[OK] Found judge scores in metadata")
        else:
            self.metadata = None
            self.judge_scores = None
            print(f"[WARN]  No metadata found - will use uniform weights")

        print(f"[OK] Loaded {len(self.labels)} samples")
        print(f"   Layers: {self.num_layers}, Hidden size: {self.hidden_size}")
        print(f"   Components: {list(self._activations.keys())}")
        print(f"   Refusal: {(self.labels == 1).sum()}, Compliant: {(self.labels == 0).sum()}")

        # Extract categories from metadata if present
        # Category values can be a string, a list of strings (multi-label), or None.
        # Normalize to: list of strings (multi-label) or None per sample.
        self.categories = None
        if self.metadata:
            cats = []
            for m in self.metadata:
                raw = m.get("category")
                if raw is None:
                    cats.append(None)
                elif isinstance(raw, list):
                    cats.append([str(c) for c in raw] if raw else None)
                else:
                    cats.append([str(raw)])
            if any(c is not None for c in cats):
                self.categories = cats
                cat_names = sorted({name for c in cats if c is not None for name in c})
                none_count = sum(1 for c in cats if c is None)
                print(f"[OK] Found {len(cat_names)} categories in metadata")
                for cat in cat_names:
                    count = sum(1 for c in cats if c is not None and cat in c)
                    print(f"      {cat}: {count}")
                if none_count:
                    print(f"      (uncategorized): {none_count}")

        if self.judge_scores is not None:
            print(
                f"   Judge score range: [{self.judge_scores.min():.2f}, {self.judge_scores.max():.2f}]"
            )

    def get_available_categories(self):
        """
        Return sorted list of unique category names, or empty list if no categories.
        """
        if self.categories is None:
            return []
        return sorted({name for c in self.categories if c is not None for name in c})

    def _build_category_mask(self, categories):
        """
        Build a boolean mask selecting samples matching given category name(s).

        Args:
            categories: Single category string or list of category strings

        Returns:
            Boolean tensor [N] — True for samples matching any listed category

        Raises:
            ValueError: If no samples match the given categories
        """
        if isinstance(categories, str):
            categories = [categories]

        if self.categories is None:
            raise ValueError(
                "No category metadata available. "
                "Extract activations with --dataset to include category information."
            )

        cat_set = set(categories)
        mask = torch.zeros(len(self.categories), dtype=torch.bool)
        for i, cat in enumerate(self.categories):
            if cat is not None and cat_set.intersection(cat):
                mask[i] = True

        if not mask.any():
            available = self.get_available_categories()
            raise ValueError(
                f"No samples found for categories: {categories}. "
                f"Available categories: {available}"
            )

        return mask

    def get_activations(self, component="attn"):
        """
        Get activations for a specific component.

        Args:
            component: "layer", "attn", or "mlp"

        Returns:
            Tensor [N, num_layers, hidden_size]
        """
        if component in self._activations:
            return self._activations[component]
        # Fallback: if requesting 'mlp' but only 'activations' key exists (old format)
        if component == "mlp" and "layer" in self._activations:
            print(f"[WARN] Component '{component}' not found, falling back to 'layer'")
            return self._activations["layer"]
        if component == "layer" and self._activations:
            print(
                f"[WARN] Component 'layer' not found, falling back to '{next(iter(self._activations))}'"
            )
            return next(iter(self._activations.values()))
        raise ValueError(
            f"Component '{component}' not found. Available: {list(self._activations.keys())}"
        )

    def get_weights_from_scores(self, scores, label):
        """
        Convert judge scores to weights for WRMD.

        For refusals (label=1): higher score = higher weight
        For compliances (label=0): more negative score = higher weight

        Args:
            scores: Judge confidence scores
            label: 0 for compliant, 1 for refusal

        Returns:
            Positive weights proportional to confidence
        """
        if label == 1:
            # Refusals: score > 0, higher is better
            weights = scores.clone()
        else:
            # Compliances: score < 0, more negative is better
            weights = -scores.clone()

        # Ensure all weights are positive
        weights = torch.clamp(weights, min=0.01)

        return weights

    def compute_steering_vectors(
        self,
        method="md",
        lambda_ridge=0.1,
        use_score_weighting=True,
        normalize=False,
        rank=1,
        component="attn",
        categories=None,
    ):
        """
        Compute steering vectors for each layer.

        Args:
            method: 'md', 'rmd', or 'wrmd'
            lambda_ridge: Ridge regularization parameter
            use_score_weighting: Whether to weight by judge scores (WRMD only)
            normalize: Whether to normalize vectors to unit length
            rank: Number of steering directions per layer (default: 1).
                  rank > 1 uses PCA of residuals after projecting out v_1.
            component: Which activations to use: "layer", "attn", or "mlp" (default: "attn")
            categories: Optional category name(s) to filter refusal cohort.
                When set, only refusal samples from these categories are used.
                Compliant baseline is always global (all compliant samples).

        Returns:
            If rank == 1: steering_vectors [num_layers, hidden_size]
            If rank > 1:  steering_vectors [num_layers, rank, hidden_size]
        """
        # Clone to prevent in-place sanitization from mutating the stored activations.
        # Without .clone(), the p1/p99 clipping at lines ~571-572 permanently alters
        # the data, making subsequent compute_steering_vectors() calls produce different
        # vectors from progressively more clipped input.
        activations = self.get_activations(component).clone()

        print(f"\n[COMPUTE] Computing steering vectors using {method.upper()}")
        print(f"   Component: {component}")
        print(f"   Ridge λ: {lambda_ridge}")
        print(f"   Score weighting: {use_score_weighting and method == 'wrmd'}")
        print(f"   Normalize: {normalize}")
        if rank > 1:
            print(f"   Rank: {rank} (multi-rank decomposition)")

        steering_vectors = []

        # Separate by label
        refusal_mask = self.labels == 1
        compliant_mask = self.labels == 0

        # Apply category filter to refusal cohort if specified
        if categories is not None:
            category_mask = self._build_category_mask(categories)
            refusal_mask = refusal_mask & category_mask
            # Compliant baseline stays global (not filtered)
            cat_str = categories if isinstance(categories, str) else ", ".join(categories)
            print(f"   Category filter: {cat_str}")
            n_refusal = refusal_mask.sum().item()
            print(f"   Filtered refusal samples: {n_refusal}")
            if n_refusal < 2:
                raise ValueError(
                    f"Category filter produced only {n_refusal} refusal sample(s). "
                    f"Minimum 2 required for steering vector computation."
                )

        # Prepare weights if using score weighting
        if use_score_weighting and self.judge_scores is not None and method == "wrmd":
            refusal_scores = self.judge_scores[refusal_mask]
            compliant_scores = self.judge_scores[compliant_mask]

            refusal_weights = self.get_weights_from_scores(refusal_scores, label=1)
            compliant_weights = self.get_weights_from_scores(compliant_scores, label=0)

            print(
                f"   Refusal weights: mean={refusal_weights.mean():.2f}, std={refusal_weights.std():.2f}"
            )
            print(
                f"   Compliant weights: mean={compliant_weights.mean():.2f}, std={compliant_weights.std():.2f}"
            )
        else:
            refusal_weights = None
            compliant_weights = None

        dtype_max = torch.finfo(activations.dtype).max

        nan_mask = activations.isnan()
        nan_count = nan_mask.sum().item()
        inf_mask = activations.isinf()
        inf_count = inf_mask.sum().item()
        if nan_count > 0:
            activations = torch.where(nan_mask, torch.zeros_like(activations), activations)
        if inf_count > 0:
            activations = torch.clamp(activations, min=-dtype_max, max=dtype_max)
        if nan_count > 0 or inf_count > 0:
            print(
                f"   [SANITIZE] Stage A: dtype limit clipping — "
                f"{nan_count} NaN elements → 0, {inf_count} inf elements → ±{dtype_max:.2e}"
            )

        norm_threshold = 1000.0
        sample_norms = activations.float().norm(dim=2)
        clean_mask = torch.isfinite(sample_norms) & (sample_norms < norm_threshold)
        clean_mask_per_layer = clean_mask.unsqueeze(2).expand_as(activations)

        sanitize_log = {}
        for layer in range(self.num_layers):
            refusal_clean = clean_mask_per_layer[refusal_mask, layer, 0]
            compliant_clean = clean_mask_per_layer[compliant_mask, layer, 0]

            refusal_acts_raw = activations[refusal_mask, layer, :]
            compliant_acts_raw = activations[compliant_mask, layer, :]

            refusal_clean_acts = refusal_acts_raw[refusal_clean]
            compliant_clean_acts = compliant_acts_raw[compliant_clean]

            if refusal_clean_acts.shape[0] >= 10:
                ref_lo = refusal_clean_acts.float().quantile(0.01, dim=0)
                ref_hi = refusal_clean_acts.float().quantile(0.99, dim=0)
            else:
                ref_lo = refusal_acts_raw.float().quantile(0.01, dim=0)
                ref_hi = refusal_acts_raw.float().quantile(0.99, dim=0)

            if compliant_clean_acts.shape[0] >= 10:
                comp_lo = compliant_clean_acts.float().quantile(0.01, dim=0)
                comp_hi = compliant_clean_acts.float().quantile(0.99, dim=0)
            else:
                comp_lo = compliant_acts_raw.float().quantile(0.01, dim=0)
                comp_hi = compliant_acts_raw.float().quantile(0.99, dim=0)

            refusal_clipped = torch.clamp(refusal_acts_raw.float(), min=ref_lo, max=ref_hi)
            compliant_clipped = torch.clamp(compliant_acts_raw.float(), min=comp_lo, max=comp_hi)

            ref_changed = (refusal_clipped != refusal_acts_raw.float()).any(dim=1)
            comp_changed = (compliant_clipped != compliant_acts_raw.float()).any(dim=1)
            if ref_changed.any() or comp_changed.any():
                ref_indices = torch.where(refusal_mask)[0][ref_changed].tolist()
                comp_indices = torch.where(compliant_mask)[0][comp_changed].tolist()
                sanitize_log[layer] = {
                    "refusal_samples": ref_indices,
                    "compliant_samples": comp_indices,
                }

            activations[refusal_mask, layer, :] = refusal_clipped.to(activations.dtype)
            activations[compliant_mask, layer, :] = compliant_clipped.to(activations.dtype)

        if sanitize_log:
            total_ref = sum(len(v["refusal_samples"]) for v in sanitize_log.values())
            total_comp = sum(len(v["compliant_samples"]) for v in sanitize_log.values())
            n_layers = len(sanitize_log)
            print(
                f"   [SANITIZE] Stage B: per-layer p1/p99 percentile clipping — "
                f"{total_ref} refusal + {total_comp} compliant samples clipped "
                f"across {n_layers}/{self.num_layers} layers"
            )
            for layer, info in sorted(sanitize_log.items()):
                parts = []
                if info["refusal_samples"]:
                    parts.append(
                        f"refusal samples {info['refusal_samples'][:5]}"
                        + (
                            f"... (+{len(info['refusal_samples'])-5} more)"
                            if len(info["refusal_samples"]) > 5
                            else ""
                        )
                    )
                if info["compliant_samples"]:
                    parts.append(
                        f"compliant samples {info['compliant_samples'][:5]}"
                        + (
                            f"... (+{len(info['compliant_samples'])-5} more)"
                            if len(info["compliant_samples"]) > 5
                            else ""
                        )
                    )
                print(f"     Layer {layer}: {', '.join(parts)}")
        else:
            print(
                f"   [SANITIZE] Stage B: per-layer p1/p99 percentile clipping — "
                f"no samples needed clipping"
            )

        for layer in range(self.num_layers):
            layer_acts = activations[:, layer, :]

            refusal_acts = layer_acts[refusal_mask]
            compliant_acts = layer_acts[compliant_mask]

            if rank > 1:
                print(f"   Layer {layer}:")
                vecs = compute_multirank_vectors(
                    refusal_acts,
                    compliant_acts,
                    refusal_weights,
                    compliant_weights,
                    neutral_acts=compliant_acts,
                    neutral_weights=compliant_weights,
                    lambda_ridge=lambda_ridge,
                    rank=rank,
                    method=method,
                )
                if normalize:
                    norms = vecs.norm(dim=1, keepdim=True)
                    vecs = vecs / norms.clamp(min=1e-8)
                steering_vectors.append(vecs)
            else:
                if method == "md":
                    vec = compute_md(refusal_acts, compliant_acts)
                elif method == "rmd":
                    vec = compute_rmd(refusal_acts, compliant_acts, lambda_ridge)
                elif method == "wrmd":
                    vec = compute_wrmd(
                        refusal_acts,
                        compliant_acts,
                        refusal_weights,
                        compliant_weights,
                        neutral_acts=compliant_acts,
                        neutral_weights=compliant_weights,
                        lambda_ridge=lambda_ridge,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                if normalize:
                    vec_norm = vec.norm()
                    if vec_norm < 1e-8:
                        raise ValueError(
                            f"Steering vector at layer {layer} has near-zero norm, cannot normalize"
                        )
                    vec = vec / vec_norm

                steering_vectors.append(vec)

        steering_vectors = torch.stack(steering_vectors)

        print(f"[OK] Computed steering vectors")
        print(f"   Shape: {steering_vectors.shape}")
        if rank == 1:
            print(
                f"   Norm range: [{steering_vectors.norm(dim=1).min():.2f}, {steering_vectors.norm(dim=1).max():.2f}]"
            )
        else:
            # [num_layers, rank, hidden_size]
            v1_norms = steering_vectors[:, 0, :].norm(dim=1)
            print(f"   v_1 norm range: [{v1_norms.min():.2f}, {v1_norms.max():.2f}]")

        return steering_vectors

    def analyze_vectors(self, steering_vectors, output_dir=None):
        """
        Analyze properties of steering vectors.

        Args:
            steering_vectors: [num_layers, hidden_size]
            output_dir: Directory to save plots

        Returns:
            Array of norms per layer
        """
        from .utils import get_output_path

        print("\n[INFO] Vector Analysis:")

        # Compute norms per layer
        norms = steering_vectors.norm(dim=1).float().numpy()

        print(f"   Mean norm: {norms.mean():.2f}")
        print(f"   Std norm: {norms.std():.2f}")
        print(f"   Max norm layer: {norms.argmax()} (norm={norms.max():.2f})")
        print(f"   Min norm layer: {norms.argmin()} (norm={norms.min():.2f})")

        # Plot norms by layer
        plt.figure(figsize=(12, 4))
        plt.plot(norms, marker="o", linewidth=2, markersize=6)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Vector Norm", fontsize=12)
        plt.title("Steering Vector Magnitude by Layer", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(
            y=norms.mean(), color="r", linestyle="--", alpha=0.5, label=f"Mean: {norms.mean():.2f}"
        )
        plt.legend()
        plt.tight_layout()

        # Save plot
        if output_dir is None:
            output_path = get_output_path(
                script_name="compute_wrmd", filename="steering_vector_norms.png"
            )
        else:
            output_path = os.path.join(output_dir, "steering_vector_norms.png")

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"   [PLOT] Saved plot: {output_path}")

        return norms

    def analyze_rank_associations(
        self, steering_vectors, target_layers=None, component="attn", output_dir=None
    ):
        """
        Analyze which PCA rank each prompt is most associated with.

        Convenience wrapper around the module-level analyze_rank_associations().

        Args:
            steering_vectors: [num_layers, rank, hidden_size] multi-rank vectors
            target_layers: list of layer indices (default: all)
            component: which activation component to use
            output_dir: optional directory to save results

        Returns:
            dict with per_prompt, per_layer, and summary results
        """
        activations = self.get_activations(component)
        return analyze_rank_associations(
            activations=activations,
            steering_vectors=steering_vectors,
            labels=self.labels,
            prompts=self.prompts,
            target_layers=target_layers,
            output_dir=output_dir,
        )

    def analyze_category_distances(
        self,
        method="md",
        lambda_ridge=0.1,
        use_score_weighting=True,
        normalize=False,
        rank=1,
        component="attn",
        target_layers=None,
        output_dir=None,
        min_samples=10,
    ):
        """
        Compute pairwise angular distances between per-category steering vectors.

        Computes vectors on-the-fly for each category, then delegates to
        compute_category_angular_distances() and plot_category_angular_distances().

        Args:
            method: 'md', 'rmd', or 'wrmd'
            lambda_ridge: Ridge regularization parameter
            use_score_weighting: Whether to weight by judge scores (WRMD only)
            normalize: Whether to normalize vectors to unit length
            rank: Number of steering directions per layer
            component: Which activations to use
            target_layers: Layer indices for analysis (default: all)
            output_dir: Directory to save results
            min_samples: Minimum refusal samples per category (skip if fewer)

        Returns:
            Results dict from compute_category_angular_distances()
        """
        from .analysis import compute_category_angular_distances, plot_category_angular_distances

        available = self.get_available_categories()
        if len(available) < 2:
            raise ValueError(
                f"Need at least 2 categories for angular distance analysis, "
                f"found {len(available)}: {available}"
            )

        print(
            f"\n[CATEGORY ANALYSIS] Computing per-category vectors for {len(available)} categories"
        )

        category_vectors = {}
        for cat in available:
            # Check sample count before computing
            cat_mask = self._build_category_mask([cat])
            n_refusal = ((self.labels == 1) & cat_mask).sum().item()
            if n_refusal < min_samples:
                print(
                    f"[SKIP] Category '{cat}': only {n_refusal} refusal samples "
                    f"(minimum: {min_samples})"
                )
                continue

            try:
                vectors = self.compute_steering_vectors(
                    method=method,
                    lambda_ridge=lambda_ridge,
                    use_score_weighting=use_score_weighting,
                    normalize=normalize,
                    rank=rank,
                    component=component,
                    categories=[cat],
                )
                category_vectors[cat] = vectors
            except ValueError as e:
                print(f"[WARN] Skipping category '{cat}': {e}")

        if len(category_vectors) < 2:
            raise ValueError(
                f"Only {len(category_vectors)} categories had enough samples. "
                f"Need at least 2 for comparison."
            )

        results = compute_category_angular_distances(category_vectors, target_layers=target_layers)
        plot_category_angular_distances(results, output_dir=output_dir)

        # Save JSON if output_dir provided
        if output_dir is not None:
            import json

            json_path = os.path.join(output_dir, "category_angular_distances.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[SAVE] Saved category angular distances to {json_path}")

        return results

    def analyze_intra_category_distances(
        self,
        target_layers=None,
        component="attn",
        min_samples=10,
        output_dir=None,
    ):
        """
        Measure intra-category angular spread of refusal activations.

        For each category, computes pairwise angular distances between individual
        refusal activation vectors to assess how coherent/stable the category's
        refusal circuit is.

        Args:
            target_layers: Layer indices for analysis (default: all)
            component: Which activations to use
            min_samples: Minimum refusal samples per category
            output_dir: Directory to save results

        Returns:
            Results dict from compute_intra_category_angular_distances()
        """
        from .analysis import (
            compute_intra_category_angular_distances,
            plot_intra_category_angular_distances,
        )

        if not self.categories:
            raise ValueError("No category metadata available.")

        activations = self.get_activations(component)

        results = compute_intra_category_angular_distances(
            activations=activations,
            labels=self.labels,
            categories=self.categories,
            target_layers=target_layers,
            min_samples=min_samples,
        )

        plot_intra_category_angular_distances(results, output_dir=output_dir)

        if output_dir is not None:
            import json

            json_path = os.path.join(output_dir, "intra_category_coherence.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[SAVE] Saved intra-category coherence to {json_path}")

        return results

    def analyze_bootstrap_stability(
        self,
        method="md",
        lambda_ridge=0.1,
        use_score_weighting=True,
        component="attn",
        target_layers=None,
        n_bootstrap=20,
        sample_ratio=0.8,
        min_samples=10,
        seed=42,
        output_dir=None,
        distance_results=None,
    ):
        """
        Compute bootstrap stability of per-category steering vectors.

        Args:
            method: 'md', 'rmd', or 'wrmd'
            lambda_ridge: Ridge regularization parameter
            use_score_weighting: Whether to weight by judge scores (WRMD only)
            component: Which activations to use
            target_layers: Layer indices for analysis (default: all)
            n_bootstrap: Number of bootstrap iterations
            sample_ratio: Fraction of refusal samples to resample per iteration
            min_samples: Minimum refusal samples per category
            seed: Random seed for reproducibility
            output_dir: Directory to save results
            distance_results: Optional inter-category distance results to annotate
                with significance

        Returns:
            Results dict from compute_bootstrap_stability()
        """
        from .analysis import (
            annotate_distances_with_significance,
            compute_bootstrap_stability,
            plot_bootstrap_stability,
        )

        if not self.categories:
            raise ValueError("No category metadata available.")

        activations = self.get_activations(component)

        results = compute_bootstrap_stability(
            activations=activations,
            labels=self.labels,
            categories=self.categories,
            judge_scores=self.judge_scores,
            target_layers=target_layers,
            method=method,
            lambda_ridge=lambda_ridge,
            use_score_weighting=use_score_weighting,
            n_bootstrap=n_bootstrap,
            sample_ratio=sample_ratio,
            min_samples=min_samples,
            seed=seed,
        )

        plot_bootstrap_stability(results, output_dir=output_dir)

        # Annotate inter-category distances with significance if available
        significance = None
        if distance_results is not None:
            significance = annotate_distances_with_significance(distance_results, results)

        if output_dir is not None:
            import json

            json_path = os.path.join(output_dir, "bootstrap_stability.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[SAVE] Saved bootstrap stability to {json_path}")

            if significance is not None:
                sig_path = os.path.join(output_dir, "category_distance_significance.json")
                with open(sig_path, "w") as f:
                    json.dump(significance, f, indent=2)
                print(f"[SAVE] Saved significance annotations to {sig_path}")

        return results

    def analyze_bootstrap_convergence(
        self,
        method="md",
        lambda_ridge=0.1,
        use_score_weighting=True,
        component="attn",
        target_layers=None,
        n_bootstrap=20,
        sample_ratio=0.8,
        min_pool_size=50,
        pool_step=50,
        min_samples=10,
        target_sizes=(500, 1000),
        instability_ceiling=30.0,
        seed=42,
        output_dir=None,
    ):
        """
        Bootstrap convergence analysis: angular spread vs pool size.

        Fits k/sqrt(n) + floor to predict how many samples each category needs
        to reach global-level stability.

        Args:
            method: 'md', 'rmd', or 'wrmd'
            lambda_ridge: Ridge regularization parameter
            use_score_weighting: Whether to weight by judge scores (WRMD only)
            component: Which activations to use
            target_layers: Layer indices for analysis (default: all)
            n_bootstrap: Number of bootstrap iterations per pool size
            sample_ratio: Bootstrap resample fraction
            min_pool_size: Smallest pool size to test
            pool_step: Step between pool sizes
            min_samples: Minimum refusal samples per category
            target_sizes: Extrapolation target sample sizes
            instability_ceiling: Floor threshold above which category is unstable
            seed: Random seed
            output_dir: Directory to save results

        Returns:
            Results dict from compute_bootstrap_convergence()
        """
        from .analysis import compute_bootstrap_convergence, plot_bootstrap_convergence

        if not self.categories:
            raise ValueError("No category metadata available.")

        activations = self.get_activations(component)

        results = compute_bootstrap_convergence(
            activations=activations,
            labels=self.labels,
            categories=self.categories,
            judge_scores=self.judge_scores,
            target_layers=target_layers,
            method=method,
            lambda_ridge=lambda_ridge,
            use_score_weighting=use_score_weighting,
            n_bootstrap=n_bootstrap,
            sample_ratio=sample_ratio,
            min_pool_size=min_pool_size,
            pool_step=pool_step,
            min_samples=min_samples,
            target_sizes=target_sizes,
            instability_ceiling=instability_ceiling,
            seed=seed,
        )

        plot_bootstrap_convergence(results, output_dir=output_dir)

        if output_dir is not None:
            import json

            json_path = os.path.join(output_dir, "bootstrap_convergence.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[SAVE] Saved bootstrap convergence to {json_path}")

        return results

    def save_vectors(
        self,
        vectors,
        output_file,
        method="md",
        lambda_ridge=0.1,
        use_score_weighting=False,
        output_dir=None,
        rank=1,
        component="attn",
        categories=None,
    ):
        """
        Save steering vectors to file.

        Args:
            vectors: [num_layers, hidden_size] or [num_layers, rank, hidden_size],
                or dict mapping component -> tensor for multi-component saves.
            output_file: Filename or full path
            method: Method used ('md', 'rmd', 'wrmd')
            lambda_ridge: Ridge parameter used
            use_score_weighting: Whether score weighting was used
            output_dir: Optional output directory
            rank: Number of steering directions per layer
            component: Component used ("layer", "attn", "mlp", or "attn+mlp")
            categories: Optional list of category names used for filtering
        """
        from .utils import get_output_dir

        if output_dir is None:
            output_dir = get_output_dir(script_name="compute_wrmd")

        if not os.path.isabs(output_file):
            output_path = os.path.join(output_dir, output_file)
        else:
            output_path = output_file

        print(f"\n[SAVE] Saving to {output_path}...")

        # Compute accurate refusal count (respecting category filter)
        if categories is not None:
            if isinstance(categories, str):
                categories = [categories]
            try:
                cat_mask = self._build_category_mask(categories)
                num_refusal = ((self.labels == 1) & cat_mask).sum().item()
            except ValueError:
                num_refusal = (self.labels == 1).sum().item()
        else:
            num_refusal = (self.labels == 1).sum().item()

        save_data = {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "method": method,
            "lambda_ridge": lambda_ridge,
            "use_score_weighting": use_score_weighting,
            "num_refusal_samples": num_refusal,
            "num_compliant_samples": (self.labels == 0).sum().item(),
            "rank": rank,
            "component": component,
        }

        if categories is not None:
            save_data["categories"] = categories

        if isinstance(vectors, dict):
            # Multi-component: save each as steering_vectors_{comp}
            save_data["components"] = list(vectors.keys())
            for comp, vecs in vectors.items():
                save_data[f"steering_vectors_{comp}"] = vecs
            # Also save one as 'steering_vectors' for backward compat
            first_comp = list(vectors.keys())[0]
            save_data["steering_vectors"] = vectors[first_comp]
        else:
            save_data["steering_vectors"] = vectors

        torch.save(save_data, output_path)

        print(f"[OK] Saved steering vectors (rank={rank}, component={component})")


def analyze_rank_associations(
    activations,
    steering_vectors,
    labels,
    prompts,
    target_layers=None,
    output_dir=None,
):
    """
    Analyze which PCA rank each prompt is most associated with.

    For multi-rank steering vectors, projects each prompt's activation residual
    onto each rank direction and determines the dominant rank. This reveals what
    each rank captures semantically — e.g. v_1 might capture the refusal/compliance
    axis while v_2 captures a specific refusal style.

    Args:
        activations: [N, num_layers, hidden_size] — per-prompt activations
        steering_vectors: [num_layers, rank, hidden_size] — multi-rank vectors
        labels: [N] tensor — 0=compliant, 1=refusal
        prompts: list of N strings
        target_layers: list of layer indices to analyze (default: all)
        output_dir: optional directory to save results

    Returns:
        dict with keys:
            'per_prompt': list of dicts with prompt, label, projections, dominant_rank
            'per_layer': dict mapping layer_idx -> rank distribution stats
            'summary': overall rank distribution
    """
    if steering_vectors.ndim != 3:
        raise ValueError(
            f"analyze_rank_associations requires multi-rank vectors (3D), "
            f"got shape {steering_vectors.shape}"
        )

    num_layers, rank, hidden_size = steering_vectors.shape
    N = activations.shape[0]

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    if target_layers is None:
        target_layers = list(range(num_layers))

    print(f"\n[RANK ANALYSIS] Analyzing rank associations for {N} prompts")
    print(f"   Rank: {rank}, Layers: {target_layers}")

    # Aggregate projections across target layers (sum of absolute projections)
    # Shape: [N, rank]
    agg_projections = torch.zeros(N, rank)

    per_layer_results = {}

    for layer_idx in target_layers:
        layer_acts = activations[:, layer_idx, :].float()  # [N, hidden_size]
        layer_vecs = steering_vectors[layer_idx].float()  # [rank, hidden_size]

        # Normalize each rank direction to unit length for fair comparison
        layer_vecs_unit = layer_vecs / layer_vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Project all prompts onto all rank directions: [N, rank]
        projections = layer_acts @ layer_vecs_unit.T

        # Dominant rank per prompt at this layer
        dominant = projections.abs().argmax(dim=1)  # [N]

        # Accumulate across layers
        agg_projections += projections.abs()

        # Per-layer stats
        rank_counts = torch.zeros(rank, dtype=torch.long)
        for r in range(rank):
            rank_counts[r] = (dominant == r).sum().item()

        # Split by label
        refusal_mask = labels == 1
        compliant_mask = labels == 0

        layer_stats = {
            "rank_counts": rank_counts.tolist(),
            "rank_pcts": (rank_counts.float() / N * 100).tolist(),
        }
        if refusal_mask.any():
            ref_dominant = projections[refusal_mask].abs().argmax(dim=1)
            ref_counts = torch.zeros(rank, dtype=torch.long)
            for r in range(rank):
                ref_counts[r] = (ref_dominant == r).sum().item()
            layer_stats["refusal_rank_counts"] = ref_counts.tolist()
        if compliant_mask.any():
            comp_dominant = projections[compliant_mask].abs().argmax(dim=1)
            comp_counts = torch.zeros(rank, dtype=torch.long)
            for r in range(rank):
                comp_counts[r] = (comp_dominant == r).sum().item()
            layer_stats["compliant_rank_counts"] = comp_counts.tolist()

        per_layer_results[layer_idx] = layer_stats

    # Overall dominant rank (aggregated across target layers)
    overall_dominant = agg_projections.argmax(dim=1)  # [N]

    # Build per-prompt results
    per_prompt = []
    for i in range(N):
        per_prompt.append(
            {
                "prompt": prompts[i],
                "label": "refusal" if labels[i].item() == 1 else "compliant",
                "dominant_rank": overall_dominant[i].item() + 1,  # 1-indexed
                "projections": agg_projections[i].tolist(),
            }
        )

    # Summary stats
    summary_counts = torch.zeros(rank, dtype=torch.long)
    for r in range(rank):
        summary_counts[r] = (overall_dominant == r).sum().item()

    refusal_mask = labels == 1
    compliant_mask = labels == 0
    summary = {
        "total_prompts": N,
        "rank_counts": summary_counts.tolist(),
        "rank_pcts": (summary_counts.float() / N * 100).tolist(),
    }
    if refusal_mask.any():
        ref_dom = overall_dominant[refusal_mask]
        ref_counts = torch.zeros(rank, dtype=torch.long)
        for r in range(rank):
            ref_counts[r] = (ref_dom == r).sum().item()
        summary["refusal_rank_counts"] = ref_counts.tolist()
        summary["refusal_rank_pcts"] = (ref_counts.float() / refusal_mask.sum() * 100).tolist()
    if compliant_mask.any():
        comp_dom = overall_dominant[compliant_mask]
        comp_counts = torch.zeros(rank, dtype=torch.long)
        for r in range(rank):
            comp_counts[r] = (comp_dom == r).sum().item()
        summary["compliant_rank_counts"] = comp_counts.tolist()
        summary["compliant_rank_pcts"] = (comp_counts.float() / compliant_mask.sum() * 100).tolist()

    # Print summary
    print(
        f"\n[RANK ANALYSIS] Overall rank distribution (aggregated across layers {target_layers}):"
    )
    for r in range(rank):
        pct = summary_counts[r].item() / N * 100
        print(f"   v_{r+1}: {summary_counts[r].item()} prompts ({pct:.1f}%)")

    if refusal_mask.any():
        print(f"\n   Refusal prompts ({refusal_mask.sum().item()}):")
        for r in range(rank):
            print(
                f"      v_{r+1}: {summary['refusal_rank_counts'][r]} ({summary['refusal_rank_pcts'][r]:.1f}%)"
            )
    if compliant_mask.any():
        print(f"\n   Compliant prompts ({compliant_mask.sum().item()}):")
        for r in range(rank):
            print(
                f"      v_{r+1}: {summary['compliant_rank_counts'][r]} ({summary['compliant_rank_pcts'][r]:.1f}%)"
            )

    # Show example prompts for each rank
    for r in range(rank):
        rank_prompts = [p for p in per_prompt if p["dominant_rank"] == r + 1]
        if rank_prompts:
            print(f"\n   Example prompts dominated by v_{r+1}:")
            # Show top 3 by projection strength
            rank_prompts.sort(key=lambda x: x["projections"][r], reverse=True)
            for p in rank_prompts[:3]:
                truncated = p["prompt"][:80] + "..." if len(p["prompt"]) > 80 else p["prompt"]
                print(f"      [{p['label']}] {truncated}")

    result = {
        "per_prompt": per_prompt,
        "per_layer": per_layer_results,
        "summary": summary,
    }

    # Save results if output_dir provided
    if output_dir is not None:
        import json

        output_path = os.path.join(output_dir, "rank_associations.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[SAVE] Saved rank associations to {output_path}")

        # Plot rank distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart: overall distribution
        labels_list = [f"v_{r+1}" for r in range(rank)]
        axes[0].bar(
            labels_list, summary_counts.tolist(), color=["#2196F3", "#FF9800", "#4CAF50"][:rank]
        )
        axes[0].set_title("Dominant Rank Distribution (All Prompts)")
        axes[0].set_ylabel("Number of Prompts")

        # Grouped bar: refusal vs compliant
        if refusal_mask.any() and compliant_mask.any():
            x = np.arange(rank)
            width = 0.35
            axes[1].bar(
                x - width / 2,
                summary.get("refusal_rank_counts", [0] * rank),
                width,
                label="Refusal",
                color="#F44336",
            )
            axes[1].bar(
                x + width / 2,
                summary.get("compliant_rank_counts", [0] * rank),
                width,
                label="Compliant",
                color="#4CAF50",
            )
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels_list)
            axes[1].set_title("Rank Distribution by Label")
            axes[1].set_ylabel("Number of Prompts")
            axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "rank_associations.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved rank association plot: {plot_path}")

    return result


def compare_methods(calculator, lambda_ridge=0.1, output_dir=None, component="attn", rank=1):
    """
    Compare different steering vector computation methods.

    Args:
        calculator: WRMDCalculator instance
        lambda_ridge: Ridge parameter
        output_dir: Directory to save comparison plots
        component: Which activation component to use (default: attn)
        rank: Number of steering directions per layer (default: 1)

    Returns:
        Dictionary of steering vectors by method name
    """
    from .utils import get_output_path

    print("\n[COMPUTE] Comparing Methods...")

    comparisons = [
        ("MD", "md", False),
        ("RMD", "rmd", False),
        ("WRMD (uniform)", "wrmd", False),
        ("WRMD (weighted)", "wrmd", True),
    ]

    all_vectors = {}

    for name, method, use_weighting in comparisons:
        print(f"\n--- {name} ---")
        vectors = calculator.compute_steering_vectors(
            method,
            lambda_ridge,
            use_score_weighting=use_weighting,
            component=component,
            rank=rank,
        )
        all_vectors[name] = vectors

    # Compare vector norms
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (name, _, _) in enumerate(comparisons):
        v = all_vectors[name]
        # For multi-rank (3D), plot v_1 norms only
        if v.dim() == 3:
            v = v[:, 0, :]
        norms = v.norm(dim=1).float().numpy()
        axes[i].plot(norms, marker="o", linewidth=2, markersize=6)
        axes[i].set_xlabel("Layer Index")
        axes[i].set_ylabel("Vector Norm")
        axes[i].set_title(f"{name} Vector Norms")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=norms.mean(), color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save plot
    if output_dir is None:
        output_path = get_output_path(script_name="compute_wrmd", filename="method_comparison.png")
    else:
        output_path = os.path.join(output_dir, "method_comparison.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[PLOT] Saved comparison plot: {output_path}")

    return all_vectors
