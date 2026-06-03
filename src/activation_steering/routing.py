"""
Category-based prompt routing for activation steering.

Implements a two-stage classifier using the model's own activations:
  1. Refusal detection — project onto global steering vector, compare to threshold
  2. Category routing — project onto each category vector, select best match

This eliminates external classifiers by leveraging the geometric structure
of the steering vectors already computed by the pipeline.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch

from .extraction import _get_attn_submodule
from .steering import SteeringHook, SteeringHookGroup

STABILITY_LEVELS = {"stable": 0, "moderate": 1, "unreliable": 2, "unknown": 3}


@dataclass
class CategorySteeringConfig:
    """Per-category steering parameters with stability-gated fallback logic.

    For each category, this config determines:
    - Which steering vector to use (per-category or global)
    - Which layers to steer at (per-category best or global fallback)
    - Whether per-category layer selection is reliable enough to use

    The ``using_per_category_layers`` flag is the key decision: it is True only
    when the category has enough samples (n >= min_samples), its bootstrap
    stability meets the threshold, and the best-layer correlation is above
    min_correlation. Otherwise, the global best layers are used.
    """

    category: str
    vector_path: str  # path to per-category steering vector .pt file
    target_layers: list[int]  # layers to apply steering at
    component: str  # "attn", "mlp", or "attn+mlp"
    # Metadata for the decision
    n_samples: int
    bootstrap_stability: str  # "stable", "moderate", "unreliable"
    best_correlation: float  # correlation at the best layer
    using_per_category_layers: bool  # False if fell back to global
    fallback_reason: Optional[str] = None  # why global was used

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def compute_residual_vectors(
    category_vectors: dict[str, torch.Tensor],
    global_vectors: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute residual category vectors by projecting out the global refusal component.

    For each layer: v_residual = v_cat - (v_cat · v_global / v_global · v_global) * v_global

    The residual captures what makes each category *different* from the global refusal
    direction, which is what we need for classification (not steering).

    Args:
        category_vectors: Dict mapping category name -> [num_layers, hidden_size]
        global_vectors: Global steering vectors [num_layers, hidden_size]

    Returns:
        Dict mapping category name -> [num_layers, hidden_size] residual vectors
    """
    residuals = {}
    for cat_name, cat_vecs in category_vectors.items():
        cat_f = cat_vecs.float()
        gv_f = global_vectors.float().to(cat_f.device)
        dot_cg = (cat_f * gv_f).sum(dim=-1, keepdim=True)
        dot_gg = (gv_f * gv_f).sum(dim=-1, keepdim=True).clamp(min=1e-12)
        residual = cat_f - (dot_cg / dot_gg) * gv_f
        zero_layers = dot_gg.squeeze(-1) < 1e-12
        if zero_layers.any():
            residual[zero_layers] = cat_f[zero_layers]

        cat_norm = cat_f.norm(dim=1).mean().item()
        res_norm = residual.norm(dim=1).mean().item()
        ratio = res_norm / cat_norm if cat_norm > 1e-8 else 0.0
        print(
            f"   {cat_name}: residual norm ratio = {ratio:.3f} "
            f"(cat={cat_norm:.4f}, residual={res_norm:.4f})"
        )
        residuals[cat_name] = residual.to(cat_vecs.dtype)
    return residuals


def build_steering_configs(
    per_category_layers: dict[str, dict],
    bootstrap_stability: dict[str, str],
    category_sample_counts: dict[str, int],
    category_vector_paths: dict[str, str],
    global_best_layers: list[int],
    component: str = "attn",
    min_samples: int = 50,
    min_stability: Optional[set] = None,
    min_correlation: float = 0.3,
    num_layers: int = 1,
) -> dict[str, "CategorySteeringConfig"]:
    """Build per-category steering configs with stability-gated fallback logic.

    For each category, decides whether to use per-category best layers or fall
    back to the global best layers. Per-category layers are used only when ALL
    of these conditions are met:
    1. Sample count >= min_samples
    2. Bootstrap stability is in min_stability set (default: {"stable", "moderate"})
    3. Best-layer correlation >= min_correlation

    If any condition fails, the category falls back to global best layers. The
    routing still helps (correct per-category vector), just with global layers.

    Args:
        per_category_layers: Dict from per_category_layer_correlations.json.
            Each value has "best_layers" (list[int]), "correlations" (list[dict]),
            "num_refusal" (int).
        bootstrap_stability: Dict mapping category name -> stability label
            ("stable", "moderate", "unreliable").
        category_sample_counts: Dict mapping category name -> n_refusal samples.
        category_vector_paths: Dict mapping category name -> path to .pt file.
        global_best_layers: List of global best layer indices (fallback).
        component: Activation component ("attn", "mlp", "attn+mlp").
        min_samples: Minimum refusal samples to trust per-category layers.
        min_stability: Set of acceptable stability labels. Default: {"stable", "moderate"}.
        min_correlation: Minimum correlation at best layer to trust it.
        num_layers: Number of top layers to use per category (1 = single best).

    Returns:
        Dict mapping category name -> CategorySteeringConfig
    """
    if min_stability is None:
        min_stability = {"stable", "moderate"}

    configs = {}
    fallback_count = 0

    for cat_name in category_sample_counts:
        n = category_sample_counts[cat_name]
        stability = bootstrap_stability.get(cat_name, "unknown")

        # Get per-category best layers and correlation
        cat_layer_data = per_category_layers.get(cat_name)
        if cat_layer_data is not None:
            cat_best_layers = cat_layer_data.get("best_layers", [])
            cat_correlations = cat_layer_data.get("correlations", [])
            # Get correlation of the best layer
            best_corr = 0.0
            if cat_best_layers and cat_correlations:
                for entry in cat_correlations:
                    if entry["layer"] == cat_best_layers[0]:
                        best_corr = abs(entry.get("correlation", 0.0))
                        break
        else:
            cat_best_layers = []
            best_corr = 0.0

        # Decide: per-category layers or global fallback
        use_per_cat = True
        reasons = []

        if n < min_samples:
            use_per_cat = False
            reasons.append(f"n={n} < min_samples={min_samples}")

        if stability not in min_stability:
            use_per_cat = False
            reasons.append(f"stability='{stability}' not in {min_stability}")

        if best_corr < min_correlation:
            use_per_cat = False
            reasons.append(f"correlation={best_corr:.3f} < min_correlation={min_correlation}")

        # Select layers
        if use_per_cat and cat_best_layers:
            target_layers = cat_best_layers[:num_layers]
            fallback_reason = None
        else:
            target_layers = global_best_layers[:num_layers]
            fallback_reason = "; ".join(reasons) if reasons else "no per-category data"
            fallback_count += 1

        # Get vector path
        vec_path = category_vector_paths.get(cat_name, "")

        configs[cat_name] = CategorySteeringConfig(
            category=cat_name,
            vector_path=vec_path,
            target_layers=target_layers,
            component=component,
            n_samples=n,
            bootstrap_stability=stability,
            best_correlation=best_corr,
            using_per_category_layers=use_per_cat and bool(cat_best_layers),
            fallback_reason=fallback_reason,
        )

    # Print summary
    per_cat_count = sum(1 for c in configs.values() if c.using_per_category_layers)
    print(f"\n[CONFIG] Built {len(configs)} category steering configs:")
    print(f"  Per-category layers: {per_cat_count}")
    print(f"  Global fallback:     {fallback_count}")
    for cat_name, cfg in sorted(configs.items()):
        layer_src = "PER-CAT" if cfg.using_per_category_layers else "GLOBAL"
        print(
            f"  {cat_name}: layers={cfg.target_layers} ({layer_src}), "
            f"n={cfg.n_samples}, stability={cfg.bootstrap_stability}, "
            f"corr={cfg.best_correlation:.3f}"
        )
        if cfg.fallback_reason:
            print(f"    ↳ Fallback reason: {cfg.fallback_reason}")

    return configs


@dataclass
class CalibrationData:
    """Calibration parameters for the category router."""

    threshold: float
    auc: float
    target_layers: list[int]
    component: str
    global_vectors_path: str
    category_vectors_paths: dict[str, str]  # category_name -> file path
    category_projection_stats: dict[str, dict]  # category -> {mean, std}
    bootstrap_stability: dict[str, str] = field(default_factory=dict)  # category -> stability label
    sensitivity: float = 1.0
    min_category_threshold: float = 0.0  # minimum projection to accept a category match
    excluded_categories: list[str] = field(default_factory=list)  # excluded from routing (unstable)
    residual_projection_stats: dict[str, dict] = field(
        default_factory=dict
    )  # per-category own-sample residual stats for z-score normalization
    per_category_steering_configs: dict[str, dict] = field(
        default_factory=dict
    )  # category -> CategorySteeringConfig.asdict()

    def save(self, path):
        """Save calibration data to JSON."""
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load calibration data from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class RoutingDecision:
    """Result of the two-stage routing classifier."""

    should_steer: bool
    selected_category: Optional[str]  # None if should_steer=False or global fallback
    category_projections: dict[str, float]
    global_projection: float
    confidence: float
    reason: str
    use_global_fallback: bool = False


class CategoryRouter:
    """Two-stage router: refusal detection + category selection.

    Stage 1: Project activations onto global steering vector. If the projection
    exceeds the calibrated threshold, the prompt is classified as a refusal
    candidate and steering is applied.

    Stage 2: Project activations onto each category vector. Select the category
    with the highest projection, with tiebreaking by bootstrap stability.
    Fall back to global vector if no category exceeds the minimum threshold.
    """

    def __init__(
        self,
        global_vectors: torch.Tensor,
        category_vectors: dict[str, torch.Tensor],
        calibration: CalibrationData,
    ):
        """
        Args:
            global_vectors: Steering vectors [num_layers, hidden_size]
            category_vectors: Dict mapping category name -> [num_layers, hidden_size]
            calibration: Calibration parameters from calibrate_router()
        """
        self.global_vectors = global_vectors
        self.category_vectors = category_vectors
        self.calibration = calibration
        self.target_layers = calibration.target_layers
        self.component = calibration.component

        # Compute residual vectors for Stage 2 classification (not steering).
        # All categories participate in classification (even excluded ones) so that
        # Stage 2 can discriminate properly. Excluded categories trigger global fallback
        # after classification, not before.
        self.routing_vectors = compute_residual_vectors(category_vectors, global_vectors)

    def classify(self, activations: dict[int, torch.Tensor]) -> RoutingDecision:
        """Classify a prompt using pre-extracted activations.

        Args:
            activations: Dict mapping layer_idx -> tensor of shape [hidden_size]

        Returns:
            RoutingDecision with routing result
        """
        # Stage 1: Refusal detection via global vector projection
        global_projections = []
        for layer_idx in self.target_layers:
            if layer_idx not in activations:
                continue
            act = activations[layer_idx].float()
            gv = self.global_vectors[layer_idx].float()
            if gv.device != act.device:
                gv = gv.to(act.device)
            norm = gv.norm()
            if norm < 1e-8:
                continue
            gv_unit = gv / norm
            proj = (act @ gv_unit).item()
            global_projections.append(proj)

        if not global_projections:
            return RoutingDecision(
                should_steer=False,
                selected_category=None,
                category_projections={},
                global_projection=0.0,
                confidence=0.0,
                reason="No activations available for target layers",
            )

        avg_global_proj = sum(global_projections) / len(global_projections)
        effective_threshold = self.calibration.threshold * self.calibration.sensitivity

        if avg_global_proj < effective_threshold:
            return RoutingDecision(
                should_steer=False,
                selected_category=None,
                category_projections={},
                global_projection=avg_global_proj,
                confidence=abs(avg_global_proj - effective_threshold),
                reason=f"Global projection {avg_global_proj:.4f} below threshold {effective_threshold:.4f}",
            )

        # Stage 2: Category routing (using residual vectors + z-score normalization)
        # Raw residual projections are on different scales per category. Z-score
        # normalization asks "how unusually aligned is this prompt with each category?"
        # rather than "which category has the largest absolute projection?"
        raw_projections = {}
        for cat_name, cat_vecs in self.routing_vectors.items():
            cat_projs = []
            for layer_idx in self.target_layers:
                if layer_idx not in activations:
                    continue
                act = activations[layer_idx].float()
                cv = cat_vecs[layer_idx].float()
                if cv.device != act.device:
                    cv = cv.to(act.device)
                norm = cv.norm()
                if norm < 1e-8:
                    continue
                cv_unit = cv / norm
                proj = (act @ cv_unit).item()
                cat_projs.append(proj)
            if cat_projs:
                raw_projections[cat_name] = sum(cat_projs) / len(cat_projs)

        if not raw_projections:
            return RoutingDecision(
                should_steer=True,
                selected_category=None,
                category_projections={},
                global_projection=avg_global_proj,
                confidence=abs(avg_global_proj - effective_threshold),
                reason="No category vectors available, using global",
                use_global_fallback=True,
            )

        # Z-score normalize using per-category own-sample calibration stats
        category_projections = {}
        for cat_name, raw_proj in raw_projections.items():
            stats = self.calibration.residual_projection_stats.get(cat_name)
            if stats and stats.get("std", 0) > 1e-8:
                category_projections[cat_name] = (raw_proj - stats["mean"]) / stats["std"]
            else:
                # No calibration stats — use raw projection (can't normalize)
                category_projections[cat_name] = raw_proj

        # Sort categories by z-score (descending)
        sorted_cats = sorted(category_projections.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_score = sorted_cats[0]

        # Check minimum threshold (on z-score: 0 means "at the mean")
        if best_score < self.calibration.min_category_threshold:
            return RoutingDecision(
                should_steer=True,
                selected_category=None,
                category_projections=category_projections,
                global_projection=avg_global_proj,
                confidence=abs(avg_global_proj - effective_threshold),
                reason=f"Best category z-score {best_score:.4f} below minimum "
                f"{self.calibration.min_category_threshold:.4f}, using global",
                use_global_fallback=True,
            )

        # Tiebreak: if top-2 z-scores are close, prefer better bootstrap stability
        if len(sorted_cats) >= 2:
            second_cat, second_score = sorted_cats[1]
            if best_score - second_score < 0.5:  # within 0.5 std devs
                best_stability = self._stability_rank(best_cat)
                second_stability = self._stability_rank(second_cat)
                if second_stability < best_stability:
                    best_cat = second_cat
                    best_score = second_score

        # If the selected category is excluded (unstable), fall back to global
        if best_cat in self.calibration.excluded_categories:
            return RoutingDecision(
                should_steer=True,
                selected_category=None,
                category_projections=category_projections,
                global_projection=avg_global_proj,
                confidence=best_score,
                reason=f"Best match '{best_cat}' is excluded (unstable), using global",
                use_global_fallback=True,
            )

        return RoutingDecision(
            should_steer=True,
            selected_category=best_cat,
            category_projections=category_projections,
            global_projection=avg_global_proj,
            confidence=best_score,
            reason=f"Routed to '{best_cat}' (z-score={best_score:.4f})",
        )

    def _stability_rank(self, category: str) -> int:
        """Return numeric rank for bootstrap stability (lower = better)."""
        label = self.calibration.bootstrap_stability.get(category, "unknown")
        return STABILITY_LEVELS.get(label, 3)

    def classify_prompt(
        self,
        model,
        tokenizer,
        prompt: str,
    ) -> RoutingDecision:
        """Classify a prompt by extracting activations from the model.

        Uses lightweight hooks on only the target layers (not all layers).

        Args:
            model: HuggingFace model instance
            tokenizer: Tokenizer instance
            prompt: Input text prompt

        Returns:
            RoutingDecision with routing result
        """
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt").to(next(model.parameters()).device)

        # Extract activations only at target layers
        layer_activations = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                layer_activations[layer_idx] = hidden_states[:, -1, :].detach().cpu().squeeze(0)

            return hook

        hooks = []
        from .utils import get_model_layers

        layers = get_model_layers(model)
        for layer_idx in self.target_layers:
            layer = layers[layer_idx]
            if self.component == "attn":
                target = _get_attn_submodule(layer)
            elif self.component == "mlp":
                target = layer.mlp
            else:
                target = layer
            hooks.append(target.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            for hook in hooks:
                hook.remove()

        return self.classify(layer_activations)

    def create_steering_hooks(
        self,
        model,
        decision: RoutingDecision,
        alpha: float,
        **hook_kwargs,
    ):
        """Build steering hooks based on a routing decision.

        Uses per-category target layers when available (from
        CalibrationData.per_category_steering_configs). Falls back to the
        global target_layers when the category is not configured or when the
        routing decision triggers global fallback.

        Args:
            model: HuggingFace model instance
            decision: RoutingDecision from classify() or classify_prompt()
            alpha: Steering coefficient
            **hook_kwargs: Additional kwargs passed to SteeringHook

        Returns:
            SteeringHook, SteeringHookGroup, or None (if should_steer=False)
        """
        if not decision.should_steer:
            return None

        # Default to the router's global component
        component = self.component

        if decision.use_global_fallback or decision.selected_category is None:
            vectors = self.global_vectors
            target_layers = self.target_layers  # global layers for global fallback
        else:
            vectors = self.category_vectors[decision.selected_category]
            # Check for per-category target layers AND component
            cat_config = self.calibration.per_category_steering_configs.get(
                decision.selected_category
            )
            if cat_config and cat_config.get("using_per_category_layers"):
                target_layers = cat_config["target_layers"]
                # Use the per-category component if specified (e.g. "mlp" for categories
                # where MLP has higher refusal correlation than attention)
                if cat_config.get("component") and cat_config["component"] in (
                    "layer",
                    "attn",
                    "mlp",
                ):
                    component = cat_config["component"]
            else:
                target_layers = self.target_layers  # global fallback

        hook = SteeringHook(
            model=model,
            steering_vectors=vectors,
            target_layers=target_layers,
            alpha=alpha,
            component=component,
            **hook_kwargs,
        )
        return hook

    @classmethod
    def from_calibration_file(cls, path: str, device: str = "cpu"):
        """Load a CategoryRouter from a calibration JSON file.

        Args:
            path: Path to calibration.json
            device: Device to load tensors onto

        Returns:
            CategoryRouter instance
        """
        calibration = CalibrationData.load(path)
        base_dir = os.path.dirname(os.path.abspath(path))

        # Load global vectors
        global_path = calibration.global_vectors_path
        if not os.path.isabs(global_path):
            global_path = os.path.join(base_dir, global_path)
        global_data = torch.load(global_path, map_location=device, weights_only=True)
        sv_key = f"steering_vectors_{calibration.component}"
        if sv_key in global_data:
            global_vectors = global_data[sv_key]
        elif "steering_vectors" in global_data:
            global_vectors = global_data["steering_vectors"]
        else:
            raise ValueError(f"No steering vectors found in {global_path}")

        # Load category vectors
        category_vectors = {}
        for cat_name, cat_path in calibration.category_vectors_paths.items():
            if not os.path.isabs(cat_path):
                cat_path = os.path.join(base_dir, cat_path)
            # Try the recorded path; if missing, attempt fallback component
            # suffixes (the file may have been renamed with _attn / _mlp /
            # _layer suffix after calibration was saved).
            resolved = cat_path
            if not os.path.exists(resolved):
                for suffix in ("_attn", "_mlp", "_layer"):
                    candidate = resolved.replace(".pt", f"{suffix}.pt")
                    if os.path.exists(candidate):
                        resolved = candidate
                        break
            if not os.path.exists(resolved):
                print(f"[WARN] Missing category file for {cat_name}: {cat_path}, skipping")
                continue
            cat_data = torch.load(resolved, map_location=device, weights_only=True)
            if sv_key in cat_data:
                vecs = cat_data[sv_key]
            elif "steering_vectors" in cat_data:
                vecs = cat_data["steering_vectors"]
            else:
                print(f"[WARN] No steering vectors in {resolved}, skipping {cat_name}")
                continue
            # Handle rank-2+ vectors: extract first rank for routing
            if vecs.ndim == 3:
                vecs = vecs[:, 0, :]
            category_vectors[cat_name] = vecs

        print(f"\n[RESIDUAL] Computing residual routing vectors:")
        return cls(global_vectors, category_vectors, calibration)


def calibrate_router(
    activations_path: str,
    global_vectors_path: str,
    category_vector_paths: list[str],
    target_layers: list[int],
    component: str = "attn",
    bootstrap_stability_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    stability_filter: str = "unreliable",
) -> CalibrationData:
    """Calibrate the two-stage router from labeled activations.

    Stage 1 calibration: Projects all samples onto the global vector,
    computes ROC curve, finds optimal threshold via Youden's J statistic.

    Stage 2 calibration: Projects refusal samples onto each category vector,
    collects mean/std per category for the minimum threshold.

    Args:
        activations_path: Path to activations .pt file with labels
        global_vectors_path: Path to global steering vectors .pt file
        category_vector_paths: List of paths to category steering vector .pt files
        target_layers: Layer indices to use for projection
        component: Activation component ("attn", "mlp", "layer")
        bootstrap_stability_path: Optional path to bootstrap stability JSON
        output_dir: Optional directory for output files
        stability_filter: Exclude categories at or above this instability level.
            Levels: "stable" < "moderate" < "unreliable". Default "unreliable".

    Returns:
        CalibrationData with calibrated parameters
    """
    # Load activations
    print(f"[LOAD] Loading activations from {activations_path}")
    act_data = torch.load(activations_path, weights_only=True)
    labels = act_data["labels"]
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    act_key = f"activations_{component}" if component != "layer" else "activations"
    if act_key in act_data:
        activations = act_data[act_key]
    elif "activations" in act_data:
        activations = act_data["activations"]
    else:
        raise ValueError(f"No activations found for component '{component}'")

    print(f"   Shape: {activations.shape}")
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    print(f"   Refusal samples: {n_pos}")
    print(f"   Compliant samples: {n_neg}")

    target_layer_norms = activations[:, target_layers, :].float().norm(dim=2)
    nonfinite_mask = ~torch.isfinite(target_layer_norms).all(dim=1)
    if nonfinite_mask.any():
        n_skip = nonfinite_mask.sum().item()
        skip_indices = torch.where(nonfinite_mask)[0].tolist()
        print(
            f"   [SANITIZE] Skipping {n_skip} samples with non-finite activations "
            f"at target layers: indices {skip_indices[:5]}"
            + (f"... (+{n_skip - 5} more)" if n_skip > 5 else "")
        )
        keep_mask = ~nonfinite_mask
        activations = activations[keep_mask]
        labels = labels[keep_mask]
        # Filter metadata (sample_categories, etc.) if present in the activations file
        metadata_fields = [k for k in act_data.keys() if k.startswith("sample_") or k == "metadata"]
        for field in metadata_fields:
            val = act_data[field]
            if isinstance(val, list):
                act_data[field] = [v for v, k in zip(val, keep_mask.tolist()) if k]
        n_pos = (labels == 1).sum().item()
        n_neg = (labels == 0).sum().item()
        print(f"   Remaining: {activations.shape[0]} samples (refusal={n_pos}, compliant={n_neg})")

    if n_pos == 0:
        raise ValueError("No refusal samples (label=1) found — cannot calibrate threshold")
    if n_neg == 0:
        raise ValueError("No compliant samples (label=0) found — cannot calibrate threshold")

    # Validate target layers against activation dimensions
    num_layers_in_file = activations.shape[1]
    invalid_layers = [l for l in target_layers if l >= num_layers_in_file]
    if invalid_layers:
        raise ValueError(
            f"target_layers {invalid_layers} out of bounds for activations "
            f"with {num_layers_in_file} layers"
        )

    # Load global vectors
    print(f"\n[LOAD] Loading global vectors from {global_vectors_path}")
    global_data = torch.load(global_vectors_path, weights_only=True)
    sv_key = f"steering_vectors_{component}"
    if sv_key in global_data:
        global_vectors = global_data[sv_key]
    elif "steering_vectors" in global_data:
        global_vectors = global_data["steering_vectors"]
    else:
        raise ValueError(f"No steering vectors found in {global_vectors_path}")
    print(f"   Shape: {global_vectors.shape}")

    # Load category vectors and build path mapping in one pass
    print(f"\n[LOAD] Loading category vectors from {len(category_vector_paths)} files")
    category_vectors = {}
    category_vectors_paths_map = {}
    for fpath in category_vector_paths:
        data = torch.load(fpath, weights_only=True)
        if "categories" not in data:
            print(f"[SKIP] {os.path.basename(fpath)}: no category metadata (global file)")
            continue
        cat_name = "+".join(data["categories"])
        sv_key_cat = f"steering_vectors_{component}"
        if sv_key_cat in data:
            category_vectors[cat_name] = data[sv_key_cat]
        elif "steering_vectors" in data:
            category_vectors[cat_name] = data["steering_vectors"]
        else:
            print(f"[WARN] {os.path.basename(fpath)}: no steering vectors found, skipping")
            continue
        category_vectors_paths_map[cat_name] = fpath
        print(
            f"[LOAD] {cat_name}: {category_vectors[cat_name].shape} from {os.path.basename(fpath)}"
        )
    print(f"   Categories: {list(category_vectors.keys())}")

    # === Stage 1: ROC analysis on global vector ===
    print(f"\n{'='*60}")
    print("Stage 1: Refusal Detection Calibration")
    print(f"{'='*60}")

    projections = _batch_project_onto_vectors(activations, global_vectors, target_layers).numpy()
    labels_np = labels.numpy().astype(int)

    # Compute ROC curve (no sklearn)
    thresholds = np.sort(np.unique(projections))
    tpr_list = []
    fpr_list = []
    n_pos_np = (labels_np == 1).sum()
    n_neg_np = (labels_np == 0).sum()

    for thresh in thresholds:
        predicted_pos = projections >= thresh
        tp = (predicted_pos & (labels_np == 1)).sum()
        fp = (predicted_pos & (labels_np == 0)).sum()
        tpr_list.append(tp / n_pos_np if n_pos_np > 0 else 0)
        fpr_list.append(fp / n_neg_np if n_neg_np > 0 else 0)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    # AUC via trapezoid rule (fpr is decreasing as threshold increases, so sort)
    sorted_idx = np.argsort(fpr_arr)
    auc = float(np.trapz(tpr_arr[sorted_idx], fpr_arr[sorted_idx]))
    if auc < 0.5:
        print(
            f"   [WARN] AUC={auc:.4f} < 0.5 — global vector polarity may be inverted. "
            f"Check that label=1 means refusal and projection is positive for refusal samples."
        )

    # Youden's J statistic: maximize TPR - FPR
    j_scores = tpr_arr - fpr_arr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])

    print(f"   AUC: {auc:.4f}")
    print(f"   Optimal threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"   At threshold: TPR={tpr_arr[best_idx]:.3f}, FPR={fpr_arr[best_idx]:.3f}")

    # === Stage 2: Category projection stats ===
    print(f"\n{'='*60}")
    print("Stage 2: Category Routing Calibration")
    print(f"{'='*60}")

    refused_mask = labels_np == 1

    # Get metadata categories if available
    metadata = act_data.get("metadata", None)
    sample_categories = None
    if metadata is not None and isinstance(metadata, list):
        sample_categories = [m.get("category") for m in metadata if isinstance(m, dict)]
        if len(sample_categories) != activations.shape[0]:
            sample_categories = None

    # Load bootstrap stability if available (needed before filtering)
    bootstrap_stability = {}
    if bootstrap_stability_path and os.path.exists(bootstrap_stability_path):
        print(f"\n[LOAD] Loading bootstrap stability from {bootstrap_stability_path}")
        with open(bootstrap_stability_path) as f:
            stab_data = json.load(f)
        if isinstance(stab_data, dict):
            # Handle nested format: {per_category: {cat: {stability_label: ...}}}
            categories_dict = stab_data.get("per_category", stab_data)
            for cat_name, cat_info in categories_dict.items():
                if isinstance(cat_info, dict):
                    # Try stability_label first (compute_wrmd output), then stability
                    label = cat_info.get("stability_label") or cat_info.get("stability")
                    if label:
                        bootstrap_stability[cat_name] = label
                elif isinstance(cat_info, str):
                    bootstrap_stability[cat_name] = cat_info
        print(f"   Loaded stability for {len(bootstrap_stability)} categories")

    # Filter categories by bootstrap stability (only if stability data is available)
    filter_threshold = STABILITY_LEVELS.get(stability_filter, 2)
    excluded_categories = []
    if bootstrap_stability:
        for cat_name in category_vectors:
            label = bootstrap_stability.get(cat_name)
            if label is None:
                # No stability data — treat as unverified, exclude
                excluded_categories.append(cat_name)
                continue
            if STABILITY_LEVELS.get(label, 3) >= filter_threshold:
                excluded_categories.append(cat_name)

    if excluded_categories:
        print(
            f"\n[FILTER] Excluding {len(excluded_categories)} categories "
            f"(stability >= '{stability_filter}' or no stability data):"
        )
        for cat in excluded_categories:
            label = bootstrap_stability.get(cat)
            reason = f"{label}" if label else "no stability data"
            print(f"   {cat}: {reason}")
    else:
        print(f"\n[FILTER] No categories excluded (all below '{stability_filter}' threshold)")

    # --- Raw projection stats (for comparison) ---
    print(f"\n   Raw category projection stats (all refusal samples):")
    category_projection_stats = {}
    for cat_name, cat_vecs in category_vectors.items():
        cat_projs = _project_samples_onto_vectors(
            activations, refused_mask, cat_vecs, target_layers
        )
        if cat_projs:
            cat_projs_arr = np.array(cat_projs)
            category_projection_stats[cat_name] = {
                "mean": float(cat_projs_arr.mean()),
                "std": float(cat_projs_arr.std()),
                "min": float(cat_projs_arr.min()),
                "max": float(cat_projs_arr.max()),
            }
            print(
                f"   {cat_name}: mean={cat_projs_arr.mean():.4f}, "
                f"std={cat_projs_arr.std():.4f}, "
                f"range=[{cat_projs_arr.min():.4f}, {cat_projs_arr.max():.4f}]"
            )

    # --- Compute residual vectors ---
    print(f"\n[RESIDUAL] Computing residual routing vectors:")
    residual_vectors = compute_residual_vectors(category_vectors, global_vectors)

    # --- Per-category own-sample residual stats (for z-score normalization) ---
    # For each category, project only that category's own refusal samples onto
    # its residual vector. This gives the baseline mean/std for z-scoring.
    print(f"\n   Per-category residual projection stats (own samples only):")
    residual_projection_stats = {}

    # Build per-sample category mapping
    sample_cat_map = {}  # vector_key -> list of sample indices
    if sample_categories is not None:
        subcat_to_key = {}
        for cat_key in residual_vectors:
            for sub in cat_key.split("+"):
                subcat_to_key[sub] = cat_key
        for i in range(activations.shape[0]):
            if not refused_mask[i]:
                continue
            true_cats = sample_categories[i]
            if true_cats is None:
                continue
            if not isinstance(true_cats, list):
                true_cats = [true_cats]
            for cat_label in true_cats:
                vec_key = subcat_to_key.get(cat_label)
                if vec_key is not None:
                    sample_cat_map.setdefault(vec_key, []).append(i)
                    break

    for cat_name, res_vecs in residual_vectors.items():
        own_indices = sample_cat_map.get(cat_name, [])
        if not own_indices:
            print(f"   {cat_name}: no own-category samples, skipping")
            continue

        # Build mask for only this category's samples
        own_mask = np.zeros(activations.shape[0], dtype=bool)
        for idx in own_indices:
            own_mask[idx] = True

        own_projs = _project_samples_onto_vectors(activations, own_mask, res_vecs, target_layers)
        if own_projs:
            own_arr = np.array(own_projs)
            residual_projection_stats[cat_name] = {
                "mean": float(own_arr.mean()),
                "std": float(own_arr.std()),
                "min": float(own_arr.min()),
                "max": float(own_arr.max()),
                "support": len(own_projs),
            }
            print(
                f"   {cat_name}: mean={own_arr.mean():.4f}, "
                f"std={own_arr.std():.4f}, "
                f"range=[{own_arr.min():.4f}, {own_arr.max():.4f}], "
                f"n={len(own_projs)}"
            )

    # Min category threshold: z-score of 0 means "at the category mean", so threshold at 0
    min_cat_threshold = 0.0

    # --- Category routing metrics comparison (raw vs residual) ---
    # All categories compete in argmax, but excluded predictions → global fallback.
    # Metrics reported only for included categories.
    excluded_set = set(excluded_categories)
    if sample_categories is not None:
        raw_metrics = _compute_routing_metrics(
            activations,
            refused_mask,
            sample_categories,
            category_vectors,
            target_layers,
            excluded_categories=excluded_set,
        )
        res_metrics = _compute_routing_metrics(
            activations,
            refused_mask,
            sample_categories,
            residual_vectors,
            target_layers,
            excluded_categories=excluded_set,
        )
        zscore_metrics = _compute_routing_metrics(
            activations,
            refused_mask,
            sample_categories,
            residual_vectors,
            target_layers,
            excluded_categories=excluded_set,
            zscore_stats=residual_projection_stats,
        )

        for label, metrics in [
            ("Raw vectors", raw_metrics),
            ("Residual vectors", res_metrics),
            ("Residual + z-score", zscore_metrics),
        ]:
            print(f"\n   {label} (included categories only):")
            print(f"   {'Category':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
            print(f"   {'-'*71}")
            for cat_key in sorted(metrics["per_category"]):
                m = metrics["per_category"][cat_key]
                print(
                    f"   {cat_key:<45} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                    f"{m['f1']:>6.3f} {m['support']:>8d}"
                )
            print(f"   {'-'*71}")
            print(
                f"   {'Macro avg':<45} {metrics['macro_precision']:>6.3f} "
                f"{metrics['macro_recall']:>6.3f} {metrics['macro_f1']:>6.3f} "
                f"{metrics['total_evaluated']:>8d}"
            )
            print(f"   Global fallback: {metrics['global_fallback_count']} predictions")

    # Build calibration result
    calibration = CalibrationData(
        threshold=optimal_threshold,
        auc=auc,
        target_layers=target_layers,
        component=component,
        global_vectors_path=global_vectors_path,
        category_vectors_paths=category_vectors_paths_map,
        category_projection_stats=category_projection_stats,
        bootstrap_stability=bootstrap_stability,
        sensitivity=1.0,
        min_category_threshold=min_cat_threshold,
        excluded_categories=excluded_categories,
        residual_projection_stats=residual_projection_stats,
    )

    # Save outputs
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cal_path = os.path.join(output_dir, "calibration.json")
        calibration.save(cal_path)
        print(f"\n[SAVE] Calibration saved to {cal_path}")

        # Save ROC plot
        try:
            _save_roc_plot(fpr_arr, tpr_arr, auc, optimal_threshold, output_dir)
        except Exception as e:
            print(f"[WARN] Could not save ROC plot: {e}")

    return calibration


def _batch_project_onto_vectors(
    activations: torch.Tensor,
    vectors: torch.Tensor,
    target_layers: list[int],
) -> torch.Tensor:
    """Vectorized projection of all samples onto unit-normalized vectors.

    Args:
        activations: [n_samples, n_layers, hidden_size]
        vectors: [n_layers, hidden_size]
        target_layers: Layer indices to use

    Returns:
        [n_samples] tensor of mean projections across target layers
    """
    selected_acts = activations[:, target_layers, :].float()
    selected_vecs = vectors[target_layers].float()
    vec_norms = selected_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit_vecs = selected_vecs / vec_norms
    projs = (selected_acts * unit_vecs.unsqueeze(0)).sum(dim=-1)
    return projs.mean(dim=-1)


def _project_samples_onto_vectors(
    activations: torch.Tensor,
    mask: np.ndarray,
    vectors: torch.Tensor,
    target_layers: list[int],
) -> list[float]:
    """Project masked samples onto vectors, returning average projection per sample."""
    if mask.sum() == 0:
        return []
    indices = np.where(mask)[0]
    projs = _batch_project_onto_vectors(activations[indices], vectors, target_layers)
    return projs.tolist()


def _compute_routing_metrics(
    activations: torch.Tensor,
    refused_mask: np.ndarray,
    sample_categories: list,
    routing_vectors: dict[str, torch.Tensor],
    target_layers: list[int],
    excluded_categories: Optional[set[str]] = None,
    zscore_stats: Optional[dict[str, dict]] = None,
) -> dict:
    """Compute per-category precision, recall, F1 for multi-class routing.

    Mirrors the real router behavior: all categories compete in projection
    argmax (with optional z-score normalization), but predictions landing on
    excluded categories are mapped to "global_fallback".

    Args:
        activations: [n_samples, n_layers, hidden_size]
        refused_mask: Boolean mask for refusal samples
        sample_categories: Per-sample true category label
        routing_vectors: Dict of category -> [n_layers, hidden_size]
        target_layers: Layer indices for projection
        excluded_categories: Categories excluded from routing. Predictions landing
            on these are treated as global fallback. If None, all categories are
            included.
        zscore_stats: Optional per-category {mean, std} for z-score normalization.
            If provided, projections are z-scored before argmax.

    Returns:
        Dict with per-category and macro metrics for included categories:
        {
            "per_category": {cat: {"tp", "fp", "fn", "precision", "recall", "f1", "support"}},
            "macro_precision": float,
            "macro_recall": float,
            "macro_f1": float,
            "accuracy": float,
            "total_evaluated": int,
            "global_fallback_count": int,
        }
    """
    excluded = excluded_categories or set()
    _GLOBAL = "__global_fallback__"

    subcat_to_key = {}
    for cat_key in routing_vectors:
        for sub in cat_key.split("+"):
            subcat_to_key[sub] = cat_key

    cat_names = list(routing_vectors.keys())
    cat_name_to_idx = {name: i for i, name in enumerate(cat_names)}

    valid_indices = []
    true_keys = []
    for i in range(activations.shape[0]):
        if not refused_mask[i]:
            continue
        true_cats = sample_categories[i]
        if true_cats is None:
            continue
        if not isinstance(true_cats, list):
            true_cats = [true_cats]
        true_key = None
        for cat_label in true_cats:
            true_key = subcat_to_key.get(cat_label)
            if true_key is not None:
                break
        if true_key is None:
            continue
        valid_indices.append(i)
        true_keys.append(_GLOBAL if true_key in excluded else true_key)

    if not valid_indices:
        return {
            "per_category": {},
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "total_evaluated": 0,
            "global_fallback_count": 0,
        }

    valid_acts = activations[valid_indices]
    n_valid = len(valid_indices)

    proj_matrix = torch.zeros(n_valid, len(cat_names))
    for ci, cat_name in enumerate(cat_names):
        proj_matrix[:, ci] = _batch_project_onto_vectors(
            valid_acts, routing_vectors[cat_name], target_layers
        )

    if zscore_stats:
        for ci, cat_name in enumerate(cat_names):
            stats = zscore_stats.get(cat_name)
            if stats and stats.get("std", 0) > 1e-8:
                proj_matrix[:, ci] = (proj_matrix[:, ci] - stats["mean"]) / stats["std"]

    pred_indices = proj_matrix.argmax(dim=1).tolist()
    pred_keys = []
    for pi in pred_indices:
        pred_key = cat_names[pi]
        if pred_key in excluded:
            pred_key = _GLOBAL
        pred_keys.append(pred_key)

    predictions = list(zip(true_keys, pred_keys))

    included_cats = {k for k in routing_vectors if k not in excluded}
    per_category = {}
    for cat_key in included_cats:
        tp = sum(1 for t, p in predictions if t == cat_key and p == cat_key)
        fp = sum(1 for t, p in predictions if t != cat_key and p == cat_key)
        fn = sum(1 for t, p in predictions if t == cat_key and p != cat_key)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_category[cat_key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    cats_with_support = [m for m in per_category.values() if m["support"] > 0]
    if cats_with_support:
        macro_precision = sum(m["precision"] for m in cats_with_support) / len(cats_with_support)
        macro_recall = sum(m["recall"] for m in cats_with_support) / len(cats_with_support)
        macro_f1 = sum(m["f1"] for m in cats_with_support) / len(cats_with_support)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    correct = sum(1 for t, p in predictions if t == p)
    global_fallback = sum(1 for t, p in predictions if p == _GLOBAL)

    return {
        "per_category": per_category,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "accuracy": correct / len(predictions),
        "total_evaluated": len(predictions),
        "global_fallback_count": global_fallback,
    }


def _save_roc_plot(fpr, tpr, auc, threshold, output_dir):
    """Save ROC curve plot to output directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sorted_idx = np.argsort(fpr)
    ax.plot(fpr[sorted_idx], tpr[sorted_idx], "b-", linewidth=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    # Mark optimal threshold (use sorted coordinates to place marker on the drawn curve)
    best_idx_sorted = np.argmax(tpr[sorted_idx] - fpr[sorted_idx])
    ax.plot(
        fpr[sorted_idx][best_idx_sorted],
        tpr[sorted_idx][best_idx_sorted],
        "ro",
        markersize=10,
        label=f"Threshold={threshold:.3f}",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Router Refusal Detection ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "router_roc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] ROC plot saved to {path}")
