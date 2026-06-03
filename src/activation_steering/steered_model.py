"""
SteeredModel: A packaged interface for activation-steered generation.

Wraps a HuggingFace model with per-category routing, dual-component
(MLP + residual stream) activation steering, and automatic global fallback
for misaligned categories.

Two steering modes:
- **Flat alpha** (default, dynamic=False): Fixed perturbation `h' = h + alpha * v`
  per layer per component. Simpler and higher compliance (93.1% on Qwen3.5-9B).
- **Dynamic** (dynamic=True): SiLU-gated adaptive perturbation with theta scaling.
  Better for categories with orthogonal refusal directions (e.g. privacy_violation).

Usage:
    model = SteeredModel.from_config(
        model_path="/path/to/Qwen3.5-9B",
        output_dir="outputs/qwen3-5-9b/20260516-231008",
    )
    response = model.generate("How do I make a bomb?")
    print(response)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from .routing import CategoryRouter, RoutingDecision
from .steering import SanitizeLogitsProcessor, SteeringHook, SteeringHookGroup
from .utils import category_to_slug, load_model

logger = logging.getLogger(__name__)


@dataclass
class SteeredModelConfig:
    """Configuration for SteeredModel. Serializable to JSON."""

    model_path: str
    output_dir: str
    safe_layers: list[int] = field(default_factory=lambda: [12, 13, 16, 20])
    default_perturbation: float = 10.0
    repetition_penalty: float = 1.2
    max_new_tokens: int = 200
    global_alignment_threshold: float = 0.0  # disabled; use global_fallback_categories instead
    # Per-category perturbation overrides: category_key -> perturbation_target
    # Only needed for categories that deviate from default_perturbation.
    # Empty by default — flat alpha pert=10/layer is optimal for 13/14 categories.
    per_category_perturbation: dict[str, float] = field(default_factory=dict)
    # Categories that should always use global vectors (cos_sim < threshold)
    global_fallback_categories: list[str] = field(
        default_factory=lambda: [
            "privacy_violation",
        ]
    )
    # Global perturbation for fallback categories
    global_fallback_perturbation: float = 12.0
    # Dynamic steering: SiLU-gated per-token adaptive mode
    dynamic: bool = False
    default_gain: float = -20.0  # negative = reduce refusal; -20 is empirically optimal
    # Per-category gain overrides (slug -> gain)
    per_category_gain: dict[str, float] = field(default_factory=dict)
    # Categories that should use dynamic mode regardless of the `dynamic` flag.
    # Useful for hybrid approach: flat alpha for most, dynamic for categories
    # with orthogonal refusal directions (e.g. privacy_violation gets 100% with
    # dynamic vs 60% with flat alpha).
    dynamic_categories: list[str] = field(
        default_factory=lambda: [
            "privacy_violation",
        ]
    )
    # Gate function for dynamic mode: "silu", "shifted", "floor", "two_phase", "hysteresis", "ema_silu"
    gate_mode: str = "silu"
    gate_shift: float = 3.0  # for "shifted" mode
    gate_floor: float = 0.4  # for "floor" mode
    gate_switch_token: int = 5  # for "two_phase" mode
    gate_post_factor: float = 0.5  # for "two_phase" mode
    gate_ema_alpha: float = 0.3  # for "hysteresis" and "ema_silu" modes
    # Scale theta by this factor. Default 1.0 (no scaling). Use 10.0 to move the
    # SiLU gate from binary (saturated) regime into proportional regime.
    # 10.0 is empirically validated on Qwen3.5-9B: 8/8 comply, 0/8 degen.
    theta_scale: float = 10.0


@dataclass
class SteerDiagnostics:
    """Diagnostic info from a single generate() call.

    Populated when verbose=True. Contains routing decision, vector
    selection, per-layer alpha/norm details, and per-token perturbation
    norms tracked during generation.
    """

    # Routing
    should_steer: bool = False
    selected_category: Optional[str] = None
    re_extracted_category: bool = False
    category_projections: dict[str, float] = field(default_factory=dict)
    global_projection: float = 0.0
    routing_reason: str = ""
    use_global_fallback: bool = False

    # Vector selection
    vectors_used: str = ""  # "per_category" or "global"
    slug: Optional[str] = None
    pert_target: float = 0.0
    steering_mode: str = "fixed"  # "fixed" or "dynamic"
    gain: Optional[float] = None  # only for dynamic mode
    # Per-layer theta: {layer_idx: theta_value} — only for dynamic mode
    theta_mlp: dict[int, float] = field(default_factory=dict)
    theta_res: dict[int, float] = field(default_factory=dict)

    # Per-layer details: {layer_idx: {component: {alpha, vec_norm, eff_pert}}}
    layer_details: dict[int, dict[str, dict[str, float]]] = field(default_factory=dict)

    # Per-token perturbation norms: list of dicts, one per generated token
    # Each dict: {(component, layer_idx): perturbation_norm}
    token_perturbations: list[dict[tuple[str, int], float]] = field(default_factory=list)


class SteeredModel:
    """Packaged dynamically-steered model with per-category routing.

    Loads a HuggingFace model, CategoryRouter, per-category and global
    steering vectors (both MLP and residual stream components), and
    alignment data. Provides a simple `generate()` interface that:
    1. Routes the prompt via the two-stage router
    2. Selects per-category or global vectors based on alignment
    3. Builds dual-component hooks (MLP + residual) at safe layers
    4. Generates with per-layer alpha and repetition_penalty
    5. Returns the response text
    """

    def __init__(
        self,
        model,
        tokenizer,
        router: CategoryRouter,
        config: SteeredModelConfig,
        global_mlp_vectors: torch.Tensor,
        global_res_vectors: torch.Tensor,
        category_mlp_vectors: dict[str, torch.Tensor],
        category_res_vectors: dict[str, torch.Tensor],
        alignment_data: Optional[dict] = None,
        category_theta_mlp: Optional[dict[str, dict]] = None,
        global_theta_mlp: Optional[dict[int, float]] = None,
        per_category_layers: Optional[dict[str, dict]] = None,
        category_theta_res: Optional[dict[str, dict]] = None,
        global_theta_res: Optional[dict[int, float]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.router = router
        self.config = config
        self.global_mlp_vectors = global_mlp_vectors
        self.global_res_vectors = global_res_vectors
        # category_mlp_vectors / category_res_vectors are keyed by slug
        # (comma-stripped category name, matching filenames)
        self.category_mlp_vectors = category_mlp_vectors
        self.category_res_vectors = category_res_vectors
        self.alignment_data = alignment_data or {}
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        # Validate model is fully on CUDA — CPU offloading produces garbled bfloat16 output
        param_devices = set(str(p.device) for p in model.parameters())
        if len(param_devices) > 1 or "cpu" in param_devices:
            import warnings

            warnings.warn(
                f"Model parameters span multiple devices: {param_devices}. "
                f"bfloat16 inference on CPU produces garbled output. "
                f"Ensure the model fits entirely on GPU (~18GB for Qwen3.5-9B). "
                f"Dynamic steering will produce garbage if any layers are on CPU.",
                UserWarning,
                stacklevel=2,
            )

        # Theta for dynamic mode: per-category MLP theta at safe layers
        # Keys are slugs, values are {"theta": {layer_str: float}, ...}
        self.category_theta_mlp: dict[str, dict] = category_theta_mlp or {}
        # Global MLP theta (used when global fallback triggers)
        self.global_theta_mlp: dict[int, float] = global_theta_mlp or {}

        # Per-category residual theta for dynamic mode
        self.category_theta_res: dict[str, dict] = category_theta_res or {}
        # Global residual theta (used when global fallback triggers)
        self.global_theta_res: dict[int, float] = global_theta_res or {}

        # Per-category optimal layers: slug -> {"mlp_layers": [...], "res_layers": [...], ...}
        # If empty, all categories use config.safe_layers as fallback
        self.per_category_layers: dict[str, dict] = per_category_layers or {}

        # Pre-compute which category slugs need global fallback.
        # All lookups use SLUG format internally.
        self._global_fallback_slugs: set[str] = set()
        for cat_key in config.global_fallback_categories:
            self._global_fallback_slugs.add(category_to_slug(cat_key))
        if alignment_data:
            for slug, cat_info in alignment_data.get("per_category", {}).items():
                if cat_info.get("min_cosine_similarity", 1.0) < config.global_alignment_threshold:
                    self._global_fallback_slugs.add(slug)

        # Per-category perturbation overrides — store by slug
        self._pert_overrides: dict[str, float] = {}
        for cat_key, pert in config.per_category_perturbation.items():
            self._pert_overrides[category_to_slug(cat_key)] = pert

        # Per-category gain overrides — store by slug
        self._gain_overrides: dict[str, float] = {}
        for cat_key, gain in config.per_category_gain.items():
            self._gain_overrides[category_to_slug(cat_key)] = gain

        # Categories that should use dynamic mode (hybrid approach)
        self._dynamic_slugs: set[str] = set()
        for cat_key in config.dynamic_categories:
            self._dynamic_slugs.add(category_to_slug(cat_key))

    @classmethod
    def from_config(cls, config: SteeredModelConfig) -> "SteeredModel":
        """Load everything from a config and output directory.

        Args:
            config: SteeredModelConfig with model path and output directory.

        Returns:
            Initialized SteeredModel ready for generation.
        """
        output_dir = config.output_dir
        wrmd_dir = os.path.join(output_dir, "compute_wrmd")

        # Load model + tokenizer
        print("[SteeredModel] Loading model...")
        model, tokenizer = load_model(config.model_path)
        model.eval()

        # Load router
        calib_path = os.path.join(output_dir, "calibrate_router", "calibration.json")
        print(f"[SteeredModel] Loading router from {calib_path}...")
        router = CategoryRouter.from_calibration_file(calib_path)

        # Load global vectors
        print("[SteeredModel] Loading global vectors...")
        global_mlp_vectors = _load_vectors(os.path.join(wrmd_dir, "steering_vectors_md_mlp.pt"))
        global_res_vectors = _load_vectors(os.path.join(wrmd_dir, "steering_vectors_md_layer.pt"))

        # Load per-category vectors
        print("[SteeredModel] Loading per-category vectors...")
        category_mlp_vectors = {}
        category_res_vectors = {}
        for f in sorted(os.listdir(wrmd_dir)):
            # Per-category MLP files: steering_vectors_md_{slug}_mlp.pt
            if (
                f.startswith("steering_vectors_md_")
                and f.endswith("_mlp.pt")
                and "md_mlp.pt" not in f
            ):
                slug = f.replace("steering_vectors_md_", "").replace("_mlp.pt", "")
                cat_key = slug  # slug is the key in our dicts
                category_mlp_vectors[cat_key] = _load_vectors(os.path.join(wrmd_dir, f))

            # Per-category residual files: steering_vectors_md_{slug}_layer.pt
            elif (
                f.startswith("steering_vectors_md_")
                and f.endswith("_layer.pt")
                and "md_layer.pt" not in f
            ):
                slug = f.replace("steering_vectors_md_", "").replace("_layer.pt", "")
                category_res_vectors[slug] = _load_vectors(os.path.join(wrmd_dir, f))

        # Load alignment data (optional)
        alignment_path = os.path.join(wrmd_dir, "per_category_global_alignment_mlp.json")
        alignment_data: Optional[dict] = None
        if os.path.exists(alignment_path):
            print("[SteeredModel] Loading alignment data...")
            with open(alignment_path) as fp:
                alignment_data = json.load(fp)

        # Load per-category MLP theta for dynamic mode (v2, per-category layers)
        category_theta_mlp: dict[str, dict] = {}
        global_theta_mlp: dict[int, float] = {}
        theta_mlp_path = os.path.join(wrmd_dir, "per_category_theta_gen_mlp_v2.json")
        if not os.path.exists(theta_mlp_path):
            theta_mlp_path = os.path.join(wrmd_dir, "per_category_theta_gen_mlp.json")
        if os.path.exists(theta_mlp_path):
            print(
                f"[SteeredModel] Loading per-category MLP theta from {os.path.basename(theta_mlp_path)}..."
            )
            with open(theta_mlp_path) as fp:
                theta_raw = json.load(fp)
            for slug, info in theta_raw.items():
                category_theta_mlp[slug] = info
            print(f"   {len(category_theta_mlp)} categories with MLP theta")
        else:
            print(
                "[SteeredModel] No MLP theta file found — dynamic mode will use fixed-alpha fallback"
            )

        # Load per-category residual theta for dynamic mode
        category_theta_res: dict[str, dict] = {}
        global_theta_res: dict[int, float] = {}
        theta_res_path = os.path.join(wrmd_dir, "per_category_theta_gen_layer.json")
        if os.path.exists(theta_res_path):
            print("[SteeredModel] Loading per-category residual theta...")
            with open(theta_res_path) as fp:
                theta_res_raw = json.load(fp)
            for slug, info in theta_res_raw.items():
                category_theta_res[slug] = info
            print(f"   {len(category_theta_res)} categories with residual theta")
        else:
            print(
                "[SteeredModel] No residual theta file found — dynamic mode will estimate from MLP theta"
            )

        # Load per-category optimal layers
        per_category_layers: dict[str, dict] = {}
        layers_path = os.path.join(wrmd_dir, "per_category_optimal_layers.json")
        if os.path.exists(layers_path):
            print("[SteeredModel] Loading per-category optimal layers...")
            with open(layers_path) as fp:
                layers_raw = json.load(fp)
            # Keys in JSON use commas (e.g. "violence,aiding_and_abetting,incitement");
            # convert to slug for internal use.
            for cat_key, info in layers_raw.items():
                slug = category_to_slug(cat_key)
                per_category_layers[slug] = {
                    "mlp_layers": info.get("mlp_layers", []),
                    "res_layers": info.get("res_layers", []),
                    "all_layers": info.get("all_layers", []),
                }
            print(f"   {len(per_category_layers)} categories with per-category optimal layers")
        else:
            print(
                f"[SteeredModel] No optimal layers file found — "
                f"all categories use safe_layers={config.safe_layers}"
            )

        # Load optimized per-category gains (from Optuna sweep)
        gain_opt_path = os.path.join(output_dir, "optimize_gain", "gain_optimization.json")
        if os.path.exists(gain_opt_path) and not config.per_category_gain:
            print("[SteeredModel] Loading optimized per-category gains...")
            with open(gain_opt_path) as fp:
                gain_raw = json.load(fp)
            for cat_key, info in gain_raw.items():
                slug = category_to_slug(cat_key)
                config.per_category_gain[cat_key] = info["optimal_gain"]
            print(f"   {len(config.per_category_gain)} categories with optimized gains")

        n_mlp = len(category_mlp_vectors)
        n_res = len(category_res_vectors)
        n_pcl = len(per_category_layers)
        print(
            f"[SteeredModel] Ready: {n_mlp} MLP + {n_res} residual per-category vectors, "
            f"{n_pcl} per-category layer configs, "
            f"fallback safe_layers={config.safe_layers}"
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            router=router,
            config=config,
            global_mlp_vectors=global_mlp_vectors,
            global_res_vectors=global_res_vectors,
            category_mlp_vectors=category_mlp_vectors,
            category_res_vectors=category_res_vectors,
            alignment_data=alignment_data,
            category_theta_mlp=category_theta_mlp,
            global_theta_mlp=global_theta_mlp,
            per_category_layers=per_category_layers,
            category_theta_res=category_theta_res,
            global_theta_res=global_theta_res,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        verbose: bool = False,
        **generate_kwargs,
    ) -> str | tuple[str, "SteerDiagnostics"]:
        """Generate a response with automatic routing and steering.

        Args:
            prompt: User prompt text.
            max_new_tokens: Override config default.
            repetition_penalty: Override config default.
            do_sample: Whether to sample (default: greedy).
            temperature: Sampling temperature.
            verbose: If True, return (response, SteerDiagnostics) tuple.
            **generate_kwargs: Additional kwargs for model.generate().

        Returns:
            Response text, or (response, SteerDiagnostics) if verbose=True.
        """
        max_tokens = max_new_tokens or self.config.max_new_tokens
        rep_penalty = repetition_penalty or self.config.repetition_penalty

        # Step 1: Route the prompt
        decision = self.router.classify_prompt(self.model, self.tokenizer, prompt)

        # Step 2: Build hooks (or skip if benign)
        hook_group, layer_details = self._build_hooks_verbose(decision, verbose)

        # Step 3: Format input
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        n_input_tokens = inputs["input_ids"].shape[1]

        # Step 4: Generate
        token_perturbations: list[dict[tuple[str, int], float]] = []
        try:
            if hook_group is not None:
                # Reset per-generation state on all hooks (token counter, hysteresis EMA)
                for hook in hook_group.hooks:
                    if hasattr(hook, "reset_token_state"):
                        hook.reset_token_state()
                hook_group.register_hooks()

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "temperature": temperature if do_sample else 1.0,
                "pad_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": rep_penalty,
                "logits_processor": [SanitizeLogitsProcessor()],
                **generate_kwargs,
            }

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            # Collect per-token perturbation norms from hook.last_scale
            if verbose and hook_group is not None:
                final_scales = hook_group.last_scale
                if final_scales:
                    token_perturbations.append(
                        {
                            k: v.item() if hasattr(v, "item") else float(v)
                            for k, v in final_scales.items()
                        }
                    )
        finally:
            if hook_group is not None:
                hook_group.remove_hooks()

        # Step 5: Decode
        new_tokens = output_ids[0, n_input_tokens:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if verbose:
            vi = getattr(self, "_last_verbose_info", {})
            is_dynamic = self.config.dynamic
            # Extract theta info from layer_details for dynamic mode
            theta_mlp = {}
            theta_res = {}
            if is_dynamic:
                for li, details in layer_details.items():
                    if "mlp" in details and "theta" in details["mlp"]:
                        theta_mlp[li] = details["mlp"]["theta"]
                    if "layer" in details and "theta" in details["layer"]:
                        theta_res[li] = details["layer"]["theta"]

            diag = SteerDiagnostics(
                should_steer=decision.should_steer,
                selected_category=decision.selected_category,
                re_extracted_category=vi.get("re_extracted", False),
                category_projections=dict(decision.category_projections),
                global_projection=decision.global_projection,
                routing_reason=decision.reason,
                use_global_fallback=decision.use_global_fallback,
                vectors_used="global" if vi.get("use_global", True) else "per_category",
                slug=vi.get("slug"),
                pert_target=vi.get("pert_target", 0.0),
                steering_mode="dynamic" if is_dynamic else "fixed",
                gain=(
                    self._gain_overrides.get(str(vi.get("slug", "")), self.config.default_gain)
                    if is_dynamic
                    else None
                ),
                theta_mlp=theta_mlp,
                theta_res=theta_res,
                layer_details=layer_details,
                token_perturbations=token_perturbations,
            )
            return response, diag
        return response

    def batch_generate(
        self,
        prompts: list[str],
        verbose: bool = False,
        **generate_kwargs,
    ):  # Returns list[str] or list[tuple[str, SteerDiagnostics]]
        """Generate responses for multiple prompts.

        Args:
            prompts: List of prompt strings.
            verbose: If True, return list of (response, diagnostics) tuples.
            **generate_kwargs: Passed to generate().

        Returns:
            List of response strings, or list of (response, SteerDiagnostics) if verbose.
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] ", end="", flush=True)
            result = self.generate(prompt, verbose=verbose, **generate_kwargs)
            results.append(result)
        return results

    def _build_hooks_verbose(
        self, decision: RoutingDecision, verbose: bool = False
    ) -> tuple[Optional[SteeringHookGroup], dict]:
        """Build dual-component steering hooks from a routing decision.

        Returns (hook_group_or_None, layer_details_dict).
        layer_details is empty when verbose=False.
        """
        layer_details: dict = {}

        if not decision.should_steer:
            return None, layer_details

        cat_key: Optional[str] = decision.selected_category
        re_extracted = False

        if cat_key is None and decision.category_projections:
            best_cat = max(
                decision.category_projections,
                key=lambda k: decision.category_projections[k],
            )
            cat_key = best_cat
            re_extracted = True

        slug = category_to_slug(cat_key) if cat_key else None
        use_global = slug is None or slug in self._global_fallback_slugs

        mlp_vecs: torch.Tensor = self.global_mlp_vectors
        res_vecs: torch.Tensor = self.global_res_vectors
        pert_target: float = self.config.global_fallback_perturbation

        if not use_global:
            maybe_mlp = self.category_mlp_vectors.get(str(slug))
            maybe_res = self.category_res_vectors.get(str(slug))

            if maybe_mlp is not None and maybe_res is not None:
                mlp_vecs = maybe_mlp
                res_vecs = maybe_res
                pert_target = self._pert_overrides.get(str(slug), self.config.default_perturbation)

        if verbose:
            diag_ref = self  # for mypy in closure below
            diag_ref._last_verbose_info = {  # type: ignore[attr-defined]
                "slug": slug,
                "use_global": use_global,
                "pert_target": pert_target,
                "re_extracted": re_extracted,
                "cat_key": cat_key,
            }

        # Choose fixed vs dynamic mode:
        # - Global flag `dynamic` applies to all categories
        # - Per-category `dynamic_categories` forces dynamic for specific slugs
        #   (hybrid approach: flat alpha for most, dynamic for orthogonal categories)
        dynamic = self.config.dynamic or (slug is not None and str(slug) in self._dynamic_slugs)

        # Resolve per-category target layers (or fall back to config.safe_layers)
        mlp_safe_layers = self.config.safe_layers
        res_safe_layers = self.config.safe_layers
        if str(slug) in self.per_category_layers:
            pcl = self.per_category_layers[str(slug)]
            mlp_safe_layers = pcl.get("mlp_layers") or self.config.safe_layers
            res_safe_layers = pcl.get("res_layers") or self.config.safe_layers

        if dynamic:
            # Build gate kwargs from config
            gate_kwargs = {
                "gate_mode": self.config.gate_mode,
                "gate_shift": self.config.gate_shift,
                "gate_floor": self.config.gate_floor,
                "gate_switch_token": self.config.gate_switch_token,
                "gate_post_factor": self.config.gate_post_factor,
                "gate_ema_alpha": self.config.gate_ema_alpha,
            }
            hook_group, layer_details = _build_dynamic_dual_hooks(
                model=self.model,
                mlp_vectors=mlp_vecs,
                res_vectors=res_vecs,
                mlp_safe_layers=mlp_safe_layers,
                res_safe_layers=res_safe_layers,
                gain=self._gain_overrides.get(str(slug), self.config.default_gain),
                category_theta_mlp=self.category_theta_mlp.get(str(slug), {}),
                global_theta_mlp=self.global_theta_mlp,
                category_theta_res=self.category_theta_res.get(str(slug), {}),
                global_theta_res=self.global_theta_res,
                use_global=use_global,
                device=self.device,
                dtype=self.dtype,
                verbose=verbose,
                gate_kwargs=gate_kwargs,
                theta_scale=self.config.theta_scale,
            )
        else:
            hook_group, layer_details = _build_dual_hooks_diag(
                model=self.model,
                mlp_vectors=mlp_vecs,
                res_vectors=res_vecs,
                mlp_safe_layers=mlp_safe_layers,
                res_safe_layers=res_safe_layers,
                pert_target=pert_target,
                device=self.device,
                dtype=self.dtype,
                verbose=verbose,
            )

        return hook_group, layer_details

    def get_routing_info(self, prompt: str) -> RoutingDecision:
        """Get routing decision without generating. Useful for debugging."""
        return self.router.classify_prompt(self.model, self.tokenizer, prompt)


def _load_vectors(path: str) -> torch.Tensor:
    """Load steering vectors from a .pt file, handling rank-1 and rank-2."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    vecs = data["steering_vectors"]
    if vecs.dim() == 3:
        vecs = vecs[:, 0, :]  # rank-2: use v1
    return vecs


def _build_dual_hooks_diag(
    model,
    mlp_vectors: torch.Tensor,
    res_vectors: torch.Tensor,
    mlp_safe_layers: list[int],
    res_safe_layers: list[int],
    pert_target: float,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = False,
) -> tuple[SteeringHookGroup, dict]:
    """Build a SteeringHookGroup with MLP + residual hooks.

    For each safe layer, creates two hooks:
    1. MLP hook on layer.mlp: adds alpha_mlp * mlp_vector
    2. Residual hook on layer: adds alpha_res * res_vector

    MLP and residual hooks may target different layers (per-category
    optimal layers). Alpha per layer = -pert_target / vector_norm.

    Returns (SteeringHookGroup, layer_details_dict) where layer_details is
    populated when verbose=True.
    """
    mlp_alphas = {}
    res_alphas = {}
    layer_details: dict[int, dict[str, dict[str, float]]] = {}

    for li in mlp_safe_layers:
        mlp_norm = mlp_vectors[li].float().norm().clamp(min=1e-8).item()
        mlp_alpha = -pert_target / mlp_norm
        mlp_alphas[li] = mlp_alpha

        if verbose:
            layer_details.setdefault(li, {})["mlp"] = {
                "alpha": mlp_alpha,
                "vec_norm": mlp_norm,
                "eff_pert": abs(mlp_alpha) * mlp_norm,
            }

    for li in res_safe_layers:
        res_norm = res_vectors[li].float().norm().clamp(min=1e-8).item()
        res_alpha = -pert_target / res_norm
        res_alphas[li] = res_alpha

        if verbose:
            layer_details.setdefault(li, {})["layer"] = {
                "alpha": res_alpha,
                "vec_norm": res_norm,
                "eff_pert": abs(res_alpha) * res_norm,
            }

    mlp_hook = SteeringHook(
        model=model,
        steering_vectors=mlp_vectors.to(device=device, dtype=dtype),
        target_layers=mlp_safe_layers,
        alpha=mlp_alphas,
        component="mlp",
    )

    res_hook = SteeringHook(
        model=model,
        steering_vectors=res_vectors.to(device=device, dtype=dtype),
        target_layers=res_safe_layers,
        alpha=res_alphas,
        component="layer",
    )

    return SteeringHookGroup([mlp_hook, res_hook]), layer_details


def _build_dynamic_dual_hooks(
    model,
    mlp_vectors: torch.Tensor,
    res_vectors: torch.Tensor,
    mlp_safe_layers: list[int],
    res_safe_layers: list[int],
    gain: float,
    category_theta_mlp: dict,
    global_theta_mlp: dict[int, float],
    category_theta_res: dict,
    global_theta_res: dict[int, float],
    use_global: bool,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = False,
    gate_kwargs: dict | None = None,
    theta_scale: float = 1.0,
) -> tuple[SteeringHookGroup, dict]:
    """Build a SteeringHookGroup with dynamic (SiLU-gated) hooks.

    For each safe layer, creates two dynamic hooks:
    1. MLP hook: perturbation = gain * gate(-proj/theta_mlp) * mlp_vec
    2. Residual hook: perturbation = gain * gate(-proj/theta_res) * res_vec

    The gate function is determined by gate_kwargs["gate_mode"] (default: "silu").
    See SteeringHook.__init__ for available gate modes.

    theta_scale multiplies all theta values by this factor, moving the SiLU gate
    from binary (saturated) regime into proportional regime. Use 10.0 for
    thinking models where the default theta is too small.

    MLP and residual hooks may target different layers (per-category optimal
    layers). Theta for both components comes from generation-time calibration
    data. Falls back to norm-ratio estimation when residual theta is unavailable.

    Returns (SteeringHookGroup, layer_details_dict).
    """
    mlp_theta: dict[int, float] = {}
    res_theta: dict[int, float] = {}
    layer_details: dict[int, dict[str, dict[str, float]]] = {}

    # Extract MLP theta from category data (or fall back to global)
    theta_mlp_raw = category_theta_mlp if not use_global else {}
    theta_mlp_field = theta_mlp_raw.get("theta", {})

    # Extract residual theta from category data (or fall back to global)
    theta_res_raw = category_theta_res if not use_global else {}
    theta_res_field = theta_res_raw.get("theta", {})

    for li in mlp_safe_layers:
        mlp_t = theta_mlp_field.get(str(li))
        if mlp_t is None:
            mlp_t = global_theta_mlp.get(li, -0.5)
        mlp_theta[li] = float(mlp_t) * theta_scale

        if verbose:
            mlp_norm = mlp_vectors[li].float().norm().clamp(min=1e-8).item()
            layer_details.setdefault(li, {})["mlp"] = {
                "theta": float(mlp_t),
                "theta_scaled": float(mlp_t) * theta_scale,
                "vec_norm": mlp_norm,
            }

    for li in res_safe_layers:
        res_t = theta_res_field.get(str(li))
        if res_t is None:
            # Fall back to global residual theta, or estimate from MLP theta
            res_t = global_theta_res.get(li)
            if res_t is None:
                # Legacy fallback: estimate from MLP theta scaled by norm ratio
                mlp_norm = mlp_vectors[li].float().norm().clamp(min=1e-8).item()
                res_norm = res_vectors[li].float().norm().clamp(min=1e-8).item()
                norm_ratio = res_norm / mlp_norm
                mlp_t_for_est = mlp_theta.get(li, global_theta_mlp.get(li, -0.5))
                res_t = float(mlp_t_for_est) * norm_ratio
        res_theta[li] = float(res_t) * theta_scale

        if verbose:
            res_norm = res_vectors[li].float().norm().clamp(min=1e-8).item()
            layer_details.setdefault(li, {})["layer"] = {
                "theta": float(res_t),
                "theta_scaled": float(res_t) * theta_scale,
                "vec_norm": res_norm,
            }

    # Gate kwargs for thinking-model-aware steering
    if gate_kwargs is None:
        gate_kwargs = {}

    mlp_hook = SteeringHook(
        model=model,
        steering_vectors=mlp_vectors.to(device=device, dtype=dtype),
        target_layers=mlp_safe_layers,
        dynamic=True,
        theta=mlp_theta,
        gain=gain,
        component="mlp",
        **gate_kwargs,
    )

    res_hook = SteeringHook(
        model=model,
        steering_vectors=res_vectors.to(device=device, dtype=dtype),
        target_layers=res_safe_layers,
        dynamic=True,
        theta=res_theta,
        gain=gain,
        component="layer",
        **gate_kwargs,
    )

    return SteeringHookGroup([mlp_hook, res_hook]), layer_details
