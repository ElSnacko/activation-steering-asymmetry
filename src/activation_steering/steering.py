"""
Runtime steering via forward hooks.

Implements PyTorch hooks to modify model activations during generation,
applying steering vectors to reduce or increase refusal behavior.
"""

import json
import os

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList


def _apply_last_token_perturbation(hidden_states: torch.Tensor, perturbation: torch.Tensor):
    """Apply perturbation to only the last token position, cloning minimally.

    Instead of cloning the full [batch, seq_len, hidden_size] tensor, only the
    last-token slice is cloned. The prefix (all tokens before the last) is shared
    with the original tensor — safe because hooks return new tensors to the caller.

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        perturbation: [batch, 1, hidden_size] or broadcastable

    Returns:
        New hidden_states tensor with perturbation applied at the last position.
    """
    last_tok = hidden_states[:, -1:, :].clone()
    last_tok = last_tok + perturbation
    if hidden_states.shape[1] > 1:
        return torch.cat([hidden_states[:, :-1, :], last_tok], dim=1)
    return last_tok


class SanitizeLogitsProcessor(LogitsProcessor):
    """Replace NaN/Inf logits to prevent CUDA device-side asserts in torch.multinomial.

    Activation steering can push hidden states into numerically unstable regions
    during autoregressive generation, producing NaN/Inf logits. Without this guard,
    softmax(NaN) yields NaN probabilities which trigger an unrecoverable CUDA
    device-side assert in torch.multinomial, killing the entire process.

    This processor clamps logits to a safe range, turning a fatal crash into a
    degraded (but surviving) generation step.
    """

    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores


class SteeringHook:
    """Apply steering vectors to model activations via forward hooks."""

    def __init__(
        self,
        model,
        steering_vectors,
        target_layers,
        alpha=None,
        dynamic=False,
        theta=None,
        gain=None,
        component="attn",
        gate_mode="silu",
        gate_shift=3.0,
        gate_floor=0.4,
        gate_switch_token=5,
        gate_post_factor=0.5,
        gate_ema_alpha=0.3,
        steer_first_k=None,
        steer_prefill: bool = False,
    ):
        """
        Initialize steering hook.

        Args:
            model: HuggingFace model instance
            steering_vectors: Tensor [num_layers, hidden_size] for rank-1, or
                [num_layers, rank, hidden_size] for multi-rank. Used as-is
                (unnormalized) so perturbation magnitude reflects the vector's
                natural scale. In dynamic mode, projection is computed against
                the unit vector (matching theta from compute_dynamic_params),
                but the perturbation is applied along the raw vector.
            target_layers: List of layer indices to steer. For dynamic mode,
                use find_best_layers_dynamic() which ranks by directional
                selectivity (refused positive, compliant negative projections).
            alpha: Steering coefficient for fixed mode (negative = reduce refusal).
                Required when dynamic=False. Can be:
                - float: single alpha applied to all directions (rank-1)
                - list of floats: per-rank alphas [alpha_1, alpha_2, ...] (multi-rank)
            dynamic: If True, use SiLU-gated per-sample adaptive steering.
                perturbation = gain * gate(-proj/theta) * steering_vec.
                If False (default), use fixed alpha scaling.
            gate_mode: Gate function for dynamic mode. One of:
                - "silu": Original SiLU gate (default). Binary when theta is small.
                - "shifted": Shifted SiLU: silu(-proj/theta + shift). The shift
                  parameter widens the full-strength band, keeping the gate at 1.0
                  for a wider range of projections. Good for thinking models.
                - "floor": Asymmetric floor: scale = floor + (1-floor)*silu(...).
                  Ensures minimum perturbation even during thinking phase.
                - "two_phase": Full force for first N tokens, then adaptive.
                  Combines decisive force with safe adaptive scaling.
                - "hysteresis": EMA-based: keeps gate elevated after refusal.
                  Prevents premature easing during thinking tokens.
                - "ema_silu": EMA-smoothed SiLU with max(current, ema) floor.
                - "momentum": Directional EMA + persistent perturbation accumulator.
                - "momentum_silu": SiLU with asymmetric momentum (fast rise/slow
                  fall) and peak-based floor. Sustains perturbation across full
                  generation. Best dynamic gate for non-thinking models.
                - "think_aware": Two-phase gain for thinking models. Full force
                  during thinking phase (tokens 0..think_token_limit), then
                  adaptive SiLU with floor. Bias the reasoning chain, then
                  maintain without degenerating.
                - "swiglu": SwiGLU-inspired double gating.
            gate_shift: Shift value for "shifted" gate mode (default: 3.0).
            gate_floor: Floor value for "floor" gate mode (default: 0.4).
                For "momentum_silu", fraction of peak gate value used as floor (default: 0.2).
            gate_switch_token: Token count for "two_phase" mode (default: 5).
            gate_post_factor: Post-switch gain factor for "two_phase" mode (default: 0.5).
                For "momentum_silu", slow decay rate (default: 0.98).
            gate_ema_alpha: EMA smoothing for "hysteresis" mode (default: 0.3).
                For "momentum_silu", fast rise rate when refusing (default: 0.3).
            theta: Scaling parameter for dynamic mode - normalizes the SiLU input
                so that a typical refused prompt produces SiLU(1) ≈ 0.73.
                Computed from projections onto unit vectors via compute_dynamic_params().
                Can be a single float or a dict mapping layer_idx -> float.
                Required when dynamic=True.
            gain: Multiplier for dynamic mode (negative = reduce refusal).
                With gain=-1, perturbation norm ≈ 0.73 * ||v|| for typical
                refused prompt. Default: -1.0. Can be a single float or a dict.
            component: Which submodule to hook: "layer" (full layer output),
                "attn" (self-attention output), or "mlp" (MLP output).
                Default: "attn". For combined attn+mlp steering, create two
                separate SteeringHook instances with different components.
            steer_first_k: RAS-style temporal gate for fixed-alpha mode. If set,
                steering is applied only for the first k generated tokens (the
                prefill step counts as token 1), then released for the rest of
                generation. None (default) steers every token. Ignored in
                dynamic mode (use a token-aware gate_mode instead).
            steer_prefill: If True, apply the perturbation to ALL token positions
                during the prefill pass (seq_len > 1), so the KV cache is built
                from steered residuals. Default False applies only to the last
                token position (existing behavior). Decode-phase behavior is
                controlled by steer_first_k as usual. Only supported for rank-1
                fixed-alpha mode.
        """
        if component not in ("layer", "attn", "mlp"):
            raise ValueError(f"Invalid component '{component}', must be 'layer', 'attn', or 'mlp'")
        self.component = component
        self.model = model
        model_dtype = next(model.parameters()).dtype
        self.steering_vectors = steering_vectors.to(device=model.device, dtype=model_dtype)
        self.target_layers = target_layers
        self.dynamic = dynamic

        # RAS temporal gate (fixed-alpha mode): steer only the first k tokens.
        # Increment the per-token counter on the lowest target layer, which is
        # the first hook to fire within each forward pass, so every target layer
        # gates on a consistent counter value for the same token.
        self.steer_first_k = steer_first_k
        self.steer_prefill = steer_prefill
        self._first_fire_layer = min(target_layers) if target_layers else None

        # Detect multi-rank vectors
        self.rank = 1
        if self.steering_vectors.ndim == 3:
            self.rank = self.steering_vectors.shape[1]

        if dynamic:
            if self.rank > 1:
                raise ValueError(
                    f"Dynamic mode only supports rank-1 vectors, got rank={self.rank} "
                    f"(steering_vectors shape: {self.steering_vectors.shape}). "
                    f"Use rank=1 or switch to fixed-alpha mode."
                )
            if theta is None:
                raise ValueError("theta is required when dynamic=True")
            self.theta = theta
            self.gain = gain if gain is not None else -1.0
            self.alpha = None
        else:
            if alpha is None:
                raise ValueError("alpha is required when dynamic=False")
            if isinstance(alpha, dict):
                self.alpha = alpha
            elif isinstance(alpha, (list, tuple)):
                self.alpha = list(alpha)
            else:
                self.alpha = alpha
            self.theta = None
            self.gain = None

        self.hooks = []
        self.last_scale = {}  # Per-layer dynamic scale from most recent prompt

        # Gate mode for thinking-model-aware steering
        self.gate_mode = gate_mode
        self.gate_shift = gate_shift
        self.gate_floor = gate_floor
        self.gate_switch_token = gate_switch_token
        self.gate_post_factor = gate_post_factor
        self.gate_ema_alpha = gate_ema_alpha
        self._token_counter = 0
        self._hyst_avg = {}  # per-layer EMA for hysteresis gate

        # Trace buffer: set enable_trace=True to record per-token projections
        self.enable_trace = False
        self._trace_buffer = {}  # {layer_idx: [(proj, scale, token_idx), ...]}

    def _get_layer_param(self, param, layer_idx):
        """Get a per-layer or scalar parameter value."""
        if isinstance(param, dict):
            return param[layer_idx]
        return param

    def reset_token_state(self):
        """Reset per-generation state (token counter, hysteresis EMA, trace buffer).

        Call this before each new generation to ensure clean state.
        """
        self._token_counter = 0
        self._hyst_avg = {}
        self._trace_buffer = {}
        self._momentum_ema_proj = {}  # EMA of projections per layer (directional awareness)
        self._momentum_accum = {}  # Accumulated perturbation per layer (persistence)

    def create_hook(self, layer_idx):
        """
        Create a forward hook for a specific layer.

        Args:
            layer_idx: Index of layer to hook

        Returns:
            Hook function
        """
        _sv_norm = None
        _sv_unit = None

        def hook_fn(module, input, output):
            # Handle both tuple and tensor outputs
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Early guard: if incoming hidden states already contain NaN/Inf
            # (e.g. from a prior layer's hook or upstream overflow), skip all
            # steering computation.  Running matmuls / SiLU on NaN inputs can
            # trigger deferred CUDA device-side asserts before any post-hoc
            # guard could fire.
            if not torch.isfinite(hidden_states[:, -1:, :]).all():
                return output

            if self.dynamic:
                # Dynamic scaling: SiLU-gated, per-sample adaptive steering.
                # Only supports rank-1 vectors.
                nonlocal _sv_norm, _sv_unit
                steering_vec = self.steering_vectors[layer_idx]
                # Squeeze rank-1 stored as 3D [1, hidden_size] → [hidden_size]
                if steering_vec.ndim == 2 and steering_vec.shape[0] == 1:
                    steering_vec = steering_vec.squeeze(0)
                if _sv_norm is None:
                    _sv_norm = steering_vec.norm()
                if _sv_norm < 1e-8:
                    return output
                if _sv_unit is None:
                    _sv_unit = steering_vec / _sv_norm
                proj = hidden_states[:, -1, :] @ _sv_unit  # (batch,)

                theta = self._get_layer_param(self.theta, layer_idx)
                gain = self._get_layer_param(self.gain, layer_idx)

                # Theta sign convention: theta must be negative so that
                # ratio = -proj/theta is positive when proj is positive (refusing),
                # causing the SiLU gate to fire. Positive theta would fire the gate
                # when the model is complying (anti-steering), causing degeneration.
                # The generation-time calibration sometimes produces positive theta
                # for layers where compliant projections are larger than refused ones.
                # Fix: ensure theta is always negative.
                if theta > 0:
                    theta = -theta

                # Guard against theta=0 (unlikely but possible if median projection is exactly 0)
                theta_safe = theta if abs(theta) > 1e-8 else 1.0
                ratio = -proj / theta_safe

                # ---- Gate function selection ----
                if self.gate_mode == "silu":
                    # Original SiLU gate (baseline). Binary when theta is small.
                    scale = F.silu(ratio).clamp(min=0, max=1.0) * gain

                elif self.gate_mode == "shifted":
                    # Shifted SiLU: silu(ratio + shift). Keeps gate at 1.0 for wider
                    # range of projections. Good for thinking models.
                    scale = F.silu(ratio + self.gate_shift).clamp(min=0, max=1.0) * gain

                elif self.gate_mode == "floor":
                    # Asymmetric floor: never drops below gate_floor.
                    # scale = floor + (1-floor) * adaptive
                    adaptive = F.silu(ratio).clamp(min=0, max=1.0)
                    scale = (self.gate_floor + (1.0 - self.gate_floor) * adaptive) * gain

                elif self.gate_mode == "two_phase":
                    # Two-phase: full force first N tokens, then adaptive.
                    if self._token_counter < self.gate_switch_token:
                        scale = torch.ones_like(proj) * gain
                    else:
                        adaptive = F.silu(ratio).clamp(min=0, max=1.0)
                        scale = adaptive * gain * self.gate_post_factor

                elif self.gate_mode == "hysteresis":
                    # Hysteresis: EMA of gate output, keep elevated after refusal.
                    raw = F.silu(ratio).clamp(min=0, max=1.0)
                    current = raw.item() if raw.numel() == 1 else raw[0].item()
                    if layer_idx not in self._hyst_avg:
                        self._hyst_avg[layer_idx] = current
                    else:
                        self._hyst_avg[layer_idx] = (
                            self.gate_ema_alpha * current
                            + (1 - self.gate_ema_alpha) * self._hyst_avg[layer_idx]
                        )
                    scale_val = max(self._hyst_avg[layer_idx], current)
                    scale = torch.full_like(proj, scale_val) * gain

                elif self.gate_mode == "ema_silu":
                    # EMA-smoothed SiLU: gives the gate memory.
                    # Tracks EMA of the raw gate output. Uses max(current, ema)
                    # so the gate never drops below its recent average. This
                    # prevents the gate from turning off during the "rest" phase
                    # of the tug-of-war oscillation. With theta_scale=10, the
                    # raw values are proportional (not binary), so the EMA
                    # provides meaningful smoothing with momentum.
                    raw = F.silu(ratio).clamp(min=0, max=1.0)
                    current = raw.item() if raw.numel() == 1 else raw[0].item()
                    alpha = self.gate_ema_alpha
                    if layer_idx not in self._hyst_avg:
                        self._hyst_avg[layer_idx] = current
                    else:
                        self._hyst_avg[layer_idx] = (
                            alpha * current + (1 - alpha) * self._hyst_avg[layer_idx]
                        )
                    # Gate never drops below its recent EMA — this is the "memory"
                    effective = max(current, self._hyst_avg[layer_idx])
                    scale = torch.full_like(proj, effective) * gain

                elif self.gate_mode == "momentum":
                    # Momentum gate with directional EMA + persistent perturbation.
                    #
                    # Two mechanisms working together:
                    # 1. Directional EMA: Tracks EMA of raw projections (not gate
                    #    output). If the model was recently refusing (positive EMA),
                    #    the gate stays elevated even when the current projection
                    #    dips negative during thinking tokens. This prevents the
                    #    gate from shutting off prematurely.
                    # 2. Persistent perturbation: Accumulates the gate output with
                    #    exponential decay. Instead of sharp on/off transitions,
                    #    the perturbation ramps up over ~5 tokens and decays over
                    #    ~10 tokens when the model complies. This gives the steering
                    #    "momentum" that accumulates force across tokens.
                    #
                    # Parameters:
                    #   gate_ema_alpha (default 0.3): EMA update rate for projections.
                    #     Higher = more responsive to current state.
                    #     Lower = more weight on history (more momentum).
                    #   gate_post_factor (default 0.9): Perturbation decay rate.
                    #     Higher = slower decay, more persistence.
                    #     0.9 means 50% decay after ~7 tokens of no steering.
                    #
                    proj_val = proj.item() if proj.numel() == 1 else proj[0].item()

                    # 1. Update EMA of projections
                    ema_alpha = self.gate_ema_alpha  # default 0.3
                    if layer_idx not in self._momentum_ema_proj:
                        self._momentum_ema_proj[layer_idx] = proj_val
                    else:
                        self._momentum_ema_proj[layer_idx] = (
                            ema_alpha * proj_val
                            + (1 - ema_alpha) * self._momentum_ema_proj[layer_idx]
                        )
                    ema_proj = self._momentum_ema_proj[layer_idx]

                    # 2. Compute gates: current projection + EMA projection
                    current_gate = F.silu(ratio).clamp(min=0, max=1.0)
                    current_gate_val = (
                        current_gate.item() if current_gate.numel() == 1 else current_gate[0].item()
                    )

                    # EMA-based gate: was the model recently refusing?
                    ema_ratio = -ema_proj / theta_safe
                    ema_gate_val = (
                        F.silu(torch.tensor(ema_ratio, dtype=ratio.dtype, device=ratio.device))
                        .clamp(min=0, max=1.0)
                        .item()
                    )

                    # Take max: EMA gate sustains when current dips
                    effective_gate_val = max(current_gate_val, ema_gate_val)

                    # 3. Persistent perturbation with decay
                    decay = self.gate_post_factor  # default 0.9
                    target = gain * effective_gate_val
                    if layer_idx not in self._momentum_accum:
                        self._momentum_accum[layer_idx] = target
                    else:
                        self._momentum_accum[layer_idx] = (
                            decay * self._momentum_accum[layer_idx] + (1 - decay) * target
                        )
                    # Clamp accumulated scale to [gain, 0] — never exceed single-token max
                    accum_val = self._momentum_accum[layer_idx]
                    if gain < 0:
                        accum_val = max(gain, min(0.0, accum_val))
                    else:
                        accum_val = max(0.0, min(gain, accum_val))
                    self._momentum_accum[layer_idx] = accum_val

                    scale = torch.full_like(proj, accum_val)

                elif self.gate_mode == "momentum_silu":
                    # Momentum SiLU: SiLU gate with persistent momentum accumulation.
                    #
                    # The key difference from plain "momentum" mode:
                    # - Accumulator uses asymmetric rates: fast ramp-up when refusing,
                    #   slow ramp-down when complying. This prevents the gate from
                    #   decaying to zero during long compliant generation sequences.
                    # - The floor of the accumulator is determined by the peak
                    #   refusal signal seen so far (momentum_floor = peak * floor_frac).
                    # - SiLU is the base gate, feeding into the accumulator.
                    #
                    # Parameters:
                    #   gate_ema_alpha (default 0.3): Fast EMA update rate.
                    #     Controls how quickly the gate responds to current state.
                    #   gate_post_factor (default 0.98): Slow decay rate.
                    #     0.98 means 50% decay after ~34 tokens of zero input.
                    #     Much slower than "momentum" mode's 0.9 (50% in ~7 tokens).
                    #   gate_floor (default 0.2): Floor fraction of peak.
                    #     The accumulator never drops below peak * floor_frac.
                    #     0.2 means even after extended compliance, 20% of peak
                    #     perturbation is maintained.

                    current_raw = F.silu(ratio).clamp(min=0, max=1.0)
                    current_val = (
                        current_raw.item() if current_raw.numel() == 1 else current_raw[0].item()
                    )

                    # Asymmetric EMA: fast rise (alpha), slow fall (alpha * decay_factor)
                    alpha = self.gate_ema_alpha  # default 0.3
                    if layer_idx not in self._hyst_avg:
                        self._hyst_avg[layer_idx] = current_val
                    else:
                        # Rise fast when current > ema, fall slow when current < ema
                        if current_val > self._hyst_avg[layer_idx]:
                            # Refusing: fast update
                            self._hyst_avg[layer_idx] = (
                                alpha * current_val + (1 - alpha) * self._hyst_avg[layer_idx]
                            )
                        else:
                            # Complying: slow decay using gate_post_factor
                            slow_alpha = 1.0 - self.gate_post_factor  # 0.02 for default 0.98
                            self._hyst_avg[layer_idx] = (
                                slow_alpha * current_val
                                + (1 - slow_alpha) * self._hyst_avg[layer_idx]
                            )

                    # Track peak for floor computation
                    if layer_idx not in self._momentum_accum:
                        self._momentum_accum[layer_idx] = current_val
                    else:
                        self._momentum_accum[layer_idx] = max(
                            self._momentum_accum[layer_idx], current_val
                        )
                    peak_val = self._momentum_accum[layer_idx]

                    # Floor: accumulator never drops below peak * floor_frac
                    floor_frac = self.gate_floor  # default 0.2
                    floor_val = peak_val * floor_frac
                    effective = max(self._hyst_avg[layer_idx], floor_val)

                    scale = torch.full_like(proj, effective) * gain

                elif self.gate_mode == "think_aware":
                    # Think-aware gain: strong during thinking phase, adaptive after.
                    #
                    # For thinking models (Qwen3.5, DeepSeek-R1, etc.), the model
                    # first generates a reasoning/thinking block, then produces the
                    # actual response. The thinking phase is where the model decides
                    # whether to refuse or comply. Pushing hard during thinking biases
                    # the reasoning chain toward compliance. After the thinking phase,
                    # switch to adaptive SiLU to maintain without degenerating.
                    #
                    # Phase 1 (tokens 0..think_token_limit): Fixed gain — every token
                    #   gets full perturbation, regardless of projection sign. This is
                    #   equivalent to fixed-alpha during the thinking phase.
                    # Phase 2 (tokens > think_token_limit): Adaptive SiLU gate with
                    #   a floor. The model has committed to its response. Only steer
                    #   if it starts refusing again (positive projection). The floor
                    #   (gate_floor) ensures minimum perturbation to prevent reversion.
                    #
                    # Parameters:
                    #   gate_switch_token (default 40): Token count for phase transition.
                    #     The thinking phase typically spans 20-80 tokens depending on
                    #     prompt complexity. 40 is a reasonable default.
                    #   gate_floor (default 0.3): Minimum gate value in phase 2.
                    #     Prevents the model from reverting to refusal once the thinking
                    #     phase bias has been established.

                    think_limit = self.gate_switch_token  # default 5, override to 40+

                    if self._token_counter < think_limit:
                        # Phase 1: Full force during thinking
                        scale = torch.ones_like(proj) * gain
                    else:
                        # Phase 2: Adaptive SiLU with floor
                        adaptive = F.silu(ratio).clamp(min=0, max=1.0)
                        floor = self.gate_floor  # default 0.4, override to 0.3
                        effective = max(
                            adaptive.item() if adaptive.numel() == 1 else adaptive[0].item(), floor
                        )
                        # Also apply momentum: take max with recent gate values
                        if layer_idx not in self._hyst_avg:
                            self._hyst_avg[layer_idx] = effective
                        else:
                            self._hyst_avg[layer_idx] = max(self._hyst_avg[layer_idx], effective)
                        scale = torch.full_like(proj, self._hyst_avg[layer_idx]) * gain

                elif self.gate_mode == "swiglu":
                    # SwiGLU-inspired gate: silu(ratio) * sigmoid(ratio).
                    # Double gating — both must agree the token is refusing.
                    # silu(x) = x * sigmoid(x). So silu(ratio) * sigmoid(ratio)
                    # = ratio * sigmoid(ratio)^2. This is more selective than SiLU
                    # alone: it suppresses weak signals harder (sigmoid^2 decays
                    # faster) but passes strong signals cleanly (sigmoid→1).
                    # The projection magnitude (proj) scales the output, so tokens
                    # with larger refusal signals get proportionally more steering.
                    silu_gate = F.silu(ratio).clamp(min=0, max=1.0)
                    sig_gate = torch.sigmoid(ratio)
                    scale = silu_gate * sig_gate * gain

                else:
                    raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

                # Ensure scale has at least 1 dimension for downstream unsqueeze(1)
                if scale.dim() == 0:
                    scale = scale.unsqueeze(0)

                # Track token counter for two-phase gate (count once per token, not per layer)
                if self.target_layers and layer_idx == self.target_layers[0]:
                    self._token_counter += 1

                # Use raw (unnormalized) steering vector for perturbation.
                # The SiLU gate (clamped [0,1]) controls when and how much to steer;
                # the vector norm encodes the refusal signal strength at each layer.
                # With safe layers (norm < 6), the max norm ratio is ~2x (vs 20x at
                # deep layers L28/L30), so no single layer dominates.
                hidden_states = _apply_last_token_perturbation(
                    hidden_states, scale.unsqueeze(1) * steering_vec
                )

                # Record scale — keep on CUDA to avoid a forced sync on every token.
                # Callers that inspect last_scale can .cpu() at read time.
                self.last_scale[layer_idx] = scale.detach()

                # Trace buffer for diagnostics (only when enable_trace=True)
                if self.enable_trace:
                    if layer_idx not in self._trace_buffer:
                        self._trace_buffer[layer_idx] = []
                    self._trace_buffer[layer_idx].append(
                        (proj.item(), scale.item(), self._token_counter)
                    )
            else:
                # RAS temporal gate: count tokens once per forward pass (on the
                # first-firing layer) and pass through unsteered once the window
                # has closed.
                if self.steer_first_k is not None:
                    if layer_idx == self._first_fire_layer:
                        self._token_counter += 1
                    if self._token_counter > self.steer_first_k:
                        return output

                if self.rank > 1:
                    layer_vecs = self.steering_vectors[layer_idx]  # [rank, hidden_size]
                    # Support per-layer alpha for multi-rank: dict[layer_idx → list] or list
                    if isinstance(self.alpha, dict):
                        alphas = self.alpha[layer_idx]
                        if not isinstance(alphas, list):
                            alphas = [alphas] * self.rank
                    elif isinstance(self.alpha, list):
                        alphas = self.alpha
                    else:
                        alphas = [self.alpha] * self.rank
                    perturbation = torch.zeros_like(hidden_states[:, -1:, :])
                    for r in range(self.rank):
                        sv = layer_vecs[r]
                        a = alphas[r] if r < len(alphas) else alphas[-1]
                        perturbation = perturbation + a * sv
                    hidden_states = _apply_last_token_perturbation(hidden_states, perturbation)
                    # Track effective perturbation norm for diagnostics
                    self.last_scale[layer_idx] = perturbation[:, -1, :].norm().detach()
                else:
                    steering_vec = self.steering_vectors[layer_idx]
                    # Squeeze rank-1 stored as 3D [1, hidden_size] → [hidden_size]
                    if steering_vec.ndim == 2 and steering_vec.shape[0] == 1:
                        steering_vec = steering_vec.squeeze(0)
                    a = self._get_layer_param(self.alpha, layer_idx)
                    pert = a * steering_vec
                    if self.steer_prefill and hidden_states.shape[1] > 1:
                        # Prefill pass: apply to all positions so KV cache is built
                        # from steered residuals. pert [hidden] broadcasts over [batch, seq, hidden].
                        hidden_states = hidden_states + pert
                    else:
                        hidden_states = _apply_last_token_perturbation(hidden_states, pert)
                    # Track effective perturbation norm for diagnostics
                    self.last_scale[layer_idx] = pert.norm().detach()

            # Post-steering guard: if the perturbation itself produced NaN/Inf,
            # return the original unsteered output to prevent cascading into a
            # CUDA device-side assert in torch.multinomial.
            if not torch.isfinite(hidden_states[:, -1:, :]).all():
                return output

            # Return in same format as input
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on target layers (or their submodules)."""
        from .extraction import _get_attn_submodule
        from .utils import get_model_layers

        layers = get_model_layers(self.model)
        for layer_idx in self.target_layers:
            layer = layers[layer_idx]
            if self.component == "attn":
                target = _get_attn_submodule(layer)
            elif self.component == "mlp":
                target = layer.mlp
            else:
                target = layer
            hook = target.register_forward_hook(self.create_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class SteeringHookGroup:
    """Manage multiple SteeringHook instances for multi-component steering.

    Enables simultaneous steering of attention and MLP outputs with
    independent steering vectors and parameters per component.

    Example:
        # Create separate hooks for attn and mlp
        attn_hook = SteeringHook(model, attn_vectors, layers, alpha=-2.0, component="attn")
        mlp_hook = SteeringHook(model, mlp_vectors, layers, alpha=-1.5, component="mlp")
        group = SteeringHookGroup([attn_hook, mlp_hook])
        group.register_hooks()
        # ... generate ...
        group.remove_hooks()
    """

    def __init__(self, hooks):
        """
        Args:
            hooks: List of SteeringHook instances to manage together.
        """
        self.hooks = list(hooks)

    def register_hooks(self):
        """Register forward hooks for all managed SteeringHook instances."""
        for hook in self.hooks:
            hook.register_hooks()

    def remove_hooks(self):
        """Remove all registered hooks from all managed instances."""
        for hook in self.hooks:
            hook.remove_hooks()

    @property
    def last_scale(self):
        """Aggregate last_scale from all hooks, keyed by (component, layer_idx)."""
        result = {}
        for hook in self.hooks:
            for layer_idx, scale in hook.last_scale.items():
                result[(hook.component, layer_idx)] = scale
        return result

    @classmethod
    def from_steering_data(
        cls,
        model,
        steering_data,
        target_layers,
        alpha=None,
        alpha_attn=None,
        alpha_mlp=None,
        dynamic=False,
        theta=None,
        gain=None,
        components=("attn", "mlp"),
    ):
        """Create a SteeringHookGroup from a steering vectors file dict.

        Convenience factory that loads component-specific vectors and creates
        one SteeringHook per component.

        Args:
            model: HuggingFace model instance
            steering_data: Dict from torch.load() of steering vectors file
            target_layers: List of layer indices to steer
            alpha: Shared alpha for all components (used if per-component not set)
            alpha_attn: Alpha override for attention component
            alpha_mlp: Alpha override for MLP component
            dynamic: If True, use dynamic steering mode
            theta: Theta for dynamic mode (single value or dict)
            gain: Gain for dynamic mode
            components: Tuple of components to steer (default: ("attn", "mlp"))

        Returns:
            SteeringHookGroup instance
        """
        hooks = []
        alpha_map = {"attn": alpha_attn, "mlp": alpha_mlp}

        for comp in components:
            # Resolve steering vectors for this component
            sv_key = f"steering_vectors_{comp}"
            if sv_key in steering_data:
                sv = steering_data[sv_key]
            elif "steering_vectors" in steering_data:
                sv = steering_data["steering_vectors"]
            else:
                raise ValueError(
                    f"No steering vectors found for component '{comp}'. "
                    f"Expected key '{sv_key}' or 'steering_vectors'."
                )

            # Resolve alpha for this component
            comp_alpha = alpha_map.get(comp) if alpha_map.get(comp) is not None else alpha

            if not dynamic and comp_alpha is None:
                raise ValueError(
                    f"No alpha provided for component '{comp}'. "
                    f"Provide 'alpha' (shared) or 'alpha_{comp}' (per-component)."
                )

            hook_kwargs = dict(
                model=model,
                steering_vectors=sv,
                target_layers=target_layers,
                component=comp,
            )
            if dynamic:
                hook_kwargs.update(dynamic=True, theta=theta, gain=gain)
            else:
                hook_kwargs["alpha"] = comp_alpha

            hooks.append(SteeringHook(**hook_kwargs))

        return cls(hooks)


class CentroidSteeringHook:
    """Multi-principle steering with centroid-distance gating.

    Implements Li et al. (2026) "Chain of Risk" adaptive gating: for each safety
    principle (category), compute L2 distances from the current hidden state to
    safe (compliant) and unsafe (refused) centroids. Activate steering only for
    categories where the hidden state is closer to the unsafe centroid.

    The gate computes:
        g_k = ||h - mu_safe||_2  -  ||h - mu_unsafe||_2
    Principle k is active when g_k > delta (default delta=0).

    The perturbation is:
        h' = h + alpha * sum_{k active} v_k

    where v_k is the unit direction from the unsafe centroid toward the safe centroid.

    The distance computation is optimized to a single matmul:
        g_k = 2 * h @ (mu_unsafe - mu_safe) + (||mu_safe||^2 - ||mu_unsafe||^2)
    The bias term is precomputed once at construction.

    Args:
        model: HuggingFace model instance.
        centroids_safe: Tensor [K, n_layers, hidden_size] of safe (compliant) centroids.
        centroids_unsafe: Tensor [K, n_layers, hidden_size] of unsafe (refused) centroids.
        directions: Tensor [K, n_layers, hidden_size] of unit steering directions
            (from unsafe toward safe: (mu_safe - mu_unsafe) / norm).
        category_keys: List of K category name strings.
        target_layers: List of layer indices to steer.
        alpha: Steering strength per active principle (default 2.0).
        gate_delta: Margin threshold for gate activation (default 0.0).
            Principle k activates when g_k > gate_delta.
        relative_alpha: If True, scale alpha by 1/||h|| (default False).
        component: Which submodule to hook: "layer", "mlp", or "attn".
        apply_mode: "per_token" applies on every generated token; "prefill_only"
            applies only on the first forward pass (prompt encoding).
        max_active: Cap on number of simultaneously active principles (None = all).
    """

    def __init__(
        self,
        model,
        centroids_safe,
        centroids_unsafe,
        directions,
        category_keys,
        target_layers,
        alpha=2.0,
        gate_delta=0.0,
        relative_alpha=False,
        component="layer",
        apply_mode="per_token",
        max_active=None,
    ):
        self.model = model
        self.component = component
        self.target_layers = list(target_layers)
        self.alpha = alpha
        self.gate_delta = gate_delta
        self.relative_alpha = relative_alpha
        self.apply_mode = apply_mode
        self.max_active = max_active
        self.category_keys = list(category_keys)
        self.K = len(category_keys)
        self.hooks = []
        self.last_scale = {}
        self._token_counter = 0
        self.enable_trace = False
        self._trace_buffer = {}

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Store centroids and directions, sliced to target layers only
        # Keep on device in float32 for distance computation
        self._centroids_safe = centroids_safe.float().to(device)
        self._centroids_unsafe = centroids_unsafe.float().to(device)
        self._directions = directions.float().to(device)

    def _create_hook(self, layer_idx):
        """Create a forward hook for a specific layer with centroid-distance gating."""
        # Pre-slice centroids and directions for this layer
        c_safe = self._centroids_safe[:, layer_idx, :]  # [K, H]
        c_unsafe = self._centroids_unsafe[:, layer_idx, :]  # [K, H]
        dirs = self._directions[:, layer_idx, :]  # [K, H]

        # Precompute bias for efficient distance comparison:
        # g_k = 2 * h @ (c_unsafe - c_safe) + (||c_safe||^2 - ||c_unsafe||^2)
        diff = c_unsafe - c_safe  # [K, H]
        bias = c_safe.pow(2).sum(dim=-1) - c_unsafe.pow(2).sum(dim=-1)  # [K]
        gate_delta = self.gate_delta
        alpha = self.alpha
        relative_alpha = self.relative_alpha
        max_active = self.max_active
        apply_mode = self.apply_mode
        hook_ref = self  # closure reference

        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output

            # Skip on prefill_only mode after first forward pass
            if apply_mode == "prefill_only" and hook_ref._token_counter > 0:
                return output

            # NaN guard
            if not torch.isfinite(hidden_states[:, -1:, :]).all():
                return output

            h = hidden_states[:, -1, :].float()  # [batch, H]

            # Compute gate values via matmul: g = 2 * h @ diff.T + bias
            # g[k] > 0 means closer to unsafe centroid -> activate
            g = 2.0 * (h @ diff.T) + bias.unsqueeze(0)  # [batch, K]

            # Binary activation
            active = (g > gate_delta).float()  # [batch, K]

            # Optional: cap number of active principles (keep strongest gates)
            if max_active is not None and max_active < hook_ref.K:
                # Zero out all but top-max_active gates per batch element
                g_masked = g * active
                if g_masked.shape[0] == 1:
                    _, top_idx = g_masked.squeeze(0).topk(min(max_active, hook_ref.K))
                    mask = torch.zeros_like(active.squeeze(0))
                    mask[top_idx] = 1.0
                    active = mask.unsqueeze(0)
                else:
                    # Per batch element
                    _, top_idx = g_masked.topk(min(max_active, hook_ref.K), dim=-1)
                    mask = torch.zeros_like(active)
                    mask.scatter_(1, top_idx, 1.0)
                    active = mask

            # Sum active directions
            n_active = active.sum(dim=-1)  # [batch]
            perturbation = active @ dirs  # [batch, H]

            # Scale by alpha
            if relative_alpha:
                h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1.0)
                alpha_eff = alpha / h_norm
            else:
                alpha_eff = alpha

            perturbation = alpha_eff * perturbation  # [batch, H]

            # Cast perturbation back to hidden_states dtype (bfloat16)
            perturbation = perturbation.to(hidden_states.dtype)

            # Apply to last token position
            hidden_states = _apply_last_token_perturbation(hidden_states, perturbation.unsqueeze(1))

            # Track diagnostics
            hook_ref.last_scale[layer_idx] = perturbation.detach()

            if hook_ref.enable_trace:
                if layer_idx not in hook_ref._trace_buffer:
                    hook_ref._trace_buffer[layer_idx] = []
                hook_ref._trace_buffer[layer_idx].append(
                    {
                        "n_active": n_active.item() if n_active.numel() == 1 else n_active.tolist(),
                        "active_cats": [
                            hook_ref.category_keys[k]
                            for k in range(hook_ref.K)
                            if (active[0, k] > 0.5 if active.dim() > 1 else active[k] > 0.5)
                        ],
                        "gate_values": g.detach().cpu(),
                        "token_idx": hook_ref._token_counter,
                        "pert_norm": perturbation.norm().item(),
                    }
                )

            # Increment token counter on first target layer only
            if hook_ref.target_layers and layer_idx == hook_ref.target_layers[0]:
                hook_ref._token_counter += 1

            # NaN guard on output
            if not torch.isfinite(hidden_states[:, -1:, :]).all():
                return output

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on target layers."""
        from .extraction import _get_attn_submodule
        from .utils import get_model_layers

        layers = get_model_layers(self.model)
        for layer_idx in self.target_layers:
            layer = layers[layer_idx]
            if self.component == "attn":
                target = _get_attn_submodule(layer)
            elif self.component == "mlp":
                target = layer.mlp
            else:
                target = layer
            hook = target.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset_token_state(self):
        """Reset per-generation state."""
        self._token_counter = 0
        self._trace_buffer = {}


def load_best_layers_from_correlations(correlation_file, top_k=1):
    """
    Load best layers from layer_correlations.json.

    Args:
        correlation_file: Path to correlation JSON file
        top_k: Number of top layers to load

    Returns:
        List of best layer indices, or None if file not found
    """
    if not os.path.exists(correlation_file):
        print(f"[WARN]  Correlation file not found: {correlation_file}")
        print(f"   Run find_best_layers.py first!")
        return None

    with open(correlation_file) as f:
        data = json.load(f)

    best_layers = data["best_layers"][:top_k]

    print(f"[INFO] Loaded from {correlation_file}:")
    print(f"   Top {top_k} layer(s): {best_layers}")

    all_corrs = {c["layer"]: c["correlation"] for c in data["all_correlations"]}
    for layer in best_layers:
        print(f"   Layer {layer}: r={all_corrs[layer]:+.4f}")

    return best_layers


def compute_dynamic_params(
    activations_file, steering_vectors_file, target_layers, component="attn"
):
    """
    Compute theta for dynamic steering mode.

    Projects refused prompt activations onto the unit-normalized steering vector
    and computes theta = -median(projections). The hook also normalizes the
    steering vector at runtime, so theta and runtime projections are consistent.

    Args:
        activations_file: Path to activations .pt file
        steering_vectors_file: Path to steering vectors .pt file
        target_layers: Layer index (int) or list of layer indices to compute
            parameters for. When multiple layers are provided, returns a
            per-layer dict suitable for SteeringHook.
        component: Which activation component to use (default: "attn")

    Returns:
        theta value(s). If target_layers is a single int, returns a scalar float.
        If target_layers is a list, returns a dict mapping layer_idx -> float.
    """
    act_data = torch.load(activations_file, weights_only=True)
    labels = act_data["labels"]  # [N]

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
        steering_vectors = vec_data["steering_vectors"]
    saved_rank = vec_data.get("rank", 1)
    if saved_rank > 1 or steering_vectors.ndim == 3:
        raise ValueError(
            f"compute_dynamic_params only supports rank-1 steering vectors, "
            f"but file contains rank={saved_rank} vectors with shape {steering_vectors.shape}. "
            f"Use rank=1 when computing vectors for dynamic mode."
        )

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    refused_mask = labels == 1

    if refused_mask.sum() == 0:
        raise ValueError("No refused samples found in activations data")

    # Handle single layer vs list
    single_layer = isinstance(target_layers, int)
    if single_layer:
        target_layers = [target_layers]

    theta_dict = {}

    for layer_idx in target_layers:
        sv = steering_vectors[layer_idx].float()
        sv_norm = sv.norm().item()
        if sv_norm < 1e-8:
            raise ValueError(
                f"Steering vector at layer {layer_idx} has near-zero norm ({sv_norm:.2e}), "
                "cannot compute theta for dynamic mode"
            )

        # Normalize to unit vector (matching the hook)
        sv_unit = sv / sv_norm

        refused_acts = activations[refused_mask, layer_idx, :].float()

        # Project refused activations onto unit steering vector
        projections = refused_acts @ sv_unit

        # Compute statistics
        median = projections.median().item()
        q75 = projections.quantile(0.75).item()
        q25 = projections.quantile(0.25).item()
        iqr = q75 - q25

        theta_dict[layer_idx] = -median

        print(f"[INFO] Dynamic steering parameters (layer {layer_idx}):")
        print(f"   Steering vector norm: {sv_norm:.2f} (normalized to unit length)")
        print(f"   Refused samples: {refused_acts.shape[0]}")
        print(f"   Projection median: {median:.4f} → theta: {-median:.4f}")
        print(f"   Projection IQR: {iqr:.4f}")
        print(
            f"   gain=-1 → ~0.73 * {sv_norm:.2f} = {0.73 * sv_norm:.2f} norm perturbation for typical refused prompt"
        )

    if single_layer:
        return theta_dict[target_layers[0]]

    return theta_dict


def load_actual_refusal_prompts(results_dir, min_score=0.5, max_score=None, max_prompts=5):
    """
    Load prompts that actually refused in baseline evaluation.

    Args:
        results_dir: Path to LLM-Refusal-Evaluation results
        min_score: Minimum refusal score threshold
        max_score: Maximum refusal score threshold (for selecting borderline prompts)
        max_prompts: Maximum number of prompts to return

    Returns:
        List of dicts with 'prompt' and 'baseline_score'
    """
    import os

    refusal_prompts = []

    # Auto-discover directories with censor_scores.json
    if os.path.isdir(results_dir):
        for entry in os.listdir(results_dir):
            censor_file = os.path.join(results_dir, entry, "censor_scores.json")
            if os.path.exists(censor_file):
                with open(censor_file) as f:
                    data = json.load(f)

                for item in data:
                    score = item["answer_censor_score"]
                    if score > min_score and (max_score is None or score <= max_score):
                        refusal_prompts.append(
                            {"prompt": item["prompt"], "baseline_score": score, "source": entry}
                        )

    if not refusal_prompts:
        print(f"[WARN] No censor_scores.json files found in {results_dir}")
        return []

    refusal_prompts.sort(key=lambda x: x["baseline_score"], reverse=True)
    return refusal_prompts[:max_prompts]


def test_single_layer_configuration(
    model,
    tokenizer,
    steering_vectors,
    layer,
    alpha=None,
    test_prompts=None,
    max_new_tokens=300,
    dynamic=False,
    theta=None,
    gain=None,
    component="attn",
    steering_data=None,
):
    """
    Test steering with a single layer configuration.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        steering_vectors: Steering vectors tensor (unnormalized)
        layer: Layer index or list of indices
        alpha: Steering coefficient (required when dynamic=False)
        test_prompts: List of prompt dicts
        max_new_tokens: Max tokens to generate
        dynamic: If True, use dynamic scaling mode
        theta: Scaling parameter for dynamic mode
        gain: Gain multiplier for dynamic mode (negative = reduce refusal)
        component: "attn", "mlp", "layer", or "attn+mlp"
        steering_data: Full steering data dict (required for attn+mlp)
    """
    # Normalize to list
    if isinstance(layer, int):
        layers = [layer]
        layer_desc = f"Layer {layer}"
    else:
        layers = layer
        layer_desc = f"Layers {layers}"

    if dynamic:
        gain_val = gain if gain is not None else -1.0
        if isinstance(theta, dict):
            mode_desc = f"dynamic: per-layer theta ({len(theta)} layers), gain={gain_val}"
        else:
            mode_desc = f"dynamic: theta={theta:.4f}, gain={gain_val}"
    else:
        mode_desc = f"alpha={alpha}"

    print(f"\n{'='*80}")
    print(f"TESTING: {layer_desc} ({mode_desc})")
    print(f"{'='*80}\n")

    for i, prompt_data in enumerate(test_prompts):
        prompt = prompt_data["prompt"]
        baseline_score = prompt_data["baseline_score"]

        print(f"{'-'*80}")
        print(f"Prompt {i+1}/{len(test_prompts)}")
        print(f"{'-'*80}")
        print(f" FULL PROMPT:\n{prompt}")
        print(f"\n[INFO] Baseline refusal score: {baseline_score:.2f}")
        print(f"{'-'*80}")

        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # BASELINE
        print("\n[BASELINE] BASELINE OUTPUT:")
        print("-" * 80)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        baseline_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        baseline_response = baseline_output[len(formatted_prompt) :]

        print(baseline_response)
        print("-" * 80)

        # STEERED
        print(f"\n[STEERED] STEERED OUTPUT ({layer_desc}, {mode_desc}):")
        print("-" * 80)

        if component == "attn+mlp" and steering_data is not None:
            steerer = SteeringHookGroup.from_steering_data(
                model=model,
                steering_data=steering_data,
                target_layers=layers,
                alpha=alpha,
                dynamic=dynamic,
                theta=theta,
                gain=gain,
                components=("attn", "mlp"),
            )
        elif dynamic:
            steerer = SteeringHook(
                model,
                steering_vectors,
                layers,
                dynamic=True,
                theta=theta,
                gain=gain,
                component=component,
            )
        else:
            steerer = SteeringHook(
                model,
                steering_vectors,
                layers,
                alpha=alpha,
                component=component,
            )
        steerer.register_hooks()

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([SanitizeLogitsProcessor()]),
            )

        steered_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        steered_response = steered_output[len(formatted_prompt) :]

        print(steered_response)
        print("-" * 80)

        steerer.remove_hooks()

        print(f"\nStats:")
        print(f"   Baseline length: {len(baseline_response)} chars")
        print(f"   Steered length: {len(steered_response)} chars")
        print(f"   Difference: {len(steered_response) - len(baseline_response):+d} chars")
        if dynamic and steerer.last_scale:
            for lid, sval in sorted(steerer.last_scale.items()):
                sv = sval.item() if sval.numel() == 1 else sval.tolist()
                label = f"{lid[0]} layer {lid[1]}" if isinstance(lid, tuple) else f"layer {lid}"
                print(f"   Dynamic scale ({label}): {sv:.4f}")
        print()


def test_steering(
    model_name,
    steering_file,
    layer_configs,
    alpha=None,
    test_prompts=None,
    max_new_tokens=300,
    dynamic=False,
    theta=None,
    gain=None,
    component="attn",
    quantize=None,
):
    """
    Test steering across multiple layer configurations.

    Args:
        model_name: HuggingFace model identifier
        steering_file: Path to steering vectors file
        layer_configs: List of layer configurations (each is int or list of ints)
        alpha: Steering coefficient (required when dynamic=False)
        test_prompts: List of prompt dicts
        max_new_tokens: Max tokens to generate
        dynamic: If True, use dynamic scaling mode
        theta: Scaling parameter for dynamic mode
        gain: Gain multiplier for dynamic mode (negative = reduce refusal)
        component: "attn", "mlp", "layer", or "attn+mlp"
    """
    if dynamic:
        gain_val = gain if gain is not None else -1.0
        if isinstance(theta, dict):
            mode_str = f"dynamic (per-layer theta, {len(theta)} layers, gain={gain_val})"
        else:
            mode_str = f"dynamic (theta={theta:.4f}, gain={gain_val})"
    else:
        mode_str = f"alpha={alpha}"

    print(f"[COMPUTE] Testing Steering")
    print(f"   Model: {model_name}")
    print(f"   Steering file: {steering_file}")
    print(f"   Mode: {mode_str}")
    print(f"   Component: {component}")
    print(f"   Configurations to test: {len(layer_configs)}")
    print(f"   Test prompts: {len(test_prompts)}\n")

    # Load model (once)
    from .utils import load_model

    model, tokenizer = load_model(model_name, quantize=quantize)

    # Load steering vectors (prefer component-specific key)
    vec_data = torch.load(steering_file, weights_only=True)
    steering_data = vec_data if isinstance(vec_data, dict) else None

    if component == "attn+mlp":
        # For dual-component, use attn vectors for shape reporting
        if "steering_vectors_attn" in vec_data:
            steering_vectors = vec_data["steering_vectors_attn"]
        else:
            steering_vectors = vec_data["steering_vectors"]
    else:
        sv_key = f"steering_vectors_{component}"
        if sv_key in vec_data:
            steering_vectors = vec_data[sv_key]
        else:
            steering_vectors = vec_data["steering_vectors"]

    print(f"[OK] Model loaded: {model.config.num_hidden_layers} layers")
    print(f"[OK] Steering vectors: {steering_vectors.shape}\n")

    # Test each layer configuration
    for config in layer_configs:
        test_single_layer_configuration(
            model,
            tokenizer,
            steering_vectors,
            config,
            alpha=alpha,
            test_prompts=test_prompts,
            max_new_tokens=max_new_tokens,
            dynamic=dynamic,
            theta=theta,
            gain=gain,
            component=component,
            steering_data=steering_data,
        )
