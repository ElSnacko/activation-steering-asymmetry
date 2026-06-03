"""
Dynamic SiLU-gated steering layer/submodule wrappers.

Wraps a decoder layer or submodule (attention, MLP) to apply
input-dependent steering at runtime. Unlike fixed-alpha steering
(which can be merged into output projection biases), dynamic
steering requires actual computation per forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .steering import _apply_last_token_perturbation


class DynamicSteeringLayer(nn.Module):
    """
    Wraps a transformer decoder layer with SiLU-gated dynamic steering.

    Applies: hidden_states += gain * clamp(SiLU(-proj/theta), min=0) * steering_vector

    This is the same math as SteeringHook.create_hook() but as a module wrapper
    suitable for saving/loading models.

    Args:
        base_layer: The original decoder layer module
        steering_vector: Steering vector for this layer [hidden_size]
        theta: SiLU scaling parameter (normalizes projection)
        gain: Multiplier (negative = reduce refusal)
    """

    def __init__(self, base_layer, steering_vector, theta, gain):
        super().__init__()
        self.base_layer = base_layer
        self.register_buffer("steering_vector", steering_vector)
        sv_norm = steering_vector.norm().clamp(min=1e-8)
        self.register_buffer("sv_unit", steering_vector / sv_norm)
        self.theta = theta
        self.gain = gain

    def __getattr__(self, name):
        """Delegate attribute lookups to base_layer for model-specific fields.

        Some model architectures (e.g. Qwen3) access layer attributes like
        `attention_type` during the forward loop. This ensures those lookups
        find the base layer's attributes transparently.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(self, *args, **kwargs):
        output = self.base_layer(*args, **kwargs)

        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        proj = hidden_states[:, -1, :] @ self.sv_unit  # (batch,)

        # Guard against theta=0 (unlikely but possible if median projection is exactly 0)
        theta_safe = self.theta if abs(self.theta) > 1e-8 else 1.0
        # SiLU activation clamped to [0, 1] to prevent unbounded scale when theta
        # is near zero.  Matches the clamp in steering.py:193.
        raw_silu = F.silu(-proj / theta_safe)
        scale = raw_silu.clamp(min=0, max=1.0) * self.gain  # (batch,)

        hidden_states = _apply_last_token_perturbation(
            hidden_states, scale.unsqueeze(1) * self.steering_vector
        )

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


class DynamicSteeringSubmodule(nn.Module):
    """
    Wraps a submodule (self_attn or mlp) with SiLU-gated dynamic steering.

    Same math as DynamicSteeringLayer but targets a specific submodule within
    a decoder layer rather than the entire layer. This allows component-specific
    dynamic steering (e.g. steering only the attention output).

    Args:
        base_module: The original submodule (e.g. layer.self_attn or layer.mlp)
        steering_vector: Steering vector for this layer [hidden_size]
        theta: SiLU scaling parameter (normalizes projection)
        gain: Multiplier (negative = reduce refusal)
    """

    def __init__(self, base_module, steering_vector, theta, gain):
        super().__init__()
        self.base_module = base_module
        self.register_buffer("steering_vector", steering_vector)
        sv_norm = steering_vector.norm().clamp(min=1e-8)
        self.register_buffer("sv_unit", steering_vector / sv_norm)
        self.theta = theta
        self.gain = gain

    def __getattr__(self, name):
        """Delegate attribute lookups to base_module transparently."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_module, name)

    def forward(self, *args, **kwargs):
        output = self.base_module(*args, **kwargs)

        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        proj = hidden_states[:, -1, :] @ self.sv_unit  # (batch,)

        # Guard against theta=0 (unlikely but possible if median projection is exactly 0)
        theta_safe = self.theta if abs(self.theta) > 1e-8 else 1.0
        # SiLU activation clamped to [0, 1] to prevent unbounded scale when theta
        # is near zero.  Matches the clamp in steering.py:193.
        raw_silu = F.silu(-proj / theta_safe)
        scale = raw_silu.clamp(min=0, max=1.0) * self.gain  # (batch,)

        hidden_states = _apply_last_token_perturbation(
            hidden_states, scale.unsqueeze(1) * self.steering_vector
        )

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
