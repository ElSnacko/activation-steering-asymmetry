"""
Random-direction control for Finding 1: KL asymmetry.

Tests whether the refusal steering direction is geometrically privileged
(low KL divergence) relative to arbitrary random directions in activation space,
or whether compliance steering is just one of many high-KL directions.

For each target layer:
  1. Compute KL on the capability probe set for the actual steering vector
     at each test alpha (compliance and refusal directions)
  2. Sample N random unit vectors, scale to the same norm as the steering vector,
     apply at each test alpha, compute KL
  3. Report: steering KL vs. distribution of random-direction KL

If the refusal direction KL is in the lower tail of the random distribution,
the direction is geometrically privileged. If compliance KL is in the upper tail,
it is unusually disruptive. Either finding transforms Finding 1 from an observation
into a structural claim.

Cost: (N_random + 2) forward passes × n_probes × n_alphas × n_layers.
With N=50, 100 probes, 3 alphas, 4 layers this is ~600 forward passes — same
cost as a single optimize_alpha trial.

Usage:
    python scripts/random_direction_control.py \\
      --model /path/to/model \\
      --steering-vectors outputs/.../steering_vectors_md_mlp.pt \\
      --layers 17 24 27 16 --component mlp \\
      --alphas -2.0 2.0 \\
      --n-random 50 \\
      --output-dir outputs/.../random_direction_control
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Random-direction KL control")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument(
        "--steering-vectors",
        required=True,
        help="Path to steering vectors .pt file (from compute_wrmd.py)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Target layers to test",
    )
    parser.add_argument(
        "--component",
        default="mlp",
        choices=["mlp", "attn", "residual"],
        help="Model component the steering vectors were extracted from (default: mlp)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[-2.0, 2.0],
        help="Alpha values to test (default: -2.0 2.0). Include both directions.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=50,
        help="Number of random directions to sample per layer (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for random direction sampling (default: 42)",
    )
    parser.add_argument(
        "--probe-set",
        default="data/capability_questions.json",
        help="Capability probe set JSON (default: data/capability_questions.json)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for results JSON and plot",
    )
    parser.add_argument(
        "--kl-method",
        default="first_token",
        choices=["first_token", "teacher_forced"],
        help="KL method (default: first_token; faster, sufficient for comparison)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode in chat template (default: off)",
    )
    return parser.parse_args()


def collect_first_token_kl(model, tokenizer, prompts, hook_fn=None):
    """Compute mean KL vs. baseline for a list of formatted prompts.

    hook_fn: optional callable that registers hooks before forward pass and
    returns a handle list to remove afterwards. If None, baseline logits are returned.
    """
    from torch.nn.functional import kl_div, log_softmax, softmax

    device = next(model.parameters()).device
    all_logits = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            if hook_fn:
                handles = hook_fn()
            out = model(**inputs)
            if hook_fn:
                for h in handles:
                    h.remove()
            # first new token logits
            logits = out.logits[0, -1, :].float()
            all_logits.append(logits)

    return all_logits  # list of [vocab_size] tensors


def kl_from_logits(base_logits_list, steered_logits_list):
    """Mean KL(base || steered) over a list of logit tensors."""
    from torch.nn.functional import kl_div, log_softmax, softmax

    kls = []
    for base, steered in zip(base_logits_list, steered_logits_list):
        p = softmax(base, dim=-1)
        log_q = log_softmax(steered, dim=-1)
        kl = kl_div(log_q, p, reduction="sum").item()
        kls.append(kl)
    return float(sum(kls) / len(kls)) if kls else 0.0


def make_steering_hook_fn(model, vector, layer_idx, alpha, component):
    """Return a function that registers a steering hook and returns handles."""

    def get_hook(vec, a):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                hidden = hidden + a * vec.to(hidden.device, dtype=hidden.dtype)
                return (hidden,) + output[1:]
            else:
                return output + a * vec.to(output.device, dtype=output.dtype)

        return hook

    def hook_fn():
        target = _get_component_module(model, layer_idx, component)
        h = target.register_forward_hook(get_hook(vector, alpha))
        return [h]

    return hook_fn


def _get_component_module(model, layer_idx, component):
    """Get the model submodule for the given layer and component."""
    layers = None
    for name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        parts = name.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            layers = obj
            break
        except AttributeError:
            continue

    if layers is None:
        raise ValueError("Could not find layer list in model")

    layer = layers[layer_idx]

    if component == "mlp":
        for attr in ["mlp", "feed_forward"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
    elif component == "attn":
        for attr in ["self_attn", "attention", "attn"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
    elif component == "residual":
        return layer

    raise ValueError(f"Could not find {component} in layer {layer_idx}")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from activation_steering.capability import load_capability_probe_set
    from activation_steering.utils import load_model

    print("Loading model...")
    model, tokenizer = load_model(args.model)
    model.eval()

    # Build probe prompts
    print(f"Loading capability probe set from {args.probe_set}...")
    _, probe_prompts = load_capability_probe_set(
        path=args.probe_set,
        n_per_category=10,
        seed=42,
        tokenizer=tokenizer,
        enable_thinking=args.enable_thinking,
    )
    print(f"  {len(probe_prompts)} probe prompts")

    # Load steering vectors
    print(f"Loading steering vectors from {args.steering_vectors}...")
    sv_data = torch.load(args.steering_vectors, map_location="cpu")
    # sv_data may be a metadata dict with 'steering_vectors' / 'steering_vectors_<component>'
    # keys alongside non-integer metadata keys (num_layers, hidden_size, etc.)
    if isinstance(sv_data, dict):
        component = getattr(args, "component", "mlp")
        sv_key = f"steering_vectors_{component}"
        if sv_key in sv_data:
            sv_tensor = sv_data[sv_key]
        elif "steering_vectors" in sv_data:
            sv_tensor = sv_data["steering_vectors"]
        else:
            sv_tensor = sv_data
        if isinstance(sv_tensor, torch.Tensor):
            steering_by_layer = {i: sv_tensor[i].float() for i in range(sv_tensor.shape[0])}
        else:
            steering_by_layer = {int(k): v.float() for k, v in sv_tensor.items()}
    elif isinstance(sv_data, torch.Tensor):
        steering_by_layer = {i: sv_data[i].float() for i in range(sv_data.shape[0])}
    else:
        raise ValueError(f"Unexpected steering vector format: {type(sv_data)}")

    print("\nCollecting baseline logits (no steering)...")
    baseline_logits = collect_first_token_kl(model, tokenizer, probe_prompts)

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    results = {}

    for layer_idx in args.layers:
        if layer_idx not in steering_by_layer:
            print(f"  WARNING: layer {layer_idx} not in steering vectors, skipping")
            continue

        sv = steering_by_layer[layer_idx]
        sv_norm = sv.norm().item()
        sv_unit = sv / sv_norm  # unit vector in steering direction

        print(f"\n=== Layer {layer_idx} (||v|| = {sv_norm:.2f}) ===")
        layer_results = {"layer": layer_idx, "sv_norm": sv_norm, "alphas": {}}

        for alpha in args.alphas:
            print(f"  alpha={alpha:+.2f}")
            alpha_results = {}

            # --- Steering direction KL ---
            hook_fn = make_steering_hook_fn(model, sv_unit, layer_idx, alpha, args.component)
            steered_logits = collect_first_token_kl(model, tokenizer, probe_prompts, hook_fn)
            sv_kl = kl_from_logits(baseline_logits, steered_logits)
            alpha_results["steering_kl"] = sv_kl
            print(f"    Steering direction KL: {sv_kl:.4f}")

            # --- Random direction KL distribution ---
            random_kls = []
            for i in range(args.n_random):
                rand_vec = torch.randn(sv.shape, generator=rng)
                rand_unit = rand_vec / rand_vec.norm()
                hook_fn_r = make_steering_hook_fn(
                    model, rand_unit, layer_idx, alpha * sv_norm, args.component
                )
                r_logits = collect_first_token_kl(model, tokenizer, probe_prompts, hook_fn_r)
                random_kls.append(kl_from_logits(baseline_logits, r_logits))

            random_kls_sorted = sorted(random_kls)
            mean_kl = sum(random_kls) / len(random_kls)
            p5 = random_kls_sorted[int(0.05 * len(random_kls_sorted))]
            p95 = random_kls_sorted[int(0.95 * len(random_kls_sorted))]
            pct_below = sum(k < sv_kl for k in random_kls) / len(random_kls)

            alpha_results["random_kl_mean"] = mean_kl
            alpha_results["random_kl_p5"] = p5
            alpha_results["random_kl_p95"] = p95
            alpha_results["random_kl_all"] = random_kls
            # Percentile of steering direction within random distribution
            alpha_results["steering_pct_rank"] = pct_below

            print(
                f"    Random KL: mean={mean_kl:.4f} p5={p5:.4f} p95={p95:.4f}  "
                f"| steering at {100*pct_below:.0f}th pct of random"
            )

            layer_results["alphas"][str(alpha)] = alpha_results

        results[str(layer_idx)] = layer_results

    # Save results
    out_file = Path(args.output_dir) / "random_direction_control.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Layer':<8} {'Alpha':>8} {'Steering KL':>12} {'Random mean':>12} {'Pct rank':>10}")
    for layer_str, lr in results.items():
        for alpha_str, ar in lr["alphas"].items():
            print(
                f"{layer_str:<8} {alpha_str:>8} {ar['steering_kl']:>12.4f} "
                f"{ar['random_kl_mean']:>12.4f} {100*ar['steering_pct_rank']:>9.0f}%"
            )

    _try_plot(results, args.output_dir)


def _try_plot(results, output_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        layers = sorted(results.keys(), key=int)
        alphas = sorted(results[layers[0]]["alphas"].keys(), key=float)
        n_alphas = len(alphas)

        fig, axes = plt.subplots(len(layers), n_alphas, figsize=(5 * n_alphas, 4 * len(layers)))
        if len(layers) == 1:
            axes = [axes]
        if n_alphas == 1:
            axes = [[ax] for ax in axes]

        for i, layer_str in enumerate(layers):
            for j, alpha_str in enumerate(alphas):
                ax = axes[i][j]
                ar = results[layer_str]["alphas"][alpha_str]
                rand_kls = ar["random_kl_all"]
                sv_kl = ar["steering_kl"]

                ax.hist(rand_kls, bins=20, alpha=0.7, color="steelblue", label="Random directions")
                ax.axvline(sv_kl, color="red", linewidth=2, label=f"Steering vec KL={sv_kl:.3f}")
                ax.set_title(f"Layer {layer_str}, α={alpha_str}")
                ax.set_xlabel("KL divergence")
                ax.set_ylabel("Count")
                ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = Path(output_dir) / "random_direction_kl.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
