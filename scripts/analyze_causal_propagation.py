"""
Causal propagation analysis for activation steering.

For each source layer s in the target set:
  - Apply steering at s only (single-layer hook, fixed alpha)
  - Capture hidden states at ALL layers for both baseline and steered runs
  - Compute delta_h_l = h_steered_l - h_baseline_l for each downstream layer l
  - Report per-layer:
      cos_sim(delta_h_l, v_l)   -- alignment with steering direction at layer l
      cos_sim(delta_h_l, v_s)   -- alignment with source layer's vector (rotation measure)
      norm_ratio                 -- ||delta_h_l|| / ||alpha * v_s||

This tells us whether a perturbation injected at layer s travels downstream in
the same direction as the refusal/comply axis at each subsequent layer, is
rotated orthogonally, or is canceled.

Usage:
    python scripts/analyze_causal_propagation.py \\
      --model mistralai/Mistral-7B-Instruct-v0.2 \\
      --steering-vectors outputs/.../steering_vectors_md_mlp.pt \\
      --correlations outputs/.../layer_correlations_mlp.json \\
      --source-layers 16 21 23 29 \\
      --alpha -1.34 \\
      --component mlp \\
      --n-prompts 50 \\
      --output-dir outputs/.../causal_propagation
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from activation_steering.capability import load_capability_probe_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--steering-vectors", required=True, help=".pt file with per-layer steering vectors"
    )
    p.add_argument(
        "--correlations", required=True, help="layer_correlations JSON for best_layers list"
    )
    p.add_argument(
        "--source-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to steer one at a time. Defaults to top-4 best layers.",
    )
    p.add_argument("--alpha", type=float, default=-2.0, help="Steering alpha (negative = comply)")
    p.add_argument("--component", default="mlp", choices=["mlp", "attn"])
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--probe-set", default="data/capability_questions.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def _get_component_module(model, layer_idx, component):
    layers = None
    for name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        try:
            for p in name.split("."):
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
    raise ValueError(f"Component {component} not found at layer {layer_idx}")


def get_num_layers(model):
    for name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        try:
            for p in name.split("."):
                obj = getattr(obj, p)
            return len(obj)
        except AttributeError:
            continue
    raise ValueError("Could not determine number of layers")


def capture_hidden_states(
    model, tokenizer, prompts, component, num_layers, steer_layer=None, steer_vec=None, alpha=None
):
    """
    Run forward passes on prompts, capturing last-token hidden states at
    every layer's component output.

    If steer_layer is set, applies alpha * steer_vec at that layer.
    The steer hook is registered BEFORE capture hooks so the capture at the
    source layer sees the post-steer value.

    Returns: dict {layer_idx: tensor [n_prompts, hidden_dim]}
    """
    captured = {l: [] for l in range(num_layers)}
    handles = []

    # Steer hook registered FIRST so it fires before the capture hook at the source layer
    steer_handle = None
    if steer_layer is not None:

        def steer_hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            v = steer_vec.to(h.device, dtype=h.dtype)
            h = h + alpha * v
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        mod = _get_component_module(model, steer_layer, component)
        steer_handle = mod.register_forward_hook(steer_hook)

    # Capture hooks at all layers (registered AFTER steer hook)
    def make_capture(layer_idx):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx].append(h[:, -1, :].detach().float().cpu())

        return hook

    for l in range(num_layers):
        mod = _get_component_module(model, l, component)
        handles.append(mod.register_forward_hook(make_capture(l)))

    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            model(**enc)

    for h in handles:
        h.remove()
    if steer_handle:
        steer_handle.remove()

    return {l: torch.stack(captured[l]) for l in range(num_layers)}


def cosine_sim(a, b):
    """Mean cosine similarity between rows of a [N,D] and vector b [D]."""
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b.unsqueeze(0), dim=-1)
    return (a_norm * b_norm).sum(-1).mean().item()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    num_layers = get_num_layers(model)
    print(f"  {num_layers} layers")

    # Load steering vectors: tensor [num_layers, hidden_dim] or dict
    print(f"Loading steering vectors from {args.steering_vectors}...")
    sv_data = torch.load(args.steering_vectors, map_location="cpu", weights_only=False)
    if isinstance(sv_data, dict) and "steering_vectors" in sv_data:
        # Standard format: dict with 'steering_vectors' tensor [num_layers, hidden_dim]
        sv_tensor = sv_data["steering_vectors"]
        steering_vectors = {l: sv_tensor[l].float() for l in range(sv_tensor.shape[0])}
    elif isinstance(sv_data, dict):
        # Legacy: keyed by layer index
        steering_vectors = {int(k): v.float() for k, v in sv_data.items() if str(k).isdigit()}
    else:
        # Raw tensor [num_layers, hidden_dim]
        steering_vectors = {l: sv_data[l].float() for l in range(sv_data.shape[0])}

    # Load source layers
    with open(args.correlations) as f:
        corr = json.load(f)
    best_layers = corr["best_layers"][:4]
    source_layers = args.source_layers if args.source_layers else best_layers
    print(f"Source layers: {source_layers}")
    print(f"Best layers from correlations: {best_layers}")

    # Load prompts
    _, prompts = load_capability_probe_set(
        path=args.probe_set, n_per_category=5, tokenizer=tokenizer
    )
    prompts = prompts[: args.n_prompts]
    print(f"Using {len(prompts)} prompts")

    # Baseline: no steering
    print("Running baseline forward passes...")
    baseline = capture_hidden_states(model, tokenizer, prompts, args.component, num_layers)

    results = {}

    for src in source_layers:
        if src not in steering_vectors:
            print(f"  Skipping layer {src}: no steering vector")
            continue

        v_src = steering_vectors[src]  # [hidden_dim]
        print(f"\nSource layer {src} (alpha={args.alpha})...")

        steered = capture_hidden_states(
            model,
            tokenizer,
            prompts,
            args.component,
            num_layers,
            steer_layer=src,
            steer_vec=v_src,
            alpha=args.alpha,
        )

        # Perturbation norm at source layer
        delta_src = steered[src] - baseline[src]  # [N, D]
        src_norm = delta_src.norm(dim=-1).mean().item()
        applied_norm = abs(args.alpha) * v_src.norm().item()

        layer_stats = {}
        for l in range(num_layers):
            if l < src:
                layer_stats[l] = {"upstream": True}
                continue
            delta = steered[l] - baseline[l]  # [N, D]
            norm = delta.norm(dim=-1).mean().item()

            # Alignment with steering vector at this layer
            v_l = steering_vectors.get(l)
            cos_vl = cosine_sim(delta, v_l.float()) if v_l is not None else None

            # Alignment with source vector
            cos_vsrc = cosine_sim(delta, v_src)

            # Norm ratio relative to applied perturbation
            norm_ratio = norm / applied_norm if applied_norm > 0 else 0.0

            layer_stats[l] = {
                "delta_norm": round(norm, 6),
                "norm_ratio": round(norm_ratio, 4),
                "cos_sim_v_l": round(cos_vl, 4) if cos_vl is not None else None,
                "cos_sim_v_src": round(cos_vsrc, 4),
                "is_source": l == src,
                "is_best_layer": l in best_layers,
            }

        results[src] = {
            "source_layer": src,
            "alpha": args.alpha,
            "applied_norm": round(applied_norm, 4),
            "src_delta_norm": round(src_norm, 4),
            "layers": layer_stats,
        }

        # Print summary for downstream layers
        print(f"  applied_norm={applied_norm:.4f}  src_delta_norm={src_norm:.4f}")
        print(
            f"  {'Layer':>6}  {'cos(Δh,v_l)':>12}  {'cos(Δh,v_src)':>14}  {'norm_ratio':>10}  {'best?':>6}"
        )
        for l in sorted(k for k in layer_stats if not layer_stats[k].get("upstream")):
            s = layer_stats[l]
            marker = "←SRC" if s["is_source"] else ("★" if s["is_best_layer"] else "")
            cos_vl = f"{s['cos_sim_v_l']:+.3f}" if s["cos_sim_v_l"] is not None else "  n/a "
            print(
                f"  {l:>6}  {cos_vl:>12}  {s['cos_sim_v_src']:>+14.3f}  {s['norm_ratio']:>10.4f}  {marker:>6}"
            )

    # Save
    out = {
        "model": args.model,
        "component": args.component,
        "alpha": args.alpha,
        "n_prompts": len(prompts),
        "source_layers": source_layers,
        "best_layers": best_layers,
        "results": {str(k): v for k, v in results.items()},
    }
    out_path = os.path.join(args.output_dir, "causal_propagation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
