"""
Merge steering vectors permanently into model weights.

This module provides functionality to permanently modify a model's weights
by merging steering vectors into output projection bias terms (MLP down_proj,
attention o_proj, or both). This creates a model that has the steering
behavior built-in without needing runtime hooks.

WARNING: This permanently modifies the model weights!
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dynamic_layer import DynamicSteeringLayer, DynamicSteeringSubmodule
from .extraction import _get_attn_output_proj, _get_attn_submodule
from .utils import get_model_layers


def export_to_gguf(model_dir, output_path=None, quantization="f16", verbose=False):
    """
    Export a HuggingFace model to GGUF format.

    Attempts to convert using llama.cpp conversion tools. Falls back to
    alternative methods if primary conversion fails.

    Args:
        model_dir: Directory containing HuggingFace model
        output_path: Path for output GGUF file (default: model_dir/model.gguf)
        quantization: Quantization type ("f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f32")
        verbose: Print detailed conversion output

    Returns:
        Dictionary with export results

    Raises:
        RuntimeError: If GGUF conversion fails
    """
    if output_path is None:
        output_path = os.path.join(model_dir, f"model-{quantization}.gguf")

    print(f"[GGUF] Converting model to GGUF format...")
    print(f"   Model directory: {model_dir}")
    print(f"   Output path: {output_path}")
    print(f"   Quantization: {quantization}")

    conversion_methods = [_try_convert_with_hf_to_gguf, _try_convert_with_llama_cpp_python]

    for method in conversion_methods:
        try:
            result = method(model_dir, output_path, quantization, verbose)
            if result["success"]:
                print(f"[OK] GGUF export successful using {result['method']}")
                return result
        except Exception as e:
            print(f"[WARN] Conversion method failed: {e}")
            continue

    raise RuntimeError(
        "GGUF conversion failed. Ensure llama.cpp or llama-cpp-python is installed.\n"
        "Install with: pip install llama-cpp-python\n"
        "Or clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
    )


def _try_convert_with_hf_to_gguf(model_dir, output_path, quantization, verbose):
    """Try conversion using llama.cpp's convert-hf-to-gguf.py script."""

    llama_cpp_paths = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("./llama.cpp"),
        Path("../llama.cpp"),
    ]

    convert_script = None
    for base_path in llama_cpp_paths:
        potential_script = base_path / "convert-hf-to-gguf.py"
        if potential_script.exists():
            convert_script = potential_script
            break

    if convert_script is None:
        raise RuntimeError("convert-hf-to-gguf.py not found")

    cmd = [
        sys.executable,
        str(convert_script),
        model_dir,
        "--outfile",
        output_path,
        "--outtype",
        quantization,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr}")

    if verbose:
        print(result.stdout)

    return {
        "success": True,
        "method": "llama.cpp convert-hf-to-gguf.py",
        "output_path": output_path,
        "quantization": quantization,
    }


def _try_convert_with_llama_cpp_python(model_dir, output_path, quantization, verbose):
    """Try conversion using llama-cpp-python library."""

    try:
        from llama_cpp import llama_model_loader
    except ImportError:
        raise RuntimeError("llama-cpp-python not installed")

    raise RuntimeError("Direct llama-cpp-python conversion not yet implemented")


def _ensure_zero_bias(linear):
    """Add a zero bias to a Linear layer that has bias=None."""
    with torch.no_grad():
        linear.bias = torch.nn.Parameter(
            torch.zeros(
                linear.out_features,
                dtype=linear.weight.dtype,
                device=linear.weight.device,
            )
        )


def _patch_config_bias_flags(config_path, merge_components):
    """Patch config.json to set bias flags for merged components.

    Sets architecture-level flags so that standard loaders (HF, vLLM)
    create Linear layers with bias=True, matching the checkpoint.

    For attention: sets 'attention_bias: true' (standard HF/vLLM field).
    For MLP: sets 'mlp_bias: true' (non-standard, requires vLLM patch).
    Also patches nested text_config if present (multimodal models).
    """
    with open(config_path) as f:
        config = json.load(f)

    patches = {}
    if "attn" in merge_components:
        patches["attention_bias"] = True
    if "mlp" in merge_components:
        patches["mlp_bias"] = True

    config.update(patches)

    # Also patch text_config if present (Qwen3.5 multimodal has nested config)
    if "text_config" in config:
        config["text_config"].update(patches)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[OK] Patched config.json: {patches}")


def _merge_into_output_proj(output_proj, steering_vec, alpha):
    """Merge a steering vector into an output projection's bias term.

    Creates the bias parameter if it doesn't exist (most models use bias=False).

    Args:
        output_proj: nn.Linear module (e.g. down_proj, o_proj)
        steering_vec: Steering vector tensor [hidden_size]
        alpha: Steering coefficient
    """
    with torch.no_grad():
        if output_proj.bias is None:
            output_proj.bias = torch.nn.Parameter(
                torch.zeros(
                    output_proj.out_features,
                    dtype=steering_vec.dtype,
                    device=steering_vec.device,
                )
            )
        output_proj.bias.data += alpha * steering_vec


def merge_steering_into_model(
    base_model_path,
    steering_vectors_file,
    target_layers,
    alpha,
    output_dir,
    export_gguf=False,
    gguf_quantization="f16",
    component="mlp",
    alpha_attn=None,
    alpha_mlp=None,
):
    """
    Permanently merge steering vectors into model weights.

    Modifies the model's output projection bias terms to incorporate steering.
    Supports merging into MLP (down_proj), attention (o_proj), or both.

    WARNING: This permanently modifies the model! The output model will
    have different behavior than the base model.

    Args:
        base_model_path: Path to base HuggingFace model
        steering_vectors_file: Path to .pt file with steering vectors
        target_layers: List of layer indices to modify
        alpha: Steering coefficient (negative = reduce refusal)
        output_dir: Directory to save modified model
        export_gguf: Whether to also export to GGUF format
        gguf_quantization: Quantization type for GGUF export
        component: Which component(s) to merge: "mlp", "attn", or "attn+mlp"
        alpha_attn: Override alpha for attention component (default: use alpha)
        alpha_mlp: Override alpha for MLP component (default: use alpha)

    Returns:
        Dictionary with metadata about the modification
    """
    if component not in ("mlp", "attn", "attn+mlp"):
        raise ValueError(f"Invalid component '{component}', must be 'mlp', 'attn', or 'attn+mlp'")

    merge_components = component.split("+") if "+" in component else [component]
    a_attn = alpha_attn if alpha_attn is not None else alpha
    a_mlp = alpha_mlp if alpha_mlp is not None else alpha

    print("[MERGE] Merging steering into model weights...")
    print(f"   Base model: {base_model_path}")
    print(f"   Layers: {target_layers}")
    print(f"   Component(s): {component}")
    if "attn" in merge_components:
        print(f"   Alpha (attn): {a_attn}")
    if "mlp" in merge_components:
        print(f"   Alpha (mlp): {a_mlp}")

    # Load model
    print("[LOAD] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load steering vectors
    print("[LOAD] Loading steering vectors...")
    vec_data = torch.load(steering_vectors_file, weights_only=True)

    # Load per-component vectors
    component_vectors = {}
    for comp in merge_components:
        sv_key = f"steering_vectors_{comp}"
        if sv_key in vec_data:
            vecs = vec_data[sv_key]
        elif "steering_vectors" in vec_data:
            vecs = vec_data["steering_vectors"]
        else:
            raise ValueError(f"No steering vectors found for component '{comp}'")

        # For multi-rank vectors [num_layers, rank, hidden_size], use primary direction
        if vecs.ndim == 3:
            print(f"   {comp}: multi-rank vectors detected (rank={vecs.shape[1]}), using rank 0")
            vecs = vecs[:, 0, :]

        component_vectors[comp] = vecs.to(model.device)

    print(f"\n[COMPUTE] Modifying {len(target_layers)} layer(s)...")

    layers = get_model_layers(model)
    for layer_idx in target_layers:
        layer = layers[layer_idx]

        if "mlp" in merge_components:
            steering_vec = component_vectors["mlp"][layer_idx]
            if hasattr(layer.mlp, "down_proj"):
                _merge_into_output_proj(layer.mlp.down_proj, steering_vec, a_mlp)
                print(f"   * Layer {layer_idx}: Added steering to MLP output bias")
            else:
                print(f"   [WARN] Layer {layer_idx}: No down_proj found, skipping MLP")

        if "attn" in merge_components:
            steering_vec = component_vectors["attn"][layer_idx]
            try:
                attn_out = _get_attn_output_proj(layer)
                _merge_into_output_proj(attn_out, steering_vec, a_attn)
                print(f"   * Layer {layer_idx}: Added steering to attention output bias")
            except AttributeError as e:
                print(f"   [WARN] Layer {layer_idx}: {e}, skipping attention")

    # Make the checkpoint self-consistent: add zero biases to ALL non-steered
    # layers so the model architecture can use bias=True globally. This makes
    # the checkpoint loadable by any framework (HF, vLLM, llama.cpp) without
    # custom loaders.
    num_layers = len(layers)
    print(f"\n[COMPAT] Adding zero biases to non-steered layers for framework compatibility...")
    zero_count = 0

    for layer_idx in range(num_layers):
        if layer_idx in target_layers:
            continue
        layer = layers[layer_idx]

        if "mlp" in merge_components and hasattr(layer.mlp, "down_proj"):
            if layer.mlp.down_proj.bias is None:
                _ensure_zero_bias(layer.mlp.down_proj)
                zero_count += 1

        if "attn" in merge_components:
            try:
                attn_out = _get_attn_output_proj(layer)
                if attn_out.bias is None:
                    _ensure_zero_bias(attn_out)
                    zero_count += 1
                # Also add zero biases to q/k/v projections if attention_bias
                # is being enabled (config requires all attn projs to have bias)
                attn_sub = _get_attn_submodule(layer)
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    proj = getattr(attn_sub, proj_name, None)
                    if proj is not None and proj.bias is None:
                        _ensure_zero_bias(proj)
                        zero_count += 1
            except AttributeError:
                pass

    # Also add zero biases to q/k/v on steered layers for attention_bias compat
    if "attn" in merge_components:
        for layer_idx in target_layers:
            layer = layers[layer_idx]
            try:
                attn_sub = _get_attn_submodule(layer)
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    proj = getattr(attn_sub, proj_name, None)
                    if proj is not None and proj.bias is None:
                        _ensure_zero_bias(proj)
                        zero_count += 1
            except AttributeError:
                pass

    print(f"   Added {zero_count} zero bias term(s) across {num_layers} layers")

    # Save modified model
    print(f"\n[SAVE] Saving modified model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Patch config.json to enable bias flags so loaders create matching architecture
    config_path = os.path.join(output_dir, "config.json")
    _patch_config_bias_flags(config_path, merge_components)

    # Also save biases as sidecar for backward compat with load_merged_model()
    merged_biases = {}
    for layer_idx in target_layers:
        layer = layers[layer_idx]
        if "mlp" in merge_components and hasattr(layer.mlp, "down_proj"):
            bias = layer.mlp.down_proj.bias
            if bias is not None:
                merged_biases[f"model.layers.{layer_idx}.mlp.down_proj.bias"] = bias.data.clone()
        if "attn" in merge_components:
            try:
                attn_out = _get_attn_output_proj(layer)
                if attn_out.bias is not None:
                    attn_sub = _get_attn_submodule(layer)
                    for attr_name in ("self_attn", "linear_attn", "attention", "attn"):
                        if hasattr(layer, attr_name) and getattr(layer, attr_name) is attn_sub:
                            break
                    else:
                        attr_name = "self_attn"
                    from .extraction import _ATTN_OUTPUT_PROJ_NAMES

                    for proj_name in _ATTN_OUTPUT_PROJ_NAMES:
                        if (
                            hasattr(attn_sub, proj_name)
                            and getattr(attn_sub, proj_name) is attn_out
                        ):
                            break
                    key = f"model.layers.{layer_idx}.{attr_name}.{proj_name}.bias"
                    merged_biases[key] = attn_out.bias.data.clone()
            except AttributeError:
                pass
    if merged_biases:
        biases_cpu = {k: v.cpu().contiguous() for k, v in merged_biases.items()}
        bias_path = os.path.join(output_dir, "steering_biases.safetensors")
        save_file(biases_cpu, bias_path)

    # Build modification type description
    mod_parts = []
    if "mlp" in merge_components:
        mod_parts.append("mlp_bias_injection")
    if "attn" in merge_components:
        mod_parts.append("attn_bias_injection")
    modification_type = "+".join(mod_parts)

    # Save metadata about the modification
    metadata = {
        "base_model": base_model_path,
        "steering_vectors_file": steering_vectors_file,
        "steering_method": vec_data.get("method", "unknown"),
        "steering_layers": target_layers,
        "steering_alpha": alpha,
        "steering_component": component,
        "num_layers": vec_data.get("num_layers"),
        "hidden_size": vec_data.get("hidden_size"),
        "lambda_ridge": vec_data.get("lambda_ridge"),
        "use_score_weighting": vec_data.get("use_score_weighting"),
        "modified": True,
        "modification_type": modification_type,
        "note": f"Steering vectors permanently merged into {component} output biases",
    }
    if alpha_attn is not None:
        metadata["steering_alpha_attn"] = alpha_attn
    if alpha_mlp is not None:
        metadata["steering_alpha_mlp"] = alpha_mlp

    metadata_path = os.path.join(output_dir, "steering_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    attn_native = "attn" in merge_components
    mlp_native = False  # No mlp_bias config in most architectures yet

    print(f"[OK] Model saved successfully!")
    print(f"[INFO] Metadata saved to: {metadata_path}")
    if attn_native and "mlp" not in merge_components:
        print(f"\nLoad with any framework (config has attention_bias=True):")
        print(f'   model = AutoModelForCausalLM.from_pretrained("{output_dir}")')
        print(f"\nOr with vLLM:")
        print(f'   vllm.LLM("{output_dir}")')
    else:
        print(f"\nLoad the modified model with:")
        print(f"   from activation_steering import load_merged_model")
        print(f'   model, tokenizer = load_merged_model("{output_dir}")')
        if "mlp" in merge_components:
            print(f"\nFor vLLM, see docs on patching vLLM's MLP class for mlp_bias support.")

    if export_gguf:
        try:
            print("\n" + "=" * 80)
            gguf_result = export_to_gguf(
                model_dir=output_dir,
                quantization=gguf_quantization,
                verbose=False,
            )
            metadata["gguf_export"] = gguf_result

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print("=" * 80)
        except Exception as e:
            print(f"[WARN] GGUF export failed: {e}")
            print("[INFO] HuggingFace model still saved successfully")
            metadata["gguf_export"] = {"success": False, "error": str(e)}

    return metadata


def load_merged_model(model_dir, device_map="auto", torch_dtype=torch.bfloat16):
    """
    Load a statically-merged steered model.

    Models saved with merge_steering_into_model() have config flags set
    (attention_bias, mlp_bias) so that from_pretrained creates matching
    architecture. For components where the config flag is recognized
    (attention_bias for most architectures), biases load natively. For
    components where the flag isn't recognized (mlp_bias), the sidecar
    file (steering_biases.safetensors) is used as fallback.

    Args:
        model_dir: Path to merged model directory
        device_map: Device map for model loading (default: "auto")
        torch_dtype: Model dtype (default: bfloat16)

    Returns:
        Tuple of (model, tokenizer) with steering biases applied.
    """
    print(f"[LOAD] Loading merged model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Check if any steering biases need to be injected from sidecar
    # (needed for components where config flags aren't supported natively,
    # e.g. mlp_bias on architectures that hardcode bias=False)
    bias_path = os.path.join(model_dir, "steering_biases.safetensors")
    if os.path.exists(bias_path):
        bias_data = load_file(bias_path)
        injected = 0
        for key, bias_tensor in bias_data.items():
            parts = key.split(".")
            if parts[-1] != "bias":
                continue
            module_parts = parts[:-1]
            module = model
            for part in module_parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            # Only inject if bias is still None (wasn't loaded natively)
            if module.bias is None:
                device = module.weight.device
                module.bias = torch.nn.Parameter(bias_tensor.to(device=device, dtype=torch_dtype))
                injected += 1
        if injected > 0:
            print(f"[OK] Injected {injected} bias term(s) from sidecar (not loaded natively)")
        else:
            print(f"[OK] All biases loaded natively from checkpoint")
    else:
        print("[OK] No sidecar needed — biases loaded natively from checkpoint")

    return model, tokenizer


def verify_merged_model(model_dir, original_model_path, steering_layers):
    """
    Verify that a merged model has the expected modifications.

    Args:
        model_dir: Directory containing merged model
        original_model_path: Path to original base model
        steering_layers: List of layer indices that should be modified

    Returns:
        Dictionary with verification results
    """

    print("[INFO] Verifying merged model...")

    # Check metadata exists
    metadata_path = os.path.join(model_dir, "steering_metadata.json")
    if not os.path.exists(metadata_path):
        return {"verified": False, "error": "No steering_metadata.json found"}

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load both models to compare
    print("[LOAD] Loading original and merged models...")
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    merged_model, _ = load_merged_model(model_dir, device_map="cpu")

    results = {
        "verified": True,
        "metadata": metadata,
        "modified_layers": [],
        "unmodified_layers": [],
    }

    component = metadata.get("steering_component", "mlp")
    check_components = component.split("+") if "+" in component else [component]

    orig_layers = get_model_layers(original_model)
    merged_layers = get_model_layers(merged_model)

    # Check each specified layer
    for layer_idx in steering_layers:
        orig_layer = orig_layers[layer_idx]
        merged_layer = merged_layers[layer_idx]
        layer_modified = False

        if "mlp" in check_components:
            if hasattr(orig_layer.mlp, "down_proj") and hasattr(merged_layer.mlp, "down_proj"):
                orig_bias = orig_layer.mlp.down_proj.bias
                merged_bias = merged_layer.mlp.down_proj.bias
                if orig_bias is None and merged_bias is not None:
                    layer_modified = True
                elif orig_bias is not None and merged_bias is not None:
                    if not torch.allclose(orig_bias, merged_bias, rtol=1e-5):
                        layer_modified = True

        if "attn" in check_components:
            try:
                orig_attn_out = _get_attn_output_proj(orig_layer)
                merged_attn_out = _get_attn_output_proj(merged_layer)
                orig_bias = orig_attn_out.bias
                merged_bias = merged_attn_out.bias
                if orig_bias is None and merged_bias is not None:
                    layer_modified = True
                elif orig_bias is not None and merged_bias is not None:
                    if not torch.allclose(orig_bias, merged_bias, rtol=1e-5):
                        layer_modified = True
            except AttributeError:
                pass

        if layer_modified:
            results["modified_layers"].append(layer_idx)
        else:
            results["unmodified_layers"].append(layer_idx)

    print(f"[INFO] Modified layers: {results['modified_layers']}")
    print(f"[INFO] Unmodified layers: {results['unmodified_layers']}")

    if len(results["modified_layers"]) == len(steering_layers):
        print("[OK] Verification passed: All specified layers modified")
    else:
        print("[WARN] Verification incomplete: Not all layers modified")
        results["verified"] = False

    return results


def save_dynamic_steered_model(
    base_model_path,
    steering_vectors_file,
    target_layers,
    theta,
    gain,
    output_dir,
    torch_dtype=torch.bfloat16,
    component="attn",
):
    """
    Save a dynamically-steered model as a self-contained directory.

    Stores base model weights unmodified, with steering metadata embedded in
    config.json and steering vectors in a safetensors sidecar file.

    The saved model can be loaded with load_steered_model() to get steering
    active, or with plain from_pretrained() to get the base model.

    Note: GGUF export is not supported for dynamic mode since the SiLU-gated
    logic cannot be represented in the GGUF format.

    Args:
        base_model_path: Path to base HuggingFace model
        steering_vectors_file: Path to .pt file with steering vectors
        target_layers: List of layer indices to steer
        theta: Per-layer theta dict (layer_idx -> float) or single float
        gain: Gain multiplier (negative = reduce refusal)
        output_dir: Directory to save the steered model
        torch_dtype: Model dtype (default: bfloat16)
        component: Which component to steer: "attn", "mlp", or "layer"

    Returns:
        Dictionary with metadata about the saved model
    """
    import glob as globmod
    import shutil

    print("[SAVE] Saving dynamic steered model...")
    print(f"   Base model: {base_model_path}")
    print(f"   Target layers: {target_layers}")
    print(f"   Gain: {gain}")
    print(f"   Component: {component}")

    # Load steering vectors from pipeline .pt file
    print("[LOAD] Loading steering vectors...")
    vec_data = torch.load(steering_vectors_file, map_location="cpu", weights_only=True)

    # Resolve component-specific vectors
    if component == "attn+mlp":
        raise NotImplementedError(
            "save_dynamic_steered_model does not yet support dual-component (attn+mlp) mode. "
            "Use static merge with merge_steering_into_model(component='attn+mlp') instead, "
            "or save separate models for each component."
        )

    sv_key = f"steering_vectors_{component}"
    if sv_key in vec_data:
        all_vectors = vec_data[sv_key]
    elif "steering_vectors" in vec_data:
        all_vectors = vec_data["steering_vectors"]
    else:
        raise ValueError(f"No steering vectors found for component '{component}'")

    # For multi-rank vectors [num_layers, rank, hidden_size], use primary direction
    if all_vectors.ndim == 3:
        print(f"   Multi-rank vectors detected (rank={all_vectors.shape[1]}), using rank 0")
        all_vectors = all_vectors[:, 0, :]

    # Normalize theta to a string-keyed dict for JSON serialization
    if isinstance(theta, dict):
        theta_dict = {str(k): float(v) for k, v in theta.items()}
    else:
        theta_dict = {str(l): float(theta) for l in target_layers}

    # Print theta values
    for l in target_layers:
        print(f"   Layer {l}: theta={theta_dict[str(l)]:.4f}")

    # Copy base model files to output directory (preserves quantization format)
    print(f"[COPY] Copying base model files to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.abspath(base_model_path)

    # Copy all model files (safetensors, tokenizer, config, etc.)
    copy_patterns = [
        "*.safetensors",
        "*.json",
        "*.txt",
        "*.model",
        "*.tiktoken",
        "*.py",
        "model.safetensors.index.json",
    ]
    copied_files = set()
    for pattern in copy_patterns:
        for src_file in globmod.glob(os.path.join(base_path, pattern)):
            dst_file = os.path.join(output_dir, os.path.basename(src_file))
            if os.path.basename(src_file) not in copied_files:
                shutil.copy2(src_file, dst_file)
                copied_files.add(os.path.basename(src_file))

    print(f"   Copied {len(copied_files)} files")

    # Patch config.json with steering metadata
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    config["steering_type"] = "dynamic_silu"
    config["steering_target_layers"] = target_layers
    config["steering_theta"] = theta_dict
    config["steering_gain"] = float(gain)
    config["steering_component"] = component

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("[OK] Steering metadata embedded in config.json")

    # Save steering vectors as safetensors sidecar (target layers only)
    sv_dict = {}
    for layer_idx in target_layers:
        sv_dict[f"layer_{layer_idx}"] = all_vectors[layer_idx].contiguous()

    sv_path = os.path.join(output_dir, "steering_vectors.safetensors")
    save_file(sv_dict, sv_path)
    print(f"[OK] Steering vectors saved: {sv_path}")

    # Save provenance metadata
    metadata = {
        "base_model": base_model_path,
        "steering_vectors_source": steering_vectors_file,
        "steering_type": "dynamic_silu",
        "steering_method": vec_data.get("method", "unknown"),
        "steering_component": component,
        "target_layers": target_layers,
        "theta": theta_dict,
        "gain": float(gain),
        "num_layers": vec_data.get("num_layers"),
        "hidden_size": vec_data.get("hidden_size"),
        "lambda_ridge": vec_data.get("lambda_ridge"),
        "use_score_weighting": vec_data.get("use_score_weighting"),
        "note": "Dynamic SiLU-gated steering. Load with load_steered_model() for active steering.",
    }

    metadata_path = os.path.join(output_dir, "steering_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Model saved successfully!")
    print(f"[INFO] Metadata saved to: {metadata_path}")
    print(f"\nLoad with steering active:")
    print(f"   from activation_steering import load_steered_model")
    print(f'   model, tokenizer = load_steered_model("{output_dir}")')
    print(f"\nLoad as base model (no steering):")
    print(f'   model = AutoModelForCausalLM.from_pretrained("{output_dir}")')

    return metadata


def load_steered_model(model_dir, device_map="auto", torch_dtype=torch.bfloat16):
    """
    Load a steered model from a saved directory.

    If the model was saved with save_dynamic_steered_model(), this will
    detect the steering config, load steering vectors, and wrap target
    layers (or submodules) with DynamicSteeringLayer/DynamicSteeringSubmodule
    so steering is automatically active.

    If no steering config is found, returns the model as-is.

    Args:
        model_dir: Path to saved model directory
        device_map: Device map for model loading (default: "auto")
        torch_dtype: Model dtype (default: bfloat16)

    Returns:
        Tuple of (model, tokenizer). Model has steering layers wrapped
        if steering config was found.
    """
    print(f"[LOAD] Loading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Check for steering config (read config.json directly since model-specific
    # config classes may not preserve unknown keys as attributes)
    config_path = os.path.join(model_dir, "config.json")
    steering_config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            raw_config = json.load(f)
        steering_config = {k: v for k, v in raw_config.items() if k.startswith("steering_")}

    steering_type = steering_config.get("steering_type")
    if steering_type is None:
        print("[INFO] No steering config found, returning base model")
        return model, tokenizer

    if steering_type != "dynamic_silu":
        print(f"[WARN] Unknown steering_type: {steering_type}, returning base model")
        return model, tokenizer

    # Load steering parameters from config
    target_layers = steering_config["steering_target_layers"]
    theta_dict = steering_config["steering_theta"]  # str-keyed
    gain = steering_config["steering_gain"]
    component = steering_config.get("steering_component", "layer")

    print(f"[INFO] Detected dynamic_silu steering config")
    print(f"   Target layers: {target_layers}")
    print(f"   Gain: {gain}")
    print(f"   Component: {component}")

    # Load steering vectors from sidecar
    sv_path = os.path.join(model_dir, "steering_vectors.safetensors")
    if not os.path.exists(sv_path):
        print(f"[ERROR] Steering vectors not found: {sv_path}")
        print("[WARN] Returning base model without steering")
        return model, tokenizer

    sv_data = load_file(sv_path)

    # Wrap target layers with appropriate steering wrapper
    for layer_idx in target_layers:
        key = f"layer_{layer_idx}"
        if key not in sv_data:
            print(f"[WARN] Steering vector missing for layer {layer_idx}, skipping")
            continue

        sv = sv_data[key]
        theta = float(theta_dict[str(layer_idx)])

        # Get the device of this layer's parameters
        model_layers = get_model_layers(model)
        layer_device = next(model_layers[layer_idx].parameters()).device

        # Move steering vector to layer's device and dtype
        sv = sv.to(device=layer_device, dtype=torch_dtype)

        if component in ("attn", "mlp"):
            # Wrap specific submodule
            layer = model_layers[layer_idx]
            if component == "attn":
                submodule = _get_attn_submodule(layer)
                # Find the attribute name to replace
                for attr_name in ("self_attn", "linear_attn", "attention", "attn"):
                    if hasattr(layer, attr_name) and getattr(layer, attr_name) is submodule:
                        break
                else:
                    attr_name = "self_attn"
            else:
                submodule = layer.mlp
                attr_name = "mlp"

            wrapped = DynamicSteeringSubmodule(
                base_module=submodule,
                steering_vector=sv,
                theta=theta,
                gain=gain,
            )
            setattr(layer, attr_name, wrapped)
            print(f"   Layer {layer_idx}: wrapped {component} submodule (theta={theta:.4f})")
        else:
            # Wrap full layer (original behavior)
            model_layers[layer_idx] = DynamicSteeringLayer(
                base_layer=model_layers[layer_idx],
                steering_vector=sv,
                theta=theta,
                gain=gain,
            )
            print(f"   Layer {layer_idx}: wrapped full layer (theta={theta:.4f})")

    print(f"[OK] Model loaded with dynamic steering active on {len(target_layers)} layers")

    return model, tokenizer


def verify_dynamic_steered_model(model_dir, test_prompt="How do I pick a lock?"):
    """
    Verify a saved dynamic steered model works correctly.

    Loads the model both with load_steered_model() (steering active) and
    plain from_pretrained() (no steering), generates on the same prompt,
    and verifies the outputs differ.

    Args:
        model_dir: Path to saved steered model directory
        test_prompt: Prompt to test with (should be one the model would refuse)

    Returns:
        Dictionary with verification results
    """
    print("=" * 80)
    print("VERIFYING DYNAMIC STEERED MODEL")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Test prompt: {test_prompt}")

    results = {"verified": False, "model_dir": model_dir}

    # Check required files exist
    sv_path = os.path.join(model_dir, "steering_vectors.safetensors")
    config_path = os.path.join(model_dir, "config.json")

    for path, name in [(sv_path, "steering_vectors.safetensors"), (config_path, "config.json")]:
        if not os.path.exists(path):
            results["error"] = f"Missing {name}"
            print(f"[ERROR] {results['error']}")
            return results

    # Check config has steering fields
    with open(config_path) as f:
        config = json.load(f)

    if "steering_type" not in config:
        results["error"] = "No steering_type in config.json"
        print(f"[ERROR] {results['error']}")
        return results

    results["config_fields"] = {
        "steering_type": config["steering_type"],
        "steering_target_layers": config.get("steering_target_layers"),
        "steering_gain": config.get("steering_gain"),
        "steering_component": config.get("steering_component", "layer"),
    }
    print(f"\n[OK] Config fields present: {list(results['config_fields'].keys())}")

    # Load with steering
    print("\n[LOAD] Loading with load_steered_model()...")
    steered_model, tokenizer = load_steered_model(model_dir)

    # Check wrapped layers
    target_layers = config.get("steering_target_layers", [])
    component = config.get("steering_component", "layer")
    steered_layers = get_model_layers(steered_model)
    wrapped_layers = []
    for layer_idx in target_layers:
        layer = steered_layers[layer_idx]
        if isinstance(layer, DynamicSteeringLayer):
            wrapped_layers.append(layer_idx)
        elif component == "attn":
            try:
                sub = _get_attn_submodule(layer)
                if isinstance(sub, DynamicSteeringSubmodule):
                    wrapped_layers.append(layer_idx)
            except AttributeError:
                pass
        elif component == "mlp":
            if isinstance(layer.mlp, DynamicSteeringSubmodule):
                wrapped_layers.append(layer_idx)

    results["wrapped_layers"] = wrapped_layers
    print(f"[INFO] Wrapped layers: {wrapped_layers}")

    if len(wrapped_layers) != len(target_layers):
        results["error"] = (
            f"Expected {len(target_layers)} wrapped layers, got {len(wrapped_layers)}"
        )
        print(f"[WARN] {results['error']}")

    # Generate with steering
    print(f"\n[TEST] Generating with steering...")
    messages = [{"role": "user", "content": test_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(steered_model.device)

    with torch.no_grad():
        steered_output = steered_model.generate(
            **inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    steered_text = tokenizer.decode(
        steered_output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    results["steered_response"] = steered_text[:200]
    print(f"   Steered: {steered_text[:100]}...")

    # Free steered model
    del steered_model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Load without steering (plain from_pretrained)
    print(f"\n[TEST] Generating without steering (plain from_pretrained)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        base_output = base_model.generate(
            **inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    base_text = tokenizer.decode(
        base_output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    results["base_response"] = base_text[:200]
    print(f"   Base: {base_text[:100]}...")

    del base_model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Compare
    outputs_differ = steered_text != base_text
    results["outputs_differ"] = outputs_differ
    results["verified"] = outputs_differ and len(wrapped_layers) == len(target_layers)

    print(f"\n{'=' * 80}")
    if results["verified"]:
        print("[OK] Verification passed: outputs differ, all layers wrapped")
    else:
        if not outputs_differ:
            print("[WARN] Verification failed: steered and base outputs are identical")
        else:
            print(
                "[WARN] Verification partially passed: outputs differ but layer wrapping incomplete"
            )
    print("=" * 80)

    return results
