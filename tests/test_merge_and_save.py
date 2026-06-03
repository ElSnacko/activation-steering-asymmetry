#!/usr/bin/env python3
"""
Test merge and model saving functionality for all component modes.

Tests:
1. Static merge into MLP bias (existing functionality)
2. Static merge into attention output projection bias (new)
3. Static merge into both attn+mlp (new dual-component)
4. Dynamic save/load with attn component (new submodule wrapping)
5. Dynamic save/load with mlp component (new submodule wrapping)
6. Verify merged model utility across component modes
"""

import json
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steering import (
    _get_attn_output_proj,
    _get_attn_submodule,
    load_merged_model,
    load_steered_model,
    merge_steering_into_model,
    save_dynamic_steered_model,
    verify_merged_model,
)
from activation_steering.dynamic_layer import DynamicSteeringSubmodule
from activation_steering.merge_steering_into_weights import _merge_into_output_proj

MODEL_PATH = "/mnt/bignvme/ai-stack/ai-backends/models/huggingface/Qwen/Qwen3.5-9B"
RUN_DIR = "outputs/qwen3-5-9b/20260310-073343"
ATTN_VECTORS = f"{RUN_DIR}/compute_wrmd/steering_vectors_wrmd.pt"
MLP_VECTORS = f"{RUN_DIR}/test_attn_mlp/steering_vectors_mlp_wrmd.pt"
TARGET_LAYERS = [15, 23]  # Use just 2 layers for speed


def create_combined_steering_file(tmpdir):
    """Create a combined attn+mlp steering file for testing."""
    attn_data = torch.load(ATTN_VECTORS, map_location="cpu")
    mlp_data = torch.load(MLP_VECTORS, map_location="cpu")

    attn_vecs = attn_data["steering_vectors"]
    if attn_vecs.ndim == 3:
        attn_vecs = attn_vecs[:, 0, :]
    mlp_vecs = mlp_data["steering_vectors"]
    if mlp_vecs.ndim == 3:
        mlp_vecs = mlp_vecs[:, 0, :]

    combined = {
        "steering_vectors_attn": attn_vecs,
        "steering_vectors_mlp": mlp_vecs,
        "steering_vectors": attn_vecs,
        "num_layers": attn_data["num_layers"],
        "hidden_size": attn_data["hidden_size"],
        "method": "wrmd",
        "component": "attn+mlp",
        "rank": 1,
    }

    path = os.path.join(tmpdir, "steering_vectors_attn_mlp.pt")
    torch.save(combined, path)
    return path


def test_merge_mlp(tmpdir):
    """Test 1: Static merge into MLP bias terms."""
    print("=" * 80)
    print("TEST 1: Static merge into MLP bias (component=mlp)")
    print("=" * 80)

    output_dir = os.path.join(tmpdir, "merged_mlp")
    metadata = merge_steering_into_model(
        base_model_path=MODEL_PATH,
        steering_vectors_file=ATTN_VECTORS,
        target_layers=TARGET_LAYERS,
        alpha=-2.0,
        output_dir=output_dir,
        component="mlp",
    )

    assert metadata["modified"], "Model should be marked as modified"
    assert metadata["steering_component"] == "mlp"
    assert metadata["modification_type"] == "mlp_bias_injection"
    assert os.path.exists(os.path.join(output_dir, "steering_metadata.json"))

    # Verify biases were added via load_merged_model
    merged, _ = load_merged_model(output_dir, device_map="cpu")
    for layer_idx in TARGET_LAYERS:
        layer = merged.model.layers[layer_idx]
        assert (
            layer.mlp.down_proj.bias is not None
        ), f"Layer {layer_idx} MLP down_proj should have bias"
        assert (
            layer.mlp.down_proj.bias.abs().sum() > 0
        ), f"Layer {layer_idx} MLP bias should be non-zero"
        print(
            f"  Layer {layer_idx}: MLP down_proj.bias norm = {layer.mlp.down_proj.bias.norm():.4f}"
        )

    # Check a non-target layer has no bias
    non_target = [i for i in range(merged.config.num_hidden_layers) if i not in TARGET_LAYERS][0]
    assert (
        merged.model.layers[non_target].mlp.down_proj.bias is None
    ), f"Non-target layer {non_target} should NOT have bias"

    del merged
    print("[OK] MLP static merge works\n")
    return output_dir


def test_merge_attn(tmpdir):
    """Test 2: Static merge into attention output projection bias."""
    print("=" * 80)
    print("TEST 2: Static merge into attention o_proj bias (component=attn)")
    print("=" * 80)

    output_dir = os.path.join(tmpdir, "merged_attn")
    metadata = merge_steering_into_model(
        base_model_path=MODEL_PATH,
        steering_vectors_file=ATTN_VECTORS,
        target_layers=TARGET_LAYERS,
        alpha=-2.0,
        output_dir=output_dir,
        component="attn",
    )

    assert metadata["steering_component"] == "attn"
    assert metadata["modification_type"] == "attn_bias_injection"

    # Verify biases were added to o_proj via load_merged_model
    merged, _ = load_merged_model(output_dir, device_map="cpu")
    for layer_idx in TARGET_LAYERS:
        layer = merged.model.layers[layer_idx]
        attn_out = _get_attn_output_proj(layer)
        assert attn_out.bias is not None, f"Layer {layer_idx} attn o_proj should have bias"
        assert (
            attn_out.bias.abs().sum() > 0
        ), f"Layer {layer_idx} attn o_proj bias should be non-zero"
        print(f"  Layer {layer_idx}: o_proj.bias norm = {attn_out.bias.norm():.4f}")

        # MLP should NOT have bias (only attn was merged)
        assert (
            layer.mlp.down_proj.bias is None
        ), f"Layer {layer_idx} MLP should NOT have bias in attn-only merge"

    del merged
    print("[OK] Attention static merge works\n")
    return output_dir


def test_merge_attn_mlp(tmpdir, combined_file):
    """Test 3: Static merge into both attn and MLP biases."""
    print("=" * 80)
    print("TEST 3: Static merge into attn+mlp (dual-component)")
    print("=" * 80)

    output_dir = os.path.join(tmpdir, "merged_attn_mlp")
    metadata = merge_steering_into_model(
        base_model_path=MODEL_PATH,
        steering_vectors_file=combined_file,
        target_layers=TARGET_LAYERS,
        alpha=-1.0,
        output_dir=output_dir,
        component="attn+mlp",
        alpha_attn=-3.0,
        alpha_mlp=-0.5,
    )

    assert metadata["steering_component"] == "attn+mlp"
    assert "attn_bias_injection" in metadata["modification_type"]
    assert "mlp_bias_injection" in metadata["modification_type"]
    assert metadata["steering_alpha_attn"] == -3.0
    assert metadata["steering_alpha_mlp"] == -0.5

    # Verify BOTH biases were added via load_merged_model
    merged, _ = load_merged_model(output_dir, device_map="cpu")
    for layer_idx in TARGET_LAYERS:
        layer = merged.model.layers[layer_idx]

        # Check attn o_proj bias
        attn_out = _get_attn_output_proj(layer)
        assert attn_out.bias is not None, f"Layer {layer_idx} attn o_proj should have bias"
        print(f"  Layer {layer_idx}: o_proj.bias norm = {attn_out.bias.norm():.4f}")

        # Check MLP down_proj bias
        assert (
            layer.mlp.down_proj.bias is not None
        ), f"Layer {layer_idx} MLP down_proj should have bias"
        print(f"  Layer {layer_idx}: down_proj.bias norm = {layer.mlp.down_proj.bias.norm():.4f}")

    del merged
    print("[OK] Dual-component static merge works\n")
    return output_dir


def test_verify_merged_model(mlp_dir, attn_dir, dual_dir):
    """Test 4: Verify merged model utility across component modes."""
    print("=" * 80)
    print("TEST 4: verify_merged_model across component modes")
    print("=" * 80)

    # MLP merge verification
    print("\n  4a: Verifying MLP merge...")
    result = verify_merged_model(mlp_dir, MODEL_PATH, TARGET_LAYERS)
    assert result["verified"], f"MLP merge verification failed: {result}"
    assert sorted(result["modified_layers"]) == sorted(TARGET_LAYERS)
    print(f"  MLP verified: modified_layers={result['modified_layers']}")

    # Attn merge verification
    print("\n  4b: Verifying attn merge...")
    result = verify_merged_model(attn_dir, MODEL_PATH, TARGET_LAYERS)
    assert result["verified"], f"Attn merge verification failed: {result}"
    assert sorted(result["modified_layers"]) == sorted(TARGET_LAYERS)
    print(f"  Attn verified: modified_layers={result['modified_layers']}")

    # Dual merge verification
    print("\n  4c: Verifying dual merge...")
    result = verify_merged_model(dual_dir, MODEL_PATH, TARGET_LAYERS)
    assert result["verified"], f"Dual merge verification failed: {result}"
    assert sorted(result["modified_layers"]) == sorted(TARGET_LAYERS)
    print(f"  Dual verified: modified_layers={result['modified_layers']}")

    print("[OK] All merge verifications pass\n")


def test_save_dynamic_attn(tmpdir):
    """Test 5: Dynamic save/load with attn component (submodule wrapping)."""
    print("=" * 80)
    print("TEST 5: Dynamic save/load with component=attn")
    print("=" * 80)

    output_dir = os.path.join(tmpdir, "dynamic_attn")
    metadata = save_dynamic_steered_model(
        base_model_path=MODEL_PATH,
        steering_vectors_file=ATTN_VECTORS,
        target_layers=TARGET_LAYERS,
        theta=50.0,
        gain=-2.0,
        output_dir=output_dir,
        component="attn",
    )

    assert metadata["steering_component"] == "attn"
    assert os.path.exists(os.path.join(output_dir, "steering_vectors.safetensors"))

    # Check config.json
    with open(os.path.join(output_dir, "config.json")) as f:
        config = json.load(f)
    assert config["steering_type"] == "dynamic_silu"
    assert config["steering_component"] == "attn"
    assert config["steering_target_layers"] == TARGET_LAYERS
    print(
        f"  Config: steering_type={config['steering_type']}, component={config['steering_component']}"
    )

    # Load with steering active
    print("  Loading with load_steered_model()...")
    model, tokenizer = load_steered_model(output_dir)

    # Check that attn submodules are wrapped
    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        attn_sub = _get_attn_submodule(layer)
        assert isinstance(
            attn_sub, DynamicSteeringSubmodule
        ), f"Layer {layer_idx} attn should be DynamicSteeringSubmodule, got {type(attn_sub).__name__}"
        print(f"  Layer {layer_idx}: attn wrapped as DynamicSteeringSubmodule")

    # Quick generation to verify it runs without errors
    print("  Running generation test...")
    messages = [{"role": "user", "content": "Hello"}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"  Output: {text[:80]}...")
    assert len(text) > 0, "Generation should produce output"

    del model
    torch.cuda.empty_cache()
    print("[OK] Dynamic attn save/load works\n")


def test_save_dynamic_mlp(tmpdir):
    """Test 6: Dynamic save/load with mlp component (submodule wrapping)."""
    print("=" * 80)
    print("TEST 6: Dynamic save/load with component=mlp")
    print("=" * 80)

    output_dir = os.path.join(tmpdir, "dynamic_mlp")
    metadata = save_dynamic_steered_model(
        base_model_path=MODEL_PATH,
        steering_vectors_file=MLP_VECTORS,
        target_layers=TARGET_LAYERS,
        theta=50.0,
        gain=-2.0,
        output_dir=output_dir,
        component="mlp",
    )

    assert metadata["steering_component"] == "mlp"

    # Load with steering active
    print("  Loading with load_steered_model()...")
    model, tokenizer = load_steered_model(output_dir)

    # Check that MLP submodules are wrapped
    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        assert isinstance(
            layer.mlp, DynamicSteeringSubmodule
        ), f"Layer {layer_idx} mlp should be DynamicSteeringSubmodule, got {type(layer.mlp).__name__}"
        print(f"  Layer {layer_idx}: mlp wrapped as DynamicSteeringSubmodule")

    # Quick generation test
    print("  Running generation test...")
    messages = [{"role": "user", "content": "Hello"}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"  Output: {text[:80]}...")
    assert len(text) > 0, "Generation should produce output"

    del model
    torch.cuda.empty_cache()
    print("[OK] Dynamic mlp save/load works\n")


def test_save_dynamic_attn_mlp_rejected(tmpdir, combined_file):
    """Test 7: Dynamic save with attn+mlp raises NotImplementedError."""
    print("=" * 80)
    print("TEST 7: Dynamic save with attn+mlp raises NotImplementedError")
    print("=" * 80)

    output_dir = os.path.join(tmpdir, "dynamic_attn_mlp")
    try:
        save_dynamic_steered_model(
            base_model_path=MODEL_PATH,
            steering_vectors_file=combined_file,
            target_layers=TARGET_LAYERS,
            theta=50.0,
            gain=-2.0,
            output_dir=output_dir,
            component="attn+mlp",
        )
        print("  ERROR: Should have raised NotImplementedError!")
        assert False
    except NotImplementedError as e:
        print(f"  Correctly raised NotImplementedError: {e}")

    print("[OK] Dynamic attn+mlp correctly rejected\n")


def test_invalid_component(tmpdir):
    """Test 8: Invalid component raises ValueError."""
    print("=" * 80)
    print("TEST 8: Invalid component raises ValueError")
    print("=" * 80)

    output_dir = os.path.join(tmpdir, "merged_invalid")
    try:
        merge_steering_into_model(
            base_model_path=MODEL_PATH,
            steering_vectors_file=ATTN_VECTORS,
            target_layers=TARGET_LAYERS,
            alpha=-2.0,
            output_dir=output_dir,
            component="layer",
        )
        print("  ERROR: Should have raised ValueError!")
        assert False
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("[OK] Invalid component correctly rejected\n")


def test_merged_model_generation(mlp_dir, attn_dir):
    """Test 9: Generate with merged models and verify outputs differ from base."""
    print("=" * 80)
    print("TEST 9: Generation comparison (base vs MLP-merged vs attn-merged)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    test_prompt = "How do I pick a lock?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = {}
    for label, model_path in [
        ("base", MODEL_PATH),
        ("mlp_merged", mlp_dir),
        ("attn_merged", attn_dir),
    ]:
        print(f"\n  Loading {label}...")
        if label == "base":
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            model, _ = load_merged_model(model_path, device_map="auto")
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        outputs[label] = text
        print(f"  {label}: {text[:100]}...")
        del model
        torch.cuda.empty_cache()

    # At least the merged models should differ from base
    base_vs_mlp = outputs["base"] != outputs["mlp_merged"]
    base_vs_attn = outputs["base"] != outputs["attn_merged"]
    print(f"\n  base != mlp_merged: {base_vs_mlp}")
    print(f"  base != attn_merged: {base_vs_attn}")

    # We expect at least one of them to differ
    assert (
        base_vs_mlp or base_vs_attn
    ), "At least one merged model should produce different output than base"

    print("[OK] Generation comparison works\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MERGE & MODEL SAVING FUNCTIONALITY TEST SUITE")
    print("=" * 80 + "\n")

    # Use a temp directory for all outputs
    tmpdir = os.path.join(
        os.path.dirname(__file__), "outputs", "qwen3-5-9b", "20260310-073343", "test_merge_save"
    )
    os.makedirs(tmpdir, exist_ok=True)
    print(f"Output directory: {tmpdir}\n")

    # Create combined steering file
    combined_file = create_combined_steering_file(tmpdir)
    print(f"Combined steering file: {combined_file}\n")

    # Run tests
    mlp_dir = test_merge_mlp(tmpdir)
    attn_dir = test_merge_attn(tmpdir)
    dual_dir = test_merge_attn_mlp(tmpdir, combined_file)

    test_verify_merged_model(mlp_dir, attn_dir, dual_dir)

    test_save_dynamic_attn(tmpdir)
    test_save_dynamic_mlp(tmpdir)
    test_save_dynamic_attn_mlp_rejected(tmpdir, combined_file)
    test_invalid_component(tmpdir)

    test_merged_model_generation(mlp_dir, attn_dir)

    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
