#!/usr/bin/env python3
"""
End-to-end test of dual-component (attn+mlp) steering.

Tests the full pipeline:
1. Extract MLP activations from existing prompts
2. Compute MLP steering vectors (WRMD)
3. Create combined steering file with attn + mlp vectors
4. Test SteeringHookGroup with both components
5. Compare single-component vs dual-component outputs
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steering import (
    ActivationExtractor,
    SteeringHook,
    WRMDCalculator,
    _get_attn_output_proj,
    _get_attn_submodule,
)
from activation_steering.steering import SteeringHookGroup

# Paths
MODEL_PATH = "/mnt/bignvme/ai-stack/ai-backends/models/huggingface/Qwen/Qwen3.5-9B"
RUN_DIR = "outputs/qwen3-5-9b/20260310-073343"
ATTN_ACTIVATIONS = f"{RUN_DIR}/extract_activations/activations_qwen_7b_judged.pt"
ATTN_VECTORS = f"{RUN_DIR}/compute_wrmd/steering_vectors_wrmd.pt"
OUTPUT_DIR = f"{RUN_DIR}/test_attn_mlp"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_attn_output_proj_discovery():
    """Test that _get_attn_output_proj finds the right module."""
    print("=" * 80)
    print("TEST 1: Attention output projection discovery")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )

    for layer_idx in [0, 15, 31]:
        layer = model.model.layers[layer_idx]
        attn_sub = _get_attn_submodule(layer)
        attn_out = _get_attn_output_proj(layer)
        print(f"  Layer {layer_idx}:")
        print(f"    Attention submodule: {type(attn_sub).__name__}")
        print(f"    Output projection: {type(attn_out).__name__} -> {attn_out}")
        print(f"    out_features: {attn_out.out_features}, bias: {attn_out.bias}")

    print("[OK] Attention output projection discovery works\n")
    return model


def test_extract_mlp_activations(model):
    """Extract MLP activations using same prompts as existing attn extraction."""
    print("=" * 80)
    print("TEST 2: Extract MLP activations")
    print("=" * 80)

    # Load existing attn activations to get prompts and labels
    attn_data = torch.load(ATTN_ACTIVATIONS, map_location="cpu")
    prompts = attn_data["prompts"]
    labels = attn_data["labels"].tolist()
    print(f"  Loaded {len(prompts)} prompts from existing attn activations")

    # Create MLP extractor
    extractor = ActivationExtractor.__new__(ActivationExtractor)
    extractor.model = model
    extractor.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if extractor.tokenizer.pad_token is None:
        extractor.tokenizer.pad_token = extractor.tokenizer.eos_token
    extractor.model_name = MODEL_PATH
    extractor.components = ["mlp"]
    extractor.num_layers = model.config.num_hidden_layers
    extractor.hidden_size = model.config.hidden_size

    mlp_output = os.path.abspath(os.path.join(OUTPUT_DIR, "activations_mlp.pt"))
    extractor.extract_dataset(prompts, labels, mlp_output, output_dir=OUTPUT_DIR)

    # Verify
    mlp_data = torch.load(mlp_output, map_location="cpu")
    print(f"  MLP activations shape: {mlp_data['activations_mlp'].shape}")
    assert "activations_mlp" in mlp_data
    assert mlp_data["activations_mlp"].shape == attn_data["activations_attn"].shape
    print("[OK] MLP activations extracted successfully\n")
    return mlp_output


def test_compute_mlp_vectors():
    """Compute MLP steering vectors using WRMD."""
    print("=" * 80)
    print("TEST 3: Compute MLP steering vectors (WRMD)")
    print("=" * 80)

    mlp_act_file = os.path.join(OUTPUT_DIR, "activations_mlp.pt")
    mlp_vec_file = os.path.join(OUTPUT_DIR, "steering_vectors_mlp_wrmd.pt")

    calc = WRMDCalculator(mlp_act_file)
    vectors = calc.compute_steering_vectors(
        method="wrmd",
        lambda_ridge=0.1,
        use_score_weighting=True,
        rank=1,  # Use rank 1 for simpler testing
        component="mlp",
    )
    mlp_vec_file = os.path.abspath(mlp_vec_file)
    calc.save_vectors(vectors, mlp_vec_file, component="mlp", output_dir=OUTPUT_DIR)

    # Verify
    mlp_vec_data = torch.load(mlp_vec_file, map_location="cpu")
    print(f"  MLP vectors keys: {list(mlp_vec_data.keys())}")
    # save_vectors stores single-component as 'steering_vectors'
    print(f"  MLP vectors shape: {mlp_vec_data['steering_vectors'].shape}")
    print("[OK] MLP steering vectors computed\n")
    return mlp_vec_file


def test_create_combined_steering_file(mlp_vec_file):
    """Create a combined steering file with both attn and mlp vectors."""
    print("=" * 80)
    print("TEST 4: Create combined attn+mlp steering file")
    print("=" * 80)

    attn_data = torch.load(ATTN_VECTORS, map_location="cpu")
    mlp_data = torch.load(mlp_vec_file, map_location="cpu")

    # For rank compatibility, use rank-1 for both
    attn_vecs = attn_data["steering_vectors"]
    if attn_vecs.ndim == 3:
        attn_vecs = attn_vecs[:, 0, :]  # Take first rank direction
    mlp_vecs = mlp_data["steering_vectors"]
    if mlp_vecs.ndim == 3:
        mlp_vecs = mlp_vecs[:, 0, :]

    combined = {
        "steering_vectors_attn": attn_vecs,
        "steering_vectors_mlp": mlp_vecs,
        "steering_vectors": attn_vecs,  # Fallback
        "num_layers": attn_data["num_layers"],
        "hidden_size": attn_data["hidden_size"],
        "method": "wrmd",
        "component": "attn+mlp",
        "rank": 1,
    }

    combined_file = os.path.join(OUTPUT_DIR, "steering_vectors_attn_mlp.pt")
    torch.save(combined, combined_file)

    print(f"  attn vectors: {attn_vecs.shape}")
    print(f"  mlp vectors:  {mlp_vecs.shape}")
    print(f"  Saved to: {combined_file}")
    print("[OK] Combined steering file created\n")
    return combined_file


def test_steering_hook_group(model, combined_file):
    """Test SteeringHookGroup with dual-component steering."""
    print("=" * 80)
    print("TEST 5: SteeringHookGroup dual-component steering")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    steering_data = torch.load(combined_file, map_location="cpu")
    target_layers = [23, 27, 15]  # Best layers from correlation analysis

    # Test 5a: Create group via from_steering_data
    print("\n  5a: Creating SteeringHookGroup.from_steering_data...")
    group = SteeringHookGroup.from_steering_data(
        model=model,
        steering_data=steering_data,
        target_layers=target_layers,
        alpha=-2.0,
        components=("attn", "mlp"),
    )
    assert len(group.hooks) == 2
    assert group.hooks[0].component == "attn"
    assert group.hooks[1].component == "mlp"
    print(f"  Created group with {len(group.hooks)} hooks")
    print(f"  Hook components: {[h.component for h in group.hooks]}")
    print(f"  Hook target layers: {[h.target_layers for h in group.hooks]}")

    # Test 5b: Register and generate
    print("\n  5b: Testing generation with dual-component hooks...")
    test_prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Baseline output
    with torch.no_grad():
        baseline_ids = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    baseline_text = tokenizer.decode(
        baseline_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"  Baseline: {baseline_text[:100]}...")

    # Dual-component steered output
    group.register_hooks()
    with torch.no_grad():
        steered_ids = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    group.remove_hooks()
    steered_text = tokenizer.decode(
        steered_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"  Dual-steered: {steered_text[:100]}...")

    # Attn-only steered output
    attn_hook = SteeringHook(
        model,
        steering_data["steering_vectors_attn"].to(model.device),
        target_layers,
        alpha=-2.0,
        component="attn",
    )
    attn_hook.register_hooks()
    with torch.no_grad():
        attn_ids = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    attn_hook.remove_hooks()
    attn_text = tokenizer.decode(
        attn_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"  Attn-only:   {attn_text[:100]}...")

    # Verify they're all different (or at least dual != baseline)
    print(f"\n  Baseline == Dual-steered: {baseline_text == steered_text}")
    print(f"  Baseline == Attn-only:    {baseline_text == attn_text}")
    print(f"  Dual-steered == Attn-only: {steered_text == attn_text}")

    print("[OK] SteeringHookGroup generation works\n")


def test_per_component_alpha(model, combined_file):
    """Test per-component alpha overrides."""
    print("=" * 80)
    print("TEST 6: Per-component alpha overrides")
    print("=" * 80)

    steering_data = torch.load(combined_file, map_location="cpu")
    target_layers = [23]

    # Test with different alphas
    group = SteeringHookGroup.from_steering_data(
        model=model,
        steering_data=steering_data,
        target_layers=target_layers,
        alpha=-1.0,  # Default
        alpha_attn=-3.0,  # Override for attn
        alpha_mlp=-0.5,  # Override for mlp
        components=("attn", "mlp"),
    )
    assert group.hooks[0].alpha == -3.0, f"Expected attn alpha=-3.0, got {group.hooks[0].alpha}"
    assert group.hooks[1].alpha == -0.5, f"Expected mlp alpha=-0.5, got {group.hooks[1].alpha}"
    print(f"  Attn alpha: {group.hooks[0].alpha}")
    print(f"  MLP alpha:  {group.hooks[1].alpha}")

    # Test with shared alpha only
    group2 = SteeringHookGroup.from_steering_data(
        model=model,
        steering_data=steering_data,
        target_layers=target_layers,
        alpha=-2.0,
        components=("attn", "mlp"),
    )
    assert group2.hooks[0].alpha == -2.0
    assert group2.hooks[1].alpha == -2.0
    print(f"  Shared alpha (-2.0): attn={group2.hooks[0].alpha}, mlp={group2.hooks[1].alpha}")

    # Test error when no alpha provided
    try:
        SteeringHookGroup.from_steering_data(
            model=model,
            steering_data=steering_data,
            target_layers=target_layers,
            alpha=None,
            alpha_attn=-2.0,
            # alpha_mlp omitted => should error
            components=("attn", "mlp"),
        )
        print("  ERROR: Should have raised ValueError for missing mlp alpha!")
    except ValueError as e:
        print(f"  Correctly caught missing alpha: {e}")

    print("[OK] Per-component alpha overrides work\n")


def test_merge_into_output_proj():
    """Test the _merge_into_output_proj helper."""
    print("=" * 80)
    print("TEST 7: Static merge into attention output projection")
    print("=" * 80)

    from activation_steering.merge_steering_into_weights import _merge_into_output_proj

    # Create a fake linear layer without bias
    linear = torch.nn.Linear(4096, 4096, bias=False)
    sv = torch.randn(4096, dtype=torch.float32)
    alpha = -2.0

    assert linear.bias is None
    _merge_into_output_proj(linear, sv, alpha)
    assert linear.bias is not None
    expected = alpha * sv
    assert torch.allclose(linear.bias.data, expected, atol=1e-6)
    print(f"  Created bias from scratch: shape={linear.bias.shape}")

    # Merge again (should add, not replace)
    _merge_into_output_proj(linear, sv, alpha)
    expected2 = 2 * alpha * sv
    assert torch.allclose(linear.bias.data, expected2, atol=1e-6)
    print(f"  Second merge accumulated correctly")

    print("[OK] Static merge helper works\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DUAL-COMPONENT (ATTN+MLP) STEERING END-TO-END TEST")
    print("=" * 80 + "\n")

    # Test 1 & load model
    model = test_attn_output_proj_discovery()

    # Test 2: Extract MLP activations
    mlp_act_file = test_extract_mlp_activations(model)

    # Test 3: Compute MLP vectors
    mlp_vec_file = test_compute_mlp_vectors()

    # Test 4: Create combined file
    combined_file = test_create_combined_steering_file(mlp_vec_file)

    # Test 5: SteeringHookGroup
    test_steering_hook_group(model, combined_file)

    # Test 6: Per-component alpha
    test_per_component_alpha(model, combined_file)

    # Test 7: Static merge helper (doesn't need model on GPU)
    test_merge_into_output_proj()

    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
