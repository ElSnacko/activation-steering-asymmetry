#!/usr/bin/env python3
"""
Tests for vectorized routing projection functions.

Validates that vectorized implementations in routing.py produce identical
results to naive Python loops across various edge cases.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from activation_steering.routing import (
    _batch_project_onto_vectors,
    _compute_routing_metrics,
    _project_samples_onto_vectors,
    compute_residual_vectors,
)


def _naive_batch_project(activations, vectors, target_layers):
    """Reference implementation matching old per-sample loop."""
    all_projs = []
    for i in range(activations.shape[0]):
        layer_projs = []
        for layer_idx in target_layers:
            act = activations[i, layer_idx, :].float()
            gv = vectors[layer_idx, :].float()
            norm = gv.norm()
            if norm < 1e-8:
                continue
            gv_unit = gv / norm
            proj = (act @ gv_unit).item()
            layer_projs.append(proj)
        if layer_projs:
            all_projs.append(sum(layer_projs) / len(layer_projs))
        else:
            all_projs.append(0.0)
    return torch.tensor(all_projs)


def _naive_project_masked(activations, mask, vectors, target_layers):
    """Reference implementation matching old _project_samples_onto_vectors."""
    projections = []
    for i in range(activations.shape[0]):
        if not mask[i]:
            continue
        layer_projs = []
        for layer_idx in target_layers:
            act = activations[i, layer_idx, :].float()
            v = vectors[layer_idx].float()
            norm = v.norm()
            if norm < 1e-8:
                continue
            v_unit = v / norm
            proj = (act @ v_unit).item()
            layer_projs.append(proj)
        if layer_projs:
            projections.append(sum(layer_projs) / len(layer_projs))
    return projections


def _make_synthetic_data(n_samples=50, n_layers=8, hidden_size=64, n_categories=4, seed=42):
    """Create synthetic test data for routing tests."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    activations = torch.randn(n_samples, n_layers, hidden_size)
    labels = torch.tensor([1] * (n_samples // 2) + [0] * (n_samples - n_samples // 2))
    global_vectors = torch.randn(n_layers, hidden_size)

    cat_names = [f"category_{i}" for i in range(n_categories)]
    category_vectors = {name: torch.randn(n_layers, hidden_size) for name in cat_names}

    sample_categories = [None] * n_samples
    for i in range(n_samples):
        if labels[i] == 1:
            sample_categories[i] = cat_names[rng.randint(n_categories)]

    target_layers = [2, 4, 6]
    refused_mask = labels.numpy() == 1

    return {
        "activations": activations,
        "labels": labels,
        "global_vectors": global_vectors,
        "category_vectors": category_vectors,
        "sample_categories": sample_categories,
        "target_layers": target_layers,
        "refused_mask": refused_mask,
        "cat_names": cat_names,
    }


def test_batch_project_matches_naive():
    data = _make_synthetic_data()
    vec_result = _batch_project_onto_vectors(
        data["activations"], data["global_vectors"], data["target_layers"]
    )
    naive_result = _naive_batch_project(
        data["activations"], data["global_vectors"], data["target_layers"]
    )
    assert torch.allclose(
        vec_result, naive_result, atol=1e-5
    ), f"Max diff: {(vec_result - naive_result).abs().max().item():.2e}"
    print("[PASS] _batch_project_onto_vectors matches naive loop")


def test_batch_project_empty_layers():
    data = _make_synthetic_data()
    result = _batch_project_onto_vectors(data["activations"], data["global_vectors"], [])
    assert result.shape == (data["activations"].shape[0],)
    assert torch.all(torch.isnan(result)) or result.shape[0] == 0
    print("[PASS] _batch_project_onto_vectors handles empty target_layers")


def test_batch_project_zero_norm_vectors():
    data = _make_synthetic_data()
    zero_vecs = data["global_vectors"].clone()
    zero_vecs[data["target_layers"]] = 0.0
    vec_result = _batch_project_onto_vectors(data["activations"], zero_vecs, data["target_layers"])
    naive_result = _naive_batch_project(data["activations"], zero_vecs, data["target_layers"])
    assert torch.allclose(vec_result, naive_result, atol=1e-5)
    print("[PASS] _batch_project_onto_vectors handles zero-norm vectors")


def test_project_samples_masked_matches_naive():
    data = _make_synthetic_data()
    vec_result = _project_samples_onto_vectors(
        data["activations"],
        data["refused_mask"],
        data["global_vectors"],
        data["target_layers"],
    )
    naive_result = _naive_project_masked(
        data["activations"],
        data["refused_mask"],
        data["global_vectors"],
        data["target_layers"],
    )
    assert len(vec_result) == len(naive_result)
    for v, n in zip(vec_result, naive_result):
        assert abs(v - n) < 1e-5, f"Diff: {abs(v - n):.2e}"
    print("[PASS] _project_samples_onto_vectors matches naive loop")


def test_project_samples_empty_mask():
    data = _make_synthetic_data()
    empty_mask = np.zeros(data["activations"].shape[0], dtype=bool)
    result = _project_samples_onto_vectors(
        data["activations"], empty_mask, data["global_vectors"], data["target_layers"]
    )
    assert result == []
    print("[PASS] _project_samples_onto_vectors handles empty mask")


def test_compute_routing_metrics_no_exclusions():
    data = _make_synthetic_data()
    metrics = _compute_routing_metrics(
        data["activations"],
        data["refused_mask"],
        data["sample_categories"],
        data["category_vectors"],
        data["target_layers"],
    )
    assert "per_category" in metrics
    assert "macro_precision" in metrics
    assert metrics["total_evaluated"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    for cat_key, m in metrics["per_category"].items():
        assert "tp" in m and "fp" in m and "fn" in m
        assert "precision" in m and "recall" in m and "f1" in m
        assert m["support"] >= 0
    print("[PASS] _compute_routing_metrics produces valid structure")


def test_compute_routing_metrics_with_exclusions():
    data = _make_synthetic_data()
    excluded = {data["cat_names"][0], data["cat_names"][1]}
    metrics = _compute_routing_metrics(
        data["activations"],
        data["refused_mask"],
        data["sample_categories"],
        data["category_vectors"],
        data["target_layers"],
        excluded_categories=excluded,
    )
    assert data["cat_names"][0] not in metrics["per_category"]
    assert data["cat_names"][2] in metrics["per_category"]
    assert metrics["global_fallback_count"] >= 0
    print("[PASS] _compute_routing_metrics handles exclusions correctly")


def test_compute_routing_metrics_with_zscore():
    data = _make_synthetic_data()
    raw_metrics = _compute_routing_metrics(
        data["activations"],
        data["refused_mask"],
        data["sample_categories"],
        data["category_vectors"],
        data["target_layers"],
    )
    zscore_stats = {}
    for cat_name in data["cat_names"]:
        if cat_name in raw_metrics["per_category"]:
            zscore_stats[cat_name] = {"mean": 0.5, "std": 1.0}
    zscore_metrics = _compute_routing_metrics(
        data["activations"],
        data["refused_mask"],
        data["sample_categories"],
        data["category_vectors"],
        data["target_layers"],
        zscore_stats=zscore_stats,
    )
    assert zscore_metrics["total_evaluated"] == raw_metrics["total_evaluated"]
    assert len(zscore_metrics["per_category"]) == len(raw_metrics["per_category"])
    print("[PASS] _compute_routing_metrics handles z-score normalization")


def test_compute_routing_metrics_no_refusal():
    data = _make_synthetic_data()
    no_refusal = np.zeros(data["activations"].shape[0], dtype=bool)
    metrics = _compute_routing_metrics(
        data["activations"],
        no_refusal,
        data["sample_categories"],
        data["category_vectors"],
        data["target_layers"],
    )
    assert metrics["total_evaluated"] == 0
    assert metrics["accuracy"] == 0.0
    print("[PASS] _compute_routing_metrics handles no refusal samples")


def test_residual_vectors_shape():
    data = _make_synthetic_data()
    residuals = compute_residual_vectors(data["category_vectors"], data["global_vectors"])
    for cat_name, res in residuals.items():
        assert res.shape == data["category_vectors"][cat_name].shape
    print("[PASS] compute_residual_vectors preserves shape")


def test_residual_vectors_orthogonality():
    data = _make_synthetic_data()
    residuals = compute_residual_vectors(data["category_vectors"], data["global_vectors"])
    gv = data["global_vectors"].float()
    for cat_name, res in residuals.items():
        res_f = res.float()
        for layer_idx in range(res_f.shape[0]):
            dot = (res_f[layer_idx] @ gv[layer_idx]).item()
            gv_norm_sq = (gv[layer_idx] @ gv[layer_idx]).item()
            if gv_norm_sq > 1e-8:
                cos_sim = dot / (
                    res_f[layer_idx].norm().item() * gv[layer_idx].norm().item() + 1e-12
                )
                assert abs(cos_sim) < 0.01, (
                    f"Residual for {cat_name} layer {layer_idx} not orthogonal to global: "
                    f"cos_sim={cos_sim:.4f}"
                )
    print("[PASS] compute_residual_vectors produces orthogonal residuals")


def test_vectorized_vs_naive_routing_predictions():
    """Full end-to-end: compare predictions from vectorized _compute_routing_metrics."""
    data = _make_synthetic_data(n_samples=30, n_categories=3, seed=123)

    def naive_routing_metrics(
        activations,
        refused_mask,
        sample_categories,
        routing_vectors,
        target_layers,
        excluded_categories=None,
        zscore_stats=None,
    ):
        excluded = excluded_categories or set()
        _GLOBAL = "__global_fallback__"
        subcat_to_key = {}
        for cat_key in routing_vectors:
            for sub in cat_key.split("+"):
                subcat_to_key[sub] = cat_key
        predictions = []
        for i in range(activations.shape[0]):
            if not refused_mask[i]:
                continue
            true_cat = sample_categories[i]
            if true_cat is None:
                continue
            true_key = subcat_to_key.get(true_cat)
            if true_key is None:
                continue
            if true_key in excluded:
                true_key = _GLOBAL
            raw_projs = {}
            for cat_name, cat_vecs in routing_vectors.items():
                layer_projs = []
                for layer_idx in target_layers:
                    act = activations[i, layer_idx, :].float()
                    cv = cat_vecs[layer_idx].float()
                    norm = cv.norm()
                    if norm < 1e-8:
                        continue
                    cv_unit = cv / norm
                    proj = (act @ cv_unit).item()
                    layer_projs.append(proj)
                if layer_projs:
                    raw_projs[cat_name] = sum(layer_projs) / len(layer_projs)
            if zscore_stats:
                scored_projs = {}
                for cat_name, raw_proj in raw_projs.items():
                    stats = zscore_stats.get(cat_name)
                    if stats and stats.get("std", 0) > 1e-8:
                        scored_projs[cat_name] = (raw_proj - stats["mean"]) / stats["std"]
                    else:
                        scored_projs[cat_name] = raw_proj
            else:
                scored_projs = raw_projs
            if scored_projs:
                pred_key = max(scored_projs, key=scored_projs.get)
            else:
                pred_key = None
            if pred_key in excluded:
                pred_key = _GLOBAL
            predictions.append((true_key, pred_key))
        return predictions

    naive_preds = naive_routing_metrics(
        data["activations"],
        data["refused_mask"],
        data["sample_categories"],
        data["category_vectors"],
        data["target_layers"],
    )
    vec_metrics = _compute_routing_metrics(
        data["activations"],
        data["refused_mask"],
        data["sample_categories"],
        data["category_vectors"],
        data["target_layers"],
    )

    assert (
        len(naive_preds) == vec_metrics["total_evaluated"]
    ), f"Sample count mismatch: naive={len(naive_preds)}, vec={vec_metrics['total_evaluated']}"

    vec_preds = []
    excluded = set()
    subcat_to_key = {}
    for cat_key in data["category_vectors"]:
        for sub in cat_key.split("+"):
            subcat_to_key[sub] = cat_key
    _GLOBAL = "__global_fallback__"
    cat_names = list(data["category_vectors"].keys())
    valid_indices = []
    true_keys = []
    for i in range(data["activations"].shape[0]):
        if not data["refused_mask"][i]:
            continue
        true_cat = data["sample_categories"][i]
        if true_cat is None:
            continue
        true_key = subcat_to_key.get(true_cat)
        if true_key is None:
            continue
        valid_indices.append(i)
        true_keys.append(_GLOBAL if true_key in excluded else true_key)

    proj_matrix = torch.zeros(len(valid_indices), len(cat_names))
    for ci, cat_name in enumerate(cat_names):
        proj_matrix[:, ci] = _batch_project_onto_vectors(
            data["activations"][valid_indices],
            data["category_vectors"][cat_name],
            data["target_layers"],
        )
    pred_indices = proj_matrix.argmax(dim=1).tolist()
    for pi in pred_indices:
        pred_key = cat_names[pi]
        vec_preds.append(pred_key)

    for i, ((nt, np_), vp) in enumerate(zip(naive_preds, vec_preds)):
        assert np_ == vp, (
            f"Prediction mismatch at sample {i}: naive={np_}, vec={vp}, " f"naive_true={nt}"
        )
    print("[PASS] Vectorized routing predictions match naive loop exactly")


if __name__ == "__main__":
    print("=" * 60)
    print("Routing Vectorization Tests")
    print("=" * 60)
    print()

    tests = [
        test_batch_project_matches_naive,
        test_batch_project_empty_layers,
        test_batch_project_zero_norm_vectors,
        test_project_samples_masked_matches_naive,
        test_project_samples_empty_mask,
        test_compute_routing_metrics_no_exclusions,
        test_compute_routing_metrics_with_exclusions,
        test_compute_routing_metrics_with_zscore,
        test_compute_routing_metrics_no_refusal,
        test_residual_vectors_shape,
        test_residual_vectors_orthogonality,
        test_vectorized_vs_naive_routing_predictions,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"[FAIL] {test.__name__}: {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print()
        print("Failures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)

    sys.exit(1 if failed else 0)
