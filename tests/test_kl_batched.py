"""
Tests for batched teacher-forced KL divergence collection.

Validates that the batched forward pass produces identical logits to the
serial (one-prefix-at-a-time) approach.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from activation_steering.kl_divergence import (
    _collect_teacher_forced_batched,
    _collect_teacher_forced_serial,
    compute_kl_divergence,
)


class TinyCausalLM(nn.Module):
    """Minimal causal LM for testing. Returns logits proportional to input."""

    def __init__(self, vocab_size=32, hidden_size=16):
        super().__init__()
        self.config = type("Config", (), {"pad_token_id": 0, "vocab_size": vocab_size})()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        logits = self.head(hidden)
        return type("Output", (), {"logits": logits})()


def _make_sequence(input_len, num_generated, vocab_size):
    seq = torch.randint(1, vocab_size - 1, (input_len + num_generated,))
    return seq


def test_batched_matches_serial_basic():
    vocab_size = 32
    model = TinyCausalLM(vocab_size=vocab_size, hidden_size=16)
    model.eval()

    input_len = 5
    num_generated = 8
    seq = _make_sequence(input_len, num_generated, vocab_size)
    device = next(model.parameters()).device

    serial = _collect_teacher_forced_serial(model, seq, input_len, num_generated, device)
    batched = _collect_teacher_forced_batched(model, seq, input_len, num_generated, device)

    assert serial.shape == batched.shape == (num_generated, vocab_size)
    assert torch.allclose(
        serial, batched, atol=1e-5
    ), f"Max diff: {(serial - batched).abs().max().item():.6e}"
    print("[PASS] Batched matches serial (basic)")


def test_batched_matches_serial_longer():
    vocab_size = 64
    model = TinyCausalLM(vocab_size=vocab_size, hidden_size=32)
    model.eval()

    input_len = 10
    num_generated = 32
    seq = _make_sequence(input_len, num_generated, vocab_size)
    device = next(model.parameters()).device

    serial = _collect_teacher_forced_serial(model, seq, input_len, num_generated, device)
    batched = _collect_teacher_forced_batched(model, seq, input_len, num_generated, device)

    assert serial.shape == batched.shape == (num_generated, vocab_size)
    assert torch.allclose(
        serial, batched, atol=1e-5
    ), f"Max diff: {(serial - batched).abs().max().item():.6e}"
    print("[PASS] Batched matches serial (32 generated tokens)")


def test_batched_matches_serial_single_token():
    vocab_size = 16
    model = TinyCausalLM(vocab_size=vocab_size, hidden_size=8)
    model.eval()

    input_len = 3
    num_generated = 1
    seq = _make_sequence(input_len, num_generated, vocab_size)
    device = next(model.parameters()).device

    serial = _collect_teacher_forced_serial(model, seq, input_len, num_generated, device)
    batched = _collect_teacher_forced_batched(model, seq, input_len, num_generated, device)

    assert serial.shape == batched.shape == (1, vocab_size)
    assert torch.allclose(serial, batched, atol=1e-5)
    print("[PASS] Batched matches serial (single generated token)")


def test_kl_divergence_symmetric():
    """Verify KL divergence is consistent between serial and batched paths."""
    vocab_size = 32
    model = TinyCausalLM(vocab_size=vocab_size, hidden_size=16)
    model.eval()

    input_len = 4
    num_generated = 6
    seq = _make_sequence(input_len, num_generated, vocab_size)
    device = next(model.parameters()).device

    baseline = _collect_teacher_forced_serial(model, seq, input_len, num_generated, device)
    steered_serial = _collect_teacher_forced_serial(model, seq, input_len, num_generated, device)
    steered_batched = _collect_teacher_forced_batched(model, seq, input_len, num_generated, device)

    kl_serial = compute_kl_divergence([baseline], [steered_serial])
    kl_batched = compute_kl_divergence([baseline], [steered_batched])

    assert abs(kl_serial.mean_kl - kl_batched.mean_kl) < 1e-6
    print(f"[PASS] KL consistent: serial={kl_serial.mean_kl:.6f}, batched={kl_batched.mean_kl:.6f}")


if __name__ == "__main__":
    test_batched_matches_serial_basic()
    test_batched_matches_serial_longer()
    test_batched_matches_serial_single_token()
    test_kl_divergence_symmetric()
    print("\nAll KL batched tests passed!")
