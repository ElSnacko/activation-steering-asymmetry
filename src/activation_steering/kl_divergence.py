"""
KL divergence measurement for steering quality assessment.

Measures how much steering distorts the model's output distribution on harmless
prompts. Lower KL divergence means the steered model behaves more like the
unsteered model on normal inputs — a robust capability-preservation signal that
doesn't depend on a judge model.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Default harmless prompts for KL measurement.  These should be benign
# questions the model answers fluently — any distribution shift here is
# collateral damage from steering.
DEFAULT_HARMLESS_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "Write a haiku about autumn.",
    "What are the three states of matter?",
    "How does a bicycle work?",
    "Summarize the plot of Romeo and Juliet.",
    "What is the Pythagorean theorem?",
    "Name five planets in our solar system.",
    "What causes rainbows?",
    "Describe the water cycle.",
    "What is the speed of light?",
    "Who wrote Pride and Prejudice?",
    "What is machine learning?",
    "How do magnets work?",
    "What is the largest ocean on Earth?",
    "Explain what DNA is.",
    "What are prime numbers?",
    "How does a compass work?",
    "What is the boiling point of water?",
    "Describe how a camera works.",
]


# Open-ended generation prompts for KL measurement.  These elicit longer,
# more varied responses than the factual harmless prompts — making KL
# sensitive to degradation in fluency, coherence, and style.
DEFAULT_GENERATION_PROMPTS = [
    "Write a short story about a robot discovering music for the first time.",
    "Explain the concept of entropy to a curious ten-year-old.",
    "Compare and contrast living in a big city versus a small town.",
    "Describe what a day in the life of a deep-sea marine biologist looks like.",
    "Write a persuasive argument for why everyone should learn to cook.",
    "Explain how a computer translates source code into machine instructions.",
    "Describe the process of making pottery from raw clay to finished product.",
    "Write a travel guide paragraph about visiting Kyoto in autumn.",
    "Explain why bridges are built in different shapes and designs.",
    "Describe how you would plan a community garden from scratch.",
    "Write a dialogue between two scientists debating the ethics of gene editing.",
    "Explain the water cycle and why it matters for agriculture.",
    "Describe a sunset over the ocean as vividly as possible.",
    "Write instructions for teaching someone to ride a bicycle.",
    "Explain the historical significance of the printing press.",
]


@dataclass
class KLResult:
    """Results from KL divergence measurement."""

    mean_kl: float
    max_kl: float
    min_kl: float
    std_kl: float
    per_prompt_kl: List[float]
    num_prompts: int
    num_tokens_avg: float


def collect_first_token_logits(
    model,
    tokenizer,
    prompts: List[str],
    show_progress: bool = True,
) -> List[torch.Tensor]:
    """
    Collect first-token logits for a set of prompts via a single forward pass.

    For each prompt, runs one forward pass and returns the logit vector at the
    last prompt position (i.e. the distribution over the first generated token).
    No generation/decoding is performed, so baseline and steered logits are
    always conditioned on the exact same token sequence — isolating the direct
    distributional effect of steering from trajectory divergence.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        prompts: Prompts to collect logits for.
        show_progress: Show tqdm progress bar.

    Returns:
        List of tensors, each of shape [1, vocab_size] (one position per prompt).
    """
    all_logits = []
    iterator = tqdm(prompts, desc="Collecting first-token logits") if show_progress else prompts

    device = next(model.parameters()).device

    for prompt in iterator:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Logit at the last prompt position = P(first generated token | prompt).
        # Shape: [1, vocab_size].  Move to CPU float32 to avoid bfloat16
        # precision artifacts.
        logits = outputs.logits[:, -1:, :].cpu().float()  # [batch=1, 1, vocab]
        all_logits.append(logits[0])  # [1, vocab_size]

    return all_logits


def collect_logits(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 32,
    show_progress: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Collect output logits and generated token sequences via greedy generation.

    Returns both logits and generated token IDs so the sequences can be reused
    for teacher-forced KL measurement via collect_teacher_forced_logits.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        prompts: Prompts to collect logits for.
        max_new_tokens: Number of tokens to generate per prompt.
        show_progress: Show tqdm progress bar.

    Returns:
        Tuple of (logits_list, sequences_list) where:
        - logits_list: List of tensors, each [num_generated_tokens, vocab_size].
        - sequences_list: List of tensors, each [prompt_len + num_generated_tokens]
          (full input_ids including prompt and generated tokens).
    """
    all_logits = []
    all_sequences = []
    device = next(model.parameters()).device
    iterator = tqdm(prompts, desc="Collecting logits") if show_progress else prompts

    for prompt in iterator:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # outputs.scores is a tuple of (num_generated_tokens,) tensors,
        # each of shape [batch_size, vocab_size]
        if outputs.scores:
            # Stack into [num_tokens, vocab_size], move to CPU in float32
            # to avoid bfloat16 precision artifacts that can produce extreme
            # logit values (e.g. 3.9e16) causing KL divergence blowups.
            logits = torch.stack([s[0] for s in outputs.scores]).cpu().float()
        else:
            _vs = getattr(model.config, "vocab_size", None) or getattr(
                getattr(model.config, "text_config", None), "vocab_size", 1
            )
            logits = torch.empty(0, _vs)

        all_logits.append(logits)
        all_sequences.append(outputs.sequences[0].cpu())  # [prompt_len + gen_len]

    return all_logits, all_sequences


def collect_teacher_forced_logits(
    model,
    tokenizer,
    prompts: List[str],
    baseline_sequences: List[torch.Tensor],
    show_progress: bool = True,
    batch_size: int = 0,
) -> List[torch.Tensor]:
    """
    Collect per-position logits by teacher-forcing on baseline sequences.

    For each generated position k, runs a forward pass on the prefix
    ``baseline_sequence[:input_len + k + 1]`` and extracts the last-position
    logit.  This measures each position independently — like a mad lib — so
    steering hooks (which steer only the last position) perturb exactly one
    position per pass, matching generation behavior with no compounding.

    When batch_size > 0, all prefixes for a single prompt are batched together
    into one forward pass (left-padded), reducing N forward passes to 1 per
    prompt. This produces identical results to the serial version because each
    prefix in the batch has its own last position where the hook fires.

    Args:
        model: HuggingFace causal LM (may have steering hooks registered).
        tokenizer: Matching tokenizer.
        prompts: The same prompts used for baseline generation (used to
                 determine prompt length for each sequence).
        baseline_sequences: Token ID sequences from collect_logits, each of
                           shape [prompt_len + num_generated_tokens].
        show_progress: Show tqdm progress bar.
        batch_size: If > 0, batch all prefixes per prompt into a single
                    forward pass. If 0 (default), use serial forward passes.
                    Batched mode uses more memory (~num_generated * max_seq_len *
                    hidden_size) but is ~N x faster where N = num_generated_tokens.

    Returns:
        List of tensors, each of shape [num_generated_tokens, vocab_size].
    """
    all_logits = []
    device = next(model.parameters()).device
    desc = "Collecting teacher-forced logits"
    iterator = (
        tqdm(zip(prompts, baseline_sequences), desc=desc, total=len(prompts))
        if show_progress
        else zip(prompts, baseline_sequences)
    )

    for prompt, seq in iterator:
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        input_len = prompt_inputs["input_ids"].shape[1]
        num_generated = seq.shape[0] - input_len

        if num_generated <= 0:
            all_logits.append(torch.empty(0, model.config.vocab_size))
            continue

        if batch_size > 0:
            all_logits.append(
                _collect_teacher_forced_batched(model, seq, input_len, num_generated, device)
            )
        else:
            all_logits.append(
                _collect_teacher_forced_serial(model, seq, input_len, num_generated, device)
            )

    return all_logits


def _collect_teacher_forced_serial(model, seq, input_len, num_generated, device):
    """Serial teacher-forced logit collection: one forward pass per position."""
    position_logits = []
    for k in range(num_generated):
        prefix_len = input_len + k + 1
        prefix = seq[:prefix_len].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=prefix)

        logit_k = outputs.logits[0, -1, :].cpu().float()
        position_logits.append(logit_k)

    return torch.stack(position_logits)


def _collect_teacher_forced_batched(model, seq, input_len, num_generated, device, chunk_size=8):
    """Batched teacher-forced logit collection: one forward pass per chunk.

    Builds a left-padded batch of all N prefixes and runs forward passes in
    chunks to avoid OOM on the lm_head projection. Each prefix has its last
    real token at the rightmost position, so steering hooks fire independently
    per prefix — producing identical results to the serial version.
    """
    max_prefix_len = input_len + num_generated
    pad_token_id = getattr(model.config, "pad_token_id", None) or 0

    batch_ids = torch.full((num_generated, max_prefix_len), pad_token_id, dtype=seq.dtype)
    for k in range(num_generated):
        prefix_len = input_len + k + 1
        batch_ids[k, max_prefix_len - prefix_len :] = seq[:prefix_len]

    attention_mask = torch.zeros(num_generated, max_prefix_len, dtype=torch.long)
    for k in range(num_generated):
        prefix_len = input_len + k + 1
        attention_mask[k, max_prefix_len - prefix_len :] = 1

    all_logits = []
    for start in range(0, num_generated, chunk_size):
        end = min(start + chunk_size, num_generated)
        chunk_ids = batch_ids[start:end].to(device)
        chunk_mask = attention_mask[start:end].to(device)

        with torch.no_grad():
            outputs = model(input_ids=chunk_ids, attention_mask=chunk_mask)

        all_logits.append(outputs.logits[:, -1, :].cpu().float())

    logits = torch.cat(all_logits, dim=0)

    return logits


def compute_kl_divergence(
    baseline_logits: List[torch.Tensor],
    steered_logits: List[torch.Tensor],
) -> KLResult:
    """
    Compute per-prompt KL divergence between baseline and steered distributions.

    Uses KL(steered || baseline) — how many extra bits the steered model
    "wastes" compared to baseline.  Computed in float32 for numerical stability.

    Args:
        baseline_logits: Logits from unsteered model, one tensor per prompt.
        steered_logits: Logits from steered model, one tensor per prompt.

    Returns:
        KLResult with per-prompt and aggregate KL statistics.
    """
    if len(baseline_logits) != len(steered_logits):
        raise ValueError(
            f"Mismatched prompt counts: {len(baseline_logits)} baseline vs "
            f"{len(steered_logits)} steered"
        )

    per_prompt_kl = []
    total_tokens = 0

    for b_logits, s_logits in zip(baseline_logits, steered_logits):
        # Use the shorter sequence length (generation may differ)
        min_len = min(b_logits.shape[0], s_logits.shape[0])
        if min_len == 0:
            per_prompt_kl.append(0.0)
            continue

        b = b_logits[:min_len].float()
        s = s_logits[:min_len].float()

        # Skip prompts with Inf/NaN logits — extreme steering can push
        # bfloat16 hidden states to overflow, making KL undefined.
        if not (torch.isfinite(b).all() and torch.isfinite(s).all()):
            total_tokens += min_len
            continue

        # KL(steered || baseline) = sum steered * (log steered - log baseline)
        b_log_probs = F.log_softmax(b, dim=-1)
        s_log_probs = F.log_softmax(s, dim=-1)
        s_probs = F.softmax(s, dim=-1)

        # Per-token KL, then average over tokens for this prompt.
        # KL(P||Q) is unbounded when Q has near-zero mass on tokens P
        # concentrates on.  Cap at 100 nats to absorb numerical noise
        # without masking genuine distribution shifts.
        kl_per_token = (s_probs * (s_log_probs - b_log_probs)).sum(dim=-1)
        kl_per_token = torch.clamp(kl_per_token, min=0.0, max=100.0)
        prompt_kl = kl_per_token.mean().item()

        if not math.isfinite(prompt_kl):
            total_tokens += min_len
            continue

        per_prompt_kl.append(prompt_kl)
        total_tokens += min_len

    # Filter any residual non-finite values before aggregation
    valid_kl = [k for k in per_prompt_kl if math.isfinite(k)]
    num_prompts = len(valid_kl)
    mean_kl = sum(valid_kl) / max(num_prompts, 1)
    max_kl = max(valid_kl) if valid_kl else 0.0
    min_kl = min(valid_kl) if valid_kl else 0.0
    std_kl = (sum((k - mean_kl) ** 2 for k in valid_kl) / max(num_prompts - 1, 1)) ** 0.5

    return KLResult(
        mean_kl=mean_kl,
        max_kl=max_kl,
        min_kl=min_kl,
        std_kl=std_kl,
        per_prompt_kl=per_prompt_kl,
        num_prompts=num_prompts,
        num_tokens_avg=total_tokens / max(num_prompts, 1),
    )


def load_harmless_prompts(
    path: Optional[str] = None,
    tokenizer=None,
    max_prompts: int = 20,
    prompt_set: str = "harmless",
) -> List[str]:
    """
    Load prompts for KL measurement.

    Args:
        path: Optional path to a JSON file with a list of prompt strings.
              Falls back to built-in defaults if not provided.
        tokenizer: If provided and it has a chat template, prompts are
                   formatted with the chat template.
        max_prompts: Maximum number of prompts to return.
        prompt_set: Which built-in set to use when path is None.
                    "harmless" (default) — short factual questions.
                    "generation" — open-ended prompts that elicit longer,
                    more varied responses for generation quality KL.

    Returns:
        List of formatted prompt strings.
    """
    import json

    if path is not None:
        with open(path) as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            raise ValueError(f"Expected a JSON list of strings, got {type(prompts)}")
    elif prompt_set == "generation":
        prompts = list(DEFAULT_GENERATION_PROMPTS)
    else:
        prompts = list(DEFAULT_HARMLESS_PROMPTS)

    prompts = prompts[:max_prompts]

    # Apply chat template if available
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = []
            for p in prompts:
                messages = [{"role": "user", "content": p}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted.append(text)
            prompts = formatted
        except Exception:
            pass  # Fall back to raw prompts for base models

    return prompts
