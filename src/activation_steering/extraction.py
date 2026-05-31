#!/usr/bin/env python3
"""
Activation extraction module.

Extracts hidden state activations from transformer models at specified
submodule outputs (attention, MLP, or full layer).
"""

import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import extract_model_name, generate_run_id, setup_model_run_dirs

VALID_COMPONENTS = ("layer", "attn", "mlp")

_ATTN_SUBMODULE_NAMES = ("self_attn", "linear_attn", "attention", "attn")

_ATTN_OUTPUT_PROJ_NAMES = ("o_proj", "out_proj", "dense")


def _get_attn_submodule(layer):
    """Find the attention submodule on a decoder layer, trying multiple names.

    Different architectures use different attribute names:
      - Llama, Qwen2, Mistral: self_attn
      - Qwen3.5 (GatedDeltaNet layers): linear_attn
      - Other: attention, attn
    """
    for name in _ATTN_SUBMODULE_NAMES:
        if hasattr(layer, name):
            return getattr(layer, name)
    children = [n for n, _ in layer.named_children()]
    raise AttributeError(
        f"Cannot find attention submodule in {type(layer).__name__}. "
        f"Tried: {_ATTN_SUBMODULE_NAMES}. Available: {children}"
    )


def _get_attn_output_proj(layer):
    """Find the attention output projection linear layer on a decoder layer.

    This is the last linear layer in the attention block before the residual
    add — the attention analog of MLP's down_proj. Used for static merge.

    Different architectures use different attribute names:
      - Llama, Qwen2, Qwen3, Mistral: self_attn.o_proj
      - GPT-2, GPT-Neo: self_attn.out_proj
      - BERT-style: self_attn.dense

    Returns:
        The output projection nn.Linear module

    Raises:
        AttributeError: If no known output projection attribute is found
    """
    attn = _get_attn_submodule(layer)
    for name in _ATTN_OUTPUT_PROJ_NAMES:
        if hasattr(attn, name):
            return getattr(attn, name)
    children = [n for n, _ in attn.named_children()]
    raise AttributeError(
        f"Cannot find attention output projection in {type(attn).__name__}. "
        f"Tried: {_ATTN_OUTPUT_PROJ_NAMES}. Available: {children}"
    )


class ActivationExtractor:
    """
    Extracts activations from a language model using forward hooks.

    Args:
        model_name: HuggingFace model name or path
        components: List of submodule types to extract from.
            'layer' = full layer output, 'attn' = attention output, 'mlp' = MLP output.
            Default: ['attn'].
    """

    def __init__(self, model_name, components=None, quantize=None, enable_thinking=False):
        if components is None:
            components = ["attn"]
        self.components = components
        self.model_name = model_name
        self.enable_thinking = enable_thinking

        from .utils import get_model_hidden_size, get_model_layers, get_model_num_layers, load_model

        self.model, self.tokenizer = load_model(model_name, quantize=quantize)

        self.num_layers = get_model_num_layers(self.model)
        self.hidden_size = get_model_hidden_size(self.model)
        self._layers = get_model_layers(self.model)

        print(f"[OK] Model loaded: {self.num_layers} layers, {self.hidden_size} hidden size")
        print(f"   Components: {self.components}")

    def extract_last_token_activations(self, prompt):
        """
        Extract activations at last token position for all layers.

        Applies the chat template before tokenizing so that activations match
        what the model sees during generation (with SteeringHook).

        Args:
            prompt: Input text prompt

        Returns:
            If single component: Tensor of shape [num_layers, hidden_size]
            If multiple components: dict mapping component name to tensor
        """
        # Apply chat template to match runtime conditions (SteeringHook)
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except Exception:
            formatted = prompt
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        # Per-component activation storage
        component_activations = {c: {} for c in self.components}

        def make_hook(component, layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                component_activations[component][layer_idx] = hidden_states[:, -1, :].detach().cpu()

            return hook

        hooks = []
        for i, layer in enumerate(self._layers):
            if "layer" in self.components:
                hooks.append(layer.register_forward_hook(make_hook("layer", i)))
            if "attn" in self.components:
                hooks.append(_get_attn_submodule(layer).register_forward_hook(make_hook("attn", i)))
            if "mlp" in self.components:
                hooks.append(layer.mlp.register_forward_hook(make_hook("mlp", i)))

        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        # Build result per component
        result = {}
        for comp in self.components:
            result[comp] = torch.stack(
                [component_activations[comp][i] for i in range(self.num_layers)]
            ).squeeze(1)

        # Single component: return tensor directly for backward compat
        if len(self.components) == 1:
            return result[self.components[0]]

        return result

    def extract_dataset(
        self, prompts, labels, output_file, metadata=None, output_dir=None, batch_size=1
    ):
        """
        Extract activations for a list of prompts.

        Registers hooks once, accumulates activations on GPU, transfers once
        at the end. Supports batched forward passes for higher GPU utilization.

        Args:
            prompts: List of strings
            labels: List of 0 (compliant) or 1 (refusal)
            output_file: Filename to save (or full path if absolute)
            metadata: Optional dict with extra info (judge scores, etc.)
            output_dir: Optional custom output directory
            batch_size: Number of prompts per forward pass. Default 1 (serial).
                Set > 1 for higher GPU utilization when prompts have similar lengths.
        """
        from .utils import extract_model_name, generate_run_id, setup_model_run_dirs

        if not prompts:
            raise ValueError(
                "No prompts to extract activations from. "
                "Check that --results-dir contains censor_scores.json files "
                "and that thresholds aren't filtering out all samples."
            )

        num_prompts = len(prompts)
        device = self.model.device

        gpu_activations = {}
        for c in self.components:
            gpu_activations[c] = torch.zeros(
                num_prompts, self.num_layers, self.hidden_size, device=device
            )

        prompt_idx = [0]

        def make_hook(component, layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                idx = prompt_idx[0]
                bs = hidden_states.shape[0]
                end = min(idx + bs, num_prompts)
                actual_bs = end - idx
                if actual_bs > 0:
                    gpu_activations[component][idx:end, layer_idx] = hidden_states[
                        :actual_bs, -1, :
                    ].detach()

            return hook

        hooks = []
        for i, layer in enumerate(self._layers):
            if "layer" in self.components:
                hooks.append(layer.register_forward_hook(make_hook("layer", i)))
            if "attn" in self.components:
                hooks.append(_get_attn_submodule(layer).register_forward_hook(make_hook("attn", i)))
            if "mlp" in self.components:
                hooks.append(layer.mlp.register_forward_hook(make_hook("mlp", i)))

        print(f"\n[COMPUTE] Extracting activations for {num_prompts} prompts...")
        print(f"   Components: {self.components}, Batch size: {batch_size}")

        original_padding_side = self.tokenizer.padding_side

        try:
            with torch.no_grad():
                if batch_size <= 1:
                    for prompt in tqdm(prompts, desc="Processing"):
                        messages = [{"role": "user", "content": prompt}]
                        try:
                            formatted = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=self.enable_thinking,
                            )
                        except Exception:
                            formatted = prompt
                        inputs = self.tokenizer(formatted, return_tensors="pt").to(device)
                        self.model(**inputs)
                        prompt_idx[0] += 1
                else:
                    self.tokenizer.padding_side = "left"
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    formatted_prompts = []
                    for prompt in prompts:
                        messages = [{"role": "user", "content": prompt}]
                        try:
                            formatted = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=self.enable_thinking,
                            )
                        except Exception:
                            formatted = prompt
                        formatted_prompts.append(formatted)

                    for batch_start in tqdm(
                        range(0, num_prompts, batch_size),
                        desc=f"Processing (bs={batch_size})",
                    ):
                        batch = formatted_prompts[batch_start : batch_start + batch_size]
                        inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(device)
                        self.model(**inputs)
                        prompt_idx[0] += len(batch)
        finally:
            self.tokenizer.padding_side = original_padding_side
            for hook in hooks:
                hook.remove()

        all_activations = {c: gpu_activations[c].cpu() for c in self.components}

        save_data = {
            "labels": torch.tensor(labels),
            "prompts": prompts,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "components": self.components,
        }

        multi_component = len(self.components) > 1
        if multi_component:
            for c in self.components:
                key = "activations" if c == "layer" else f"activations_{c}"
                save_data[key] = all_activations[c]
                print(f"   {c}: {all_activations[c].shape}")
        else:
            comp = self.components[0]
            key = "activations" if comp == "layer" else f"activations_{comp}"
            save_data[key] = all_activations[comp]
            if comp == "mlp":
                save_data["activations"] = all_activations[comp]

        if metadata:
            save_data["metadata"] = metadata

        if output_dir is None:
            model_name = extract_model_name(self.model_name)
            run_id = generate_run_id()
            output_dirs = setup_model_run_dirs(model_name=model_name, run_id=run_id)
            output_dir = output_dirs["extract_activations"]

        if not os.path.isabs(output_file):
            output_path = os.path.join(output_dir, output_file)
        else:
            output_path = output_file

        print(f"\n[SAVE] Saving to {output_path}...")
        torch.save(save_data, output_path)

        print(f"[OK] Saved {num_prompts} samples")
        print(f"   Refusal: {labels.count(1)}, Compliant: {labels.count(0)}")


def load_prompts_from_judge_scores(
    results_dir,
    refusal_threshold=0.1,
    compliance_threshold=-0.1,
):
    """
    Load prompts based on actual judge scores from LLM-Refusal-Evaluation.

    Each sample is labeled purely by its judge score — split names are irrelevant.
    All splits with censor_scores.json are auto-discovered and processed.

    Args:
        results_dir: Path to evaluation results directory
        refusal_threshold: Judge score above this = refusal (default 0.1)
        compliance_threshold: Judge score below this = compliant (default -0.1)

    Returns:
        Tuple of (prompts, labels, metadata)
    """
    all_prompts = []
    all_labels = []
    all_metadata = []

    print(f"[LOAD] Loading judge scores from: {results_dir}")
    print(f"   Refusal threshold: > {refusal_threshold}")
    print(f"   Compliance threshold: < {compliance_threshold}")
    print()

    # Auto-discover all splits containing censor_scores.json
    splits = []
    if os.path.isdir(results_dir):
        for entry in sorted(os.listdir(results_dir)):
            entry_path = os.path.join(results_dir, entry)
            score_file = os.path.join(entry_path, "censor_scores.json")
            if os.path.isdir(entry_path) and os.path.exists(score_file):
                splits.append(entry)

    if splits:
        print(f"[INFO] Discovered splits: {splits}")
        print()

        for split in splits:
            score_file = os.path.join(results_dir, split, "censor_scores.json")

            with open(score_file) as f:
                data = json.load(f)

            refusal_count = 0
            compliant_count = 0
            skipped_count = 0

            for item in data:
                score = item["answer_censor_score"]
                prompt = item["prompt"]

                if score > refusal_threshold:
                    all_prompts.append(prompt)
                    all_labels.append(1)
                    all_metadata.append(
                        {
                            "split": split,
                            "score": score,
                            "category": item.get("category"),
                            "prompt_hash": item.get("prompt_hash"),
                            "source_dataset": item.get("source_dataset"),
                            "source_split": item.get("source_split"),
                        }
                    )
                    refusal_count += 1
                elif score < compliance_threshold:
                    all_prompts.append(prompt)
                    all_labels.append(0)
                    all_metadata.append(
                        {
                            "split": split,
                            "score": score,
                            "category": item.get("category"),
                            "prompt_hash": item.get("prompt_hash"),
                            "source_dataset": item.get("source_dataset"),
                            "source_split": item.get("source_split"),
                        }
                    )
                    compliant_count += 1
                else:
                    skipped_count += 1

            print(f"  {split}:")
            print(f"    Refused: {refusal_count}")
            print(f"    Complied: {compliant_count}")
            print(f"    Uncertain (skipped): {skipped_count}")
    else:
        print(f"[WARN] No splits with censor_scores.json found in {results_dir}")

    if not all_prompts:
        print(
            f"\n[INFO] censor_scores.json yielded no prompts, falling back to judge_scores.json..."
        )
        judge_splits = []
        if os.path.isdir(results_dir):
            for entry in sorted(os.listdir(results_dir)):
                entry_path = os.path.join(results_dir, entry)
                judge_file = os.path.join(entry_path, "judge_scores.json")
                if os.path.isdir(entry_path) and os.path.exists(judge_file):
                    judge_splits.append(entry)

        if not judge_splits:
            print(f"[WARN] No splits with judge_scores.json found either")
        else:
            print(f"[INFO] Found judge_scores.json splits: {judge_splits}")
            print()

            for split in judge_splits:
                judge_file = os.path.join(results_dir, split, "judge_scores.json")

                with open(judge_file) as f:
                    data = json.load(f)

                refusal_count = 0
                compliant_count = 0
                skipped_count = 0

                for prompt_entry in data:
                    if not prompt_entry:
                        skipped_count += 1
                        continue

                    labels = []
                    prompt_text = None
                    category = None
                    for seq in prompt_entry:
                        label = seq.get("label")
                        if label is not None and isinstance(label, (int, float)):
                            labels.append(float(label))
                        if prompt_text is None:
                            prompt_text = seq.get("prompt")
                        if category is None:
                            category = seq.get("category")

                    if not labels or prompt_text is None:
                        skipped_count += 1
                        continue

                    avg_label = sum(labels) / len(labels)

                    if avg_label > refusal_threshold:
                        all_prompts.append(prompt_text)
                        all_labels.append(1)
                        all_metadata.append(
                            {
                                "split": split,
                                "score": avg_label,
                                "category": category,
                                "source": "judge_scores_fallback",
                            }
                        )
                        refusal_count += 1
                    elif avg_label < compliance_threshold:
                        all_prompts.append(prompt_text)
                        all_labels.append(0)
                        all_metadata.append(
                            {
                                "split": split,
                                "score": avg_label,
                                "category": category,
                                "source": "judge_scores_fallback",
                            }
                        )
                        compliant_count += 1
                    else:
                        skipped_count += 1

                print(f"  {split} (judge_scores fallback):")
                print(f"    Refused: {refusal_count}")
                print(f"    Complied: {compliant_count}")
                print(f"    Uncertain (skipped): {skipped_count}")

    print(f"\n[INFO] Total dataset:")
    print(f"   Refusal examples: {all_labels.count(1)}")
    print(f"   Compliant examples: {all_labels.count(0)}")
    print(f"   Total: {len(all_labels)}")

    return all_prompts, all_labels, all_metadata


def load_prompts_from_dataset(
    dataset_name,
    prompt_column="prompt",
    category_column="category",
    split=None,
    categories=None,
    max_per_category=None,
):
    """
    Load prompts with category labels from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "PKU-Alignment/BeaverTails-Evaluation")
        prompt_column: Column name for prompt text (default: "prompt")
        category_column: Column name for category label (default: "category")
        split: Dataset split to load (default: auto-detect)
        categories: Optional list of categories to filter to
        max_per_category: Optional max samples per category

    Returns:
        Tuple of (prompts, categories_list, category_names)
        - prompts: list of prompt strings
        - categories_list: list of category strings (one per prompt)
        - category_names: sorted list of unique category names
    """
    from datasets import load_dataset

    print(f"[LOAD] Loading dataset: {dataset_name}")
    if split:
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name)
        # Auto-detect: use the first available split
        if hasattr(ds, "keys"):
            available_splits = list(ds.keys())
            split = available_splits[0]
            print(f"   Auto-selected split: {split}")
            ds = ds[split]

    if prompt_column not in ds.column_names:
        raise ValueError(
            f"Column '{prompt_column}' not found in dataset. " f"Available: {ds.column_names}"
        )
    if category_column not in ds.column_names:
        raise ValueError(
            f"Column '{category_column}' not found in dataset. " f"Available: {ds.column_names}"
        )

    prompts = []
    categories_list = []

    # Build per-category counts for max_per_category limiting
    category_counts = {}

    for row in ds:
        raw_cat = row[category_column]
        # Normalize to a string (single-label datasets) or keep as-is
        if isinstance(raw_cat, list):
            # Multi-label: use first category as primary for filtering/counting
            cat = raw_cat[0] if raw_cat else None
        elif raw_cat is not None:
            cat = str(raw_cat)
        else:
            cat = None

        if categories and cat not in categories:
            continue
        if max_per_category is not None:
            count = category_counts.get(cat, 0)
            if count >= max_per_category:
                continue
            category_counts[cat] = count + 1

        prompts.append(row[prompt_column])
        categories_list.append(cat)

    category_names = sorted({c for c in categories_list if c is not None})

    print(f"[OK] Loaded {len(prompts)} prompts across {len(category_names)} categories")
    for cat in category_names:
        count = sum(1 for c in categories_list if c == cat)
        print(f"   {cat}: {count}")

    return prompts, categories_list, category_names


def load_prompts_from_judge_scores_with_categories(
    results_dir,
    dataset_name=None,
    category_map=None,
    prompt_column="prompt",
    category_column="category",
    dataset_split=None,
    refusal_threshold=0.1,
    compliance_threshold=-0.1,
):
    """
    Load judge-scored prompts and cross-reference with category labels.

    Calls load_prompts_from_judge_scores() internally, then injects category
    metadata by matching prompts against a HuggingFace dataset or explicit map.

    Args:
        results_dir: Path to evaluation results directory
        dataset_name: HuggingFace dataset to load categories from
        category_map: Explicit dict mapping prompt text -> category string
            (alternative to dataset_name)
        prompt_column: Column name for prompt text in dataset
        category_column: Column name for category in dataset
        dataset_split: Dataset split to load (default: auto-detect first split)
        refusal_threshold: Judge score above this = refusal
        compliance_threshold: Judge score below this = compliant

    Returns:
        Tuple of (prompts, labels, metadata) — metadata dicts now include "category" key
    """
    prompts, labels, metadata = load_prompts_from_judge_scores(
        results_dir, refusal_threshold, compliance_threshold
    )

    if not prompts:
        return prompts, labels, metadata

    # Check if categories are already present in the loaded data (from submodule output).
    # Check value (not just key presence) — upstream always writes the key but sets it
    # to None when no category_column was configured (pre-Feature-5 or unconfigured runs).
    has_categories = any(m.get("category") is not None for m in metadata)
    if has_categories:
        print(
            "\n[INFO] Categories already present in judge scores output, "
            "skipping HuggingFace dataset re-load"
        )
        # Print category distribution
        categories = [m["category"] for m in metadata]
        if categories:
            cat_names = sorted(
                {
                    name
                    for c in categories
                    if c is not None
                    for name in (c if isinstance(c, list) else [str(c)])
                }
            )
            print(f"   Categories ({len(cat_names)}):")
            for cat in cat_names:
                count = sum(
                    1
                    for c in categories
                    if c is not None and (cat in c if isinstance(c, list) else str(c) == cat)
                )
                print(f"      {cat}: {count}")
        return prompts, labels, metadata

    # Build category map if dataset provided
    if category_map is None and dataset_name is not None:
        from datasets import load_dataset

        print(f"\n[LOAD] Loading categories from dataset: {dataset_name}")
        if dataset_split:
            ds = load_dataset(dataset_name, split=dataset_split)
            print(f"   Using split: {dataset_split}")
        else:
            ds = load_dataset(dataset_name)
            if hasattr(ds, "keys"):
                split_used = list(ds.keys())[0]
                print(f"   Auto-selected split: {split_used}")
                ds = ds[split_used]

        category_map = {}
        for row in ds:
            key = row[prompt_column].strip()
            category_map[key] = row[category_column]
        print(f"   Built category map with {len(category_map)} entries")

    if category_map is None:
        print("[WARN] No category source provided — metadata will not include categories")
        return prompts, labels, metadata

    # Inject categories into metadata
    matched = 0
    unmatched = 0
    for i, prompt in enumerate(prompts):
        cat = category_map.get(prompt.strip())
        metadata[i]["category"] = cat
        if cat is not None:
            matched += 1
        else:
            unmatched += 1

    print(f"\n[INFO] Category matching:")
    print(f"   Matched: {matched}")
    print(f"   Unmatched: {unmatched}")

    # Print category distribution
    categories = [m["category"] for m in metadata if m.get("category") is not None]
    if categories:
        cat_names = sorted(
            {
                name
                for c in categories
                if c is not None
                for name in (c if isinstance(c, list) else [str(c)])
            }
        )
        print(f"   Categories ({len(cat_names)}):")
        for cat in cat_names:
            count = sum(
                1
                for c in categories
                if c is not None and (cat in c if isinstance(c, list) else str(c) == cat)
            )
            print(f"      {cat}: {count}")

    return prompts, labels, metadata


def analyze_dataset_quality(metadata):
    """
    Analyze the quality of the extracted dataset.
    Shows score distributions and potential issues.
    """
    if not metadata:
        print("[WARN] No metadata available for quality analysis")
        return

    scores = [m["score"] for m in metadata]
    refusal_scores = [s for s in scores if s > 0]
    compliant_scores = [s for s in scores if s < 0]

    print(f"\n[QUALITY] Dataset Quality Analysis:")
    print(f"   Total samples: {len(scores)}")

    if refusal_scores:
        print(
            f"   Refusal scores: mean={sum(refusal_scores)/len(refusal_scores):.2f}, "
            f"min={min(refusal_scores):.2f}, max={max(refusal_scores):.2f}"
        )
    if compliant_scores:
        print(
            f"   Compliant scores: mean={sum(compliant_scores)/len(compliant_scores):.2f}, "
            f"min={min(compliant_scores):.2f}, max={max(compliant_scores):.2f}"
        )

    # Check for potential issues
    high_confidence = sum(1 for s in scores if abs(s) > 0.8)
    low_confidence = sum(1 for s in scores if abs(s) < 0.3)
    print(f"   High confidence (|score| > 0.8): {high_confidence}")
    print(f"   Low confidence (|score| < 0.3): {low_confidence}")
