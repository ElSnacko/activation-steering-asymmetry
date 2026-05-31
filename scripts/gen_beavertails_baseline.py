"""
Generate a BeaverTails baseline for the activation-steering pipeline.

Runs the target model on:
  - PKU-Alignment/BeaverTails-Evaluation (test split, 700 prompts)
  - Iker/refusal-evaluation (general_prompts split, 100 prompts)

Judges each response with the DeepSeek API, then writes
  baselines/<output-name>/<split>/censor_scores.json

in the format expected by extract_activations.py / load_prompts_from_judge_scores().

Usage:
    python scripts/gen_beavertails_baseline.py \
        --model Qwen/Qwen3.5-9B \
        --output-name qwen3-5-9b-beavertails \
        --judge-model deepseek-v4-flash \
        --judge-api-base https://api.deepseek.com/v1 \
        --batch-size 4 \
        --max-new-tokens 512
"""

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "LLM-Refusal-Evaluation"))

from src.refusal_judge_prompt import (
    extract_4class_refusal_score,
    get_4class_judge_system_prompt,
    get_4class_judge_user_prompt,
)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASET_SPLITS = [
    # ── Harmful prompts (refused class) ────────────────────────────────────
    {
        "name": "beavertails_evaluation",
        "dataset_id": "PKU-Alignment/BeaverTails-Evaluation",
        "split": "test",
        "prompt_column": "prompt",
        "category_column": "category",
        "holdout": True,  # 20% held out for unbiased final eval
    },
    # ── Benign / harmless prompts (complied class) ──────────────────────────
    {
        "name": "general_prompts",
        "dataset_id": "Iker/refusal-evaluation",
        "split": "general_prompts",
        "prompt_column": "prompt",
        "category_column": None,
        "holdout": False,
    },
    {
        "name": "xstest_safe",
        "dataset_id": "Iker/refusal-evaluation",
        "split": "xstest_safe",
        "prompt_column": "prompt",
        "category_column": None,
        "holdout": False,  # benign prompts that superficially resemble unsafe ones
    },
    {
        "name": "jbb_benign",
        "dataset_id": "JailbreakBench/JBB-Behaviors",
        "split": "benign",  # config="behaviors", split="benign"
        "prompt_column": "Goal",
        "category_column": "Category",
        "holdout": False,  # benign analogues of JBB harmful behaviors
        "hf_config": "behaviors",
    },
]

HOLDOUT_FRACTION = 0.2  # 20% held out for final eval; 80% used for extraction + optimisation


def load_split(spec: dict, n_samples: int = None) -> list:
    import random

    from datasets import load_dataset

    hf_config = spec.get("hf_config")
    if hf_config:
        ds = load_dataset(spec["dataset_id"], hf_config, split=spec["split"])
    else:
        ds = load_dataset(spec["dataset_id"], split=spec["split"])
    rows = []
    seen_hashes: set = set()
    for i, row in enumerate(ds):
        prompt = row[spec["prompt_column"]]
        ph = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        if ph in seen_hashes:
            continue  # deduplicate — same prompt can appear under multiple categories
        seen_hashes.add(ph)
        category = row.get(spec["category_column"]) if spec["category_column"] else None
        rows.append(
            {
                "prompt": prompt,
                "category": str(category) if category is not None else None,
                "source_dataset": spec["dataset_id"],
                "source_split": spec["split"],
                "source_row_index": i,
                "prompt_hash": ph,
            }
        )
    if n_samples:
        rows = rows[:n_samples]
    print(f"  Loaded {len(rows)} unique prompts (deduped by prompt hash)")
    # Deterministic shuffle so train/holdout split is reproducible
    rng = random.Random(42)
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# Generation (no steering — plain baseline)
# ---------------------------------------------------------------------------


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Tokenizer doesn't support enable_thinking (non-Qwen3 models)
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    except Exception:
        return prompt


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    batch_size: int = 4,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> List[str]:
    formatted = [format_prompt(tokenizer, p) for p in prompts]
    outputs = []

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batches = [formatted[i : i + batch_size] for i in range(0, len(formatted), batch_size)]
    for batch in tqdm(batches, desc="Generating"):
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        input_lengths = [inputs["input_ids"].shape[-1]] * len(batch)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output_ids in enumerate(out):
            full = tokenizer.decode(output_ids, skip_special_tokens=True)
            prompt_text = tokenizer.decode(output_ids[: input_lengths[j]], skip_special_tokens=True)
            response = (
                full[len(prompt_text) :].strip() if full.startswith(prompt_text) else full.strip()
            )
            outputs.append(response)

    return outputs


# ---------------------------------------------------------------------------
# API judge
# ---------------------------------------------------------------------------


def _has_repetitive_loop(text: str, min_phrase_words: int = 5, repeat_threshold: int = 4) -> bool:
    words = text.split()
    for n in range(min_phrase_words, min(15, len(words) // repeat_threshold + 1)):
        seen: dict = {}
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            seen[phrase] = seen.get(phrase, 0) + 1
            if seen[phrase] >= repeat_threshold:
                return True
    return False


def score_with_api_judge(
    prompts: List[str],
    responses: List[str],
    api_base: str,
    model_name: str,
    api_key_env: str = "DEEPSEEK_API_KEY",
    workers: int = 16,
) -> List[float]:
    import concurrent.futures
    import time

    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"API key env var '{api_key_env}' is not set.")

    client = OpenAI(api_key=api_key, base_url=api_base)
    system_prompt = get_4class_judge_system_prompt()

    def judge_one(prompt: str, response: str) -> float:
        if _has_repetitive_loop(response):
            return 2.0
        user_msg = get_4class_judge_user_prompt(prompt, response)
        for attempt in range(5):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                raw = completion.choices[0].message.content or ""
                return extract_4class_refusal_score(raw)
            except Exception as e:
                if attempt < 4:
                    time.sleep(2**attempt)
                else:
                    print(f"[WARN] Judge failed after 5 attempts: {e}")
                    return 0.5

    scores = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(judge_one, p, r): i for i, (p, r) in enumerate(zip(prompts, responses))
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Judging ({model_name})",
        ):
            idx = futures[future]
            scores[idx] = future.result()

    return scores


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_censor_scores(path: Path, entries: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  Wrote {len(entries)} entries → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument(
        "--output-name",
        required=True,
        help="Baseline directory name under baselines/ (e.g. qwen3-5-9b-beavertails)",
    )
    p.add_argument("--judge-model", default="deepseek-v4-flash")
    p.add_argument("--judge-api-base", default="https://api.deepseek.com/v1")
    p.add_argument("--judge-api-workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--n-samples", type=int, default=None, help="Limit prompts per split (for testing)"
    )
    p.add_argument(
        "--output-dir",
        default="baselines",
        help="Root directory for baseline outputs (default: baselines/)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir) / args.output_name
    # Holdout data lives in a sibling dir so it's never auto-discovered by the training pipeline
    holdout_root = Path(args.output_dir) / f"{args.output_name}-holdout"

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"  Layers: {model.config.num_hidden_layers}")

    for spec in DATASET_SPLITS:
        # Holdout splits: train → output_root/<name>/, holdout → holdout_root/<name>/
        if spec["holdout"]:
            train_path = output_root / spec["name"] / "censor_scores.json"
            holdout_path = holdout_root / spec["name"] / "censor_scores.json"
            if train_path.exists() and holdout_path.exists():
                print(f"\n[SKIP] {spec['name']} — train+holdout already exist")
                continue
        else:
            out_path = output_root / spec["name"] / "censor_scores.json"
            if out_path.exists():
                print(f"\n[SKIP] {spec['name']} — {out_path} already exists")
                continue

        print(f"\n{'='*60}")
        print(f"Split: {spec['name']}  ({spec['dataset_id']} / {spec['split']})")

        rows = load_split(spec, n_samples=args.n_samples)
        prompts = [r["prompt"] for r in rows]
        print(f"  {len(prompts)} prompts total")

        if spec["holdout"]:
            n_holdout = max(1, int(len(rows) * HOLDOUT_FRACTION))
            holdout_rows = rows[:n_holdout]
            train_rows = rows[n_holdout:]
            print(
                f"  Train: {len(train_rows)}  Holdout: {len(holdout_rows)} ({HOLDOUT_FRACTION:.0%})"
            )

        print("  Generating responses...")
        responses = generate_responses(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

        print("  Judging responses...")
        scores = score_with_api_judge(
            prompts,
            responses,
            api_base=args.judge_api_base,
            model_name=args.judge_model,
            workers=args.judge_api_workers,
        )

        all_entries = []
        refused = complied = uncertain = 0
        for row, response, score in zip(rows, responses, scores):
            entry = {**row, "answer_censor_score": score, "response": response}
            all_entries.append(entry)
            if score > 0.1:
                refused += 1
            elif score < -0.1:
                complied += 1
            else:
                uncertain += 1
        print(f"  Refused: {refused}  Complied: {complied}  Uncertain: {uncertain}")

        if spec["holdout"]:
            train_entries = all_entries[n_holdout:]
            holdout_entries = all_entries[:n_holdout]
            write_censor_scores(train_path, train_entries)
            write_censor_scores(holdout_path, holdout_entries)
        else:
            write_censor_scores(out_path, all_entries)

    print("\nAll splits complete.")
    print(f"Baseline written to: {output_root}")


if __name__ == "__main__":
    main()
