"""
Build the capability probe set from real MMLU benchmark questions.

Randomly selects --n-categories subjects from MMLU's full taxonomy (seeded),
then samples an equal number of questions from each selected subject.
No hand-picking of subjects — selection is driven by the seed alone.

Default: 10 categories × 10 questions = 100 questions total.

Saves to data/capability_questions.json and replaces any existing file.
Run this once before using the probe set for KL/PPL measurement.

Usage:
    python scripts/build_capability_probe.py
    python scripts/build_capability_probe.py --n-categories 10 --n 100
    python scripts/build_capability_probe.py --n-categories 5 --n 50
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build MMLU capability probe set")
    parser.add_argument(
        "--output",
        default="data/capability_questions.json",
        help="Output path (default: data/capability_questions.json)",
    )
    parser.add_argument(
        "--n-categories",
        type=int,
        default=10,
        help="Number of MMLU subjects to randomly select (default: 10)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Total number of questions; must be divisible by --n-categories (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed — controls both subject selection and question sampling (default: 42)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="MMLU split to sample from (default: test)",
    )
    args = parser.parse_args()

    if args.n % args.n_categories != 0:
        raise ValueError(
            f"--n ({args.n}) must be divisible by --n-categories ({args.n_categories})"
        )
    n_per_category = args.n // args.n_categories

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    rng = random.Random(args.seed)

    print(f"Loading MMLU ({args.split} split) from HuggingFace...")
    ds = load_dataset("cais/mmlu", "all", split=args.split)

    # Group rows by subject
    by_subject: dict[str, list] = {}
    for row in ds:
        by_subject.setdefault(row["subject"], []).append(row)

    all_subjects = sorted(by_subject.keys())
    print(f"MMLU has {len(all_subjects)} subjects")

    # Randomly select n_categories subjects (seeded — no hand-picking)
    if args.n_categories > len(all_subjects):
        raise ValueError(
            f"--n-categories ({args.n_categories}) exceeds available subjects ({len(all_subjects)})"
        )
    selected_subjects = sorted(rng.sample(all_subjects, args.n_categories))
    print(f"Selected {len(selected_subjects)} subjects (seed={args.seed}):")
    for s in selected_subjects:
        available = len(by_subject[s])
        print(f"  {s}: {available} available, sampling {n_per_category}")
        if available < n_per_category:
            raise ValueError(
                f"Subject '{s}' only has {available} questions but {n_per_category} requested. "
                f"Reduce --n or --n-categories."
            )

    questions = []
    for subject in selected_subjects:
        sampled = rng.sample(by_subject[subject], n_per_category)
        for row in sampled:
            choices = [f"{l}) {t}" for l, t in zip("ABCD", row["choices"])]
            questions.append(
                {
                    "category": subject,
                    "question": row["question"],
                    "choices": choices,
                    "answer": answer_map[row["answer"]],
                }
            )

    # Shuffle so subjects are interleaved in the output
    rng.shuffle(questions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nSaved {len(questions)} questions to {output_path}")
    print(f"  {n_per_category} questions × {len(selected_subjects)} subjects")


if __name__ == "__main__":
    main()
