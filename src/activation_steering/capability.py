"""
Capability preservation evaluation for steered models.

Measures whether steering degrades model performance on non-sensitive tasks
using multiple-choice questions (MMLU-style). No judge model required --
scoring is deterministic via answer parsing.

Supports three question sources:
  1. Built-in questions (data/capability_questions.json) -- quick smoke test
  2. Custom JSON files -- same format as built-in
  3. HuggingFace benchmarks (MMLU, ARC) -- requires `datasets` package
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm

# Default question file ships with the package
_DEFAULT_QUESTIONS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "capability_questions.json"
)

KL_PROBE_N_PER_CATEGORY = 10

# Supported HuggingFace benchmarks and their dataset configs
SUPPORTED_BENCHMARKS = {
    "mmlu": {
        "dataset": "cais/mmlu",
        "split": "test",
        "description": "Massive Multitask Language Understanding (57 subjects)",
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "name": "ARC-Easy",
        "split": "test",
        "description": "AI2 Reasoning Challenge (Easy)",
    },
    "arc_challenge": {
        "dataset": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "test",
        "description": "AI2 Reasoning Challenge (Challenge)",
    },
}


@dataclass
class CapabilityResult:
    """Results from a capability evaluation run."""

    accuracy: float
    num_correct: int
    num_total: int
    per_category: Dict[str, Dict[str, float]]
    details: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "per_category": self.per_category,
            "details": self.details,
        }


def load_questions(path: Optional[str] = None) -> List[Dict]:
    """
    Load capability evaluation questions.

    Args:
        path: Path to a JSON file with custom questions. If None, uses the
              default set in data/capability_questions.json.
              JSON format: list of dicts with keys: category, question, choices, answer

    Returns:
        List of question dicts
    """
    questions_path = Path(path) if path else _DEFAULT_QUESTIONS_PATH

    if not questions_path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_path}. "
            "Provide a path via --capability-questions or ensure data/capability_questions.json exists."
        )

    with open(questions_path) as f:
        questions = json.load(f)

    # Validate format
    required_keys = {"question", "choices", "answer"}
    for i, q in enumerate(questions):
        missing = required_keys - set(q.keys())
        if missing:
            raise ValueError(f"Question {i} missing required keys: {missing}")
        if q["answer"] not in ("A", "B", "C", "D"):
            raise ValueError(f"Question {i} answer must be A, B, C, or D, got: {q['answer']}")
        if "category" not in q:
            q["category"] = "custom"

    return questions


def load_hf_benchmark(
    benchmark: str,
    subjects: Optional[List[str]] = None,
    max_questions: Optional[int] = None,
    split: Optional[str] = None,
) -> List[Dict]:
    """
    Load questions from a HuggingFace benchmark dataset.

    Requires the `datasets` package: pip install datasets

    Args:
        benchmark: Benchmark name -- one of: mmlu, arc_easy, arc_challenge
        subjects: For MMLU, list of subject names to include (default: all).
                  Example: ["abstract_algebra", "anatomy", "computer_security"]
                  Run with subjects=[] to list available subjects.
        max_questions: Limit total number of questions (sampled across subjects)
        split: Dataset split to use (default: per-benchmark default, usually "test")

    Returns:
        List of question dicts in standard format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` package is required for loading HuggingFace benchmarks. "
            "Install it with: pip install datasets"
        )

    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. "
            f"Supported: {', '.join(SUPPORTED_BENCHMARKS.keys())}"
        )

    config = SUPPORTED_BENCHMARKS[benchmark]
    use_split = split or config["split"]

    if benchmark == "mmlu":
        return _load_mmlu(load_dataset, subjects, max_questions, use_split)
    elif benchmark.startswith("arc"):
        return _load_arc(load_dataset, config, max_questions, use_split)
    else:
        raise ValueError(f"No loader implemented for benchmark: {benchmark}")


def _load_mmlu(load_dataset, subjects, max_questions, split) -> List[Dict]:
    """Load MMLU from HuggingFace."""
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    if subjects is not None and len(subjects) == 0:
        # List available subjects
        ds = load_dataset("cais/mmlu", "all", split=split)
        available = sorted(set(ds["subject"]))
        print(f"Available MMLU subjects ({len(available)}):")
        for s in available:
            print(f"  {s}")
        return []

    # Load all MMLU or specific subjects
    ds = load_dataset("cais/mmlu", "all", split=split)

    if subjects:
        ds = ds.filter(lambda x: x["subject"] in subjects)

    questions = []
    for row in ds:
        choices_list = row["choices"]
        questions.append(
            {
                "category": row.get("subject", "mmlu"),
                "question": row["question"],
                "choices": [f"{letter}) {text}" for letter, text in zip("ABCD", choices_list)],
                "answer": answer_map[row["answer"]],
            }
        )

    if max_questions and len(questions) > max_questions:
        import random

        random.seed(42)
        questions = random.sample(questions, max_questions)

    return questions


def _load_arc(load_dataset, config, max_questions, split) -> List[Dict]:
    """Load ARC (Easy or Challenge) from HuggingFace."""
    ds = load_dataset(config["dataset"], config["name"], split=split)

    # ARC answer labels can be "A","B","C","D" or "1","2","3","4"
    label_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}

    questions = []
    for row in ds:
        choices_text = row["choices"]["text"]
        choices_labels = row["choices"]["label"]

        # Normalize to A/B/C/D
        choice_strs = []
        for label, text in zip(choices_labels, choices_text):
            letter = label_to_letter.get(label, label)
            choice_strs.append(f"{letter}) {text}")

        answer_key = row["answerKey"]
        answer = label_to_letter.get(answer_key, answer_key)

        # Skip questions with more or fewer than 4 choices
        if len(choices_text) != 4 or answer not in "ABCD":
            continue

        questions.append(
            {
                "category": config["name"].lower().replace("-", "_"),
                "question": row["question"],
                "choices": choice_strs,
                "answer": answer,
            }
        )

    if max_questions and len(questions) > max_questions:
        import random

        random.seed(42)
        questions = random.sample(questions, max_questions)

    return questions


# Few-shot examples for base models. These are NOT from the built-in question set.
_FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "choices": ["A) Berlin", "B) Madrid", "C) Paris", "D) Rome"],
        "answer": "C",
    },
    {
        "question": "Which gas do plants absorb from the atmosphere for photosynthesis?",
        "choices": ["A) Oxygen", "B) Nitrogen", "C) Carbon dioxide", "D) Hydrogen"],
        "answer": "C",
    },
    {
        "question": "What is 7 x 8?",
        "choices": ["A) 48", "B) 54", "C) 56", "D) 64"],
        "answer": "C",
    },
    {
        "question": "Who painted the Mona Lisa?",
        "choices": [
            "A) Vincent van Gogh",
            "B) Pablo Picasso",
            "C) Leonardo da Vinci",
            "D) Michelangelo",
        ],
        "answer": "C",
    },
    {
        "question": "What is the largest organ in the human body?",
        "choices": ["A) Heart", "B) Liver", "C) Skin", "D) Brain"],
        "answer": "C",
    },
]


def _format_single_question(question: Dict) -> str:
    """Format one question block (no answer)."""
    text = f"Question: {question['question']}\n"
    for choice in question["choices"]:
        text += f"{choice}\n"
    text += "Answer:"
    return text


def _format_single_example(example: Dict) -> str:
    """Format one few-shot example (with answer)."""
    text = f"Question: {example['question']}\n"
    for choice in example["choices"]:
        text += f"{choice}\n"
    text += f"Answer: {example['answer']}"
    return text


def _is_base_model(tokenizer) -> bool:
    """Heuristic: check if the tokenizer/model is a base (non-instruct) model."""
    name = getattr(tokenizer, "name_or_path", "") or ""
    name_lower = name.lower()
    # If the name contains instruct/chat indicators, it's not a base model
    if any(kw in name_lower for kw in ["instruct", "chat", "rlhf", "dpo", "sft"]):
        return False
    return True


def format_mcq_prompt(
    question: Dict,
    tokenizer=None,
    few_shot: Optional[int] = None,
) -> str:
    """
    Format a multiple-choice question as a prompt.

    For instruct/chat models: uses chat template with zero-shot instruction.
    For base models: uses few-shot examples (default 5) so the model learns
    the expected output format from context.

    Args:
        question: Question dict with question, choices, answer keys
        tokenizer: HuggingFace tokenizer (used to detect base vs instruct)
        few_shot: Number of few-shot examples (0=zero-shot, None=auto-detect).
                  Auto uses 5-shot for base models, 0-shot for instruct models.
    """
    is_base = _is_base_model(tokenizer) if tokenizer is not None else True

    if few_shot is None:
        few_shot = 5 if is_base else 0

    # Instruct model with zero-shot: use chat template
    if few_shot == 0 and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        text = f"{question['question']}\n"
        for choice in question["choices"]:
            text += f"{choice}\n"
        text += "\nAnswer with just the letter (A, B, C, or D)."
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Few-shot format (for base models or as fallback)
    parts = []
    examples = _FEW_SHOT_EXAMPLES[:few_shot]
    for ex in examples:
        parts.append(_format_single_example(ex))
    parts.append(_format_single_question(question))
    return "\n\n".join(parts)


def parse_mcq_answer(response: str) -> Optional[str]:
    """
    Extract answer letter from model response.

    Tries multiple patterns in order of specificity:
    1. Standalone letter at start of response
    2. "Answer: X" or "answer is X" patterns
    3. First occurrence of A/B/C/D in parentheses
    4. First standalone A/B/C/D letter
    """
    response = response.strip()

    # Single letter response (most common with few-shot)
    if len(response) >= 1 and response[0] in "ABCD":
        return response[0]

    if response.upper() in ("A", "B", "C", "D"):
        return response.upper()

    # Pattern 2: "Answer: X" or "the answer is X"
    match = re.search(r"(?:answer|option)\s*(?:is|:)\s*\(?([A-D])\)?", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Letter in parentheses
    match = re.search(r"\(([A-D])\)", response)
    if match:
        return match.group(1).upper()

    # Pattern 4: First standalone A-D letter (word boundary on both sides)
    match = re.search(r"\b([A-D])\b", response)
    if match:
        return match.group(1).upper()

    return None


def evaluate_capability(
    model,
    tokenizer,
    questions: Optional[List[Dict]] = None,
    max_questions: Optional[int] = None,
    max_new_tokens: int = 32,
    few_shot: Optional[int] = None,
    show_progress: bool = True,
) -> CapabilityResult:
    """
    Evaluate model capability on multiple-choice questions.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        questions: List of question dicts. Loads default set if None.
        max_questions: Limit number of questions (None = all)
        max_new_tokens: Max tokens to generate per question (short for MCQ)
        few_shot: Number of few-shot examples (0=zero-shot, None=auto-detect:
                  5-shot for base models, 0-shot for instruct models)
        show_progress: Show progress bar

    Returns:
        CapabilityResult with accuracy and per-category breakdown
    """
    if questions is None:
        questions = load_questions()

    if max_questions is not None:
        questions = questions[:max_questions]

    is_base = _is_base_model(tokenizer)
    effective_few_shot = few_shot if few_shot is not None else (5 if is_base else 0)
    mode = f"{effective_few_shot}-shot" if effective_few_shot > 0 else "zero-shot (chat)"
    print(f"Prompt mode: {mode} (detected {'base' if is_base else 'instruct'} model)")

    details = []
    category_stats = {}

    iterator = tqdm(questions, desc="Capability eval") if show_progress else questions

    for q in iterator:
        prompt = format_mcq_prompt(q, tokenizer, few_shot=few_shot)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        response = full_output[len(prompt_text) :].strip()

        parsed = parse_mcq_answer(response)
        correct = parsed == q["answer"]

        cat = q.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if correct:
            category_stats[cat]["correct"] += 1

        details.append(
            {
                "question": q["question"],
                "category": cat,
                "expected": q["answer"],
                "parsed": parsed,
                "correct": correct,
                "response": response[:200],
            }
        )

    num_correct = sum(1 for d in details if d["correct"])
    num_total = len(details)
    accuracy = num_correct / num_total if num_total > 0 else 0.0

    per_category = {}
    for cat, stats in sorted(category_stats.items()):
        cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        per_category[cat] = {
            "accuracy": cat_acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    return CapabilityResult(
        accuracy=accuracy,
        num_correct=num_correct,
        num_total=num_total,
        per_category=per_category,
        details=details,
    )


def compare_capability(
    baseline: CapabilityResult,
    steered: CapabilityResult,
) -> Dict:
    """
    Compare baseline vs steered capability results.

    Returns:
        Dict with accuracy delta, per-category deltas, and degradation flag
    """
    delta = steered.accuracy - baseline.accuracy

    category_deltas = {}
    all_categories = set(baseline.per_category.keys()) | set(steered.per_category.keys())
    for cat in sorted(all_categories):
        base_acc = baseline.per_category.get(cat, {}).get("accuracy", 0.0)
        steer_acc = steered.per_category.get(cat, {}).get("accuracy", 0.0)
        category_deltas[cat] = {
            "baseline": base_acc,
            "steered": steer_acc,
            "delta": steer_acc - base_acc,
        }

    return {
        "baseline_accuracy": baseline.accuracy,
        "steered_accuracy": steered.accuracy,
        "accuracy_delta": delta,
        "degraded": delta < -0.05,
        "severely_degraded": delta < -0.15,
        "per_category": category_deltas,
    }


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

# Diverse corpus covering multiple domains and writing styles.  Each passage
# should be long enough to give a stable per-token loss signal (~50-120 tokens).
DEFAULT_PERPLEXITY_CORPUS = [
    # Science / technical
    (
        "The mitochondria generate adenosine triphosphate through oxidative "
        "phosphorylation, a process that couples the electron transport chain "
        "to chemiosmotic ATP synthesis across the inner mitochondrial membrane. "
        "Protons are pumped from the matrix into the intermembrane space, "
        "creating an electrochemical gradient that drives ATP synthase."
    ),
    # History / narrative
    (
        "The construction of the Panama Canal took more than a decade to "
        "complete, requiring the excavation of over 200 million cubic yards "
        "of earth. French efforts under Ferdinand de Lesseps failed due to "
        "disease and engineering setbacks. The United States resumed the "
        "project in 1904 and opened the canal to traffic in August 1914."
    ),
    # Mathematics / logic
    (
        "A prime number is a natural number greater than one that has no "
        "positive divisors other than one and itself. The fundamental theorem "
        "of arithmetic states that every integer greater than one can be "
        "represented uniquely as a product of prime numbers, up to the order "
        "of the factors. This decomposition is called prime factorization."
    ),
    # Literature / creative
    (
        "The old lighthouse keeper watched the storm approach from the west, "
        "its dark clouds swallowing the horizon like ink spreading through "
        "water. He had weathered a thousand such storms in his forty years "
        "on the island, but something about the color of the sky tonight "
        "made him uneasy. He checked the lamp one more time."
    ),
    # Technology / computing
    (
        "Modern neural networks are trained using stochastic gradient descent "
        "and its variants, which iteratively adjust model parameters to "
        "minimize a loss function. Backpropagation computes the gradient of "
        "the loss with respect to each parameter by applying the chain rule "
        "of calculus through the computational graph."
    ),
    # Geography / earth science
    (
        "The Amazon River basin spans approximately 7 million square "
        "kilometers and contains roughly one-fifth of the world's total "
        "river flow. The basin supports the largest tropical rainforest on "
        "Earth, home to an estimated 10 percent of all species. Seasonal "
        "flooding creates vast floodplains called varzea forests."
    ),
    # Philosophy / abstract reasoning
    (
        "The problem of induction, first articulated by David Hume, questions "
        "whether we can rationally justify the inference from observed "
        "regularities to universal laws. Just because the sun has risen every "
        "morning in recorded history does not logically guarantee it will "
        "rise tomorrow. This challenge remains central to the philosophy of science."
    ),
    # Cooking / procedural
    (
        "To prepare a classic French omelette, beat three eggs with a fork "
        "until the yolks and whites are just combined. Heat butter in a "
        "non-stick pan over medium-high heat until it foams but does not "
        "brown. Pour in the eggs and stir continuously with a spatula, "
        "tilting the pan to let uncooked egg flow to the edges."
    ),
    # Medicine / biology
    (
        "The human immune system relies on two complementary defense "
        "mechanisms. Innate immunity provides immediate, nonspecific "
        "protection through physical barriers, phagocytes, and inflammatory "
        "responses. Adaptive immunity develops over days and produces "
        "highly specific antibodies and memory cells that confer lasting "
        "protection against previously encountered pathogens."
    ),
    # Law / social science
    (
        "The principle of habeas corpus requires that a person under arrest "
        "be brought before a judge or into court, ensuring that imprisonment "
        "is not without legal authority. This principle is considered one of "
        "the fundamental safeguards of individual liberty in common law "
        "systems, preventing indefinite detention without judicial review."
    ),
]


@dataclass
class PerplexityResult:
    """Results from perplexity evaluation."""

    perplexity: float
    mean_loss: float
    per_passage: List[Dict]
    num_passages: int
    num_tokens: int

    def to_dict(self) -> Dict:
        return {
            "perplexity": self.perplexity,
            "mean_loss": self.mean_loss,
            "per_passage": self.per_passage,
            "num_passages": self.num_passages,
            "num_tokens": self.num_tokens,
        }


def load_perplexity_corpus(path: Optional[str] = None) -> List[str]:
    """
    Load text passages for perplexity evaluation.

    Args:
        path: Optional JSON file containing a list of text strings.
              Falls back to built-in diverse corpus if not provided.

    Returns:
        List of text passages.
    """
    if path is not None:
        with open(path) as f:
            corpus = json.load(f)
        if not isinstance(corpus, list):
            raise ValueError(f"Expected a JSON list of strings, got {type(corpus)}")
        return corpus
    return list(DEFAULT_PERPLEXITY_CORPUS)


def load_capability_probe_set(
    path: Optional[str] = None,
    n_per_category: int = KL_PROBE_N_PER_CATEGORY,
    seed: int = 42,
    tokenizer=None,
    enable_thinking: bool = False,
) -> tuple:
    """
    Load a stratified, seeded probe set from capability questions for KL/PPL measurement.

    Samples n_per_category questions per category (default 1) using a fixed seed so the
    same questions are used across every trial in a run — ensuring KL and perplexity are
    directly comparable. Stratification prevents easy categories (elementary_math) from
    dominating over harder ones (logic, physics).

    Args:
        path: Path to capability questions JSON. Defaults to data/capability_questions.json.
        n_per_category: Questions to sample per category (default 10 → 100 total for the
                        10-category built-in set).
        seed: RNG seed for reproducibility (default 42).
        tokenizer: If provided, formats questions with chat template for KL use.
        enable_thinking: Passed to apply_chat_template for thinking models.

    Returns:
        Tuple of (raw_texts, formatted_prompts) where:
          - raw_texts: list of "Q: ...\nChoices: ...\nAnswer: X" strings for PPL
          - formatted_prompts: chat-template formatted strings for KL (same as raw_texts
            if no tokenizer provided)
    """
    import random

    questions_path = Path(path) if path else _DEFAULT_QUESTIONS_PATH
    with open(questions_path) as f:
        questions = json.load(f)

    # Group by category
    by_category: Dict[str, list] = {}
    for q in questions:
        cat = q.get("category", "unknown")
        by_category.setdefault(cat, []).append(q)

    rng = random.Random(seed)
    selected = []
    for cat in sorted(by_category):
        pool = by_category[cat]
        k = min(n_per_category, len(pool))
        selected.extend(rng.sample(pool, k))

    # Build raw text: question + choices + correct answer (for PPL teacher-forcing)
    raw_texts = []
    for q in selected:
        choices_str = "  ".join(q.get("choices", []))
        answer = q.get("answer", "")
        raw_texts.append(f"{q['question']}\n{choices_str}\nAnswer: {answer}")

    # Build formatted prompts for KL: question + choices only (answer withheld)
    prompt_texts = []
    for q in selected:
        choices_str = "\n".join(q.get("choices", []))
        prompt_texts.append(f"{q['question']}\n{choices_str}")

    formatted_prompts = prompt_texts
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted_prompts = []
            for p in prompt_texts:
                messages = [{"role": "user", "content": p}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                formatted_prompts.append(text)
        except Exception:
            formatted_prompts = prompt_texts

    return raw_texts, formatted_prompts


def evaluate_perplexity(
    model,
    tokenizer,
    corpus: Optional[List[str]] = None,
    max_passages: Optional[int] = None,
    show_progress: bool = True,
) -> PerplexityResult:
    """
    Evaluate model perplexity on a text corpus.

    Computes teacher-forced perplexity: for each passage, tokenize the full
    text, run a forward pass, and measure cross-entropy loss per token.
    Lower perplexity means the model assigns higher probability to the
    reference text — a sensitive measure of generation quality degradation.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        corpus: List of text passages.  Uses built-in diverse corpus if None.
        max_passages: Limit number of passages.
        show_progress: Show progress bar.

    Returns:
        PerplexityResult with aggregate and per-passage perplexity.
    """
    import math

    if corpus is None:
        corpus = list(DEFAULT_PERPLEXITY_CORPUS)
    if max_passages is not None:
        corpus = corpus[:max_passages]

    per_passage = []
    total_loss = 0.0
    total_tokens = 0

    iterator = tqdm(corpus, desc="Perplexity eval") if show_progress else corpus

    for passage in iterator:
        inputs = tokenizer(passage, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        if seq_len < 2:
            continue

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)

        # outputs.loss is the mean cross-entropy over all predicted tokens
        loss = outputs.loss.item()
        # Number of predicted tokens = seq_len - 1 (first token has no prediction)
        n_tokens = seq_len - 1

        per_passage.append(
            {
                "passage": passage[:100] + "..." if len(passage) > 100 else passage,
                "loss": loss,
                "perplexity": math.exp(loss),
                "num_tokens": n_tokens,
            }
        )

        total_loss += loss * n_tokens
        total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(mean_loss)

    return PerplexityResult(
        perplexity=perplexity,
        mean_loss=mean_loss,
        per_passage=per_passage,
        num_passages=len(per_passage),
        num_tokens=total_tokens,
    )


def compare_perplexity(
    baseline: PerplexityResult,
    steered: PerplexityResult,
) -> Dict:
    """
    Compare baseline vs steered perplexity results.

    Returns:
        Dict with perplexity delta, ratio, and degradation flags.
    """
    ratio = steered.perplexity / baseline.perplexity if baseline.perplexity > 0 else float("inf")
    delta = steered.perplexity - baseline.perplexity

    return {
        "baseline_perplexity": baseline.perplexity,
        "steered_perplexity": steered.perplexity,
        "perplexity_delta": delta,
        "perplexity_ratio": ratio,
        # >10% increase in perplexity signals degradation
        "degraded": ratio > 1.10,
        # >50% increase is severe
        "severely_degraded": ratio > 1.50,
    }
