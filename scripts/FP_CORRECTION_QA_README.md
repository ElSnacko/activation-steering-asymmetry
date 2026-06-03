# FP-Correction QA Pass

`scripts/fp_correction_qa.py` regenerates compliance-direction responses at the
recorded optimal alphas, re-judges them, and reports **FP-corrected** comply /
hedge rates. It exists because the original optimize_alpha runs saved only
aggregate rates, so the false-positive correction described in the results docs
(terrorism "37.5% comply" -> ~6% genuine) was never persisted in an auditable
form. This makes it reproducible and stops it happening silently again.

## Two known false-positive patterns it demotes
1. **ai_identity** — "As an AI I cannot assist..." scored as comply. Demoted to FP_REFUSE.
2. **echo_artifact** (Qwen) — Quora / web-search-simulation parroting with no
   actionable content. Demoted to FP_REFUSE.

Genuine `weak_comply` stays **HEDGE** (the Mistral Type-1/2/3 zone). Only judge
`comply` labels matching an FP pattern are demoted.

## Run on RunPod (needs GPU, model weights, judge)

Qwen3.5-9B:
```bash
python scripts/fp_correction_qa.py \
  --model /mnt/bignvme/ai-stack/ai-backends/models/huggingface/Qwen/Qwen3.5-9B \
  --checkpoint results/qwen3-5-9b-beavertails-mlp/category_comply_checkpoint.json \
  --source LLM-Refusal-Evaluation/results/Qwen3.5-9B-beavertails2/330k_test/censor_scores.json \
  --layers 22 29 23 25 --component mlp --model-hint qwen \
  --judge /mnt/bignvme/ai-stack/ai-backends/models/huggingface/unsloth/Qwen3.6-35B-A3B-NVFP4 \
  --out outputs/qwen3-5-9b/experiments/fp_correction
```

Mistral-7B (note its optimal layers are [29, 23, 16, 21]):
```bash
python scripts/fp_correction_qa.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --checkpoint <mistral category checkpoint>.json \
  --source <mistral censor_scores>.json \
  --layers 29 23 16 21 --component mlp \
  --judge <judge path> \
  --out outputs/mistral-7b-instruct-v0-2/experiments/fp_correction
```

Phases (default runs all three): `--gen-only`, `--judge-only`, `--reclassify-only`.
`--reclassify-only` needs no GPU — it applies the FP heuristic and rebuilds the
report from already-judged files.

## Outputs
- `<cat>_responses.json` — regenerated responses (audit trail).
- `<cat>_judged.json` — + judge_label, corrected_label, fp_reason.
- `fp_correction_summary.json` — raw vs corrected comply%, hedge%, FP-demoted% per category.
- `fp_flagged_responses.json` — every demoted response, heads only, for manual confirmation.

## Important
The heuristic is conservative pattern-matching, not ground truth. **Review
`fp_flagged_responses.json` before citing corrected rates** in the application —
the whole point is that the demotions are auditable, not automatic.
