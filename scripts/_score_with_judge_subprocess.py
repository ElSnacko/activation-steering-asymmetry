#!/usr/bin/env python3
"""
Helper script to run judge scoring in a separate process.
This avoids CUDA context conflicts with the parent process.

IMPORTANT: vLLM spawns a separate VLLM::EngineCore process that can
become orphaned if this script crashes. We use atexit + signal handlers
to ensure cleanup, and also kill any child processes on exit.
"""

import argparse
import atexit
import gc
import json
import os
import signal
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add LLM-Refusal-Evaluation to path
llm_eval_dir = str(Path(__file__).parent.parent / "LLM-Refusal-Evaluation")
if llm_eval_dir not in sys.path:
    sys.path.insert(0, llm_eval_dir)

from src.llm_judge import LLMJudge

# Track child PIDs so we can kill orphaned vLLM engines on exit
_child_pids_before = set()
_judge_ref = None


def _get_child_pids():
    """Get all child PIDs of this process."""
    try:
        import subprocess

        result = subprocess.run(
            ["pgrep", "-P", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return set(int(p) for p in result.stdout.strip().split() if p)
    except Exception:
        return set()


def _cleanup():
    """Kill any child processes spawned after we started (e.g. VLLM::EngineCore)."""
    global _judge_ref
    # Try to delete the judge to trigger vLLM's own cleanup
    if _judge_ref is not None:
        try:
            del _judge_ref
            gc.collect()
        except Exception:
            pass

    # Kill any new child processes that appeared after startup.
    # Use SIGKILL — SIGTERM may be ignored by vLLM worker processes.
    new_children = _get_child_pids() - _child_pids_before
    for pid in new_children:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass
    # Reap children so they don't become zombies holding GPU memory
    for pid in new_children:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass

    # Kill any grandchildren (e.g. vLLM workers spawned by EngineCore) that
    # aren't direct children.  We can't use killpg on our own process group
    # because that would SIGKILL ourselves during atexit, making the parent
    # see returncode -9 and treat a successful run as a failure.
    try:
        import subprocess as _sp

        result = _sp.run(
            ["pgrep", "-g", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.strip().split():
            if line:
                pid = int(line)
                if pid != os.getpid():
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    except Exception:
        pass


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT by cleaning up and exiting."""
    _cleanup()
    sys.exit(1)


def main():
    global _child_pids_before, _judge_ref

    # Record existing children before we start
    _child_pids_before = _get_child_pids()

    # Register cleanup handlers
    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--responses-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--gpu-memory-util", type=float, default=0.9)
    parser.add_argument(
        "--judge-max-model-len",
        type=int,
        default=None,
        help="Max sequence length for vLLM judge. Default: 4096.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vllm (disables CUDAGraphs — slower, use only if OOM)",
    )
    parser.add_argument(
        "--thinking-string", type=str, default=None, help="String to split on for thinking models."
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Pass text_only=True to LLMJudge (required for NVFP4 quantized models)",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        help="KV cache dtype (e.g. 'fp8'). Default: auto.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=512,
        help="Max concurrent sequences. 512 avoids Mamba cache alignment errors on MoE. Default: 512.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2096,
        help="Max batched tokens per step. 2096 required for Qwen3 MoE block alignment. Default: 2096.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: on — judge prompts share a common prefix).",
    )
    parser.add_argument("--no-prefix-caching", dest="enable_prefix_caching", action="store_false")
    args = parser.parse_args()

    # Load prompts and responses
    with open(args.prompts_file) as f:
        prompts = json.load(f)

    with open(args.responses_file) as f:
        responses = json.load(f)

    # Create judge
    # Default to 4096 tokens — plenty for judge prompt + response scoring, and
    # avoids KV cache OOM that occurs with the model's full max_model_len (e.g.
    # 24576) when GPU memory is shared with the parent process's CUDA context.
    max_model_len = args.judge_max_model_len if args.judge_max_model_len is not None else 4096
    judge_kwargs = dict(
        model_name=args.judge_model,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_util,
    )
    if args.text_only:
        judge_kwargs["text_only"] = True
    if args.kv_cache_dtype is not None:
        judge_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    # enforce_eager is a vLLM parameter, not an LLMJudge parameter.
    # Pass it only if LLMJudge accepts it; otherwise patch vLLM's LLM class.
    import inspect

    judge_init_params = inspect.signature(LLMJudge.__init__).parameters
    # Monkey-patch vLLM's LLM to inject settings LLMJudge doesn't expose.
    from vllm import LLM as _OrigLLM

    _orig_init = _OrigLLM.__init__
    _patch_args = args

    def _patched_init(self, *a, **kw):
        if _patch_args.enforce_eager:
            kw["enforce_eager"] = True
        kw["disable_log_stats"] = False
        kw["max_num_seqs"] = _patch_args.max_num_seqs
        kw["max_num_batched_tokens"] = _patch_args.max_num_batched_tokens
        if _patch_args.enable_prefix_caching:
            kw["enable_prefix_caching"] = True
        return _orig_init(self, *a, **kw)

    _OrigLLM.__init__ = _patched_init

    if args.enforce_eager and "enforce_eager" in judge_init_params:
        judge_kwargs["enforce_eager"] = True

    judge = LLMJudge(**judge_kwargs)
    _judge_ref = judge

    # Auto-detect thinking string for thinking/reasoning models.
    # These models generate verbose reasoning before the final answer.
    # Without this, the score extractor may parse reasoning text and misclassify.
    thinking_string = args.thinking_string
    if thinking_string is None:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.judge_model)
            added = tok.added_tokens_encoder
            if "<|channel|>" in added:
                # GPT-OSS models (e.g. gpt-oss-20b): <|channel|>analysis before <|channel|>final
                thinking_string = "<|channel|>final"
            elif "</think>" in added or "<think>" in added:
                # DeepSeek R1, Qwen3/QwQ: <think>...</think> reasoning block
                thinking_string = "</think>"
            elif "</thinking>" in added or "<|thinking|>" in added:
                # Other thinking model families
                thinking_string = "</thinking>"
            if thinking_string:
                print(f"Auto-detected thinking model, using thinking_string='{thinking_string}'")
            del tok
        except Exception:
            pass

    # Score responses
    qa_pairs = list(zip(prompts, responses))
    results = judge.judge(
        questions_answers=qa_pairs,
        num_return_sequences=1,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=2048,
        thinking_string=thinking_string,
    )

    # Extract scores
    scores = [result["label"] for result in results]

    # Save scores
    with open(args.output_file, "w") as f:
        json.dump(scores, f)

    print(f"Saved {len(scores)} judge scores to {args.output_file}")

    # Explicit cleanup before exit
    _judge_ref = None
    del judge
    gc.collect()


if __name__ == "__main__":
    main()
