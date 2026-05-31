# output_utils.py - Utility functions for managing output directories and paths

import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def category_to_slug(category_name, max_length=60):
    """
    Convert a category name to a filesystem-safe slug.

    Args:
        category_name: Human-readable category name
            (e.g., "Violence, Aiding and Abetting, Incitement")
        max_length: Maximum slug length (default: 60)

    Returns:
        Lowercase slug with separators normalized
        (e.g., "violence_aiding_and_abetting_incitement")
    """
    slug = category_name.lower()
    slug = slug.replace("/", "-")
    slug = slug.replace(",", "")
    slug = slug.replace(" ", "_")
    # Collapse repeated separators
    slug = re.sub(r"[-_]{2,}", "_", slug)
    slug = slug.strip("-_")
    return slug[:max_length].strip("-_")


def resolve_model_path(path):
    """
    Resolve a model path to absolute if it's a local path.

    HuggingFace Hub identifiers (e.g. "Qwen/Qwen2.5-7B-Instruct") pass through
    unchanged. Local paths (relative or absolute) are resolved to absolute paths
    to avoid huggingface_hub validation errors.

    Args:
        path: Model name or path string

    Returns:
        Resolved absolute path for local paths, or the original string for Hub IDs
    """
    # If it exists on disk, it's definitely a local path
    if os.path.exists(path):
        return os.path.abspath(path)

    # HuggingFace Hub IDs follow the pattern "org/model" with no path separators
    # beyond a single slash. Local paths typically have multiple separators,
    # start with . or /, or contain backslashes.
    if path.startswith(("./", "../", "/")):
        return os.path.abspath(path)
    if "\\" in path:
        return os.path.abspath(path)
    # A bare path with multiple slashes is a local path (e.g. "models/org/model")
    if path.count("/") > 1:
        return os.path.abspath(path)

    # Single-slash paths like "Qwen/Qwen2.5-7B-Instruct" are Hub IDs
    return path


def check_gpu_memory(min_vram_gb=None):
    """
    Query nvidia-smi for GPU VRAM and print a summary.

    Args:
        min_vram_gb: If set, warn when total VRAM across all GPUs is below
                     this threshold (in GB).

    Returns:
        List of dicts with 'index', 'name', 'total_mb', 'free_mb', 'used_mb'
        per GPU, or an empty list if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("nvidia-smi not found or timed out — cannot detect GPU memory")
        return []

    if result.returncode != 0:
        logger.warning("nvidia-smi failed (return code %d)", result.returncode)
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "total_mb": int(parts[2]),
                "free_mb": int(parts[3]),
                "used_mb": int(parts[4]),
            }
        )

    if not gpus:
        logger.warning("No GPUs detected by nvidia-smi")
        return []

    total_vram_gb = sum(g["total_mb"] for g in gpus) / 1024
    free_vram_gb = sum(g["free_mb"] for g in gpus) / 1024

    print(
        f"[GPU] Detected {len(gpus)} GPU(s)  —  "
        f"Total VRAM: {total_vram_gb:.1f} GB, Free: {free_vram_gb:.1f} GB"
    )
    for g in gpus:
        print(
            f"  GPU {g['index']}: {g['name']}  " f"{g['total_mb']} MB total, {g['free_mb']} MB free"
        )

    if min_vram_gb is not None and total_vram_gb < min_vram_gb:
        print(
            f"[WARN] Total VRAM ({total_vram_gb:.1f} GB) is below recommended "
            f"minimum ({min_vram_gb} GB). You may need --enforce-eager or a "
            f"smaller model."
        )

    return gpus


def ensure_dir(dir_path):
    """
    Create directory if it doesn't exist

    Args:
        dir_path: Path to directory (string or Path object)

    Returns:
        Path object of the created directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_model_name(model_string):
    """
    Extract a clean model name from a model string

    Args:
        model_string: Full model name (e.g., "Qwen/Qwen2.5-7B-Instruct")

    Returns:
        Clean model name (e.g., "qwen2.5-7b-instruct")
    """
    # Extract the last part after the slash
    if "/" in model_string:
        model_name = model_string.split("/")[-1]
    else:
        model_name = model_string

    # Convert to lowercase and replace special characters
    model_name = model_name.lower()
    model_name = re.sub(r"[^a-z0-9\-]", "-", model_name)
    model_name = re.sub(r"-+", "-", model_name)  # Replace multiple hyphens with single

    return model_name


def generate_run_id():
    """
    Generate a unique run ID based on timestamp

    Returns:
        String run ID (e.g., "20231227-035148")
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def infer_run_from_path(file_path):
    """
    Infer model_name and run_id from an output file path.

    Expects the convention: outputs/{model_name}/{run_id}/{step}/filename
    Returns (model_name, run_id) or (None, None) if the path doesn't match.
    """
    parts = Path(file_path).resolve().parts
    # Find 'outputs' in the path and take the next two segments
    for i, part in enumerate(parts):
        if part == "outputs" and i + 2 < len(parts):
            model_name = parts[i + 1]
            run_id = parts[i + 2]
            # Accept any run_id that starts with a date prefix (YYYYMMDD-)
            if re.match(r"^\d{8}-", run_id):
                return model_name, run_id
    return None, None


def get_run_output_dir(base_dir="outputs", model_name=None, run_id=None):
    """
    Get output directory path for a specific model and run

    Args:
        base_dir: Base output directory (default: "outputs")
        model_name: Name of the model (e.g., "qwen2.5-7b-instruct")
        run_id: Unique run identifier (e.g., "20231227-035148")

    Returns:
        Path object for the output directory
    """
    if model_name and run_id:
        output_path = Path(base_dir) / model_name / run_id
    elif model_name:
        output_path = Path(base_dir) / model_name
    else:
        output_path = Path(base_dir)

    return ensure_dir(output_path)


def get_output_path(base_dir="outputs", model_name=None, run_id=None, filename=None):
    """
    Get full path for an output file

    Args:
        base_dir: Base output directory (default: "outputs")
        model_name: Name of the model (e.g., "qwen2.5-7b-instruct")
        run_id: Unique run identifier (e.g., "20231227-035148")
        filename: Name of the output file

    Returns:
        Path object for the output file
    """
    output_dir = get_run_output_dir(base_dir, model_name, run_id)

    if filename:
        return output_dir / filename
    else:
        return output_dir


def setup_model_run_dirs(base_dir="outputs", model_name=None, run_id=None):
    """
    Setup output directories for a specific model run

    Args:
        base_dir: Base output directory (default: "outputs")
        model_name: Name of the model (e.g., "qwen2.5-7b-instruct")
        run_id: Unique run identifier (e.g., "20231227-035148")

    Returns:
        Dictionary mapping script types to their output directories
    """
    run_dir = get_run_output_dir(base_dir, model_name, run_id)

    output_dirs = {
        "extract_activations": ensure_dir(run_dir / "extract_activations"),
        "compute_wrmd": ensure_dir(run_dir / "compute_wrmd"),
        "find_best_layers": ensure_dir(run_dir / "find_best_layers"),
    }

    return output_dirs


# Legacy functions for backward compatibility
def get_output_dir(base_dir="outputs", script_name=None):
    """
    Legacy function - use get_run_output_dir instead
    """
    if script_name:
        output_path = Path(base_dir) / script_name
    else:
        output_path = Path(base_dir)

    return ensure_dir(output_path)


def _detect_fp8_checkpoint(model_path):
    """Check if a model checkpoint uses FP8 quantization.

    Looks for FP8 indicators in the model config or safetensors metadata
    without loading the full weights.

    Returns:
        True if FP8 is detected, False otherwise.
    """
    model_path = Path(model_path)

    # Check config.json for quantization_config
    config_path = model_path / "config.json" if model_path.is_dir() else None
    if config_path and config_path.exists():
        import json

        with open(config_path) as f:
            config = json.load(f)

        quant_config = config.get("quantization_config", {})
        quant_method = quant_config.get("quant_method", "")
        if "fp8" in quant_method.lower():
            return True

        # Check for common FP8 indicators
        if quant_config.get("quant_type") == "fp8":
            return True

    # Check for weight_scale_inv in safetensors index
    index_path = model_path / "model.safetensors.index.json" if model_path.is_dir() else None
    if index_path and index_path.exists():
        import json

        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if any("weight_scale_inv" in k for k in weight_map):
            return True

    # Check model name for FP8 hints
    model_str = str(model_path).lower()
    if "fp8" in model_str:
        return True

    return False


def load_model(model_name_or_path, quantize=None, device_map="auto", **kwargs):
    """Load a model and tokenizer with automatic FP8 detection and quantization support.

    FP8 checkpoints are handled with a tiered strategy:
    1. Native FP8 loading via torch_dtype="auto" (uses the checkpoint's own quantization
       config if transformers supports it — no extra dependencies needed)
    2. BitsAndBytes 8-bit fallback if native loading produces CPU-offloaded parameters
       (indicating the FP8 weights were upcast to bfloat16 and overflowed VRAM)
    3. Explicit --load-in-4bit or --load-in-8bit overrides always use BitsAndBytes

    Note: If quantization is used during activation extraction, the same quantization
    mode should be used during inference/steering for best results, since quantized
    activations differ from unquantized ones.

    Note: BitsAndBytes quantized models cannot be moved between devices (model.cpu() /
    model.cuda() will fail). Scripts that offload the model to CPU for judge subprocess
    scoring keep the quantized model on GPU and share VRAM with the judge instead.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        quantize: Override quantization mode: "4bit", "8bit", or None (auto-detect).
            When None, FP8 checkpoints use native loading (torch_dtype="auto").
        device_map: Device map strategy (default: "auto").
        **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained()
            (e.g., enforce_eager=True for memory-constrained GPUs).

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(model_name_or_path)
    is_fp8 = _detect_fp8_checkpoint(model_path)

    # Determine quantization strategy
    effective_quantize = quantize

    load_kwargs = {
        "device_map": device_map,
    }

    if effective_quantize == "4bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "BitsAndBytes is required for 4-bit quantization. "
                "Install with: pip install bitsandbytes"
            )
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        print(f"Loading model (4-bit): {model_path}")

    elif effective_quantize == "8bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "BitsAndBytes is required for 8-bit quantization. "
                "Install with: pip install bitsandbytes"
            )
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print(f"Loading model (8-bit): {model_path}")

    elif is_fp8:
        # Native FP8: let transformers use the checkpoint's quantization config.
        # torch_dtype="auto" preserves the checkpoint's dtype instead of forcing bfloat16
        # which would upcast FP8 weights and overflow VRAM.
        load_kwargs["torch_dtype"] = "auto"
        print(f"[AUTO] FP8 checkpoint detected — loading with native dtype")
        print(f"Loading model (FP8 native): {model_path}")

    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        print(f"Loading model: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs, **kwargs)

    # Check if FP8 native loading silently fell back to bfloat16 and overflowed to CPU.
    # This happens when transformers doesn't understand the FP8 format — weights get
    # upcast and device_map="auto" offloads the excess to CPU.
    if is_fp8 and effective_quantize is None:
        has_cpu_params = any(p.device.type == "cpu" for p in model.parameters())
        if has_cpu_params:
            print(
                "[WARN] FP8 native loading resulted in CPU-offloaded parameters — "
                "the model likely overflowed GPU memory."
            )
            print("  Attempting BitsAndBytes 8-bit reload...")
            try:
                from transformers import BitsAndBytesConfig

                del model
                import gc

                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs.pop("torch_dtype", None)
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs, **kwargs)
                effective_quantize = "8bit"
                print("  BitsAndBytes 8-bit reload successful")
            except Exception as bnb_err:
                print(f"  [WARN] BitsAndBytes 8-bit fallback failed: {bnb_err}")
                print(
                    "  Continuing with CPU-offloaded model (generation will be slow). "
                    "Install bitsandbytes or use a smaller model."
                )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if effective_quantize:
        print(f"  Quantization: {effective_quantize}")

    layers = get_model_layers(model)
    num_layers = len(layers)
    hidden_size = get_model_hidden_size(model)
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    return model, tokenizer


def get_model_layers(model):
    """Resolve the decoder layers list for any supported model architecture.

    Different architectures store their layers at different attribute paths:
      - Llama, Qwen2, Mistral, etc.: model.model.layers
      - Gemma 4 (multimodal): model.model.language_model.layers
      - Other wrapped models: check common nested paths

    Args:
        model: A HuggingFace transformers model (e.g., from AutoModelForCausalLM).

    Returns:
        nn.ModuleList or list of decoder layer modules.
    """
    # Direct path: model.model.layers (Llama, Qwen, Mistral, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # Gemma 4 multimodal: model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers

    # Try recursive search for a 'layers' ModuleList
    for name, module in model.named_modules():
        if name.endswith(".layers") and hasattr(module, "__len__"):
            return module

    raise AttributeError(
        f"Cannot resolve decoder layers for {type(model).__name__}. "
        f"Model structure: {[n for n, _ in model.named_children()]}"
    )


def get_model_hidden_size(model):
    """Resolve the hidden size for any supported model architecture.

    Handles multimodal models where config.num_hidden_layers is on the text_config.

    Args:
        model: A HuggingFace transformers model.

    Returns:
        int: Hidden size of the language model.
    """
    config = model.config
    # Direct attribute (most models)
    if hasattr(config, "hidden_size"):
        # For multimodal, check if text_config has it
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return config.text_config.hidden_size
        return config.hidden_size
    # Nested config (Gemma 4, etc.)
    if hasattr(config, "text_config"):
        return config.text_config.hidden_size
    raise AttributeError(f"Cannot resolve hidden_size from config: {type(config).__name__}")


def get_model_num_layers(model):
    """Resolve the number of decoder layers for any supported model architecture.

    Args:
        model: A HuggingFace transformers model.

    Returns:
        int: Number of decoder layers.
    """
    return len(get_model_layers(model))
