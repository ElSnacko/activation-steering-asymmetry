"""
Activation Steering for Language Models.

A toolkit for extracting activations, computing steering vectors, and analyzing
effective layers for modifying LLM refusal behavior.
"""

__version__ = "0.1.0"

from .analysis import (
    annotate_distances_with_significance,
    compute_bootstrap_convergence,
    compute_bootstrap_stability,
    compute_category_angular_distances,
    compute_intra_category_angular_distances,
    compute_layer_correlations,
    find_best_layers,
    find_best_layers_dynamic,
    load_category_vectors_from_files,
    plot_bootstrap_convergence,
    plot_bootstrap_stability,
    plot_category_angular_distances,
    plot_correlations,
    plot_intra_category_angular_distances,
    visualize_layer_projections,
)
from .capability import (
    DEFAULT_PERPLEXITY_CORPUS,
    SUPPORTED_BENCHMARKS,
    CapabilityResult,
    PerplexityResult,
    compare_capability,
    compare_perplexity,
    evaluate_capability,
    evaluate_perplexity,
    format_mcq_prompt,
    load_capability_probe_set,
    load_hf_benchmark,
    load_perplexity_corpus,
    load_questions,
    parse_mcq_answer,
)
from .computation import (
    WRMDCalculator,
    analyze_rank_associations,
    compare_methods,
    compute_md,
    compute_multirank_vectors,
    compute_rmd,
    compute_wrmd,
)
from .dynamic_layer import DynamicSteeringLayer, DynamicSteeringSubmodule
from .extraction import (
    VALID_COMPONENTS,
    ActivationExtractor,
    _get_attn_output_proj,
    _get_attn_submodule,
    analyze_dataset_quality,
    load_prompts_from_dataset,
    load_prompts_from_judge_scores,
    load_prompts_from_judge_scores_with_categories,
)
from .kl_divergence import (
    DEFAULT_GENERATION_PROMPTS,
    DEFAULT_HARMLESS_PROMPTS,
    KLResult,
    collect_first_token_logits,
    collect_logits,
    collect_teacher_forced_logits,
    compute_kl_divergence,
    load_harmless_prompts,
)
from .merge_steering_into_weights import (
    export_to_gguf,
    load_merged_model,
    load_steered_model,
    merge_steering_into_model,
    save_dynamic_steered_model,
    verify_dynamic_steered_model,
    verify_merged_model,
)
from .routing import (
    CalibrationData,
    CategoryRouter,
    RoutingDecision,
    calibrate_router,
    compute_residual_vectors,
)
from .steered_model import SteerDiagnostics, SteeredModel, SteeredModelConfig
from .steering import (
    SanitizeLogitsProcessor,
    SteeringHook,
    SteeringHookGroup,
    compute_dynamic_params,
    load_actual_refusal_prompts,
    load_best_layers_from_correlations,
    test_steering,
)
from .utils import (
    category_to_slug,
    check_gpu_memory,
    ensure_dir,
    extract_model_name,
    generate_run_id,
    get_output_path,
    get_run_output_dir,
    infer_run_from_path,
    load_model,
    resolve_model_path,
    setup_model_run_dirs,
)

__all__ = [
    # Core classes
    "ActivationExtractor",
    "WRMDCalculator",
    "SteeringHook",
    # Extraction functions
    "load_prompts_from_judge_scores",
    "load_prompts_from_judge_scores_with_categories",
    "load_prompts_from_dataset",
    "analyze_dataset_quality",
    # Computation functions
    "compute_md",
    "compute_rmd",
    "compute_wrmd",
    "compute_multirank_vectors",
    "analyze_rank_associations",
    "compare_methods",
    # Analysis functions
    "annotate_distances_with_significance",
    "compute_bootstrap_convergence",
    "compute_bootstrap_stability",
    "compute_category_angular_distances",
    "compute_intra_category_angular_distances",
    "compute_layer_correlations",
    "load_category_vectors_from_files",
    "plot_bootstrap_convergence",
    "plot_bootstrap_stability",
    "plot_category_angular_distances",
    "plot_intra_category_angular_distances",
    "plot_correlations",
    "find_best_layers",
    "find_best_layers_dynamic",
    "visualize_layer_projections",
    # Steering functions
    "SanitizeLogitsProcessor",
    "compute_dynamic_params",
    "load_best_layers_from_correlations",
    "load_actual_refusal_prompts",
    "test_steering",
    # Packaged model
    "SteeredModel",
    "SteeredModelConfig",
    "SteerDiagnostics",
    # Dynamic steering
    "DynamicSteeringLayer",
    "DynamicSteeringSubmodule",
    "SteeringHookGroup",
    # Capability evaluation
    "CapabilityResult",
    "PerplexityResult",
    "evaluate_capability",
    "evaluate_perplexity",
    "compare_capability",
    "compare_perplexity",
    "load_capability_probe_set",
    "load_questions",
    "load_hf_benchmark",
    "load_perplexity_corpus",
    "format_mcq_prompt",
    "parse_mcq_answer",
    "SUPPORTED_BENCHMARKS",
    "DEFAULT_PERPLEXITY_CORPUS",
    # KL divergence
    "KLResult",
    "collect_first_token_logits",
    "collect_logits",
    "collect_teacher_forced_logits",
    "compute_kl_divergence",
    "load_harmless_prompts",
    "DEFAULT_HARMLESS_PROMPTS",
    "DEFAULT_GENERATION_PROMPTS",
    # Routing
    "CalibrationData",
    "CategoryRouter",
    "RoutingDecision",
    "calibrate_router",
    "compute_residual_vectors",
    # Merge functions
    "merge_steering_into_model",
    "load_merged_model",
    "verify_merged_model",
    "export_to_gguf",
    "save_dynamic_steered_model",
    "load_steered_model",
    "verify_dynamic_steered_model",
    # Utility functions
    "category_to_slug",
    "ensure_dir",
    "extract_model_name",
    "generate_run_id",
    "get_run_output_dir",
    "get_output_path",
    "setup_model_run_dirs",
    "load_model",
    "resolve_model_path",
    "infer_run_from_path",
    "check_gpu_memory",
]
