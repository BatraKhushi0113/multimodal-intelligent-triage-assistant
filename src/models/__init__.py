from src.models.ablation import (
    ABLATION_CONFIGS,
    run_ablation_study_module,
)
from src.models.baseline import (
    DenseTransformer,
    build_pipeline,
    evaluate_predictions,
    get_feature_subset,
    run_all_baseline_experiments,
)
from src.models.context_encoder import (
    ContextDataset,
    ContextEncoder,
    run_context_encoder_module,
)
from src.models.fusion import (
    AdaptiveReliabilityFusion,
    BaselineConcatenationFusion,
    ModalityDropout,
    MultimodalFusionDataset,
    evaluate_adaptive_fusion,
    evaluate_baseline_fusion,
    run_adaptive_fusion_module,
    run_baseline_fusion_module,
    run_modality_ablation_benchmark,
    train_adaptive_fusion_model,
    train_baseline_fusion_model,
)
from src.models.reliability import (
    AdaptiveCrossModalGate,
    ModalityReliabilityScorer,
)
from src.models.text_encoder import (
    TextDataset,
    TextEncoder,
    analyze_text_corpus,
    generate_and_cache_embeddings,
    run_text_modality_module,
)
from src.models.uncertainty import (
    MCDropoutPredictor,
    TriageSafetyAuditor,
    compute_expected_calibration_error,
    disable_mc_dropout,
    enable_mc_dropout,
    evaluate_selective_prediction,
    run_uncertainty_module,
)
from src.models.vitals_encoder import (
    VitalsClassifierProbe,
    VitalsDataset,
    VitalsEncoder,
    run_vitals_encoder_module,
)

__all__ = [
    "DenseTransformer",
    "build_pipeline",
    "evaluate_predictions",
    "get_feature_subset",
    "run_all_baseline_experiments",
    "TextDataset",
    "TextEncoder",
    "analyze_text_corpus",
    "generate_and_cache_embeddings",
    "run_text_modality_module",
    "VitalsEncoder",
    "VitalsDataset",
    "VitalsClassifierProbe",
    "run_vitals_encoder_module",
    "ContextEncoder",
    "ContextDataset",
    "run_context_encoder_module",
    "BaselineConcatenationFusion",
    "ModalityDropout",
    "MultimodalFusionDataset",
    "train_baseline_fusion_model",
    "evaluate_baseline_fusion",
    "run_modality_ablation_benchmark",
    "run_baseline_fusion_module",
    "ModalityReliabilityScorer",
    "AdaptiveCrossModalGate",
    "AdaptiveReliabilityFusion",
    "train_adaptive_fusion_model",
    "evaluate_adaptive_fusion",
    "run_adaptive_fusion_module",
    "ABLATION_CONFIGS",
    "run_ablation_study_module",
    "enable_mc_dropout",
    "disable_mc_dropout",
    "MCDropoutPredictor",
    "TriageSafetyAuditor",
    "compute_expected_calibration_error",
    "evaluate_selective_prediction",
    "run_uncertainty_module",
]
