"""
Chest X-ray vision models and Grad-CAM visual attention modules.
"""

from src.vision.xray_model import (
    ChestXRayModel,
    ChestXRayPreprocessor,
    ChestXRayPipeline,
    RadiographicAcuityModifier,
    generate_synthetic_chest_xray,
    NIH_CHEST_XRAY_CLASSES,
    CLINICAL_FINDING_WEIGHTS,
    DEFAULT_DETECTION_THRESHOLDS,
)
from src.vision.gradcam import (
    GradCAM,
    GradCAMVisualizer,
    GradCAMPipeline,
    create_cam_overlay,
    colorize_heatmap,
    extract_focal_regions,
    draw_focal_boxes_on_overlay,
    map_coordinates_to_thoracic_zone,
    THORACIC_ANATOMICAL_ZONES,
)
from src.vision.advanced_vision import (
    XRayImageQualityAssessor,
    ImageQualityReport,
    GradCAMPlusPlus,
    AdvancedChestXRayPipeline,
    AdvancedVisionResult,
    PathologyPrediction,
)

__all__ = [
    "ChestXRayModel",
    "ChestXRayPreprocessor",
    "ChestXRayPipeline",
    "RadiographicAcuityModifier",
    "generate_synthetic_chest_xray",
    "NIH_CHEST_XRAY_CLASSES",
    "CLINICAL_FINDING_WEIGHTS",
    "DEFAULT_DETECTION_THRESHOLDS",
    "GradCAM",
    "GradCAMVisualizer",
    "GradCAMPipeline",
    "create_cam_overlay",
    "colorize_heatmap",
    "extract_focal_regions",
    "draw_focal_boxes_on_overlay",
    "map_coordinates_to_thoracic_zone",
    "THORACIC_ANATOMICAL_ZONES",
    "XRayImageQualityAssessor",
    "ImageQualityReport",
    "GradCAMPlusPlus",
    "AdvancedChestXRayPipeline",
    "AdvancedVisionResult",
    "PathologyPrediction",
]
