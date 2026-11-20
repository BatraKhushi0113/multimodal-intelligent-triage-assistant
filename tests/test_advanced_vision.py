"""
Unit Tests for Advanced Computer Vision & IQA Module.
=====================================================
Tests Image Quality Assessment (IQA), blur and contrast detection, non-radiograph
rejection, Grad-CAM++ higher-order activation mapping, and MCDO uncertainty.
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.advanced_vision import (
    AdvancedChestXRayPipeline,
    AdvancedVisionResult,
    GradCAMPlusPlus,
    ImageQualityReport,
    XRayImageQualityAssessor,
)
from src.vision.xray_model import generate_synthetic_chest_xray


@pytest.fixture
def quality_assessor() -> XRayImageQualityAssessor:
    return XRayImageQualityAssessor()


@pytest.fixture
def synthetic_xray() -> np.ndarray:
    pil_img = generate_synthetic_chest_xray(pathology="pneumothorax", image_size=224, random_seed=42)
    return np.array(pil_img)


def test_iqa_valid_synthetic_xray(quality_assessor: XRayImageQualityAssessor, synthetic_xray: np.ndarray):
    """Verify standard procedural radiograph passes IQA with acceptable status."""
    report = quality_assessor.assess_quality(synthetic_xray)
    assert isinstance(report, ImageQualityReport)
    assert report.is_valid is True
    assert report.status in ["ACCEPTABLE", "WARNING_LOW_QUALITY"]
    assert report.blur_score > 10.0
    assert report.is_color_natural_photo is False


def test_iqa_blur_detection(quality_assessor: XRayImageQualityAssessor, synthetic_xray: np.ndarray):
    """Verify heavily blurred image is flagged with low Laplacian variance."""
    import scipy.ndimage as ndi
    # Severe Gaussian blur
    blurred = ndi.gaussian_filter(synthetic_xray.astype(np.float32), sigma=6.0).astype(np.uint8)
    report = quality_assessor.assess_quality(blurred)
    assert report.is_blurry is True
    assert report.blur_score < 40.0


def test_iqa_non_radiograph_color_photo_rejection(quality_assessor: XRayImageQualityAssessor):
    """Verify vibrant RGB natural photo is flagged and rejected as non-radiograph."""
    # Create colorful synthetic RGB pattern (e.g. rainbow flag)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # Pure red
    rgb[:112, :, 1] = 255 # Yellow upper
    rgb[:, :, 2] = 128

    report = quality_assessor.assess_quality(rgb)
    assert report.is_color_natural_photo is True
    assert report.is_valid is False
    assert report.status == "REJECTED_INVALID"
    assert any("natural color" in r for r in report.rejection_reasons)


def test_iqa_severe_overexposure_rejection(quality_assessor: XRayImageQualityAssessor):
    """Verify washed-out / white-clipped image is rejected."""
    white_img = np.full((224, 224), 250, dtype=np.uint8)
    report = quality_assessor.assess_quality(white_img)
    assert report.is_valid is False
    assert report.exposure_status == "OVEREXPOSED"


def test_gradcam_plus_plus_heatmap_and_focal_regions(synthetic_xray: np.ndarray):
    """Verify Grad-CAM++ generates normalized (224, 224) saliency and identifies anatomical zones."""
    pipeline = AdvancedChestXRayPipeline(
        checkpoint_path=None,
        device="cpu",
        mc_dropout_passes=3,
    )
    cam_engine = pipeline.cam_engine

    tensor_in = pipeline.preprocessor.preprocess(synthetic_xray)
    tensor_in.requires_grad = True

    heatmap = cam_engine.generate_heatmap(tensor_in, target_finding_idx=7)  # Pneumothorax
    assert heatmap.shape == (224, 224)
    assert 0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0

    regions = cam_engine.extract_focal_regions(heatmap, threshold=0.40)
    assert isinstance(regions, list)
    if len(regions) > 0:
        assert "thoracic_zone" in regions[0]
        assert "bbox" in regions[0]
        assert len(regions[0]["bbox"]) == 4

    cam_engine.remove_hooks()


def test_advanced_vision_pipeline_end_to_end(synthetic_xray: np.ndarray):
    """Verify end-to-end analysis produces full AdvancedVisionResult with MCDO uncertainty."""
    pipeline = AdvancedChestXRayPipeline(
        checkpoint_path=PROJECT_ROOT / "models" / "checkpoints" / "chest_xray_densenet121.pt",
        device="cpu",
        mc_dropout_passes=4,
    )
    res = pipeline.analyze_radiograph(synthetic_xray, target_finding="Pneumothorax")

    assert isinstance(res, AdvancedVisionResult)
    assert res.evaluated is True
    assert len(res.predictions) == 14
    assert res.mean_predictive_entropy >= 0.0
    assert 0.0 <= res.max_probability <= 1.0

    # Verify uncertainty intervals
    pneu_pred = next((p for p in res.predictions if p.finding == "Pneumothorax"), None)
    assert pneu_pred is not None
    assert pneu_pred.ci_lower <= pneu_pred.probability <= pneu_pred.ci_upper
    assert pneu_pred.epistemic_uncertainty >= 0.0

    # Verify CAM overlay array
    assert res.cam_overlay is not None
    assert res.cam_overlay.shape == (224, 224, 3)
