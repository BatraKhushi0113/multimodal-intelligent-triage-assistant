"""
Unit Tests for Module 14: Chest X-ray Analysis Module
=====================================================
Tests the multi-label vision model architecture (DenseNet-121, EfficientNet-B0),
image preprocessing and normalization, procedural synthetic radiograph generation,
decoupled clinical decision support acuity modifier, and end-to-end pipeline inference.
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


def test_nih_classes_and_weights_consistency():
    """Verify all 14 NIH Chest X-ray classes have corresponding clinical weights and thresholds."""
    assert len(NIH_CHEST_XRAY_CLASSES) == 14
    for finding in NIH_CHEST_XRAY_CLASSES:
        assert finding in CLINICAL_FINDING_WEIGHTS, f"Missing clinical weight for {finding}"
        assert 0.0 < CLINICAL_FINDING_WEIGHTS[finding] <= 1.0
        assert finding in DEFAULT_DETECTION_THRESHOLDS, f"Missing detection threshold for {finding}"
        assert 0.0 < DEFAULT_DETECTION_THRESHOLDS[finding] < 1.0


def test_densenet121_architecture_shapes():
    """Verify DenseNet-121 backbone forward pass, feature extraction, and probability output shapes."""
    model = ChestXRayModel(architecture="densenet121", pretrained=False, num_findings=14)
    model.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # 1. Feature extraction
    features = model.extract_features(dummy_input)
    assert features.shape == (batch_size, 1024), f"Expected (2, 1024), got {features.shape}"

    # 2. Forward logits
    logits = model(dummy_input)
    assert logits.shape == (batch_size, 14), f"Expected (2, 14), got {logits.shape}"

    # 3. Sigmoid probabilities
    probs = model.predict_proba(dummy_input)
    assert probs.shape == (batch_size, 14)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    # 4. Grad-CAM layer existence
    cam_layer = model.get_gradcam_target_layer()
    assert isinstance(cam_layer, torch.nn.Conv2d)


def test_efficientnet_b0_architecture_shapes():
    """Verify EfficientNet-B0 backbone forward pass and feature extraction shapes."""
    model = ChestXRayModel(architecture="efficientnet_b0", pretrained=False, num_findings=14)
    model.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    features = model.extract_features(dummy_input)
    assert features.shape == (batch_size, 1280), f"Expected (2, 1280), got {features.shape}"

    logits = model(dummy_input)
    assert logits.shape == (batch_size, 14), f"Expected (2, 14), got {logits.shape}"

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (batch_size, 14)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_xray_preprocessor_formats():
    """Verify preprocessor converts grayscale PIL, 2D numpy, and 3D RGB inputs to standardized tensors."""
    preprocessor = ChestXRayPreprocessor(image_size=224)

    # 1. Grayscale PIL image
    gray_pil = Image.fromarray(np.random.randint(0, 256, (128, 128), dtype=np.uint8), mode="L")
    tensor1 = preprocessor.preprocess(gray_pil)
    assert tensor1.shape == (1, 3, 224, 224)
    assert tensor1.dtype == torch.float32

    # 2. 2D numpy grayscale array
    gray_arr = np.random.rand(150, 150).astype(np.float32)
    tensor2 = preprocessor.preprocess(gray_arr)
    assert tensor2.shape == (1, 3, 224, 224)

    # 3. 3D numpy RGB array
    rgb_arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    tensor3 = preprocessor.preprocess(rgb_arr)
    assert tensor3.shape == (1, 3, 224, 224)

    # 4. Unnormalize
    unnorm = preprocessor.unnormalize(tensor1)
    assert unnorm.shape == (224, 224, 3)
    assert 0.0 <= unnorm.min() and unnorm.max() <= 1.0


def test_synthetic_chest_xray_generator():
    """Verify procedural radiograph generator creates realistic valid grayscale images for all pathologies."""
    pathologies = ["normal", "pneumothorax", "cardiomegaly", "effusion", "consolidation", "edema"]

    for path in pathologies:
        img = generate_synthetic_chest_xray(pathology=path, image_size=224, random_seed=123)
        assert isinstance(img, Image.Image)
        assert img.size == (224, 224)
        assert img.mode == "L"

        arr = np.array(img)
        assert arr.min() >= 0 and arr.max() <= 255
        # Ensure anatomical structure exists (thorax is not uniform)
        assert np.std(arr) > 20.0


def test_radiographic_acuity_modifier_critical():
    """Verify critical life threats trigger emergency escalation tier and high severity index."""
    modifier = RadiographicAcuityModifier()

    # Case 1: Acute Tension Pneumothorax
    ptx_probs = {f: 0.05 for f in NIH_CHEST_XRAY_CLASSES}
    ptx_probs["Pneumothorax"] = 0.88

    res_ptx = modifier.evaluate_findings(ptx_probs)
    assert res_ptx["acuity_tier"] == "CRITICAL_ESCALATION"
    assert "KTAS 1/2" in res_ptx["triage_impact"]
    assert "Pneumothorax" in res_ptx["critical_flags"]
    assert res_ptx["radiographic_severity_index"] >= 0.85
    assert "IMMEDIATE PHYSICIAN INTERVENTION" in res_ptx["recommended_action"]

    # Case 2: Acute Pulmonary Edema
    edema_probs = {f: 0.05 for f in NIH_CHEST_XRAY_CLASSES}
    edema_probs["Edema"] = 0.80

    res_edema = modifier.evaluate_findings(edema_probs)
    assert res_edema["acuity_tier"] == "CRITICAL_ESCALATION"
    assert "Edema" in res_edema["critical_flags"]

    # Case 3: Clear study without acute findings
    clear_probs = {f: 0.08 for f in NIH_CHEST_XRAY_CLASSES}
    res_clear = modifier.evaluate_findings(clear_probs)
    assert res_clear["acuity_tier"] == "NO_ACUTE_PATHOLOGY"
    assert res_clear["triage_impact"] == "Neutral (Preserve Primary Intake Acuity)"
    assert res_clear["num_findings_detected"] == 0
    assert res_clear["radiographic_severity_index"] < 0.25


def test_pipeline_end_to_end_inference():
    """Verify ChestXRayPipeline runs full inference on synthetic image and produces expected schema."""
    pipeline = ChestXRayPipeline(
        model_architecture="densenet121",
        pretrained=False,
        image_size=224,
    )

    synth_img = generate_synthetic_chest_xray("pneumothorax", image_size=224, random_seed=42)
    output = pipeline.predict(synth_img)

    required_keys = [
        "finding_probabilities",
        "radiographic_severity_index",
        "acuity_tier",
        "triage_impact",
        "urgency_level",
        "critical_flags",
        "detected_findings",
        "num_findings_detected",
        "recommended_action",
        "clinical_narrative",
    ]
    for key in required_keys:
        assert key in output, f"Missing required output key: {key}"

    assert len(output["finding_probabilities"]) == 14
    for finding, p in output["finding_probabilities"].items():
        assert 0.0 <= p <= 1.0
