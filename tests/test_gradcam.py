"""
Unit Tests for Module 15: Grad-CAM for X-ray Model.
===================================================
Tests PyTorch hook registration and detaching, gradient backpropagation,
heat map spatial dimensions and normalization, target finding resolution,
focal hotspot extraction, anatomical zone mapping, alpha-blended overlays,
and end-to-end GradCAMPipeline integration.
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.xray_model import (
    ChestXRayModel,
    ChestXRayPreprocessor,
    generate_synthetic_chest_xray,
    NIH_CHEST_XRAY_CLASSES,
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


@pytest.fixture(scope="module")
def xray_model():
    """Module-level ChestXRayModel fixture."""
    model = ChestXRayModel(architecture="densenet121", pretrained=False, num_findings=14)
    model.eval()
    return model


@pytest.fixture(scope="module")
def preprocessor():
    """Module-level Preprocessor fixture."""
    return ChestXRayPreprocessor(image_size=224)


def test_gradcam_hook_lifecycle(xray_model):
    """Verify hook attachment, capture of activations/gradients, and safe detaching."""
    with GradCAM(xray_model) as cam_explainer:
        assert cam_explainer.forward_handle is not None
        assert cam_explainer.backward_handle is not None

        dummy_x = torch.randn(1, 3, 224, 224, requires_grad=True)
        res = cam_explainer.generate_cam(dummy_x, target_class="Pneumothorax")

        assert cam_explainer.activations is not None
        assert cam_explainer.gradients is not None
        assert res["target_class_name"] == "Pneumothorax"

    # After context exit, hooks must be cleaned up
    assert cam_explainer.forward_handle is None
    assert cam_explainer.backward_handle is None


def test_gradcam_heatmap_geometry_and_bounds(xray_model):
    """Verify output heatmap dimensions (224, 224), float32 type, and [0, 1] normalization."""
    explainer = GradCAM(xray_model)
    dummy_x = torch.randn(1, 3, 224, 224, requires_grad=True)

    result = explainer.generate_cam(dummy_x, target_class=0)
    heatmap = result["cam_heatmap"]

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.dtype == np.float32
    assert heatmap.shape == (224, 224)
    assert 0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0
    explainer.remove_hooks()


def test_target_finding_resolution(xray_model):
    """Test specifying target class by name, index, or None (auto top-logit)."""
    explainer = GradCAM(xray_model)
    dummy_x = torch.randn(1, 3, 224, 224, requires_grad=True)

    # 1. By name
    res_name = explainer.generate_cam(dummy_x, target_class="Cardiomegaly")
    assert res_name["target_class_name"] == "Cardiomegaly"
    assert res_name["target_class_idx"] == NIH_CHEST_XRAY_CLASSES.index("Cardiomegaly")

    # 2. By index
    res_idx = explainer.generate_cam(dummy_x, target_class=2)
    assert res_idx["target_class_idx"] == 2
    assert res_idx["target_class_name"] == NIH_CHEST_XRAY_CLASSES[2]

    # 3. Auto top-logit
    res_auto = explainer.generate_cam(dummy_x, target_class=None)
    assert 0 <= res_auto["target_class_idx"] < 14
    assert res_auto["target_class_name"] in NIH_CHEST_XRAY_CLASSES

    explainer.remove_hooks()


def test_invalid_target_class_errors(xray_model):
    """Verify informative errors on invalid finding name or out-of-bounds index."""
    explainer = GradCAM(xray_model)
    dummy_x = torch.randn(1, 3, 224, 224, requires_grad=True)

    with pytest.raises(ValueError, match="Unknown finding"):
        explainer.generate_cam(dummy_x, target_class="NonExistentThoracicDisease")

    with pytest.raises(ValueError, match="Target index"):
        explainer.generate_cam(dummy_x, target_class=999)

    explainer.remove_hooks()


def test_counterfactual_cam_generation(xray_model):
    """Verify counterfactual (negative attribution) mode executes and yields normalized CAM."""
    explainer = GradCAM(xray_model)
    dummy_x = torch.randn(1, 3, 224, 224, requires_grad=True)

    cf_res = explainer.generate_cam(dummy_x, target_class="Edema", counterfactual=True)
    assert cf_res["is_counterfactual"] is True
    assert cf_res["cam_heatmap"].shape == (224, 224)
    assert 0.0 <= np.max(cf_res["cam_heatmap"]) <= 1.0

    explainer.remove_hooks()


def test_multi_finding_cams(xray_model):
    """Verify simultaneous multi-finding Grad-CAM generation for differential comparison."""
    explainer = GradCAM(xray_model)
    dummy_x = torch.randn(1, 3, 224, 224, requires_grad=True)

    findings = ["Consolidation", "Effusion", "Pneumonia"]
    multi_res = explainer.generate_multi_finding_cams(dummy_x, finding_names=findings)

    assert len(multi_res) == 3
    for i, res in enumerate(multi_res):
        assert res["target_class_name"] == findings[i]
        assert res["cam_heatmap"].shape == (224, 224)

    explainer.remove_hooks()


def test_colorize_and_create_cam_overlay():
    """Verify heatmap colorization and alpha blending with PIL and NumPy inputs."""
    # Synthetic test radiograph and heatmap
    orig_img = generate_synthetic_chest_xray("cardiomegaly", image_size=224, random_seed=42)
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[80:160, 80:160] = 0.95  # Focal square in cardiac center

    # 1. Colorize
    color_rgb = colorize_heatmap(heatmap, colormap_name="jet")
    assert color_rgb.shape == (224, 224, 3)
    assert color_rgb.dtype == np.uint8

    # 2. Overlay from PIL Image
    overlay_pil = create_cam_overlay(orig_img, heatmap, alpha=0.50, colormap_name="jet")
    assert isinstance(overlay_pil, Image.Image)
    assert overlay_pil.size == (224, 224)

    # 3. Overlay from NumPy array
    arr_orig = np.array(orig_img)
    overlay_arr = create_cam_overlay(arr_orig, heatmap, alpha=0.40, colormap_name="inferno")
    assert isinstance(overlay_arr, Image.Image)
    assert overlay_arr.size == (224, 224)


def test_focal_region_extraction_and_zone_mapping():
    """Verify extraction of high-activation focal hotspots and anatomical thoracic zone mapping."""
    heatmap = np.zeros((224, 224), dtype=np.float32)
    # Simulate focal apical lesion (upper thorax, right side: x in [40, 80], y in [30, 70])
    heatmap[30:70, 40:80] = 0.92

    regions = extract_focal_regions(heatmap, threshold=0.60, min_area_pixels=20)
    assert len(regions) >= 1
    top_reg = regions[0]

    assert top_reg["peak_activation"] >= 0.90
    assert top_reg["area_fraction"] > 0.0
    assert "Upper Thoracic Zone (Apical)" in top_reg["anatomical_zone"]

    # Test coordinate mapping directly
    assert map_coordinates_to_thoracic_zone(0.20, 0.20) == "Right Upper Thoracic Zone (Apical)"
    assert map_coordinates_to_thoracic_zone(0.80, 0.20) == "Left Upper Thoracic Zone (Apical)"
    assert map_coordinates_to_thoracic_zone(0.50, 0.55) == "Central Mediastinal / Cardiac Silhouette"
    assert map_coordinates_to_thoracic_zone(0.25, 0.85) == "Right Lower Basilar / Costophrenic Zone"
    assert map_coordinates_to_thoracic_zone(0.75, 0.85) == "Left Lower Basilar / Costophrenic Zone"


def test_draw_focal_boxes_on_overlay():
    """Verify annotation of bounding boxes and crosshairs onto overlay image."""
    base_img = Image.new("RGB", (224, 224), color="gray")
    regions = [
        {
            "bbox": (50, 40, 100, 90),
            "peak_coordinate": (75, 65),
            "peak_activation": 0.95,
            "anatomical_zone": "Right Upper Thoracic Zone (Apical)",
        }
    ]
    annotated = draw_focal_boxes_on_overlay(base_img, regions)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (224, 224)


def test_gradcam_pipeline_end_to_end():
    """Verify full end-to-end GradCAMPipeline inference and diagnostic report generation."""
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "chest_xray_densenet121.pt"
    if not ckpt_path.exists():
        pytest.skip("Calibrated vision checkpoint not found; run Module 14 evaluation first.")

    pipeline = GradCAMPipeline(
        model_architecture="densenet121",
        pretrained=False,
        checkpoint_path=ckpt_path,
        image_size=224,
    )

    test_img = generate_synthetic_chest_xray("pneumothorax", random_seed=902)
    output = pipeline.analyze_and_explain(
        image_input=test_img,
        target_finding=None,
        colormap_name="jet",
    )

    # Required clinical explainability fields
    assert "target_finding" in output
    assert "target_confidence" in output
    assert "cam_heatmap" in output
    assert "overlay_image" in output
    assert "focal_regions" in output
    assert "anatomical_summary" in output
    assert "radiographic_severity_index" in output
    assert "acuity_tier" in output

    assert output["target_finding"] == "Pneumothorax"
    assert output["target_confidence"] > 0.90
    assert output["acuity_tier"] == "CRITICAL_ESCALATION"
    assert isinstance(output["overlay_image"], Image.Image)
    assert output["cam_heatmap"].shape == (224, 224)
