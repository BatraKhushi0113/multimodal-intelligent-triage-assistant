"""
Grad-CAM Visual Attention Module for Chest Radiography.
=======================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM, Selvaraju et al., 2017)
tailored for emergency thoracic radiograph interpretation and clinical triage explainability.

Key Capabilities:
1. GradCAM: PyTorch hook-based gradient attribution for DenseNet-121 / EfficientNet-B0.
2. Target Finding Flexibility: Compute saliency maps for specific findings (e.g. Pneumothorax,
   Cardiomegaly) or automatically target the highest-confidence predicted pathology.
3. Multi-label & Counterfactual Saliency: Contrastive and negative attribution options.
4. Alpha-blended Overlays: High-contrast colormapped saliency overlays on radiographs.
5. Anatomical Focal Region Extraction: Automated detection of high-attention focal hotspots,
   bounding boxes, and mapping to anatomical thoracic zones (Apical, Mediastinal, Basilar, Perihilar).
6. GradCAMVisualizer: Publication-grade multi-panel diagnostic figures and cohort galleries.
7. GradCAMPipeline: High-level clinical interface coupling inference with visual explainability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from src.vision.xray_model import (
    ChestXRayModel,
    ChestXRayPipeline,
    ChestXRayPreprocessor,
    RadiographicAcuityModifier,
    NIH_CHEST_XRAY_CLASSES,
    CLINICAL_FINDING_WEIGHTS,
    DEFAULT_DETECTION_THRESHOLDS,
)

logger = logging.getLogger(__name__)


# Standard Radiological Anatomical Thoracic Zones
THORACIC_ANATOMICAL_ZONES = [
    "Right Upper Thoracic Zone (Apical)",
    "Left Upper Thoracic Zone (Apical)",
    "Right Mid-Lung Zone",
    "Left Mid-Lung Zone",
    "Central Mediastinal / Cardiac Silhouette",
    "Right Lower Basilar / Costophrenic Zone",
    "Left Lower Basilar / Costophrenic Zone",
    "Diffuse / Bilateral Perihilar",
]


def map_coordinates_to_thoracic_zone(
    norm_x: float, norm_y: float, image_width: int = 224, image_height: int = 224
) -> str:
    """
    Map normalized coordinates (x in [0, 1], y in [0, 1]) to a standardized thoracic zone.

    Anatomical orientation:
    - Image Left (x < 0.45) = Anatomical Right Hemithorax
    - Image Right (x > 0.55) = Anatomical Left Hemithorax
    - Image Center (0.35 <= x <= 0.65, 0.40 <= y <= 0.78) = Cardiac / Mediastinal
    - Top (y < 0.38) = Upper / Apical Zone
    - Bottom (y > 0.68) = Lower / Costophrenic Basilar Zone
    """
    # 1. Cardiac / Mediastinum check
    if 0.35 <= norm_x <= 0.68 and 0.42 <= norm_y <= 0.78:
        return "Central Mediastinal / Cardiac Silhouette"

    # 2. Apical / Upper Thorax
    if norm_y < 0.38:
        if norm_x < 0.50:
            return "Right Upper Thoracic Zone (Apical)"
        else:
            return "Left Upper Thoracic Zone (Apical)"

    # 3. Basilar / Costophrenic
    if norm_y > 0.68:
        if norm_x < 0.50:
            return "Right Lower Basilar / Costophrenic Zone"
        else:
            return "Left Lower Basilar / Costophrenic Zone"

    # 4. Mid-lung zones
    if norm_x < 0.50:
        return "Right Mid-Lung Zone"
    else:
        return "Left Mid-Lung Zone"


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for Chest Radiographs.

    Computes activation heatmaps by projecting class-specific gradients back to the
    highest-level feature maps of a convolutional neural network.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Initialize the GradCAM explainer.

        Args:
            model: PyTorch model (e.g. ChestXRayModel).
            target_layer: Convolutional layer to hook. If None, retrieves from model.get_gradcam_target_layer().
            device: Compute device (CPU or CUDA).
        """
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.model.to(self.device)
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_gradcam_target_layer"):
                self.target_layer = model.get_gradcam_target_layer()
            elif hasattr(model, "features"):
                # Fallback to last module in features
                self.target_layer = model.features[-1]
            else:
                raise ValueError("target_layer not provided and could not be inferred from model.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = None
        self.backward_handle = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def fwd_hook(module: nn.Module, input_tensor: Any, output_tensor: torch.Tensor) -> None:
            self.activations = output_tensor.detach()

        def bwd_hook(
            module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]
        ) -> None:
            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(fwd_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(bwd_hook)

    def remove_hooks(self) -> None:
        """Cleanly remove registered hooks to avoid memory accumulation."""
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[str, int]] = None,
        counterfactual: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a normalized 2D Grad-CAM heatmap for a target finding.

        Args:
            input_tensor: Model-ready normalized tensor of shape (1, 3, H, W) or (3, H, W).
            target_class: Class name (e.g. 'Pneumothorax') or integer index. If None,
                          automatically targets the class with the highest raw logit.
            counterfactual: If True, inverts gradient weights to highlight features
                            that negatively support the target score.

        Returns:
            Dictionary containing:
            - 'cam_heatmap': 2D float32 numpy array in [0.0, 1.0] of shape (H, W).
            - 'target_class_idx': integer index of the analyzed class.
            - 'target_class_name': string name of the analyzed class.
            - 'confidence': predicted probability in [0.0, 1.0] for the target class.
            - 'raw_logit': raw model logit before sigmoid.
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device).clone()
        input_tensor.requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probabilities = torch.sigmoid(logits)[0]

        # Determine target class index
        if target_class is None:
            target_idx = int(torch.argmax(logits, dim=1).item())
        elif isinstance(target_class, str):
            if target_class in NIH_CHEST_XRAY_CLASSES:
                target_idx = NIH_CHEST_XRAY_CLASSES.index(target_class)
            else:
                raise ValueError(
                    f"Unknown finding '{target_class}'. Must be one of {NIH_CHEST_XRAY_CLASSES}"
                )
        elif isinstance(target_class, (int, np.integer)):
            target_idx = int(target_class)
            if not (0 <= target_idx < len(NIH_CHEST_XRAY_CLASSES)):
                raise ValueError(
                    f"Target index {target_idx} out of range [0, {len(NIH_CHEST_XRAY_CLASSES)-1}]"
                )
        else:
            raise TypeError(f"Invalid target_class type: {type(target_class)}")

        target_name = NIH_CHEST_XRAY_CLASSES[target_idx]
        target_score = logits[0, target_idx]
        confidence = float(probabilities[target_idx].item())
        raw_logit = float(target_score.item())

        # Backward pass on the un-squashed logit
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Global average pooling of gradients: alpha_k = (1/Z) * sum_i sum_j (d y / d A_k)
        # Shape: (1, Channels, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        if counterfactual:
            weights = -weights

        # Linear combination: L_gradcam = ReLU( sum_k alpha_k * A_k )
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Bilinear interpolation to original spatial resolution
        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)

        # Min-Max Normalization to [0.0, 1.0]
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        denom = cam_max - cam_min
        if denom > 1e-8:
            norm_cam = (cam_np - cam_min) / denom
        else:
            norm_cam = np.zeros_like(cam_np)

        return {
            "cam_heatmap": norm_cam,
            "target_class_idx": target_idx,
            "target_class_name": target_name,
            "confidence": confidence,
            "raw_logit": raw_logit,
            "is_counterfactual": counterfactual,
        }

    def generate_multi_finding_cams(
        self,
        input_tensor: torch.Tensor,
        finding_names: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate separate Grad-CAM heatmaps for multiple candidate findings.

        Args:
            input_tensor: Model-ready normalized tensor.
            finding_names: Explicit list of finding names to compute. If None,
                           selects top-k highest predicted findings.
            top_k: Number of top findings if finding_names is None.

        Returns:
            List of CAM result dictionaries.
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            probs = torch.sigmoid(self.model(input_tensor.to(self.device)))[0].cpu().numpy()

        if finding_names is None:
            sorted_indices = np.argsort(probs)[::-1][:top_k]
            targets = [NIH_CHEST_XRAY_CLASSES[idx] for idx in sorted_indices]
        else:
            targets = finding_names

        results = []
        for t in targets:
            cam_res = self.generate_cam(input_tensor, target_class=t)
            results.append(cam_res)

        return results


def colorize_heatmap(
    heatmap: np.ndarray, colormap_name: str = "jet"
) -> np.ndarray:
    """
    Convert a 2D float heatmap in [0, 1] to an RGB uint8 array of shape (H, W, 3).

    Args:
        heatmap: 2D float numpy array.
        colormap_name: Scientific colormap name ('jet', 'inferno', 'viridis', 'magma').

    Returns:
        RGB uint8 numpy array.
    """
    heatmap_clamped = np.clip(heatmap, 0.0, 1.0)

    if HAS_OPENCV:
        cv_map_dict = {
            "jet": cv2.COLORMAP_JET,
            "inferno": cv2.COLORMAP_INFERNO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "magma": cv2.COLORMAP_MAGMA,
        }
        cv_code = cv_map_dict.get(colormap_name.lower(), cv2.COLORMAP_JET)
        heatmap_uint8 = (heatmap_clamped * 255.0).astype(np.uint8)
        color_bgr = cv2.applyColorMap(heatmap_uint8, cv_code)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        return color_rgb
    else:
        # High-fidelity matplotlib fallback
        try:
            cmap = plt.get_cmap(colormap_name)
        except Exception:
            cmap = plt.get_cmap("jet")
        color_rgba = cmap(heatmap_clamped)
        return (color_rgba[:, :, :3] * 255.0).astype(np.uint8)


def create_cam_overlay(
    original_image: Union[Image.Image, np.ndarray, str, Path],
    cam_heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
    threshold_cutoff: float = 0.15,
) -> Image.Image:
    """
    Blend a Grad-CAM attention heatmap onto the original chest radiograph.

    Args:
        original_image: PIL Image, numpy array (H, W, 3) or (H, W), or image path.
        cam_heatmap: 2D float numpy array with values in [0, 1].
        alpha: Heatmap blend opacity (0.0 = original image, 1.0 = heatmap only).
        colormap_name: Colormap name ('jet', 'inferno', 'viridis').
        threshold_cutoff: Attention floor below which the heatmap is faded out
                          to preserve background anatomical detail.

    Returns:
        Blended RGB PIL Image.
    """
    # 1. Standardize original image to RGB numpy array
    if isinstance(original_image, (str, Path)):
        pil_orig = Image.open(original_image).convert("RGB")
    elif isinstance(original_image, Image.Image):
        pil_orig = original_image.convert("RGB")
    elif isinstance(original_image, np.ndarray):
        arr = original_image.copy()
        if arr.dtype != np.uint8:
            if np.max(arr) <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        pil_orig = Image.fromarray(arr).convert("RGB")
    else:
        raise TypeError(f"Unsupported original_image type: {type(original_image)}")

    target_w, target_h = pil_orig.size
    orig_np = np.array(pil_orig).astype(np.float32)

    # 2. Resize heatmap to match original image dimensions
    if cam_heatmap.shape != (target_h, target_w):
        if HAS_OPENCV:
            cam_resized = cv2.resize(
                cam_heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            cam_pil = Image.fromarray((cam_heatmap * 255.0).astype(np.uint8))
            cam_pil = cam_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
            cam_resized = np.array(cam_pil).astype(np.float32) / 255.0
    else:
        cam_resized = cam_heatmap

    # 3. Colorize heatmap
    color_heatmap = colorize_heatmap(cam_resized, colormap_name=colormap_name).astype(np.float32)

    # 4. Apply soft threshold masking to prevent washing out clear anatomical regions
    effective_alpha = np.clip(
        (cam_resized - threshold_cutoff) / (1.0 - threshold_cutoff + 1e-8), 0.0, 1.0
    )
    # Shape: (H, W, 1)
    blend_weights = (effective_alpha * alpha)[:, :, None]

    blended = blend_weights * color_heatmap + (1.0 - blend_weights) * orig_np
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(blended)


def extract_focal_regions(
    cam_heatmap: np.ndarray,
    threshold: float = 0.60,
    min_area_pixels: int = 40,
    image_shape: Tuple[int, int] = (224, 224),
) -> List[Dict[str, Any]]:
    """
    Extract focal high-activation clinical hotspot regions, bounding boxes, and anatomical zones.

    Args:
        cam_heatmap: 2D float numpy array in [0, 1].
        threshold: Saliency threshold for defining the focal hotspot mask.
        min_area_pixels: Minimum pixel count to register a focal region.
        image_shape: (Height, Width) of the reference image.

    Returns:
        List of dictionaries with bounding box, peak coordinates, area, and anatomical zone.
    """
    H, W = cam_heatmap.shape
    binary_mask = (cam_heatmap >= threshold).astype(np.uint8)

    regions: List[Dict[str, Any]] = []

    if HAS_OPENCV:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_pixels:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Find peak coordinate inside this contour
            roi = cam_heatmap[y : y + h, x : x + w]
            peak_loc_rel = np.unravel_index(np.argmax(roi), roi.shape)
            peak_y = int(y + peak_loc_rel[0])
            peak_x = int(x + peak_loc_rel[1])
            peak_val = float(cam_heatmap[peak_y, peak_x])

            norm_cx = (x + w / 2.0) / float(W)
            norm_cy = (y + h / 2.0) / float(H)
            zone_desc = map_coordinates_to_thoracic_zone(norm_cx, norm_cy, W, H)

            regions.append({
                "bbox": (int(x), int(y), int(x + w), int(y + h)),
                "bbox_normalized": (
                    round(x / float(W), 4),
                    round(y / float(H), 4),
                    round((x + w) / float(W), 4),
                    round((y + h) / float(H), 4),
                ),
                "peak_coordinate": (int(peak_x), int(peak_y)),
                "peak_activation": round(peak_val, 4),
                "pixel_area": int(area),
                "area_fraction": round(float(area) / float(H * W), 4),
                "anatomical_zone": zone_desc,
            })
    else:
        # Fallback using scipy.ndimage
        labeled, num_features = ndi.label(binary_mask)
        slices = ndi.find_objects(labeled)
        for idx, slc in enumerate(slices):
            if slc is None:
                continue
            sub_mask = labeled[slc] == (idx + 1)
            area = int(np.sum(sub_mask))
            if area < min_area_pixels:
                continue

            y_min, y_max = slc[0].start, slc[0].stop
            x_min, x_max = slc[1].start, slc[1].stop
            roi = cam_heatmap[slc]
            peak_loc_rel = np.unravel_index(np.argmax(roi), roi.shape)
            peak_y = int(y_min + peak_loc_rel[0])
            peak_x = int(x_min + peak_loc_rel[1])
            peak_val = float(cam_heatmap[peak_y, peak_x])

            norm_cx = ((x_min + x_max) / 2.0) / float(W)
            norm_cy = ((y_min + y_max) / 2.0) / float(H)
            zone_desc = map_coordinates_to_thoracic_zone(norm_cx, norm_cy, W, H)

            regions.append({
                "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
                "bbox_normalized": (
                    round(x_min / float(W), 4),
                    round(y_min / float(H), 4),
                    round(x_max / float(W), 4),
                    round(y_max / float(H), 4),
                ),
                "peak_coordinate": (int(peak_x), int(peak_y)),
                "peak_activation": round(peak_val, 4),
                "pixel_area": area,
                "area_fraction": round(float(area) / float(H * W), 4),
                "anatomical_zone": zone_desc,
            })

    # Sort descending by peak activation
    regions.sort(key=lambda r: r["peak_activation"], reverse=True)
    return regions


def draw_focal_boxes_on_overlay(
    overlay_image: Image.Image,
    focal_regions: List[Dict[str, Any]],
    box_color: str = "#FF1744",
    crosshair_color: str = "#00E676",
    line_width: int = 2,
) -> Image.Image:
    """
    Annotate overlay image with bounding boxes, crosshairs, and peak activation badges.

    Args:
        overlay_image: PIL Image of blended radiograph overlay.
        focal_regions: List of region dictionaries from extract_focal_regions.
        box_color: Hex color string for bounding box outline.
        crosshair_color: Hex color string for peak coordinate crosshair.
        line_width: Border pixel width.

    Returns:
        Annotated PIL Image.
    """
    annotated = overlay_image.copy()
    draw = ImageDraw.Draw(annotated)

    for i, reg in enumerate(focal_regions):
        x1, y1, x2, y2 = reg["bbox"]
        px, py = reg["peak_coordinate"]
        peak_val = reg["peak_activation"]
        zone = reg["anatomical_zone"]

        # 1. Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        # 2. Peak crosshair (10px arm length)
        arm = 5
        draw.line([px - arm, py, px + arm, py], fill=crosshair_color, width=line_width)
        draw.line([px, py - arm, px, py + arm], fill=crosshair_color, width=line_width)

        # 3. Label tag
        badge_text = f"ROI #{i+1}: {peak_val:.2f}"
        badge_y = max(2, y1 - 12)
        draw.rectangle([x1, badge_y, x1 + 58, badge_y + 11], fill="#1E293B")
        draw.text((x1 + 2, badge_y), badge_text, fill="#FFFFFF")

    return annotated


class GradCAMVisualizer:
    """
    Publication-Quality Clinical Visualization Suite for Radiographic Grad-CAM.
    """

    def __init__(self, style: str = "seaborn-v0_8-whitegrid") -> None:
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

    def plot_diagnostic_explanation_panel(
        self,
        original_image: Union[Image.Image, np.ndarray],
        cam_heatmap: np.ndarray,
        overlay_image: Image.Image,
        all_probabilities: Dict[str, float],
        target_finding: str,
        confidence: float,
        acuity_tier: str,
        severity_index: float,
        focal_regions: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (16, 5),
    ) -> plt.Figure:
        """
        Generate a 4-panel clinical diagnostic explanation report:
        1. Original Chest Radiograph
        2. Pure Grad-CAM Attention Heatmap
        3. Annotated Saliency Overlay (Hotspot Bounding Boxes & Peak Crosshair)
        4. Multi-Label Pathology Confidence Bar Chart with Clinical Thresholds
        """
        fig, axes = plt.subplots(1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 1, 1.4]})

        # Panel 1: Original Radiograph
        if isinstance(original_image, Image.Image):
            axes[0].imshow(original_image.convert("L"), cmap="gray")
        elif isinstance(original_image, np.ndarray):
            if original_image.ndim == 3:
                axes[0].imshow(original_image)
            else:
                axes[0].imshow(original_image, cmap="gray")
        axes[0].set_title("1. Chest Radiograph (Input)", fontsize=11, fontweight="bold", pad=8)
        axes[0].text(8, 18, "PA ERECT", color="yellow", fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
        axes[0].text(205, 18, "L", color="white", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Pure Attention Heatmap
        im_heat = axes[1].imshow(cam_heatmap, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"2. Grad-CAM: {target_finding}", fontsize=11, fontweight="bold", pad=8)
        axes[1].axis("off")
        cbar = fig.colorbar(im_heat, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Saliency Attribution", fontsize=8)

        # Panel 3: Annotated Overlay
        annotated_overlay = draw_focal_boxes_on_overlay(overlay_image, focal_regions)
        axes[2].imshow(annotated_overlay)
        axes[2].set_title("3. Anatomical Saliency Overlay", fontsize=11, fontweight="bold", pad=8)
        axes[2].axis("off")

        # Panel 4: Multi-label Probability Bar Chart
        findings = list(all_probabilities.keys())
        probs = [all_probabilities[f] for f in findings]
        y_pos = np.arange(len(findings))

        colors = []
        for f in findings:
            if f == target_finding:
                colors.append("#EF4444")  # Highlight target finding in bold red
            elif f in ("Pneumothorax", "Edema", "Consolidation", "Cardiomegaly"):
                colors.append("#F59E0B")  # Critical findings in amber
            else:
                colors.append("#3B82F6")  # Other findings in blue

        bars = axes[3].barh(y_pos, probs, color=colors, height=0.65, edgecolor="none", alpha=0.85)
        axes[3].axvline(0.40, color="#DC2626", linestyle="--", linewidth=1.2, alpha=0.7, label="Avg Detection Threshold (0.40)")
        axes[3].set_yticks(y_pos)
        axes[3].set_yticklabels(findings, fontsize=8)
        axes[3].invert_yaxis()
        axes[3].set_xlim(0.0, 1.05)
        axes[3].set_xlabel("Predicted Probability", fontsize=9, fontweight="bold")
        axes[3].set_title(
            f"4. Radiographic Acuity: {acuity_tier} (RSI: {severity_index:.2f})",
            fontsize=10,
            fontweight="bold",
            pad=8,
        )
        axes[3].legend(loc="lower right", fontsize=8, framealpha=0.8)
        axes[3].grid(axis="x", linestyle=":", alpha=0.6)

        # Value labels on bars
        for bar, prob in zip(bars, probs):
            if prob >= 0.05:
                axes[3].text(
                    prob + 0.02,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{prob*100:.1f}%",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="#1E293B",
                )

        plt.tight_layout()

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            logger.info(f"Saved diagnostic explanation figure to {output_path}")

        return fig

    def plot_multi_finding_comparison(
        self,
        original_image: Union[Image.Image, np.ndarray],
        cam_results: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """
        Compare differential visual attention patterns across multiple pathologies on the same radiograph.
        """
        num_cams = len(cam_results)
        fig, axes = plt.subplots(1, num_cams + 1, figsize=figsize)

        # Panel 0: Original Radiograph
        if isinstance(original_image, Image.Image):
            axes[0].imshow(original_image.convert("L"), cmap="gray")
        else:
            axes[0].imshow(original_image, cmap="gray")
        axes[0].set_title("Input Chest Radiograph", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panels 1..N: Individual findings
        for idx, cam_res in enumerate(cam_results):
            ax = axes[idx + 1]
            finding = cam_res["target_class_name"]
            conf = cam_res["confidence"]
            heat = cam_res["cam_heatmap"]

            overlay = create_cam_overlay(original_image, heat, alpha=0.50, colormap_name="jet")
            ax.imshow(overlay)
            ax.set_title(f"{finding}\nProb: {conf*100:.1f}%", fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            logger.info(f"Saved multi-finding comparison to {output_path}")

        return fig

    def plot_cohort_gradcam_gallery(
        self,
        cases_data: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (16, 9),
    ) -> plt.Figure:
        """
        Generate a multi-case gallery showing original radiographs, attention overlays,
        and clinical acuity badges across standard emergency presentations.
        """
        num_cases = len(cases_data)
        rows = 2
        cols = int(np.ceil(num_cases / rows))

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes_flat = axes.flatten()

        for idx, case in enumerate(cases_data):
            ax = axes_flat[idx]
            overlay = case["overlay_image"]
            finding = case.get("target_finding", "Normal")
            conf = case.get("confidence", 0.0)
            acuity = case.get("acuity_tier", "NO_ACUTE_PATHOLOGY")
            rsi = case.get("radiographic_severity_index", 0.0)
            case_title = case.get("case_label", f"Case {idx+1}")

            ax.imshow(overlay)
            ax.axis("off")

            badge_color = {
                "CRITICAL_ESCALATION": "#EF4444",
                "URGENT_EVALUATION": "#F59E0B",
                "PATHOLOGY_DETECTED": "#3B82F6",
                "NO_ACUTE_PATHOLOGY": "#10B981",
            }.get(acuity, "#6B7280")

            ax.set_title(
                f"{case_title}\nTop: {finding} ({conf*100:.1f}%) | RSI: {rsi:.2f}",
                fontsize=9,
                fontweight="bold",
                pad=6,
            )

            # Acuity badge inside subplot
            ax.text(
                8,
                214,
                acuity,
                color="white",
                fontsize=7,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_color, alpha=0.9),
            )

        # Hide any unused subplots
        for idx in range(num_cases, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            logger.info(f"Saved cohort gallery to {output_path}")

        return fig


class GradCAMPipeline:
    """
    Unified High-Level Clinical Inference & Explainability Pipeline for Chest Radiographs.
    """

    def __init__(
        self,
        model_architecture: str = "densenet121",
        pretrained: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
        image_size: int = 224,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pipeline = ChestXRayPipeline(
            model_architecture=model_architecture,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            device=device,
        )
        self.preprocessor = self.pipeline.preprocessor
        self.model = self.pipeline.model
        self.explainer = GradCAM(self.model, device=self.device)
        self.visualizer = GradCAMVisualizer()

    def analyze_and_explain(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        target_finding: Optional[str] = None,
        colormap_name: str = "jet",
        threshold_cutoff: float = 0.15,
        focal_threshold: float = 0.55,
        save_plot_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full radiographic triage inference and generate visual explanations.

        Args:
            image_input: File path, PIL Image, or numpy array.
            target_finding: Specific finding name (e.g. 'Pneumothorax'). If None,
                            automatically uses top detected finding or highest logit.
            colormap_name: Scientific heatmap colormap ('jet', 'inferno', 'viridis').
            threshold_cutoff: Attention floor for alpha-blended overlay.
            focal_threshold: Saliency threshold for hotspot bounding box extraction.
            save_plot_path: Optional file path to save 4-panel diagnostic figure.

        Returns:
            Comprehensive clinical explainability dictionary.
        """
        # 1. Run inference
        inference_res = self.pipeline.predict(image_input)
        finding_probs = inference_res["finding_probabilities"]
        detected = inference_res["detected_findings"]
        acuity_tier = inference_res["acuity_tier"]
        rsi = inference_res["radiographic_severity_index"]

        # 2. Determine target finding
        if target_finding is None:
            if len(detected) > 0:
                chosen_target = next(iter(detected.keys()))
            else:
                chosen_target = max(finding_probs.items(), key=lambda x: x[1])[0]
        else:
            chosen_target = target_finding

        # 3. Generate Grad-CAM heatmap
        tensor = self.preprocessor.preprocess(image_input)
        cam_result = self.explainer.generate_cam(tensor, target_class=chosen_target)
        cam_heatmap = cam_result["cam_heatmap"]
        confidence = cam_result["confidence"]

        # 4. Generate visual overlay
        if isinstance(image_input, (str, Path)):
            pil_original = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_original = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_original = Image.fromarray(image_input).convert("RGB")
        else:
            # Reconstruct from unnormalized tensor
            unnorm_np = self.preprocessor.unnormalize(tensor)
            pil_original = Image.fromarray((unnorm_np * 255.0).astype(np.uint8))

        overlay_img = create_cam_overlay(
            original_image=pil_original,
            cam_heatmap=cam_heatmap,
            alpha=0.48,
            colormap_name=colormap_name,
            threshold_cutoff=threshold_cutoff,
        )

        # 5. Extract focal hotspot regions
        focal_regions = extract_focal_regions(
            cam_heatmap=cam_heatmap,
            threshold=focal_threshold,
            min_area_pixels=30,
            image_shape=cam_heatmap.shape,
        )

        # 6. Formulate anatomical clinical narrative
        if focal_regions:
            primary_zone = focal_regions[0]["anatomical_zone"]
            peak_act = focal_regions[0]["peak_activation"]
            anatomical_summary = (
                f"Visual Saliency Analysis: Primary model attention for {chosen_target} "
                f"(confidence: {confidence*100:.1f}%) is localized to {primary_zone} "
                f"(peak activation: {peak_act:.2f}, {len(focal_regions)} focal cluster(s) detected)."
            )
        else:
            anatomical_summary = (
                f"Visual Saliency Analysis: Model attention for {chosen_target} "
                f"(confidence: {confidence*100:.1f}%) demonstrates diffuse parenchymal distribution "
                f"without isolated high-activation focal hotspots."
            )

        # 7. Generate diagnostic figure if requested
        fig = None
        if save_plot_path is not None:
            fig = self.visualizer.plot_diagnostic_explanation_panel(
                original_image=pil_original,
                cam_heatmap=cam_heatmap,
                overlay_image=overlay_img,
                all_probabilities=finding_probs,
                target_finding=chosen_target,
                confidence=confidence,
                acuity_tier=acuity_tier,
                severity_index=rsi,
                focal_regions=focal_regions,
                output_path=save_plot_path,
            )

        return {
            **inference_res,
            "target_finding": chosen_target,
            "target_confidence": confidence,
            "cam_heatmap": cam_heatmap,
            "overlay_image": overlay_img,
            "focal_regions": focal_regions,
            "num_focal_regions": len(focal_regions),
            "primary_anatomical_zone": focal_regions[0]["anatomical_zone"] if focal_regions else "Diffuse",
            "anatomical_summary": anatomical_summary,
            "diagnostic_figure": fig,
        }
