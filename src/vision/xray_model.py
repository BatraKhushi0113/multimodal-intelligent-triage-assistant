"""
Chest X-ray Analysis Module for the Multimodal Intelligent Triage Assistant.

This module provides:
1. Deep learning vision backbone (DenseNet-121 / EfficientNet-B0) fine-tuned for
   multi-label detection of 14 standard NIH Chest X-ray thoracic pathologies.
2. Clinical decision support acuity modifier computing a Radiographic Severity Index
   and emergency escalation recommendations decoupled from initial triage intake.
3. Realistic procedural chest radiograph generator for benchmarking, edge testing,
   and offline simulation without requiring massive external DICOM downloads.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# Standard 14 NIH Chest X-ray Pathology Classes
NIH_CHEST_XRAY_CLASSES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# Clinical Severity Weights for Acuity Modification
# Higher weight indicates immediate emergency risk at triage
CLINICAL_FINDING_WEIGHTS: Dict[str, float] = {
    "Pneumothorax": 1.00,       # Risk of tension pneumothorax / respiratory arrest
    "Edema": 0.90,              # Acute pulmonary edema / flash cardiogenic edema
    "Consolidation": 0.85,      # Severe acute lobar pneumonia / ARDS
    "Cardiomegaly": 0.75,       # Acute heart failure exacerbation / pericardial effusion
    "Effusion": 0.65,           # Massive pleural effusion / hemothorax
    "Pneumonia": 0.65,          # Acute infectious infiltrate
    "Atelectasis": 0.50,        # Significant segmental/lobar collapse
    "Infiltration": 0.50,       # Diffuse alveolar/interstitial opacities
    "Emphysema": 0.40,          # Severe hyperinflation / COPD exacerbation
    "Mass": 0.40,               # Solitary or multiple pulmonary mass (>3cm)
    "Pleural_Thickening": 0.35, # Chronic fibrothorax or acute exudative pleurisy
    "Fibrosis": 0.30,           # Established interstitial lung disease
    "Nodule": 0.30,             # Granuloma or small coin lesion (<3cm)
    "Hernia": 0.30,             # Diaphragmatic visceral displacement
}

# Calibrated Detection Thresholds for Multi-Label Inference
# Calibrated for high emergency sensitivity on acute life threats
DEFAULT_DETECTION_THRESHOLDS: Dict[str, float] = {
    "Pneumothorax": 0.35,
    "Edema": 0.35,
    "Consolidation": 0.40,
    "Cardiomegaly": 0.45,
    "Effusion": 0.40,
    "Pneumonia": 0.40,
    "Atelectasis": 0.45,
    "Infiltration": 0.45,
    "Mass": 0.50,
    "Nodule": 0.50,
    "Emphysema": 0.50,
    "Fibrosis": 0.50,
    "Pleural_Thickening": 0.50,
    "Hernia": 0.50,
}


class ChestXRayModel(nn.Module):
    """
    Multi-label Chest Radiograph Pathology Detection Network.

    Utilizes an ImageNet-pretrained DenseNet-121 backbone (standard CheXNet topology)
    or EfficientNet-B0 with custom feature extraction and a regularized multi-label
    classification head producing logits for 14 NIH thoracic findings.
    """

    def __init__(
        self,
        architecture: str = "densenet121",
        pretrained: bool = True,
        num_findings: int = 14,
        dropout_rate: float = 0.25,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        self.num_findings = num_findings
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels

        if "densenet" in self.architecture:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            try:
                base_model = models.densenet121(weights=weights)
            except Exception as e:
                logger.warning(
                    f"Could not load online weights for DenseNet-121 ({e}); initializing with random weights."
                )
                base_model = models.densenet121(weights=None)

            self.features = base_model.features
            in_features = base_model.classifier.in_features  # 1024

            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(256, num_findings),
            )
            self._target_cam_layer = self.features.denseblock4.denselayer16.conv2

        elif "efficientnet" in self.architecture:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            try:
                base_model = models.efficientnet_b0(weights=weights)
            except Exception as e:
                logger.warning(
                    f"Could not load online weights for EfficientNet-B0 ({e}); initializing with random weights."
                )
                base_model = models.efficientnet_b0(weights=None)

            self.features = base_model.features
            in_features = base_model.classifier[1].in_features  # 1280
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(256, num_findings),
            )
            self._target_cam_layer = self.features[-1]
        else:
            raise ValueError(
                f"Unsupported vision architecture '{architecture}'. Supported: 'densenet121', 'efficientnet_b0'."
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled feature vector before the final classification head.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Pooled feature tensor of shape (B, hidden_dim).
        """
        feats = self.features(x)
        feats = nn.functional.relu(feats, inplace=False)
        pooled = self.global_pool(feats)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw logits for all 14 findings.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_findings).
        """
        feats = self.features(x)
        feats = nn.functional.relu(feats, inplace=False)
        pooled = self.global_pool(feats)
        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sigmoid probabilities for all findings.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Probability tensor in [0, 1] of shape (B, num_findings).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the optimal convolutional layer for Grad-CAM attribution."""
        return self._target_cam_layer


class ChestXRayPreprocessor:
    """
    Standardized Chest Radiograph Preprocessing & Normalization Pipeline.

    Ensures single-channel grayscale radiographs or standard RGB images
    are converted to 3-channel tensors normalized via standard ImageNet parameters.
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def preprocess(
        self, image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess an image input into a model-ready batch tensor of shape (1, 3, H, W).

        Args:
            image_input: File path, PIL Image, numpy array, or existing torch Tensor.

        Returns:
            PyTorch tensor normalized and shaped (1, 3, image_size, image_size).
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Radiograph image not found: {path}")
            pil_img = Image.open(path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            arr = image_input
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)

            if arr.ndim == 2:
                pil_img = Image.fromarray(arr, mode="L").convert("RGB")
            elif arr.ndim == 3:
                if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                    # Channels first (C, H, W) -> convert to (H, W, C)
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                pil_img = Image.fromarray(arr).convert("RGB")
            else:
                raise ValueError(f"Invalid numpy array shape for radiograph: {arr.shape}")
        elif isinstance(image_input, torch.Tensor):
            t = image_input
            if t.ndim == 2:
                t = t.unsqueeze(0).repeat(3, 1, 1)
            elif t.ndim == 3 and t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            elif t.ndim == 3 and t.shape[2] in (1, 3):
                t = t.permute(2, 0, 1)
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)

            if t.ndim == 3:
                t = t.unsqueeze(0)  # (1, 3, H, W)

            if t.shape[-2:] != (self.image_size, self.image_size):
                t = nn.functional.interpolate(
                    t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
                )
            return t.float()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(pil_img)  # (3, H, W)
        return tensor.unsqueeze(0)  # (1, 3, H, W)

    def unnormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Invert ImageNet normalization for visualization purposes.

        Args:
            tensor: Normalized tensor of shape (3, H, W) or (1, 3, H, W).

        Returns:
            Numpy array of shape (H, W, 3) with values in [0, 1].
        """
        t = tensor.detach().cpu().clone()
        if t.ndim == 4:
            t = t.squeeze(0)

        for c in range(3):
            t[c] = t[c] * self.std[c] + self.mean[c]

        t = torch.clamp(t, 0.0, 1.0)
        return t.permute(1, 2, 0).numpy()


class RadiographicAcuityModifier:
    """
    Decoupled Clinical Decision Support Modifier for Chest Radiographs.

    Analyzes multi-label findings from a chest X-ray and translates them into:
    1. Continuous Radiographic Severity Index (RSI in [0.0, 1.0]).
    2. Critical Emergency Flags (e.g., Tension Pneumothorax, Acute Pulmonary Edema).
    3. Acuity Escalation Recommendation (resuscitation, emergency, monitoring).
    4. Structured Clinical Narrative for Emergency Care Teams.
    """

    def __init__(
        self,
        finding_weights: Optional[Dict[str, float]] = None,
        detection_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.finding_weights = finding_weights or CLINICAL_FINDING_WEIGHTS
        self.detection_thresholds = detection_thresholds or DEFAULT_DETECTION_THRESHOLDS

    def evaluate_findings(
        self,
        finding_probabilities: Dict[str, float],
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate predicted probabilities and formulate clinical triage recommendations.

        Args:
            finding_probabilities: Dict mapping finding name to predicted probability [0, 1].
            custom_thresholds: Optional override thresholds for detection.

        Returns:
            Dictionary containing severity index, detected findings, emergency flags,
            and clinical recommendation.
        """
        thresholds = custom_thresholds or self.detection_thresholds

        detected_findings: Dict[str, float] = {}
        for finding, prob in finding_probabilities.items():
            thresh = thresholds.get(finding, 0.50)
            if prob >= thresh:
                detected_findings[finding] = round(float(prob), 4)

        # Sort detected findings by probability descending
        detected_findings = dict(
            sorted(detected_findings.items(), key=lambda item: item[1], reverse=True)
        )

        # Identify critical life threats
        critical_candidates = ["Pneumothorax", "Edema", "Consolidation", "Cardiomegaly"]
        critical_flags: List[str] = [
            f for f in critical_candidates if f in detected_findings
        ]

        # Calculate continuous Radiographic Severity Index (RSI)
        # Combines the maximal weighted finding with secondary cumulative impact
        weighted_scores = [
            prob * self.finding_weights.get(finding, 0.3)
            for finding, prob in finding_probabilities.items()
        ]
        max_weighted = max(weighted_scores) if weighted_scores else 0.0

        # Cumulative secondary contribution
        secondary_sum = sum(
            prob * self.finding_weights.get(finding, 0.3) * 0.15
            for finding, prob in detected_findings.items()
        )
        radiographic_severity_index = min(1.0, max_weighted + secondary_sum)

        # Determine clinical recommendation
        if "Pneumothorax" in detected_findings or "Edema" in detected_findings:
            acuity_tier = "CRITICAL_ESCALATION"
            triage_impact = "High Acuity (Potential KTAS 1/2 Escalation)"
            urgency_level = "EMERGENT_IMMEDIATE"
            recommended_action = (
                "IMMEDIATE PHYSICIAN INTERVENTION: Radiographic signs of acute cardiopulmonary "
                "decompensation (tension pneumothorax / acute pulmonary edema). Initiate immediate "
                "bedside clinical evaluation, high-flow oxygenation, and decompression/diuretic prep."
            )
        elif "Consolidation" in detected_findings or "Cardiomegaly" in detected_findings:
            acuity_tier = "URGENT_EVALUATION"
            triage_impact = "Moderate-to-High Acuity (Potential KTAS 2/3 Modifier)"
            urgency_level = "VERY_URGENT"
            recommended_action = (
                "URGENT CLINICAL REVIEW: Acute lobar consolidation or marked cardiomegaly detected. "
                "Evaluate for severe respiratory compromise, sepsis secondary to pneumonia, or decompensated heart failure."
            )
        elif len(detected_findings) > 0:
            acuity_tier = "PATHOLOGY_DETECTED"
            triage_impact = "Subacute Modifier (Potential KTAS 3/4 Tracking)"
            urgency_level = "URGENT"
            recommended_action = (
                "STANDARD RADIOGRAPHIC PROTOCOL: Detected thoracic pathology without immediate "
                "life-threatening signs. Correlate with physical auscultation, clinical history, and laboratory vitals."
            )
        else:
            acuity_tier = "NO_ACUTE_PATHOLOGY"
            triage_impact = "Neutral (Preserve Primary Intake Acuity)"
            urgency_level = "STANDARD"
            recommended_action = (
                "CLEAR RADIOGRAPH: No acute thoracic abnormalities exceeding detection thresholds. "
                "Maintain baseline triage acuity determined from clinical vitals and chief complaint."
            )

        # Generate clinician narrative summary
        if detected_findings:
            findings_str = ", ".join(
                [f"{k} ({v*100:.1f}%)" for k, v in detected_findings.items()]
            )
            narrative = (
                f"Chest Radiograph Analysis: Detected {len(detected_findings)} thoracic pathology/pathologies: "
                f"[{findings_str}]. Radiographic Severity Index: {radiographic_severity_index:.2f}. "
                f"Assessment: {recommended_action}"
            )
        else:
            narrative = (
                f"Chest Radiograph Analysis: Unremarkable study with no findings exceeding clinical thresholds. "
                f"Radiographic Severity Index: {radiographic_severity_index:.2f}. "
                f"Assessment: {recommended_action}"
            )

        return {
            "radiographic_severity_index": round(float(radiographic_severity_index), 4),
            "acuity_tier": acuity_tier,
            "triage_impact": triage_impact,
            "urgency_level": urgency_level,
            "critical_flags": critical_flags,
            "detected_findings": detected_findings,
            "num_findings_detected": len(detected_findings),
            "recommended_action": recommended_action,
            "clinical_narrative": narrative,
        }


def generate_synthetic_chest_xray(
    pathology: str = "normal",
    image_size: int = 224,
    random_seed: Optional[int] = 42,
) -> Image.Image:
    """
    Generate an anatomically faithful synthetic chest radiograph for simulation and testing.

    Models anatomical chest structures:
    - Rib cage contours and thoracic perimeter
    - Radiolucent (dark) bilateral lung fields
    - Central radiopaque (white) vertebral column and mediastinum
    - Left-shifted asymmetric cardiac silhouette
    - Smooth diaphragmatic domes with acute costophrenic angles
    - Procedurally injected pathological features:
      * 'normal': Clear lung fields, sharp costophrenic angles.
      * 'pneumothorax': Unilateral peripheral hyperlucency with sharp visceral pleural line and medial compression.
      * 'cardiomegaly': Pronounced transverse cardiac enlargement (CTR > 0.60).
      * 'effusion': Basilar fluid meniscus sign and blunted costophrenic sulcus.
      * 'consolidation': Dense patchy alveolar infiltrate with irregular borders.
      * 'edema': Bilateral perihilar "batwing" vascular haziness.

    Args:
        pathology: Target pathology ('normal', 'pneumothorax', 'cardiomegaly',
                   'effusion', 'consolidation', 'edema').
        image_size: Dimensions of generated square image.
        random_seed: Seed for reproducible procedural generation.

    Returns:
        PIL Image in 'L' (grayscale) mode.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    H, W = image_size, image_size
    y, x = np.ogrid[:H, :W]
    cx, cy = W / 2.0, H / 2.0

    # 1. Base Soft Tissue Thoracic Mantle (Dark background outside thorax)
    thorax_radius_x = W * 0.44
    thorax_radius_y = H * 0.46
    thorax_mask = ((x - cx) ** 2) / (thorax_radius_x**2) + ((y - cy) ** 2) / (thorax_radius_y**2) <= 1.0
    img = np.where(thorax_mask, 0.28, 0.08)

    # 2. Bilateral Lung Fields (Radiolucent - Dark)
    # Right Lung (Anatomical right = Image left)
    rl_cx, rl_cy = W * 0.31, H * 0.44
    rl_rx, rl_ry = W * 0.16, H * 0.31
    right_lung_dist = ((x - rl_cx) ** 2) / (rl_rx**2) + ((y - rl_cy) ** 2) / (rl_ry**2)
    right_lung_mask = right_lung_dist <= 1.0

    # Left Lung (Anatomical left = Image right)
    ll_cx, ll_cy = W * 0.69, H * 0.44
    ll_rx, ll_ry = W * 0.16, H * 0.31
    left_lung_dist = ((x - ll_cx) ** 2) / (ll_rx**2) + ((y - ll_cy) ** 2) / (ll_ry**2)
    left_lung_mask = left_lung_dist <= 1.0

    # Carve out lungs with smooth gradients
    lungs_combined = np.maximum(
        np.clip(1.0 - right_lung_dist, 0.0, 1.0),
        np.clip(1.0 - left_lung_dist, 0.0, 1.0),
    )
    img = img - 0.22 * lungs_combined

    # 3. Central Mediastinum and Spine Column (Dense radiopaque central stripe)
    spine_width = W * 0.06
    spine_intensity = np.exp(-((x - cx) ** 2) / (2 * (spine_width**2)))
    img = img + 0.32 * spine_intensity * (y / H) * 0.9

    # 4. Diaphragmatic Domes (Smooth curved white boundaries at lower thorax)
    # Right dome (slightly higher)
    r_diaph = H * 0.72 - (W * 0.10) * np.sin(np.pi * np.clip((x - W * 0.15) / (W * 0.35), 0, 1))
    # Left dome
    l_diaph = H * 0.74 - (W * 0.09) * np.sin(np.pi * np.clip((x - W * 0.50) / (W * 0.35), 0, 1))
    diaph_profile = np.minimum(r_diaph, l_diaph)
    diaph_mask = y >= diaph_profile
    img = np.where(diaph_mask, np.maximum(img, 0.48 + 0.25 * ((y - diaph_profile) / (H * 0.3))), img)

    # 5. Cardiac Silhouette (Asymmetric central-left opacity)
    cardiac_w = W * 0.15 if pathology != "cardiomegaly" else W * 0.24
    cardiac_h = H * 0.19 if pathology != "cardiomegaly" else H * 0.24
    cardiac_cx = cx + W * 0.05 if pathology != "cardiomegaly" else cx + W * 0.08
    cardiac_cy = cy + H * 0.14
    cardiac_dist = ((x - cardiac_cx) ** 2) / (cardiac_w**2) + ((y - cardiac_cy) ** 2) / (cardiac_h**2)
    cardiac_opacity = np.clip(1.0 - cardiac_dist, 0.0, 1.0) ** 0.8
    img = img + 0.45 * cardiac_opacity

    # 6. Posterior Rib Contours (Subtle anatomical arched bands)
    num_ribs = 7
    thorax_float = thorax_mask.astype(float)
    for i in range(num_ribs):
        rib_y = H * (0.22 + 0.075 * i)
        rib_curve = rib_y + 0.08 * (x - cx) ** 2 / W
        rib_band = np.exp(-((y - rib_curve) ** 2) / 6.0)
        img = img + 0.06 * rib_band * thorax_float

    # 7. Clavicles (Upper horizontal bony struts)
    clav_r = np.exp(-((y - (H * 0.22 - 0.03 * (x - cx))) ** 2) / 7.0) * (x < cx).astype(float)
    clav_l = np.exp(-((y - (H * 0.22 + 0.03 * (x - cx))) ** 2) / 7.0) * (x > cx).astype(float)
    img = img + 0.15 * (clav_r + clav_l) * thorax_float

    # 8. Inject Specific Pathological Features
    path_key = pathology.lower().strip()

    if path_key == "pneumothorax":
        # Right hemithorax peripheral visceral pleural line + hyperlucent void
        ptx_region = ((x < W * 0.28) & (y > H * 0.25) & (y < H * 0.70))
        img[ptx_region] *= 0.45  # Hyperlucency (absence of lung vascular markings)
        # Sharp pleural line
        ptx_vertical = ((y > H * 0.25) & (y < H * 0.70)).astype(float)
        pleural_line = np.exp(-((x - W * 0.28) ** 2) / 3.0) * ptx_vertical
        img = img + 0.35 * pleural_line

    elif path_key == "effusion":
        # Right base blunting of costophrenic angle and fluid meniscus
        meniscus_y = H * 0.76 - 0.003 * ((x - W * 0.15) ** 2)
        effusion_mask = (y > meniscus_y) & (x < cx)
        img = np.where(effusion_mask, np.maximum(img, 0.65), img)

    elif path_key in ("consolidation", "pneumonia"):
        # Dense irregular patchy alveolar opacity in right lower/middle lobe
        patch_cx, patch_cy = W * 0.28, H * 0.52
        patch_dist = np.sqrt((x - patch_cx) ** 2 + (y - patch_cy) ** 2)
        # Add spatial noise for fluffy cloud-like alveolar consolidation
        noise = ndi.gaussian_filter(np.random.normal(0, 1, (H, W)), sigma=3.0)
        consolidation_intensity = np.clip(1.0 - patch_dist / (W * 0.13) + 0.35 * noise, 0.0, 1.0)
        img = img + 0.42 * consolidation_intensity * right_lung_mask.astype(float)

    elif path_key == "edema":
        # Bilateral perihilar batwing haze and vascular engorgement
        hilar_dist = np.sqrt((x - cx) ** 2 + 1.8 * (y - cy) ** 2)
        edema_haze = np.clip(1.0 - hilar_dist / (W * 0.28), 0.0, 1.0)
        noise = ndi.gaussian_filter(np.random.normal(0, 1, (H, W)), sigma=2.0)
        lungs_float = (right_lung_mask | left_lung_mask).astype(float)
        img = img + 0.30 * edema_haze * (0.8 + 0.2 * noise) * lungs_float

    # 9. Realistic Radiographic Noise & Anatomic Smoothing
    noise = np.random.normal(0, 0.015, (H, W))
    img = ndi.gaussian_filter(img + noise, sigma=0.8)
    img = np.clip(img, 0.0, 1.0)

    # Convert to 8-bit unsigned integer grayscale
    img_uint8 = (img * 255.0).astype(np.uint8)
    return Image.fromarray(img_uint8, mode="L")


class ChestXRayPipeline:
    """
    End-to-End Chest X-ray Triage Vision Inference Pipeline.

    Integrates:
    - Preprocessor
    - Pretrained PyTorch Vision Network
    - Decoupled Radiographic Acuity Modifier
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
        self.preprocessor = ChestXRayPreprocessor(image_size=image_size)
        self.model = ChestXRayModel(
            architecture=model_architecture,
            pretrained=pretrained,
            num_findings=len(NIH_CHEST_XRAY_CLASSES),
        ).to(self.device)

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                logger.info(f"Loading vision model weights from {ckpt}")
                state_dict = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(state_dict)

        self.model.eval()
        self.acuity_modifier = RadiographicAcuityModifier()

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full radiographic inference on an input image.

        Args:
            image_input: Path, PIL Image, numpy array, or tensor.
            custom_thresholds: Optional custom detection thresholds.

        Returns:
            Dictionary containing finding probabilities, detected pathologies,
            Radiographic Severity Index, and emergency triage modifier recommendations.
        """
        tensor = self.preprocessor.preprocess(image_input).to(self.device)

        with torch.no_grad():
            probs_tensor = self.model.predict_proba(tensor)[0].cpu().numpy()

        finding_probs = {
            finding: float(probs_tensor[i])
            for i, finding in enumerate(NIH_CHEST_XRAY_CLASSES)
        }

        clinical_eval = self.acuity_modifier.evaluate_findings(
            finding_probabilities=finding_probs,
            custom_thresholds=custom_thresholds,
        )

        return {
            "finding_probabilities": finding_probs,
            **clinical_eval,
        }

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Run batch inference over a collection of radiographs."""
        return [self.predict(img) for img in image_inputs]
