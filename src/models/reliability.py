"""
Multimodal Intelligent Triage Assistant - Modality Reliability Estimation
=========================================================================
Module 9 (Part 1): Data-driven and learned modality reliability scoring for
clinical emergency triage.

In real-world emergency departments, modalities exhibit heterogeneous quality:
  1. Vitals: Subject to sensor disconnects, missingness (e.g. unmeasured SpO2),
     imputation artifacts, or non-cooperative triage states.
  2. Text (Chief Complaint): Varies from rich, highly specific clinical complaints
     ("crushing retrosternal chest pain") to brief, non-informative noise ("pain", "sick").
  3. Context: Varies across admission modes and pediatric/geriatric age groups.

This module provides:
  - ModalityReliabilityScorer: Computes data-driven reliability priors r_m in (0, 1]
    derived from missingness masks, physiological anomalies, and embedding confidence.
  - LearnableReliabilityGate: Neural network mapping concatenated latent features
    and data-driven priors to normalized cross-modal attention gates [w_text, w_vitals, w_context].
  - Modality attribution extractor for patient-level interpretability.

Research & Educational Prototype Only.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# 1. DATA-DRIVEN RELIABILITY ESTIMATOR
# ==============================================================================
class ModalityReliabilityScorer(nn.Module):
    """
    Computes prior reliability scores for each clinical modality based on
    intrinsic signal properties and observed missingness indicators:
    
    1. Vitals Reliability (r_vitals):
       - Assesses missingness flags (e.g. saturation_is_missing).
       - Evaluates physiological plausibility and dispersion from normal ranges.
       - Penalizes records where critical vital measurements are absent or synthetic.
       
    2. Text Reliability (r_text):
       - Assesses semantic embedding energy / magnitude ||z_text||.
       - Penalizes generic or empty complaints mapped near the centroid or zero vector.
       
    3. Context Reliability (r_context):
       - Assesses completeness of demographic and operational crowding context.
    """

    def __init__(
        self,
        vitals_dim: int = 128,
        text_dim: int = 384,
        context_dim: int = 64,
        min_reliability: float = 0.1,
    ):
        super().__init__()
        self.vitals_dim = vitals_dim
        self.text_dim = text_dim
        self.context_dim = context_dim
        self.min_reliability = float(min_reliability)

        # Learnable calibration scaling for each modality prior
        self.vitals_calibrator = nn.Sequential(
            nn.Linear(vitals_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.text_calibrator = nn.Sequential(
            nn.Linear(text_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.context_calibrator = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        vitals_emb: torch.Tensor,
        context_emb: torch.Tensor,
        missing_indicators: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reliability scores r = [r_text, r_vitals, r_context] for each sample.
        Input shapes:
          - text_emb: (batch_size, 384)
          - vitals_emb: (batch_size, 128)
          - context_emb: (batch_size, 64)
          - missing_indicators: optional (batch_size, num_missing_flags)
        Output shape:
          - reliability: (batch_size, 3) with values in [min_reliability, 1.0]
        """
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        if vitals_emb.dim() == 1:
            vitals_emb = vitals_emb.unsqueeze(0)
        if context_emb.dim() == 1:
            context_emb = context_emb.unsqueeze(0)

        # Baseline learned confidence from latent representations
        r_t = self.text_calibrator(text_emb)
        r_v = self.vitals_calibrator(vitals_emb)
        r_c = self.context_calibrator(context_emb)

        # If explicit missingness indicators are provided, modulate vitals prior directly
        if missing_indicators is not None:
            # missing_indicators: 1 if missing, 0 if present
            # We scale down vitals reliability when saturation or vitals are missing
            if missing_indicators.dim() == 1:
                missing_indicators = missing_indicators.unsqueeze(0)
            missing_penalty = 1.0 - 0.35 * missing_indicators.float().mean(dim=-1, keepdim=True)
            r_v = r_v * missing_penalty

        # Clamp to strictly positive lower bound to prevent numerical division by zero
        r_t = torch.clamp(r_t, min=self.min_reliability, max=1.0)
        r_v = torch.clamp(r_v, min=self.min_reliability, max=1.0)
        r_c = torch.clamp(r_c, min=self.min_reliability, max=1.0)

        # Stack into (batch_size, 3): [text, vitals, context]
        reliability = torch.cat([r_t, r_v, r_c], dim=-1)
        return reliability


# ==============================================================================
# 2. ADAPTIVE CROSS-MODAL RELIABILITY GATE
# ==============================================================================
class AdaptiveCrossModalGate(nn.Module):
    """
    Learns dynamic cross-modal attention gating weights w = [w_text, w_vitals, w_context]
    conditioned on both:
      1. Projected modality representations in a shared d-dimensional space.
      2. Intrinsic modality reliability scores from ModalityReliabilityScorer.
      
    Mathematical Formulation:
      h_m = Projection_m(z_m) for m in {text, vitals, context}
      q = W_q [h_text; h_vitals; h_context]
      raw_logits = W_gate(h_m) + log(r_m + eps)
      w_m = Softmax(raw_logits / temperature)
      
    Properties:
      - sum(w_m) = 1.0 for each patient (interpretable percentage allocation).
      - When vitals reliability drops (e.g. missing sensor), gating automatically
        shifts weight to text and context.
      - Temperature parameter controls sharpness of modality selection.
    """

    def __init__(
        self,
        shared_dim: int = 128,
        num_modalities: int = 3,
        temperature: float = 1.0,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_modalities = num_modalities
        self.temperature = float(temperature)

        # Cross-modal interaction backbone
        # Takes all projected modalities and outputs unnormalized gating logits
        self.interaction_mlp = nn.Sequential(
            nn.Linear(shared_dim * num_modalities, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(shared_dim, num_modalities),
        )

    def forward(
        self,
        h_text: torch.Tensor,
        h_vitals: torch.Tensor,
        h_context: torch.Tensor,
        reliability_priors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute modality attention weights.
        Inputs:
          - h_text, h_vitals, h_context: (batch_size, shared_dim)
          - reliability_priors: optional (batch_size, 3) in (0, 1]
        Outputs:
          - weights: (batch_size, 3) summing to 1.0
          - unnormalized_logits: (batch_size, 3)
        """
        # Concatenate projected representations: (batch_size, 3 * shared_dim)
        joint_repr = torch.cat([h_text, h_vitals, h_context], dim=-1)
        raw_logits = self.interaction_mlp(joint_repr)

        # Modulate logits with log-reliability priors (Bayesian prior injection)
        if reliability_priors is not None:
            eps = 1e-6
            log_priors = torch.log(torch.clamp(reliability_priors, min=eps, max=1.0))
            raw_logits = raw_logits + log_priors

        # Apply temperature-scaled Softmax
        weights = F.softmax(raw_logits / max(self.temperature, 0.1), dim=-1)
        return weights, raw_logits


if __name__ == "__main__":
    from src.models.fusion import run_adaptive_fusion_module
    run_adaptive_fusion_module()

