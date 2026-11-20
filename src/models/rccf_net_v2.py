"""
RCCF-Net v2: Residual Cross-Attention Clinical Fusion Network v2.
================================================================
Module: src/models/rccf_net_v2.py

Advanced Multimodal Fusion Architecture featuring:
  1. Evidence-Conditioned Reliability (Conditioned on CECE 8-d evidence vectors).
  2. Multi-Head Cross-Modal Bidirectional Attention across Text, Vitals, and Context tokens.
  3. Missing-Modality Awareness with learned missing-modality tokens and attention key masks.
  4. Residual Fusion connecting attended multimodal features with unimodal skip projections.
  5. Stochastic Modality Dropout during training.
  6. Dual-head output: 5-class KTAS logits + learned epistemic dispersion parameter.
  7. Stratified training loop, early stopping, and comparative benchmark interface.

Research & Educational Prototype Only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. RCCF-NET V2 DATASET WITH CECE EVIDENCE AUGMENTATION
# ==============================================================================
class RCCFNetDataset(Dataset):
    """
    PyTorch Dataset for RCCF-Net v2 supporting tri-modal tensors and optional
    CECE evidence conditioning vectors.
    """

    def __init__(
        self,
        text_embeddings: torch.Tensor,
        vitals_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
        cece_evidence: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ):
        assert len(text_embeddings) == len(vitals_embeddings) == len(context_embeddings)
        self.text = text_embeddings.float()
        self.vitals = vitals_embeddings.float()
        self.context = context_embeddings.float()

        if cece_evidence is not None:
            assert len(cece_evidence) == len(self.text)
            self.cece_evidence = cece_evidence.float()
        else:
            # Default zero/neutral evidence vector (8-dim)
            self.cece_evidence = torch.zeros((len(self.text), 8), dtype=torch.float32)

        if targets is not None:
            assert len(targets) == len(self.text)
            self.targets = targets.long()
        else:
            self.targets = None

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "text": self.text[idx],
            "vitals": self.vitals[idx],
            "context": self.context[idx],
            "cece_evidence": self.cece_evidence[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


# ==============================================================================
# 2. MODALITY DROPOUT MODULE
# ==============================================================================
class StochasticModalityDropout(nn.Module):
    """
    Stochastically zeros entire modality representations during training to enforce
    robust multimodal representations under sensor drops or absent text.
    """

    def __init__(self, p: float = 0.15):
        super().__init__()
        self.p = p

    def forward(
        self,
        text: torch.Tensor,
        vitals: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns dropped representations and a binary mask tensor of shape (B, 3).
        """
        if not self.training or self.p <= 0.0:
            B = text.size(0)
            presence_mask = torch.ones((B, 3), dtype=torch.bool, device=text.device)
            return text, vitals, context, presence_mask

        B = text.size(0)
        # Random drop masks (1 = keep, 0 = drop)
        mask_t = (torch.rand(B, 1, device=text.device) > self.p).float()
        mask_v = (torch.rand(B, 1, device=vitals.device) > self.p).float()
        mask_c = (torch.rand(B, 1, device=context.device) > self.p).float()

        # Prevent all 3 modalities from being dropped simultaneously
        all_dropped = (mask_t + mask_v + mask_c) == 0.0
        if all_dropped.any():
            # Guarantee at least vitals is kept if all 3 randomly dropped
            mask_v = torch.where(all_dropped, torch.ones_like(mask_v), mask_v)

        dropped_text = text * mask_t
        dropped_vitals = vitals * mask_v
        dropped_context = context * mask_c

        presence_mask = torch.cat([mask_t, mask_v, mask_c], dim=-1).bool()
        return dropped_text, dropped_vitals, dropped_context, presence_mask


# ==============================================================================
# 3. RCCF-NET V2 MODEL ARCHITECTURE
# ==============================================================================
class RCCFNetV2(nn.Module):
    """
    Residual Cross-Attention Clinical Fusion Network v2.
    Unites:
      - Modality projection into common shared embedding space (d_shared = 128).
      - Learned modality type embeddings (Text, Vitals, Context).
      - Multi-Head Bidirectional Cross-Modal Self-Attention (4 heads).
      - CECE Evidence-Conditioned Gating Head.
      - Residual Unimodal Skip Projection.
      - Dual-Head Acuity Logits (5 classes) + Epistemic Dispersion Parameter.
    """

    def __init__(
        self,
        text_dim: int = 384,
        vitals_dim: int = 128,
        context_dim: int = 64,
        cece_dim: int = 8,
        shared_dim: int = 128,
        num_classes: int = 5,
        num_heads: int = 4,
        dropout_prob: float = 0.25,
        modality_dropout_prob: float = 0.15,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vitals_dim = vitals_dim
        self.context_dim = context_dim
        self.cece_dim = cece_dim
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.temperature = temperature

        # 1. Unimodal Projections into Shared Latent Space
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.vitals_proj = nn.Sequential(
            nn.Linear(vitals_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )

        # 2. Modality Positional / Type Embeddings
        # 3 modalities: 0=Text, 1=Vitals, 2=Context
        self.modality_type_embeddings = nn.Parameter(torch.randn(3, shared_dim) * 0.02)
        # Learned missing-modality token for masked sensor handling
        self.missing_modality_token = nn.Parameter(torch.zeros(1, shared_dim))

        # 3. Multi-Head Cross-Modal Attention Transformer Block
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(shared_dim)

        # Feed-Forward Network following attention
        self.ffn = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Dropout(dropout_prob),
        )
        self.ffn_norm = nn.LayerNorm(shared_dim)

        # 4. CECE Evidence-Conditioned Gating Head
        self.cece_encoder = nn.Sequential(
            nn.Linear(cece_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        # Latent Gating Scorer
        self.latent_gate_scorer = nn.Sequential(
            nn.Linear(shared_dim * 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        # 5. Residual Unimodal Skip Projection
        self.skip_proj = nn.Sequential(
            nn.Linear(shared_dim * 3, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.fused_norm = nn.LayerNorm(shared_dim)

        # 6. Regularization
        self.modality_dropout = StochasticModalityDropout(p=modality_dropout_prob)

        # 7. Dual Heads: KTAS Classification + Dispersion
        self.classifier_head = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, num_classes),
        )
        # Epistemic dispersion head (variance scalar >= 0)
        self.dispersion_head = nn.Sequential(
            nn.Linear(shared_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        text: torch.Tensor,
        vitals: torch.Tensor,
        context: torch.Tensor,
        cece_evidence: Optional[torch.Tensor] = None,
        mask_modalities: Optional[List[str]] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass for RCCF-Net v2.
        """
        B = text.size(0)

        # Handle explicit evaluation masking
        presence_mask = torch.ones((B, 3), dtype=torch.bool, device=text.device)
        if mask_modalities:
            if "text" in mask_modalities:
                text = torch.zeros_like(text)
                presence_mask[:, 0] = False
            if "vitals" in mask_modalities:
                vitals = torch.zeros_like(vitals)
                presence_mask[:, 1] = False
            if "context" in mask_modalities:
                context = torch.zeros_like(context)
                presence_mask[:, 2] = False
        elif self.training:
            text, vitals, context, presence_mask = self.modality_dropout(text, vitals, context)

        # 1. Project into shared latent space
        h_t = self.text_proj(text)        # (B, shared_dim)
        h_v = self.vitals_proj(vitals)    # (B, shared_dim)
        h_c = self.context_proj(context)  # (B, shared_dim)

        # Replace dropped/missing modalities with learned token
        missing_expanded = self.missing_modality_token.expand(B, -1)
        h_t = torch.where(presence_mask[:, 0:1], h_t, missing_expanded)
        h_v = torch.where(presence_mask[:, 1:2], h_v, missing_expanded)
        h_c = torch.where(presence_mask[:, 2:3], h_c, missing_expanded)

        # 2. Add modality type embeddings
        token_t = (h_t + self.modality_type_embeddings[0]).unsqueeze(1)  # (B, 1, d)
        token_v = (h_v + self.modality_type_embeddings[1]).unsqueeze(1)  # (B, 1, d)
        token_c = (h_c + self.modality_type_embeddings[2]).unsqueeze(1)  # (B, 1, d)

        tokens = torch.cat([token_t, token_v, token_c], dim=1)  # (B, 3, d)

        # Key padding mask: True indicates masked (missing)
        key_padding_mask = ~presence_mask  # (B, 3)

        # If an entire batch row is somehow masked, unmask all to prevent NaN in attention
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)
        key_padding_mask = torch.where(all_masked, torch.zeros_like(key_padding_mask), key_padding_mask)

        # 3. Bidirectional Multi-Head Cross-Modal Attention
        attn_out, attn_weights = self.cross_attention(
            query=tokens,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
        )
        tokens_attn = self.attn_norm(tokens + attn_out)

        # FFN block
        tokens_ffn = self.ffn_norm(tokens_attn + self.ffn(tokens_attn))  # (B, 3, d)

        # 4. CECE Evidence-Conditioned Attention Gating
        concat_raw = torch.cat([h_t, h_v, h_c], dim=-1)  # (B, 3d)
        latent_gate_scores = self.latent_gate_scorer(concat_raw)  # (B, 3)

        if cece_evidence is not None:
            cece_prior = self.cece_encoder(cece_evidence.to(text.device))  # (B, 3)
            combined_scores = latent_gate_scores + 0.60 * cece_prior
        else:
            combined_scores = latent_gate_scores

        # Temperature-scaled Softmax over modality gates
        # Apply -1e9 mask to dropped modalities
        masked_scores = torch.where(presence_mask, combined_scores, torch.full_like(combined_scores, -1e4))
        modality_weights = F.softmax(masked_scores / max(self.temperature, 1e-4), dim=-1)  # (B, 3)

        # Weighted combination of attended tokens
        w_expanded = modality_weights.unsqueeze(-1)  # (B, 3, 1)
        r_attended = torch.sum(tokens_ffn * w_expanded, dim=1)  # (B, d)

        # 5. Residual Skip Connection from Unimodal Inputs
        r_skip = self.skip_proj(concat_raw)  # (B, d)

        # Fused unified patient representation
        z_fused = self.fused_norm(r_attended + r_skip)  # (B, d)

        # 6. Classification & Dispersion Outputs
        logits = self.classifier_head(z_fused)  # (B, 5)
        dispersion = self.dispersion_head(z_fused)  # (B, 1)

        if return_weights:
            aux_info = {
                "modality_weights": modality_weights,  # (B, 3)
                "epistemic_dispersion": dispersion,    # (B, 1)
                "fused_embedding": z_fused,            # (B, d)
                "cross_attention_weights": attn_weights, # (B, 3, 3)
            }
            return logits, aux_info

        return logits


# ==============================================================================
# 4. TRAINING & VALIDATION PIPELINE FOR RCCF-NET V2
# ==============================================================================
def train_rccf_net_v2(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: RCCFNetV2,
    class_weights: torch.Tensor,
    learning_rate: float = 0.0005,
    weight_decay: float = 0.001,
    max_epochs: int = 50,
    patience: int = 10,
    device: str = "cpu",
    save_checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train RCCF-Net v2 with class-weighted cross-entropy and early stopping on Macro F1.
    Strictly preserves baseline checkpoints.
    """
    model.to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_val_macro_f1 = -1.0
    best_state_dict: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    logger.info(f"Starting RCCF-Net v2 Training | Device: {device} | Max Epochs: {max_epochs}")

    for epoch in range(1, max_epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            t = batch["text"].to(device)
            v = batch["vitals"].to(device)
            c = batch["context"].to(device)
            e = batch["cece_evidence"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(text=t, vitals=v, context=c, cece_evidence=e)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            running_loss += loss.item() * len(y)
            n_samples += len(y)

        train_loss = running_loss / max(n_samples, 1)

        # Validation
        model.eval()
        val_preds: List[int] = []
        val_targets: List[int] = []
        running_val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                t = batch["text"].to(device)
                v = batch["vitals"].to(device)
                c = batch["context"].to(device)
                e = batch["cece_evidence"].to(device)
                y = batch["target"].to(device)

                logits = model(text=t, vitals=v, context=c, cece_evidence=e)
                loss = criterion(logits, y)
                running_val_loss += loss.item() * len(y)
                val_samples += len(y)

                preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy().tolist())

        val_loss = running_val_loss / max(val_samples, 1)

        # Scikit-learn metrics
        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        scheduler.step(val_macro_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        if val_macro_f1 > best_val_macro_f1 + 1e-4:
            best_val_macro_f1 = val_macro_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Macro F1: {best_val_macro_f1:.4f}")
            break

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save to dedicated checkpoint path if requested
    if save_checkpoint_path is not None and best_state_dict is not None:
        save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "best_val_macro_f1": best_val_macro_f1,
                "architecture": "RCCFNetV2",
                "hyperparameters": {
                    "text_dim": model.text_dim,
                    "vitals_dim": model.vitals_dim,
                    "context_dim": model.context_dim,
                    "cece_dim": model.cece_dim,
                    "shared_dim": model.shared_dim,
                    "num_classes": model.num_classes,
                },
            },
            save_checkpoint_path,
        )
        logger.info(f"Saved RCCF-Net v2 checkpoint to: {save_checkpoint_path}")

    return {
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
        "stopped_epoch": epoch,
    }

