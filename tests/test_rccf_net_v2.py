"""
Unit Tests for RCCF-Net v2 Architecture.
========================================
Tests forward tensor dimensions, cross-attention mechanism, missing-modality
masking, CECE evidence conditioning, and gradient flow.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rccf_net_v2 import RCCFNetDataset, RCCFNetV2, StochasticModalityDropout


@pytest.fixture
def rccf_model() -> RCCFNetV2:
    return RCCFNetV2(
        text_dim=384,
        vitals_dim=128,
        context_dim=64,
        cece_dim=8,
        shared_dim=128,
        num_classes=5,
        num_heads=4,
        dropout_prob=0.1,
    )


def test_rccf_forward_shapes(rccf_model: RCCFNetV2):
    """Verify forward pass returns logits of shape (B, 5) and valid dispersion."""
    B = 8
    t = torch.randn(B, 384)
    v = torch.randn(B, 128)
    c = torch.randn(B, 64)
    e = torch.randn(B, 8)

    logits, aux = rccf_model(t, v, c, cece_evidence=e, return_weights=True)

    assert logits.shape == (B, 5)
    assert not torch.isnan(logits).any()
    assert "modality_weights" in aux
    assert aux["modality_weights"].shape == (B, 3)
    # Weights must partition to unity (sum = 1.0)
    w_sum = aux["modality_weights"].sum(dim=-1)
    assert torch.allclose(w_sum, torch.ones(B), atol=1e-4)

    assert "epistemic_dispersion" in aux
    assert aux["epistemic_dispersion"].shape == (B, 1)
    assert (aux["epistemic_dispersion"] >= 0.0).all()


def test_rccf_missing_modality_masking(rccf_model: RCCFNetV2):
    """Verify evaluation zero-masking handles missing modalities without NaNs."""
    B = 4
    t = torch.randn(B, 384)
    v = torch.randn(B, 128)
    c = torch.randn(B, 64)

    rccf_model.eval()

    # Mask text
    logits_no_text, aux_t = rccf_model(t, v, c, mask_modalities=["text"], return_weights=True)
    assert logits_no_text.shape == (B, 5)
    assert not torch.isnan(logits_no_text).any()
    assert (aux_t["modality_weights"][:, 0] < 0.05).all()  # Text weight penalized

    # Mask vitals
    logits_no_vitals, aux_v = rccf_model(t, v, c, mask_modalities=["vitals"], return_weights=True)
    assert logits_no_vitals.shape == (B, 5)
    assert not torch.isnan(logits_no_vitals).any()


def test_rccf_gradient_flow(rccf_model: RCCFNetV2):
    """Verify gradients propagate cleanly to all unimodal and attention heads."""
    rccf_model.train()
    B = 4
    t = torch.randn(B, 384, requires_grad=True)
    v = torch.randn(B, 128, requires_grad=True)
    c = torch.randn(B, 64, requires_grad=True)
    e = torch.randn(B, 8)
    targets = torch.tensor([0, 2, 4, 1], dtype=torch.long)

    logits = rccf_model(t, v, c, cece_evidence=e)
    loss = torch.nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    # Verify model parameter gradients
    assert rccf_model.text_proj[0].weight.grad is not None
    assert rccf_model.vitals_proj[0].weight.grad is not None
    assert rccf_model.classifier_head[-1].weight.grad is not None
    assert not torch.isnan(rccf_model.classifier_head[-1].weight.grad).any()


def test_stochastic_modality_dropout():
    """Verify modality dropout behaves stochastically in train and preserves in eval."""
    dropout = StochasticModalityDropout(p=0.50)
    B = 20
    t = torch.ones(B, 10)
    v = torch.ones(B, 10)
    c = torch.ones(B, 10)

    # In eval mode, nothing is dropped
    dropout.eval()
    t_eval, v_eval, c_eval, mask_eval = dropout(t, v, c)
    assert torch.equal(t_eval, t)
    assert mask_eval.all()

    # In train mode, drops occur
    dropout.train()
    t_train, v_train, c_train, mask_train = dropout(t, v, c)
    # At least some modality should be dropped across 20 samples with p=0.5
    assert not mask_train.all()

