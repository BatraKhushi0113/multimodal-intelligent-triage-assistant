"""
Unit Tests for AdaptiveReliabilityFusion, ModalityReliabilityScorer & AdaptiveCrossModalGate
==========================================================================================
Tests reliability bounds, missingness penalty, partition-of-unity gating weights,
variable batch sizes, single-sample evaluation, and gradient flow across all pathways.
"""

import sys
from pathlib import Path
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion import AdaptiveReliabilityFusion
from src.models.reliability import AdaptiveCrossModalGate, ModalityReliabilityScorer


def test_reliability_scorer_bounds():
    """Verify reliability prior scores stay within valid [0.1, 1.0] clinical bounds."""
    scorer = ModalityReliabilityScorer(vitals_dim=128, text_dim=384, context_dim=64, min_reliability=0.1)
    scorer.eval()

    text = torch.randn(8, 384)
    vitals = torch.randn(8, 128)
    context = torch.randn(8, 64)

    r = scorer(text, vitals, context)
    assert r.shape == (8, 3)
    assert (r >= 0.1).all(), "Reliability score violated minimum threshold"
    assert (r <= 1.0).all(), "Reliability score exceeded maximum threshold of 1.0"


def test_reliability_scorer_missingness_penalty():
    """Verify that unmeasured vitals reduce vitals reliability score."""
    scorer = ModalityReliabilityScorer(vitals_dim=128, text_dim=384, context_dim=64, min_reliability=0.1)
    scorer.eval()

    text = torch.randn(4, 384)
    vitals = torch.randn(4, 128)
    context = torch.randn(4, 64)

    # Missing indicators: 0 = present, 1 = missing
    present_flag = torch.zeros(4, 1)
    missing_flag = torch.ones(4, 1)

    r_present = scorer(text, vitals, context, missing_indicators=present_flag)
    r_missing = scorer(text, vitals, context, missing_indicators=missing_flag)

    # Vitals reliability (column 1) should be strictly lower when missing
    assert (r_missing[:, 1] < r_present[:, 1]).all(), (
        "Missingness flag did not penalize vitals reliability score."
    )


def test_cross_modal_gate_partition_of_unity():
    """Verify that cross-modal attention weights form a valid probability distribution summing to 1.0."""
    gate = AdaptiveCrossModalGate(shared_dim=128, num_modalities=3, temperature=1.0)
    gate.eval()

    batch_size = 12
    h_text = torch.randn(batch_size, 128)
    h_vitals = torch.randn(batch_size, 128)
    h_context = torch.randn(batch_size, 128)
    priors = torch.rand(batch_size, 3) * 0.9 + 0.1

    weights, logits = gate(h_text, h_vitals, h_context, reliability_priors=priors)

    assert weights.shape == (batch_size, 3)
    assert (weights >= 0.0).all(), "Gating weights must be non-negative"
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), (
        f"Gating weights must sum to 1.0 per patient. Got sums: {sums}"
    )


def test_adaptive_fusion_forward_shapes():
    """Verify output tensor shapes with standard batch sizes."""
    model = AdaptiveReliabilityFusion(
        text_dim=384,
        vitals_dim=128,
        context_dim=64,
        shared_dim=128,
        num_classes=5,
    )
    model.eval()

    for batch_size in [1, 4, 8, 16, 32]:
        text = torch.randn(batch_size, 384)
        vitals = torch.randn(batch_size, 128)
        context = torch.randn(batch_size, 64)

        logits = model(text, vitals, context)
        assert logits.shape == (batch_size, 5)
        assert not torch.isnan(logits).any()


def test_adaptive_fusion_weights_return():
    """Verify return_weights=True provides case-level modality attributions."""
    model = AdaptiveReliabilityFusion(text_dim=384, vitals_dim=128, context_dim=64, shared_dim=128)
    model.eval()

    text = torch.randn(5, 384)
    vitals = torch.randn(5, 128)
    context = torch.randn(5, 64)

    logits, weights = model(text, vitals, context, return_weights=True)
    assert logits.shape == (5, 5)
    assert weights.shape == (5, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_adaptive_fusion_gradient_flow():
    """Verify backpropagation flows through all modality projection, gating, and classification heads."""
    model = AdaptiveReliabilityFusion(text_dim=384, vitals_dim=128, context_dim=64, shared_dim=128)
    model.train()

    text = torch.randn(4, 384)
    vitals = torch.randn(4, 128)
    context = torch.randn(4, 64)
    targets = torch.tensor([0, 1, 2, 3])

    criterion = torch.nn.CrossEntropyLoss()
    logits = model(text, vitals, context)
    loss = criterion(logits, targets)
    loss.backward()

    # Verify gradients exist across text projection, vitals projection, gate, and classifier
    assert model.text_proj[0].weight.grad is not None
    assert model.vitals_proj[0].weight.grad is not None
    assert model.context_proj[0].weight.grad is not None
    assert model.cross_modal_gate.interaction_mlp[0].weight.grad is not None
    assert model.classifier[-1].weight.grad is not None

    assert torch.norm(model.classifier[-1].weight.grad) > 0.0
