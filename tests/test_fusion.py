"""
Unit Tests for BaselineConcatenationFusion, ModalityDropout & MultimodalFusionDataset
===================================================================================
Tests forward pass shapes, variable batch sizes, single-sample evaluation,
dict/tuple/concatenated input formats, modality masking, gradient flow, and dataset indexing.
"""

import sys
from pathlib import Path
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion import (
    BaselineConcatenationFusion,
    ModalityDropout,
    MultimodalFusionDataset,
)


def test_multimodal_dataset_indexing():
    """Verify dataset indexing, item types, and concatenated tensor shapes."""
    n_samples = 20
    text = torch.randn(n_samples, 384)
    vitals = torch.randn(n_samples, 128)
    context = torch.randn(n_samples, 64)
    targets = torch.randint(0, 5, (n_samples,))

    dataset = MultimodalFusionDataset(text, vitals, context, targets)
    assert len(dataset) == n_samples

    item = dataset[0]
    assert "text" in item and item["text"].shape == (384,)
    assert "vitals" in item and item["vitals"].shape == (128,)
    assert "context" in item and item["context"].shape == (64,)
    assert "target" in item and item["target"].dtype == torch.long

    concat = dataset.get_concatenated_tensor()
    assert concat.shape == (n_samples, 576)


def test_modality_dropout_behavior():
    """Verify modality dropout in training and evaluation modes."""
    drop = ModalityDropout(p=0.5)
    text = torch.ones(10, 10)
    vitals = torch.ones(10, 10)
    context = torch.ones(10, 10)

    # In eval mode, should be identical
    drop.eval()
    t_out, v_out, c_out = drop(text, vitals, context)
    assert torch.equal(t_out, text)
    assert torch.equal(v_out, vitals)
    assert torch.equal(c_out, context)

    # In train mode, should drop some but never all simultaneously
    drop.train()
    for _ in range(5):
        t_out, v_out, c_out = drop(text, vitals, context)
        # Check that for every sample, at least one modality has non-zero sum
        active_sum = (t_out.sum(dim=-1) > 0) | (v_out.sum(dim=-1) > 0) | (c_out.sum(dim=-1) > 0)
        assert active_sum.all(), "ModalityDropout dropped all modalities simultaneously for a sample!"


def test_fusion_forward_shapes():
    """Verify output tensor shapes with standard batch sizes."""
    model = BaselineConcatenationFusion(
        text_dim=384,
        vitals_dim=128,
        context_dim=64,
        hidden_dims=[128, 64],
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


def test_fusion_input_formats():
    """Verify model accepts tuple, dict, and pre-concatenated inputs."""
    model = BaselineConcatenationFusion(text_dim=384, vitals_dim=128, context_dim=64)
    model.eval()

    batch_size = 4
    text = torch.randn(batch_size, 384)
    vitals = torch.randn(batch_size, 128)
    context = torch.randn(batch_size, 64)

    # 1. Positional args
    out_pos = model(text, vitals, context)

    # 2. Dictionary input
    batch_dict = {"text": text, "vitals": vitals, "context": context}
    out_dict = model(batch_dict)
    assert torch.allclose(out_pos, out_dict, atol=1e-5)

    # 3. Concatenated input
    concat = torch.cat([text, vitals, context], dim=-1)
    out_concat = model(concatenated=concat)
    assert torch.allclose(out_pos, out_concat, atol=1e-5)


def test_fusion_modality_masking():
    """Verify that zero-masking individual modalities alters logits predictably."""
    model = BaselineConcatenationFusion(text_dim=384, vitals_dim=128, context_dim=64)
    model.eval()

    text = torch.randn(2, 384)
    vitals = torch.randn(2, 128)
    context = torch.randn(2, 64)

    unmasked_logits = model(text, vitals, context)
    masked_text_logits = model(text, vitals, context, mask_modalities=["text"])
    masked_vitals_logits = model(text, vitals, context, mask_modalities=["vitals"])

    # Logits should differ when a modality is zero-masked
    assert not torch.allclose(unmasked_logits, masked_text_logits)
    assert not torch.allclose(unmasked_logits, masked_vitals_logits)


def test_fusion_return_features():
    """Verify return_features returns both logits and latent features."""
    model = BaselineConcatenationFusion(text_dim=384, vitals_dim=128, context_dim=64, hidden_dims=[128, 64])
    model.eval()

    text = torch.randn(3, 384)
    vitals = torch.randn(3, 128)
    context = torch.randn(3, 64)

    logits, features = model(text, vitals, context, return_features=True)
    assert logits.shape == (3, 5)
    assert features.shape == (3, 64)


def test_fusion_gradient_flow():
    """Verify backpropagation updates model parameters."""
    model = BaselineConcatenationFusion(text_dim=384, vitals_dim=128, context_dim=64)
    model.train()

    text = torch.randn(4, 384)
    vitals = torch.randn(4, 128)
    context = torch.randn(4, 64)
    targets = torch.tensor([0, 1, 2, 3])

    criterion = torch.nn.CrossEntropyLoss()
    logits = model(text, vitals, context)
    loss = criterion(logits, targets)
    loss.backward()

    # Verify head and backbone have gradients
    assert model.head.weight.grad is not None
    assert torch.norm(model.head.weight.grad) > 0.0
