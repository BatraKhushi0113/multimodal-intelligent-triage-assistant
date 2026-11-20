"""
Unit Tests for Deep Learning ContextEncoder & ContextDataset
============================================================
Tests forward pass shapes, single-sample evaluation, boundary clipping,
dictionary vs tensor inputs, gradient flow, and dataset extraction.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.context_encoder import (
    CONTEXT_CAT_COLS,
    CONTEXT_NUM_COLS,
    ContextDataset,
    ContextEncoder,
)


def test_context_encoder_forward_shape():
    """Verify output tensor shapes with various batch sizes."""
    embedding_dim = 64
    encoder = ContextEncoder(embedding_dim=embedding_dim)
    encoder.eval()

    for batch_size in [1, 4, 8, 16, 32]:
        cat_x = torch.randint(0, 2, (batch_size, 3))
        num_x = torch.randn(batch_size, 2)
        out = encoder(cat_x, num_x)
        assert out.shape == (batch_size, embedding_dim), (
            f"Expected shape ({batch_size}, {embedding_dim}), got {out.shape}"
        )
        assert not torch.isnan(out).any()


def test_context_encoder_dict_input():
    """Verify encoder accepts dictionary input format seamlessly."""
    encoder = ContextEncoder(embedding_dim=64)
    encoder.eval()

    batch_input = {
        "cat_features": torch.tensor([[0, 2, 1], [1, 0, 0]], dtype=torch.long),
        "num_features": torch.tensor([[0.5, -1.2], [-0.3, 0.8]], dtype=torch.float32),
    }

    out = encoder(batch_input)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_context_encoder_boundary_clipping():
    """Ensure out-of-range category indices are clamped safely without IndexError."""
    encoder = ContextEncoder(embedding_dim=64)
    encoder.eval()

    # Pass categorical values that exceed normal ranges
    extreme_cat_x = torch.tensor([[999, 999, 999], [-5, -10, -1]], dtype=torch.long)
    num_x = torch.randn(2, 2)

    with torch.no_grad():
        out = encoder(extreme_cat_x, num_x)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_context_encoder_gradient_flow():
    """Verify gradients propagate back into embedding weights and MLP layers."""
    encoder = ContextEncoder(embedding_dim=64)
    encoder.train()

    cat_x = torch.tensor([[0, 1, 0], [1, 2, 1]], dtype=torch.long)
    num_x = torch.randn(2, 2)

    out = encoder(cat_x, num_x)
    loss = out.sum()
    loss.backward()

    # Check embeddings received gradients
    assert encoder.sex_embedding.weight.grad is not None
    assert encoder.arrival_embedding.weight.grad is not None
    assert encoder.injury_embedding.weight.grad is not None
    # Check linear projection received gradients
    assert encoder.fc1.weight.grad is not None
    assert encoder.fc2.weight.grad is not None


def test_context_dataset_extraction():
    """Verify ContextDataset properly parses tabular columns into tensor representations."""
    synthetic_data = {
        "Sex_cat": [0, 1, 0],
        "Arrival_mode_cat": [1, 3, 0],
        "Injury_cat": [0, 1, 0],
        "Age_scaled": [0.25, -0.75, 1.10],
        "Patients_per_hour_scaled": [-0.1, 0.4, -0.9],
        "KTAS_RN_encoded": [1, 2, 4],
    }
    df = pd.DataFrame(synthetic_data)
    ds = ContextDataset(df)

    assert len(ds) == 3
    sample = ds[0]

    assert "cat_features" in sample
    assert "num_features" in sample
    assert "target" in sample
    assert sample["cat_features"].shape == (3,)
    assert sample["num_features"].shape == (2,)
    assert sample["cat_features"].dtype == torch.long
    assert sample["num_features"].dtype == torch.float32
    assert sample["target"].item() == 1


if __name__ == "__main__":
    pytest.main([__file__])
