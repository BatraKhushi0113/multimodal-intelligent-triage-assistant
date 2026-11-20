"""
Unit Tests for Deep Learning VitalsEncoder & VitalsDataset
=========================================================
Tests forward pass, variable batch sizes, single-sample evaluation,
and configurable embedding dimensions.
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

from src.models.vitals_encoder import VitalsDataset, VitalsEncoder, VITALS_FEATURE_NAMES


def test_vitals_encoder_forward_shape():
    """Verify output tensor shapes with standard batch sizes."""
    input_dim = 10
    embedding_dim = 128
    encoder = VitalsEncoder(input_dim=input_dim, hidden_dim=64, embedding_dim=embedding_dim)
    encoder.eval()

    for batch_size in [1, 8, 16, 32, 64]:
        x = torch.randn(batch_size, input_dim)
        out = encoder(x)
        assert out.shape == (batch_size, embedding_dim), (
            f"Expected shape ({batch_size}, {embedding_dim}), got {out.shape}"
        )


def test_vitals_encoder_configurable_dimensions():
    """Verify that input, hidden, and embedding dimensions are fully configurable."""
    for in_d in [8, 10, 15]:
        for emb_d in [32, 64, 128, 256]:
            encoder = VitalsEncoder(input_dim=in_d, hidden_dim=48, embedding_dim=emb_d)
            encoder.eval()
            x = torch.randn(4, in_d)
            out = encoder(x)
            assert out.shape == (4, emb_d)


def test_vitals_encoder_single_sample_eval_mode():
    """Ensure BatchNorm does not crash on a single sample during inference."""
    encoder = VitalsEncoder(input_dim=10, hidden_dim=64, embedding_dim=128, use_batch_norm=True)
    encoder.eval()

    single_sample = torch.randn(1, 10)
    with torch.no_grad():
        out = encoder(single_sample)
    assert out.shape == (1, 128)
    assert not torch.isnan(out).any()


def test_vitals_dataset_loading():
    """Verify VitalsDataset extraction from a synthetic dataframe."""
    synthetic_data = {col: [0.5, -0.2, 1.0] for col in VITALS_FEATURE_NAMES}
    synthetic_data["KTAS_RN_encoded"] = [0, 2, 4]
    df = pd.DataFrame(synthetic_data)

    dataset = VitalsDataset(df)
    assert len(dataset) == 3

    item = dataset[0]
    assert "vitals" in item
    assert "target" in item
    assert item["vitals"].shape == (10,)
    assert item["target"].item() == 0


if __name__ == "__main__":
    pytest.main([__file__])
