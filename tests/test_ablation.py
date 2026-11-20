"""
Unit Tests for Formal Cross-Modality Ablation Study (Module 10)
==============================================================
Tests configuration integrity, bootstrap confidence interval estimation,
paired hypothesis testing, and table formatting.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ablation import (
    ABLATION_CONFIGS,
    compute_bootstrap_ci,
    compute_paired_bootstrap_pvalue,
    df_to_markdown_table,
)


def test_ablation_configs_structure():
    """Verify all 7 configurations (1 trimodal, 3 bimodal, 3 unimodal) are properly defined."""
    assert len(ABLATION_CONFIGS) == 7

    # Exactly 1 trimodal
    trimodal = [c for c in ABLATION_CONFIGS if c["num_modalities"] == 3]
    assert len(trimodal) == 1
    assert trimodal[0]["mask"] is None

    # Exactly 3 bimodal
    bimodal = [c for c in ABLATION_CONFIGS if c["num_modalities"] == 2]
    assert len(bimodal) == 3
    for b in bimodal:
        assert len(b["mask"]) == 1

    # Exactly 3 unimodal
    unimodal = [c for c in ABLATION_CONFIGS if c["num_modalities"] == 1]
    assert len(unimodal) == 3
    for u in unimodal:
        assert len(u["mask"]) == 2


def test_bootstrap_ci_computation():
    """Verify bootstrap confidence interval outputs and logical bounds."""
    y_true = np.array([0, 1, 2, 3, 4] * 20)
    y_pred = np.array([0, 1, 2, 3, 4] * 20)  # perfect predictions

    res = compute_bootstrap_ci(y_true, y_pred, n_bootstraps=50, seed=42)

    assert "macro_f1_ci" in res
    assert "accuracy_ci" in res
    assert "weighted_f1_ci" in res

    f1_low, f1_high = res["macro_f1_ci"]
    assert 0.0 <= f1_low <= f1_high <= 1.0

    acc_low, acc_high = res["accuracy_ci"]
    assert 0.0 <= acc_low <= acc_high <= 1.0


def test_paired_bootstrap_pvalue():
    """Verify paired bootstrap hypothesis test calculates a valid p-value."""
    y_true = np.array([0, 1, 2, 3, 4] * 20)
    y_pred_good = np.array([0, 1, 2, 3, 4] * 20)
    y_pred_bad = np.array([1, 2, 3, 4, 0] * 20)

    p_val = compute_paired_bootstrap_pvalue(y_true, y_pred_good, y_pred_bad, n_bootstraps=50, seed=42)
    assert 0.0 <= p_val <= 1.0


def test_markdown_table_formatter():
    """Verify pure-Python dataframe markdown conversion without external dependencies."""
    df = pd.DataFrame({
        "Model": ["A", "B"],
        "Score": [0.85, 0.92],
    })
    md = df_to_markdown_table(df)

    assert "| Model | Score |" in md
    assert "| :--- | :--- |" in md
    assert "| A | 0.85 |" in md
    assert "| B | 0.92 |" in md
