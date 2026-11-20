"""
Multimodal Intelligent Triage Assistant - Deep Learning Vitals Encoder
======================================================================
Module 6: Static physiological vitals encoding via a regularized PyTorch MLP
with Batch Normalization, GELU activations, and Dropout.

Input Features (10-d):
  - Continuous Vitals: SBP, DBP, HR, RR, BT, Saturation, NRS_pain
  - Missingness Indicator: Saturation_is_missing
  - Categorical/Discrete: Mental_raw (AVPU), Pain_flag

Research & Educational Prototype Only.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VITALS_FEATURE_NAMES = [
    "SBP_scaled",
    "DBP_scaled",
    "HR_scaled",
    "RR_scaled",
    "BT_scaled",
    "Saturation_scaled",
    "Saturation_is_missing",
    "NRS_pain_scaled",
    "Mental_raw",
    "Pain_flag",
]


# ==============================================================================
# 1. PYTORCH DATASET
# ==============================================================================
class VitalsDataset(Dataset):
    """
    PyTorch Dataset for physiological vitals and admission state indicators.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "KTAS_RN_encoded",
    ):
        self.feature_cols = feature_cols or VITALS_FEATURE_NAMES
        self.X = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)

        if target_col in df.columns:
            self.y = torch.tensor(df[target_col].values, dtype=torch.long)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"vitals": self.X[idx]}
        if self.y is not None:
            item["target"] = self.y[idx]
        return item


# ==============================================================================
# 2. VITALS ENCODER MLP ARCHITECTURE
# ==============================================================================
class VitalsEncoder(nn.Module):
    """
    Deep Learning MLP Encoder for static clinical vital signs.
    Transforms raw/scaled physiological measurements into a compact,
    regularized latent representation.

    Architecture:
      Input (d_in=10)
            ↓
      Linear Layer (d_in → d_hidden, e.g. 64)
            ↓
      Batch Normalization / Layer Normalization
            ↓
      GELU Activation
            ↓
      Dropout (p=0.2)
            ↓
      Linear Layer (d_hidden → d_embed, e.g. 128)
            ↓
      Layer Normalization
            ↓
      GELU Activation
            ↓
      Dropout (p=0.2)
            ↓
      Vitals Embedding (d_embed=128)
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        embedding_dim: int = 128,
        dropout_prob: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # First stage: Feature Expansion & Interaction
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Use LayerNorm as default or BatchNorm1d
        self.norm1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_prob)

        # Second stage: Embedding Projection
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vitals tensor.
        Input:  (batch_size, input_dim)
        Output: (batch_size, embedding_dim)
        """
        # If batch size is 1 during evaluation and using BatchNorm, LayerNorm avoids crash
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = self.fc1(x)
        # Safe batchnorm: only apply if batch size > 1 or in eval mode
        if isinstance(self.norm1, nn.BatchNorm1d) and x.size(0) == 1 and self.training:
            h = self.act1(h)
        else:
            h = self.act1(self.norm1(h))
        h = self.drop1(h)

        out = self.fc2(h)
        out = self.drop2(self.act2(self.norm2(out)))
        return out


class VitalsClassifierProbe(nn.Module):
    """
    Lightweight classification wrapper around VitalsEncoder
    to evaluate unimodal predictive power on KTAS acuity.
    """

    def __init__(self, encoder: VitalsEncoder, num_classes: int = 5):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.encoder(x)
        logits = self.head(embeds)
        return logits


# ==============================================================================
# 3. TRAINING & EVALUATION ROUTINES
# ==============================================================================
def train_vitals_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: VitalsClassifierProbe,
    class_weights: torch.Tensor,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    max_epochs: int = 40,
    patience: int = 8,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Train VitalsClassifierProbe with early stopping and class-weighted cross entropy.
    """
    model.to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1 = -1.0
    best_weights = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x, y = batch["vitals"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["vitals"].to(device), batch["target"].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(val_f1)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {"best_val_macro_f1": best_val_f1, "history": history}


def evaluate_vitals_model(
    model: VitalsClassifierProbe,
    data_loader: DataLoader,
    device: str = "cpu",
    num_classes: int = 5,
) -> Dict[str, Any]:
    """Evaluate model and return multiclass metrics."""
    model.eval()
    model.to(device)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch["vitals"].to(device), batch["target"].to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_rec = {}
    for i in range(num_classes):
        row_sum = cm[i].sum()
        per_class_rec[f"Recall_KTAS_{i+1}"] = round(float(cm[i, i] / row_sum) if row_sum > 0 else 0.0, 4)

    return {
        "Accuracy": round(acc, 4),
        "Macro_F1": round(f1_macro, 4),
        "Weighted_F1": round(f1_weighted, 4),
        "Macro_Recall": round(rec_macro, 4),
        **per_class_rec,
        "confusion_matrix": cm.tolist(),
    }


def cache_vitals_embeddings(
    encoder: VitalsEncoder,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed/",
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and serialize vitals embeddings for multimodal fusion."""
    encoder.eval()
    encoder.to(device)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def extract(df: pd.DataFrame) -> torch.Tensor:
        ds = VitalsDataset(df)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        embeds = []
        with torch.no_grad():
            for batch in loader:
                x = batch["vitals"].to(device)
                e = encoder(x).cpu()
                embeds.append(e)
        return torch.cat(embeds, dim=0)

    print("\n[EMBEDDING EXTRACTION] Extracting Vitals latent vectors...")
    emb_train = extract(train_df)
    emb_val = extract(val_df)
    emb_test = extract(test_df)

    torch.save(emb_train, out_path / "vitals_embeddings_train.pt")
    torch.save(emb_val, out_path / "vitals_embeddings_val.pt")
    torch.save(emb_test, out_path / "vitals_embeddings_test.pt")
    print(f"[SAVED] Vitals embeddings saved to: {out_path.resolve()}")

    return emb_train, emb_val, emb_test


# ==============================================================================
# 4. REPORT GENERATOR
# ==============================================================================
def generate_vitals_report(
    metrics: Dict[str, Any],
    input_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    output_path: str = "outputs/vitals_encoder_report.md",
) -> str:
    """Generate Markdown report for Module 6."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Deep Learning Vitals Encoder Report",
        "",
        "> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)",
        "> **Modality:** Static Physiological Vital Signs (`SBP`, `DBP`, `HR`, `RR`, `BT`, `Saturation`, `Mental`, `Pain`, `NRS_pain`)",
        "",
        "---",
        "",
        "## 1. Architectural Specification",
        "",
        "```",
        f"Input Features (Dimension: {input_dim})",
        "  ├── Continuous Vitals (7): SBP, DBP, HR, RR, BT, Saturation, NRS_pain",
        "  ├── Missingness Indicator (1): Saturation_is_missing",
        "  └── Clinical Discrete Indicators (2): Mental_raw (AVPU), Pain_flag",
        "        ↓",
        f"Linear Layer ({input_dim} → {hidden_dim})",
        "        ↓",
        f"BatchNorm1d({hidden_dim}) + GELU Activation",
        "        ↓",
        "Dropout (p = 0.20)",
        "        ↓",
        f"Linear Projection ({hidden_dim} → {embedding_dim})",
        "        ↓",
        f"LayerNorm({embedding_dim}) + GELU Activation",
        "        ↓",
        "Dropout (p = 0.20)",
        "        ↓",
        f"Vitals Latent Vector (d_vitals = {embedding_dim})",
        "```",
        "",
        "---",
        "",
        "## 2. Unimodal Performance on Unseen Test Set (N = 191)",
        "",
        "| Metric | Value | Clinical Benchmark Context |",
        "| :--- | :---: | :--- |",
        f"| **Test Accuracy** | **{metrics['Accuracy']:.4f}** | Pure vitals classification acuity |",
        f"| **Test Macro F1** | **{metrics['Macro_F1']:.4f}** | Balanced across all 5 KTAS levels |",
        f"| **Test Weighted F1** | **{metrics['Weighted_F1']:.4f}** | Overall population weighted F1 |",
        f"| **KTAS 1 Recall (Resuscitation)** | **{metrics['Recall_KTAS_1']:.2f}** | Critical sensitivity on most acute tier |",
        f"| **KTAS 2 Recall (Emergent)** | **{metrics['Recall_KTAS_2']:.2f}** | Sensitivity on emergent presentations |",
        f"| **KTAS 3 Recall (Urgent)** | **{metrics['Recall_KTAS_3']:.2f}** | Sensitivity on urgent presentations |",
        f"| **KTAS 4 Recall (Less Urgent)** | **{metrics['Recall_KTAS_4']:.2f}** | Sensitivity on stable presentations |",
        f"| **KTAS 5 Recall (Non-Urgent)** | **{metrics['Recall_KTAS_5']:.2f}** | Sensitivity on minor presentations |",
        "",
        "---",
        "",
        "## 3. Scientific Justification for Static MLP vs Temporal Recurrent Models",
        "",
        "1. **Nature of Emergency Intake Vitals:** Emergency triage records in this dataset represent **single snapshot measurements** taken during initial triage arrival, not continuous bedside ICU time-series telemetry.",
        "2. **No Temporal Dimension:** Applying RNN, LSTM, or temporal Transformers to static 1-point snapshots would impose fictitious sequential assumptions.",
        "3. **Regularization for Compact Datasets:** The 2-stage MLP incorporates BatchNorm, LayerNorm, GELU, and 20% Dropout to constrain parameter complexity on 886 training records.",
        "",
        "---",
        "",
        "## 4. Cached Latent Embeddings",
        "",
        "- `data/processed/vitals_embeddings_train.pt`: `torch.Size([886, 128])`",
        "- `data/processed/vitals_embeddings_val.pt`: `torch.Size([190, 128])`",
        "- `data/processed/vitals_embeddings_test.pt`: `torch.Size([191, 128])`",
        "- Model weights checkpoint: `models/checkpoints/vitals_encoder.pt`",
    ]

    report_str = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"[REPORT GENERATED] Vitals encoder report saved to: {out_file.resolve()}")
    return report_str


# ==============================================================================
# 5. ORCHESTRATION PIPELINE
# ==============================================================================
def run_vitals_encoder_module(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Execute Module 6 end-to-end."""
    print("=" * 75)
    print("MODULE 6: DEEP LEARNING VITALS ENCODER")
    print("=" * 75)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_dir = Path(cfg["data"]["processed"].get("processed_dir", "data/processed/"))
    models_dir = Path(cfg["outputs"].get("models_dir", "models/checkpoints/"))
    metrics_dir = Path(cfg["outputs"].get("metrics_dir", "outputs/metrics/"))
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    input_dim = len(VITALS_FEATURE_NAMES)  # 10
    hidden_dim = int(cfg["vitals_encoder"].get("hidden_dim", 64))
    embedding_dim = int(cfg["vitals_encoder"].get("embedding_dim", 128))
    dropout_prob = float(cfg["vitals_encoder"].get("dropout", 0.2))

    # 1. Load splits
    train_df = pd.read_parquet(proc_dir / "train.parquet")
    val_df = pd.read_parquet(proc_dir / "val.parquet")
    test_df = pd.read_parquet(proc_dir / "test.parquet")

    train_ds = VitalsDataset(train_df)
    val_ds = VitalsDataset(val_df)
    test_ds = VitalsDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 2. Build model & compute class weights
    encoder = VitalsEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout_prob=dropout_prob,
    )
    probe = VitalsClassifierProbe(encoder=encoder, num_classes=5)

    y_train = train_df["KTAS_RN_encoded"].values
    class_counts = np.bincount(y_train, minlength=5)
    class_weights = torch.tensor(
        len(y_train) / (5.0 * np.maximum(class_counts, 1)), dtype=torch.float32
    )

    # 3. Train
    print(f"\n[TRAINING] Training VitalsEncoder MLP (d_in={input_dim} -> d_hidden={hidden_dim} -> d_embed={embedding_dim})...")
    train_vitals_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=probe,
        class_weights=class_weights,
        learning_rate=0.001,
        weight_decay=0.01,
        max_epochs=40,
        patience=8,
    )

    # 4. Evaluate on test set
    test_metrics = evaluate_vitals_model(probe, test_loader)
    print("\n[EVALUATION] Unimodal Vitals Test Metrics:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  * {k}: {v}")

    # Save metrics
    with open(metrics_dir / "vitals_encoder_results.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Save trained encoder checkpoint
    torch.save(encoder.state_dict(), models_dir / "vitals_encoder.pt")
    print(f"[CHECKPOINT SAVED] Vitals encoder weights saved to: {models_dir / 'vitals_encoder.pt'}")

    # 5. Cache embeddings
    cache_vitals_embeddings(
        encoder=encoder,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=str(proc_dir),
    )

    # 6. Generate report
    generate_vitals_report(
        metrics=test_metrics,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_path="outputs/vitals_encoder_report.md",
    )

    print("\n" + "=" * 75)
    print("[MODULE 6 COMPLETE] Deep Learning Vitals Encoder finalized.")
    print("=" * 75)

    return test_metrics


if __name__ == "__main__":
    run_vitals_encoder_module()
