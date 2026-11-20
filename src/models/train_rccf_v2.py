"""
Training & Checkpoint Generation Script for RCCF-Net v2.
=======================================================
Module: src/models/train_rccf_v2.py

Generates CECE evidence conditioning features across train/val/test splits,
trains RCCF-Net v2 with class weighting and early stopping, and saves the
calibrated weights to models/checkpoints/rccf_net_v2_model.pt.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.cece import CrossModalEvidenceConflictEngine
from src.models.rccf_net_v2 import RCCFNetDataset, RCCFNetV2, train_rccf_net_v2


def compute_cece_features_for_df(df: pd.DataFrame, cece_engine: CrossModalEvidenceConflictEngine) -> torch.Tensor:
    """Extract 8-dimensional CECE evidence conditioning vectors for all records in dataframe."""
    vectors: List[List[float]] = []

    for _, row in df.iterrows():
        # Handle unscaled or scaled columns
        complaint = str(row.get("Chief_complain_clean", row.get("Chief_complain", "")))
        sbp = float(row.get("SBP_scaled", 120.0)) * 200.0 if "SBP_scaled" in row else float(row.get("SBP", 120.0))
        dbp = float(row.get("DBP_scaled", 80.0)) * 120.0 if "DBP_scaled" in row else float(row.get("DBP", 80.0))
        hr = float(row.get("HR_scaled", 80.0)) * 180.0 if "HR_scaled" in row else float(row.get("HR", 80.0))
        rr = float(row.get("RR_scaled", 18.0)) * 40.0 if "RR_scaled" in row else float(row.get("RR", 18.0))
        bt = float(row.get("BT_scaled", 36.8)) * 42.0 if "BT_scaled" in row else float(row.get("BT", 36.8))
        spo2 = 98.0 if row.get("Saturation_is_missing", 0) == 1 else (
            float(row.get("Saturation_scaled", 0.98)) * 100.0 if "Saturation_scaled" in row else float(row.get("Saturation", 98.0))
        )
        nrs = float(row.get("NRS_pain_scaled", 0.0)) * 10.0 if "NRS_pain_scaled" in row else float(row.get("NRS_pain", 0.0))
        pain_f = int(row.get("Pain_flag", row.get("Pain", 0)))
        injury_str = "Trauma" if int(row.get("Injury_cat", row.get("Injury", 0))) == 1 else "Non-trauma"

        report = cece_engine.audit_intake(
            chief_complaint=complaint,
            sbp=sbp,
            dbp=dbp,
            hr=hr,
            rr=rr,
            bt=bt,
            saturation=spo2,
            nrs_pain=nrs,
            pain_flag=pain_f,
            injury=injury_str,
        )
        vectors.append(report.cece_conditioning_vector)

    return torch.tensor(vectors, dtype=torch.float32)


def main():
    print("=" * 78)
    print("RCCF-Net v2 Training & Checkpoint Generation Pipeline")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")

    # 1. Load Data Splits & Embeddings
    data_dir = PROJECT_ROOT / "data" / "processed"
    df_train = pd.read_csv(data_dir / "train.csv")
    df_val = pd.read_csv(data_dir / "val.csv")
    df_test = pd.read_csv(data_dir / "test.csv")

    t_train = torch.load(data_dir / "text_embeddings_train.pt")
    v_train = torch.load(data_dir / "vitals_embeddings_train.pt")
    c_train = torch.load(data_dir / "context_embeddings_train.pt")
    y_train = torch.tensor(df_train["KTAS_RN_encoded"].values, dtype=torch.long)

    t_val = torch.load(data_dir / "text_embeddings_val.pt")
    v_val = torch.load(data_dir / "vitals_embeddings_val.pt")
    c_val = torch.load(data_dir / "context_embeddings_val.pt")
    y_val = torch.tensor(df_val["KTAS_RN_encoded"].values, dtype=torch.long)

    t_test = torch.load(data_dir / "text_embeddings_test.pt")
    v_test = torch.load(data_dir / "vitals_embeddings_test.pt")
    c_test = torch.load(data_dir / "context_embeddings_test.pt")
    y_test = torch.tensor(df_test["KTAS_RN_encoded"].values, dtype=torch.long)

    print(f"Train samples: {len(t_train)} | Val samples: {len(t_val)} | Test samples: {len(t_test)}")

    # 2. Extract CECE Conditioning Vectors
    cece_engine = CrossModalEvidenceConflictEngine()
    print("Computing CECE evidence conditioning vectors...")
    e_train = compute_cece_features_for_df(df_train, cece_engine)
    e_val = compute_cece_features_for_df(df_val, cece_engine)
    e_test = compute_cece_features_for_df(df_test, cece_engine)

    # Save cached CECE evidence tensors
    torch.save(e_train, data_dir / "cece_evidence_train.pt")
    torch.save(e_val, data_dir / "cece_evidence_val.pt")
    torch.save(e_test, data_dir / "cece_evidence_test.pt")
    print("Saved cached CECE evidence vectors.")

    # 3. Create Datasets & DataLoaders
    train_ds = RCCFNetDataset(t_train, v_train, c_train, cece_evidence=e_train, targets=y_train)
    val_ds = RCCFNetDataset(t_val, v_val, c_val, cece_evidence=e_val, targets=y_val)
    test_ds = RCCFNetDataset(t_test, v_test, c_test, cece_evidence=e_test, targets=y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 4. Compute Balanced Class Weights
    classes = np.array([0, 1, 2, 3, 4])
    cw = compute_class_weight("balanced", classes=classes, y=y_train.numpy())
    class_weights_t = torch.tensor(cw, dtype=torch.float32)

    # 5. Initialize RCCF-Net v2
    model = RCCFNetV2(
        text_dim=384,
        vitals_dim=128,
        context_dim=64,
        cece_dim=8,
        shared_dim=128,
        num_classes=5,
        num_heads=4,
        dropout_prob=0.20,
        modality_dropout_prob=0.15,
        temperature=1.0,
    )

    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "rccf_net_v2_model.pt"

    # 6. Train model
    res = train_rccf_net_v2(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        class_weights=class_weights_t,
        learning_rate=0.0007,
        weight_decay=0.001,
        max_epochs=45,
        patience=10,
        device=device,
        save_checkpoint_path=ckpt_path,
    )
    print(f"Training Complete! Best Validation Macro F1: {res['best_val_macro_f1']:.4f}")

    # 7. Evaluate on Held-out Test Set
    model.eval()
    test_preds: List[int] = []
    test_targets: List[int] = []

    with torch.no_grad():
        for batch in test_loader:
            t = batch["text"].to(device)
            v = batch["vitals"].to(device)
            c = batch["context"].to(device)
            e = batch["cece_evidence"].to(device)
            y = batch["target"].to(device)

            logits = model(text=t, vitals=v, context=c, cece_evidence=e)
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            test_preds.extend(preds)
            test_targets.extend(y.cpu().numpy().tolist())

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    test_acc = accuracy_score(test_targets, test_preds)
    test_macro_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    print("\n" + "=" * 50)
    print("RCCF-Net v2 TEST EVALUATION RESULTS:")
    print("=" * 50)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print(classification_report(test_targets, test_preds, target_names=[f"KTAS {i+1}" for i in range(5)], zero_division=0))


if __name__ == "__main__":
    main()

