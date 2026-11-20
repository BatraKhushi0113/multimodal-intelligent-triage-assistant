"""
Multimodal Intelligent Triage Assistant - Text Modality & Pretrained Embeddings
===============================================================================
Module 5: Clinical chief complaint vocabulary analysis, frozen sentence-transformer
embedding generation (MiniLM), PyTorch TextEncoder, and TF-IDF vs MiniLM comparison.

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
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# 1. TEXT DATASET & PYTORCH MODULE
# ==============================================================================
class TextDataset(Dataset):
    """
    PyTorch Dataset for text modality.
    Supports either precomputed dense embedding tensors or raw text strings.
    """

    def __init__(
        self,
        embeddings: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        targets: Optional[np.ndarray] = None,
    ):
        self.embeddings = embeddings
        self.texts = texts
        self.targets = torch.tensor(targets, dtype=torch.long) if targets is not None else None

        if self.embeddings is not None and self.targets is not None:
            assert len(self.embeddings) == len(self.targets), "Embeddings and targets length mismatch."
        elif self.texts is not None and self.targets is not None:
            assert len(self.texts) == len(self.targets), "Texts and targets length mismatch."

    def __len__(self) -> int:
        if self.embeddings is not None:
            return len(self.embeddings)
        elif self.texts is not None:
            return len(self.texts)
        return 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {}
        if self.embeddings is not None:
            item["embedding"] = self.embeddings[idx]
        if self.texts is not None:
            item["text"] = self.texts[idx]
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


class TextEncoder(nn.Module):
    """
    Deep Learning Text Encoder for clinical chief complaints.
    Projects dense sentence embeddings (e.g. 384-d from all-MiniLM-L6-v2)
    into a configurable latent space with normalization, non-linearity, and dropout.
    """

    def __init__(
        self,
        input_dim: int = 384,
        output_dim: int = 128,
        dropout_prob: float = 0.2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [nn.Linear(input_dim, output_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout_prob))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input shape:  (batch_size, input_dim)
        Output shape: (batch_size, output_dim)
        """
        return self.projection(x)


# ==============================================================================
# 2. VOCABULARY & LENGTH ANALYSIS
# ==============================================================================
def analyze_text_corpus(
    df: pd.DataFrame,
    text_col: str = "Chief_complain_clean",
    plots_dir: str = "outputs/plots/",
) -> Dict[str, Any]:
    """
    Analyze chief complaint vocabulary, token lengths, character lengths,
    and term frequency distributions.
    """
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    texts = df[text_col].fillna("unspecified").astype(str).tolist()
    char_lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]

    # Term frequencies
    word_freq: Dict[str, int] = {}
    for t in texts:
        for w in t.split():
            clean_w = w.strip(".,/()[]-")
            if clean_w:
                word_freq[clean_w] = word_freq.get(clean_w, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    top_20 = sorted_words[:20]

    # 1. Plot length distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(word_counts, bins=range(1, 15), ax=ax1, color="#2980b9", edgecolor="black", discrete=True)
    ax1.set_title("Chief Complaint Word Count Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Words", fontsize=10)
    ax1.set_ylabel("Patient Count", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    sns.histplot(char_lengths, bins=25, ax=ax2, color="#16a085", edgecolor="black")
    ax2.set_title("Character Length Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Characters", fontsize=10)
    ax2.set_ylabel("Patient Count", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plots_path / "text_length_distribution.png", dpi=300)
    plt.close()

    # 2. Plot top complaints
    fig, ax = plt.subplots(figsize=(8, 6))
    words, freqs = zip(*top_20)
    y_pos = np.arange(len(words))
    ax.barh(y_pos, freqs, color="#34495e", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency Count Across Patients", fontsize=10, fontweight="semibold")
    ax.set_title("Top 20 Frequent Clinical Complaint Terms", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for i, count in enumerate(freqs):
        ax.annotate(f" {count:,}", xy=(count, i), va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_path / "top_chief_complaints.png", dpi=300)
    plt.close()

    return {
        "total_records": len(texts),
        "vocab_size": len(word_freq),
        "mean_word_count": round(float(np.mean(word_counts)), 2),
        "median_word_count": int(np.median(word_counts)),
        "max_word_count": int(np.max(word_counts)),
        "mean_char_length": round(float(np.mean(char_lengths)), 2),
        "median_char_length": int(np.median(char_lengths)),
        "top_terms": dict(top_20),
    }


# ==============================================================================
# 3. PRETRAINED EMBEDDINGS EXTRACTION & CACHING
# ==============================================================================
def generate_and_cache_embeddings(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "data/processed/",
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract dense sentence representations using a frozen pretrained MiniLM model.
    Caches tensors locally for efficient multimodal reuse.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_pt = out_path / "text_embeddings_train.pt"
    val_pt = out_path / "text_embeddings_val.pt"
    test_pt = out_path / "text_embeddings_test.pt"

    # Check if cached
    if train_pt.exists() and val_pt.exists() and test_pt.exists():
        print(f"[CACHE HIT] Loading cached text embeddings from: {out_path.resolve()}")
        emb_train = torch.load(train_pt, weights_only=True)
        emb_val = torch.load(val_pt, weights_only=True)
        emb_test = torch.load(test_pt, weights_only=True)
        return emb_train, emb_val, emb_test

    print(f"\n[ENCODING] Loading pretrained sentence-transformer: '{model_name}' on {device}...")
    model = SentenceTransformer(model_name, device=device)

    # Freeze encoder parameters
    for param in model.parameters():
        param.requires_grad = False

    print("[ENCODING] Extracting embeddings for Train, Val, and Test splits...")
    emb_train_np = model.encode(train_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    emb_val_np = model.encode(val_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    emb_test_np = model.encode(test_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    emb_train = torch.tensor(emb_train_np, dtype=torch.float32)
    emb_val = torch.tensor(emb_val_np, dtype=torch.float32)
    emb_test = torch.tensor(emb_test_np, dtype=torch.float32)

    # Save to disk
    torch.save(emb_train, train_pt)
    torch.save(emb_val, val_pt)
    torch.save(emb_test, test_pt)
    print(f"[SAVED] Cached embeddings saved to: {out_path.resolve()}")

    return emb_train, emb_val, emb_test


# ==============================================================================
# 4. COMPARISON: TF-IDF VS MINILM
# ==============================================================================
def compare_text_representations(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    emb_train: torch.Tensor,
    emb_val: torch.Tensor,
    emb_test: torch.Tensor,
    plots_dir: str = "outputs/plots/",
) -> pd.DataFrame:
    """
    Compare classification performance using TF-IDF vs frozen MiniLM embeddings.
    """
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    results = []

    # -------------------------------------------------------------
    # 1. TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)

    clf_tfidf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_tfidf.fit(X_train_tfidf, y_train)
    preds_tfidf = clf_tfidf.predict(X_test_tfidf)

    acc_tfidf = float(accuracy_score(y_test, preds_tfidf))
    f1_macro_tfidf = float(f1_score(y_test, preds_tfidf, average="macro", zero_division=0))
    f1_weighted_tfidf = float(f1_score(y_test, preds_tfidf, average="weighted", zero_division=0))
    rec_macro_tfidf = float(recall_score(y_test, preds_tfidf, average="macro", zero_division=0))

    results.append({
        "Text_Representation": "TF-IDF (300 n-grams)",
        "Classifier": "Logistic Regression (Balanced)",
        "Embedding_Dim": 300,
        "Test_Accuracy": round(acc_tfidf, 4),
        "Test_Macro_F1": round(f1_macro_tfidf, 4),
        "Test_Weighted_F1": round(f1_weighted_tfidf, 4),
        "Test_Macro_Recall": round(rec_macro_tfidf, 4),
    })

    # -------------------------------------------------------------
    # 2. MiniLM Embeddings + Logistic Regression (Linear Probe)
    # -------------------------------------------------------------
    clf_minilm = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, C=1.0)
    clf_minilm.fit(emb_train.numpy(), y_train)
    preds_minilm = clf_minilm.predict(emb_test.numpy())

    acc_minilm = float(accuracy_score(y_test, preds_minilm))
    f1_macro_minilm = float(f1_score(y_test, preds_minilm, average="macro", zero_division=0))
    f1_weighted_minilm = float(f1_score(y_test, preds_minilm, average="weighted", zero_division=0))
    rec_macro_minilm = float(recall_score(y_test, preds_minilm, average="macro", zero_division=0))

    results.append({
        "Text_Representation": "Frozen MiniLM-L6-v2 (Dense)",
        "Classifier": "Logistic Regression Probe",
        "Embedding_Dim": emb_train.shape[1],
        "Test_Accuracy": round(acc_minilm, 4),
        "Test_Macro_F1": round(f1_macro_minilm, 4),
        "Test_Weighted_F1": round(f1_weighted_minilm, 4),
        "Test_Macro_Recall": round(rec_macro_minilm, 4),
    })

    # -------------------------------------------------------------
    # 3. MiniLM + PyTorch MLP Probe
    # -------------------------------------------------------------
    # Train lightweight MLP on frozen embeddings
    torch.manual_seed(42)
    mlp = nn.Sequential(
        nn.Linear(emb_train.shape[1], 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, 5),
    )
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=0.01)
    
    # Class weights for CrossEntropyLoss
    class_counts = np.bincount(y_train, minlength=5)
    class_weights = torch.tensor(len(y_train) / (5.0 * np.maximum(class_counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = TextDataset(embeddings=emb_train, targets=y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    mlp.train()
    for _ in range(30):
        for batch in train_loader:
            optimizer.zero_grad()
            logits = mlp(batch["embedding"])
            loss = criterion(logits, batch["target"])
            loss.backward()
            optimizer.step()

    mlp.eval()
    with torch.no_grad():
        test_logits = mlp(emb_test)
        preds_mlp = test_logits.argmax(dim=1).numpy()

    acc_mlp = float(accuracy_score(y_test, preds_mlp))
    f1_macro_mlp = float(f1_score(y_test, preds_mlp, average="macro", zero_division=0))
    f1_weighted_mlp = float(f1_score(y_test, preds_mlp, average="weighted", zero_division=0))
    rec_macro_mlp = float(recall_score(y_test, preds_mlp, average="macro", zero_division=0))

    results.append({
        "Text_Representation": "Frozen MiniLM + MLP Projection",
        "Classifier": "PyTorch MLP (Linear-LN-GELU-Linear)",
        "Embedding_Dim": 128,
        "Test_Accuracy": round(acc_mlp, 4),
        "Test_Macro_F1": round(f1_macro_mlp, 4),
        "Test_Weighted_F1": round(f1_weighted_mlp, 4),
        "Test_Macro_Recall": round(rec_macro_mlp, 4),
    })

    comp_df = pd.DataFrame(results)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(comp_df))
    width = 0.35
    ax.bar(x - width / 2, comp_df["Test_Accuracy"], width, label="Accuracy", color="#2980b9", edgecolor="black")
    ax.bar(x + width / 2, comp_df["Test_Macro_F1"], width, label="Macro F1", color="#e67e22", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Text_Representation"], rotation=15, ha="right", fontsize=9, fontweight="semibold")
    ax.set_ylabel("Test Score", fontsize=10, fontweight="semibold")
    ax.set_title("Text Modality Benchmark: TF-IDF vs Frozen MiniLM Embeddings", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plots_path / "text_model_comparison.png", dpi=300)
    plt.close()

    return comp_df


# ==============================================================================
# 5. SCIENTIFIC REPORT GENERATOR
# ==============================================================================
def generate_text_modality_report(
    vocab_stats: Dict[str, Any],
    comp_df: pd.DataFrame,
    output_path: str = "outputs/text_modality_report.md",
) -> str:
    """Generate Markdown report for Module 5."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Text Modality & Pretrained Embeddings Report",
        "",
        "> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)",
        "> **Modality:** Clinical Chief Complaint (`Chief_complain`)",
        "",
        "---",
        "",
        "## 1. Corpus & Vocabulary Characteristics",
        "",
        f"- **Total Triage Records:** `{vocab_stats['total_records']:,}`",
        f"- **Distinct Vocabulary Terms:** `{vocab_stats['vocab_size']:,}`",
        f"- **Mean Words per Complaint:** `{vocab_stats['mean_word_count']}` words (Median: `{vocab_stats['median_word_count']}`)",
        f"- **Mean Character Length:** `{vocab_stats['mean_char_length']}` characters (Median: `{vocab_stats['median_char_length']}`)",
        f"- **Maximum Word Count:** `{vocab_stats['max_word_count']}` words",
        "",
        "> **Clinical Linguistic Observation:** Emergency chief complaints are terse, telegraphic phrases "
        "(e.g., *'chest pain'*, *'dyspnea'*, *'right forearm burn'*). Most complaints contain only 2 to 4 words. "
        "Training a Transformer from scratch would lead to catastrophic overfitting on such a compact dataset (~1,267 records).",
        "",
        "### Top 10 Clinical Complaint Terms Across Patients:",
        "",
        "| Term | Patient Occurrence Count | Clinical Relevance |",
        "| :--- | :---: | :--- |"
    ]

    for term, count in list(vocab_stats["top_terms"].items())[:10]:
        lines.append(f"| `{term}` | {count:,} | Frequent acute symptom or anatomical site |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Representation Comparison: TF-IDF vs Frozen MiniLM",
        "",
        "| Text Representation | Classifier Architecture | Output Dim | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Macro Recall |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, row in comp_df.iterrows():
        lines.append(
            f"| `{row['Text_Representation']}` | {row['Classifier']} | {row['Embedding_Dim']} | "
            f"**{row['Test_Accuracy']:.4f}** | **{row['Test_Macro_F1']:.4f}** | {row['Test_Weighted_F1']:.4f} | {row['Test_Macro_Recall']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Scientific Justification for Frozen MiniLM Embeddings",
        "",
        "1. **Semantic Generalization vs Exact Lexical Matches:** TF-IDF relies on exact character n-grams. "
        "Pretrained sentence transformers (`all-MiniLM-L6-v2`) embed clinical synonyms and related terminology "
        "(e.g., *'chest tightness'*, *'sternal ache'*, *'substernal discomfort'*) into proximate regions in the 384-d latent space.",
        "2. **Overfitting Mitigation:** Freezing the transformer weights preserves foundational semantic knowledge "
        "while preventing parameter divergence on a 1,267-record dataset.",
        "3. **Modular Computational Efficiency:** Precomputing and caching the 384-d embeddings to disk "
        "reduces downstream multimodal training epoch runtimes from minutes to seconds.",
        "",
        "---",
        "",
        "## 4. TextEncoder Architecture Specification",
        "",
        "```",
        "Chief Complaint Text",
        "        ↓",
        "Frozen MiniLM-L6-v2 (Embedding Dim = 384)",
        "        ↓",
        "Linear Projection (384 → 128)",
        "        ↓",
        "LayerNorm(128)",
        "        ↓",
        "GELU Activation",
        "        ↓",
        "Dropout (p = 0.2)",
        "        ↓",
        "Text Latent Vector (d_text = 128)",
        "```",
        "",
        "Cached files: `data/processed/text_embeddings_{train,val,test}.pt`."
    ])

    report_str = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"[REPORT GENERATED] Text modality report saved to: {out_file.resolve()}")
    return report_str


# ==============================================================================
# 6. ORCHESTRATION PIPELINE
# ==============================================================================
def run_text_modality_module(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Execute Module 5 end-to-end."""
    print("=" * 75)
    print("MODULE 5: TEXT MODALITY & PRETRAINED EMBEDDINGS")
    print("=" * 75)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_dir = Path(cfg["data"]["processed"].get("processed_dir", "data/processed/"))
    plots_dir = cfg["outputs"].get("plots_dir", "outputs/plots/")
    metrics_dir = Path(cfg["outputs"].get("metrics_dir", "outputs/metrics/"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["text_encoder"].get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load splits
    train_df = pd.read_parquet(proc_dir / "train.parquet")
    val_df = pd.read_parquet(proc_dir / "val.parquet")
    test_df = pd.read_parquet(proc_dir / "test.parquet")

    # 2. Vocabulary and length analysis
    print("\n[ANALYSIS] Analyzing vocabulary and complaint length...")
    all_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    vocab_stats = analyze_text_corpus(all_df, text_col="Chief_complain_clean", plots_dir=plots_dir)
    print(f"  * Total vocabulary terms: {vocab_stats['vocab_size']:,}")
    print(f"  * Mean word count: {vocab_stats['mean_word_count']} words")

    train_texts = train_df["Chief_complain_clean"].tolist()
    val_texts = val_df["Chief_complain_clean"].tolist()
    test_texts = test_df["Chief_complain_clean"].tolist()

    y_train = train_df["KTAS_RN_encoded"].values
    y_val = val_df["KTAS_RN_encoded"].values
    y_test = test_df["KTAS_RN_encoded"].values

    # 3. Pretrained sentence embeddings extraction & caching
    emb_train, emb_val, emb_test = generate_and_cache_embeddings(
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        model_name=model_name,
        output_dir=str(proc_dir),
        device=device,
    )

    # 4. Compare TF-IDF vs MiniLM
    print("\n[BENCHMARK] Comparing TF-IDF vs Frozen MiniLM representations...")
    comp_df = compare_text_representations(
        train_texts, val_texts, test_texts,
        y_train, y_val, y_test,
        emb_train, emb_val, emb_test,
        plots_dir=plots_dir,
    )
    print("\n" + comp_df.to_string(index=False))

    # Save metrics
    comp_df.to_csv(metrics_dir / "text_modality_comparison.csv", index=False)

    # 5. Unit test TextEncoder forward pass
    print("\n[VERIFICATION] Verifying PyTorch TextEncoder forward pass and tensor shapes...")
    encoder = TextEncoder(input_dim=emb_train.shape[1], output_dim=128, dropout_prob=0.2)
    sample_batch = emb_train[:8]
    output_tensor = encoder(sample_batch)
    print(f"  * Input shape:  {sample_batch.shape} (batch_size=8, input_dim={emb_train.shape[1]})")
    print(f"  * Output shape: {output_tensor.shape} (batch_size=8, projected_dim=128)")
    assert output_tensor.shape == (8, 128), "Shape mismatch in TextEncoder output."
    print("  * Forward pass verification PASSED.")

    # 6. Generate report
    generate_text_modality_report(vocab_stats, comp_df, output_path="outputs/text_modality_report.md")

    print("\n" + "=" * 75)
    print("[MODULE 5 COMPLETE] Text modality pipeline and embeddings finalized.")
    print("=" * 75)

    return {
        "vocab_size": vocab_stats["vocab_size"],
        "embedding_dim": emb_train.shape[1],
        "test_results": comp_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    run_text_modality_module()
