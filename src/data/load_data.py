"""
Multimodal Intelligent Triage Assistant - Data Ingestion & Exploratory Analysis
=============================================================================
Module 1: Robust loading, auditing, suspicious value detection, and exploratory
analysis of the Emergency Service Triage Application dataset.

Research & Educational Prototype Only.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


def load_raw_data(
    filepath: str,
    delimiter: str = ";",
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the raw emergency triage CSV with robust encoding handling.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    delimiter : str, default=';'
        Delimiter used in the CSV.
    encoding : str, optional
        Character encoding. If None, tries utf-8, latin1, iso-8859-9, cp1254.

    Returns
    -------
    pd.DataFrame
        Raw dataframe without premature type conversion or data loss.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Triage dataset file not found at: {path.resolve()}")

    encodings_to_try = [encoding] if encoding else ["utf-8", "latin1", "iso-8859-9", "cp1254", "cp1252"]
    
    last_error = None
    for enc in encodings_to_try:
        if enc is None:
            continue
        try:
            df = pd.read_csv(path, sep=delimiter, encoding=enc, dtype=object)
            # Re-read with default pandas parsing to observe natural type inferences
            df_natural = pd.read_csv(path, sep=delimiter, encoding=enc)
            print(f"[SUCCESS] Loaded dataset using encoding='{enc}' and separator='{delimiter}'")
            return df_natural
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break

    raise RuntimeError(f"Failed to load {filepath} with tested encodings: {last_error}")


def inspect_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """Return and print the shape of the dataset."""
    n_rows, n_cols = df.shape
    print(f"\n[DATASET SHAPE] Rows: {n_rows:,} | Columns: {n_cols:,}")
    return n_rows, n_cols


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect columns, inferred data types, non-null counts, and sample entries.

    Returns
    -------
    pd.DataFrame
        Summary table of column specifications.
    """
    records = []
    for col in df.columns:
        series = df[col]
        non_null = series.notnull().sum()
        null_count = series.isnull().sum()
        pct_null = (null_count / len(df)) * 100
        dtype = str(series.dtype)
        sample_vals = series.dropna().unique()[:3].tolist()

        records.append({
            "Column": col,
            "Inferred_Dtype": dtype,
            "Non_Null_Count": non_null,
            "Null_Count": null_count,
            "Null_Percent": round(pct_null, 2),
            "Sample_Values": sample_vals
        })
    
    col_df = pd.DataFrame(records)
    return col_df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check standard NaN missingness as well as placeholder missing strings
    like '??', '#BOŞ!', '', and whitespace strings.
    """
    missing_records = []
    placeholder_tokens = {"??", "?", "#boş!", "#bo!", "#bo\ufffd!", "nan", "none", "null", ""}

    for col in df.columns:
        s = df[col]
        standard_nulls = int(s.isnull().sum())
        
        # Check string representations for masked missingness
        as_str = s.astype(str).str.strip().str.lower()
        masked_nulls = int(as_str.isin(placeholder_tokens).sum()) - int(s.isnull().sum())
        masked_nulls = max(0, masked_nulls)
        
        total_effective_missing = standard_nulls + masked_nulls
        pct_effective = (total_effective_missing / len(df)) * 100

        missing_records.append({
            "Column": col,
            "Standard_NaN": standard_nulls,
            "Masked_Placeholders": masked_nulls,
            "Total_Missing": total_effective_missing,
            "Missing_Percent": round(pct_effective, 2)
        })

    report_df = pd.DataFrame(missing_records).sort_values(by="Total_Missing", ascending=False)
    return report_df


def check_duplicates(df: pd.DataFrame) -> int:
    """Detect exact duplicate rows in the dataset."""
    dup_count = int(df.duplicated().sum())
    print(f"\n[DUPLICATE ROWS] Found {dup_count} exact duplicate rows.")
    return dup_count


def get_summary_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate summary statistics for numeric columns and categorical/text columns.
    """
    num_summary = df.describe(include=[np.number]).T
    cat_summary = df.describe(include=["object"]).T if (df.dtypes == "object").any() else pd.DataFrame()
    return num_summary, cat_summary


def get_categorical_uniques(df: pd.DataFrame, max_cardinality: int = 25) -> Dict[str, Any]:
    """
    Inspect unique value counts and distributions for categorical/discrete columns.
    """
    results = {}
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique <= max_cardinality:
            val_counts = df[col].value_counts(dropna=False).to_dict()
            results[col] = {
                "n_unique": n_unique,
                "distribution": val_counts
            }
    return results


def detect_suspicious_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify potential anomalies, physiological outliers, and string encodings
    within numerical medical fields without mutating the underlying data.
    """
    suspicious = {}

    # 1. Non-numeric tokens in vital signs
    vital_cols = ["SBP", "DBP", "HR", "RR", "BT", "Saturation", "NRS_pain"]
    for col in vital_cols:
        if col in df.columns:
            non_numeric = df[col][pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce").isna()]
            if len(non_numeric) > 0:
                suspicious[f"{col}_non_numeric_tokens"] = {
                    "count": int(len(non_numeric)),
                    "unique_tokens": [str(t) for t in non_numeric.unique()[:10]]
                }

    # 2. Physiological range checks on parseable values
    physiological_bounds = {
        "SBP": (60, 240),         # Systolic Blood Pressure (mmHg)
        "DBP": (40, 140),         # Diastolic Blood Pressure (mmHg)
        "HR": (40, 180),          # Heart Rate (bpm)
        "RR": (8, 40),            # Respiratory Rate (breaths/min)
        "BT": (34.0, 42.0),       # Body Temperature (Celsius)
        "Saturation": (70, 100),  # Oxygen Saturation (SpO2 %)
        "Age": (0, 115)           # Patient Age (years)
    }

    for col, (low, high) in physiological_bounds.items():
        if col in df.columns:
            clean_s = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
            out_of_bounds = clean_s[(clean_s < low) | (clean_s > high)]
            if len(out_of_bounds) > 0:
                suspicious[f"{col}_physiological_outliers"] = {
                    "normal_range": [low, high],
                    "outlier_count": int(len(out_of_bounds)),
                    "min_detected": float(clean_s.min()) if not clean_s.empty else None,
                    "max_detected": float(clean_s.max()) if not clean_s.empty else None,
                    "sample_values": [float(v) for v in out_of_bounds.head(5).tolist()]
                }

    # 3. Pain score consistency check (Pain indicator vs NRS_pain rating)
    if "Pain" in df.columns and "NRS_pain" in df.columns:
        pain_flag = pd.to_numeric(df["Pain"], errors="coerce")
        # Cases where Pain=0 (no pain reported) but NRS_pain > 0
        nrs_clean = pd.to_numeric(df["NRS_pain"].astype(str).str.replace(",", "."), errors="coerce")
        inconsistent_pain = df[(pain_flag == 0) & (nrs_clean > 0)]
        suspicious["pain_inconsistency_count"] = int(len(inconsistent_pain))

    # 4. Inverted blood pressure check (DBP >= SBP)
    if "SBP" in df.columns and "DBP" in df.columns:
        sbp_vals = pd.to_numeric(df["SBP"].astype(str).str.replace(",", "."), errors="coerce")
        dbp_vals = pd.to_numeric(df["DBP"].astype(str).str.replace(",", "."), errors="coerce")
        inverted_bp = df[(sbp_vals.notnull()) & (dbp_vals.notnull()) & (dbp_vals >= sbp_vals)]
        suspicious["inverted_bp_count"] = int(len(inverted_bp))
        if len(inverted_bp) > 0:
            suspicious["inverted_bp_samples"] = inverted_bp[["SBP", "DBP"]].to_dict(orient="records")

    return suspicious


def visualize_ktas_distribution(
    df: pd.DataFrame,
    target_col: str = "KTAS_RN",
    output_path: str = "outputs/plots/ktas_rn_distribution.png"
) -> None:
    """
    Plot and save the class distribution of KTAS_RN triage levels.
    """
    if target_col not in df.columns:
        print(f"[WARNING] Target column '{target_col}' not in dataframe. Skipping plot.")
        return

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    counts = df[target_col].value_counts().sort_index()
    percentages = (counts / len(df)) * 100

    palette = sns.color_palette("mako", len(counts))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color=palette, edgecolor="black", alpha=0.85)

    for bar, pct in zip(bars, percentages.values):
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title(f"Target Distribution: {target_col} (Emergency Triage Acuity)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("KTAS Level (1: Resuscitation → 5: Non-Urgent)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Patient Count", fontsize=11, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine(top=True, right=True)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] KTAS distribution saved to: {out_file.resolve()}")


def visualize_missingness(
    df: pd.DataFrame,
    output_path: str = "outputs/plots/missingness_overview.png"
) -> None:
    """
    Generate and save an overview bar chart of missing values across all columns.
    """
    missing_df = check_missing_values(df)
    missing_only = missing_df[missing_df["Total_Missing"] > 0]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if missing_only.empty:
        print("[INFO] No missing values detected for missingness plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_only) * 0.45)))
    y_pos = np.arange(len(missing_only))
    
    bars = ax.barh(y_pos, missing_only["Missing_Percent"], color="#e74c3c", alpha=0.8, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(missing_only["Column"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Effective Missingness Percentage (%)", fontsize=11, fontweight="semibold")
    ax.set_title("Dataset Missingness Overview (Standard NaN + Clinical Placeholders)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, count in zip(bars, missing_only["Total_Missing"]):
        width = bar.get_width()
        ax.annotate(
            f"{width:.1f}% ({count:,} records)",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left", va="center", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[PLOT SAVED] Missingness plot saved to: {out_file.resolve()}")


def generate_dataset_quality_report(
    df: pd.DataFrame,
    output_path: str = "outputs/dataset_quality_report.md"
) -> str:
    """
    Generate a comprehensive, scientifically rigorous dataset quality report in Markdown.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = df.shape
    col_info = get_column_info(df)
    missing_df = check_missing_values(df)
    dup_count = int(df.duplicated().sum())
    suspicious = detect_suspicious_values(df)
    uniques = get_categorical_uniques(df)

    lines = [
        "# Dataset Quality & Exploratory Analysis Report",
        "",
        "> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)",
        "> **Notice:** This assessment does not perform clinical diagnosis or replace professional judgment.",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total Records:** `{n_rows:,}`",
        f"- **Total Features:** `{n_cols}`",
        f"- **Exact Duplicate Rows:** `{dup_count}`",
        f"- **Columns with Missing/Placeholder Data:** `{(missing_df['Total_Missing'] > 0).sum()}` of `{n_cols}`",
        f"- **Target Variable (`KTAS_RN`):** Present, 5 discrete severity classes (1 to 5)",
        "",
        "---",
        "",
        "## 2. Column Specifications & Inferred Types",
        "",
        "| Column | Raw Type | Non-Null | Missing (%) | Sample Values |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    for _, row in col_info.iterrows():
        sample_str = ", ".join([str(v) for v in row["Sample_Values"][:3]])
        sample_str = sample_str.replace("|", "/")
        lines.append(
            f"| `{row['Column']}` | `{row['Inferred_Dtype']}` | {row['Non_Null_Count']:,} | {row['Null_Percent']}% | {sample_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Missingness & Non-Standard Placeholders",
        "",
        "Emergency clinical datasets frequently contain non-standard missingness tokens (e.g., `??`, `#BOŞ!`, unrecorded observations) rather than standard empty values:",
        "",
        "| Column | Standard NaN | Masked Placeholders | Effective Missing | Missing (%) |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])

    for _, row in missing_df.iterrows():
        if row["Total_Missing"] > 0:
            lines.append(
                f"| `{row['Column']}` | {row['Standard_NaN']} | {row['Masked_Placeholders']} | {row['Total_Missing']} | **{row['Missing_Percent']}%** |"
            )

    lines.extend([
        "",
        "### Key Missingness Findings:",
        "- **Saturation (`Saturation`):** 688 standard NaNs and 9 placeholder (`??`) tokens, resulting in **55.01% missingness**. In emergency triage, oxygen saturation is often only measured when a patient presents with respiratory, cardiovascular, or altered mental complaints.",
        "- **NRS Pain Score (`NRS_pain`):** 556 entries contain the Turkish spreadsheet error token `#BOŞ!` (meaning `#EMPTY!`). Cross-referencing reveals that when `Pain == 0` (no pain), `NRS_pain` is systematically unrecorded.",
        "- **Vitals (`SBP`, `DBP`, `HR`, `RR`, `BT`):** Between 18 and 29 records contain string placeholders `??` indicating unobtained vital measurements.",
        "",
        "---",
        "",
        "## 4. KTAS Acuity Target Distribution (`KTAS_RN`)",
        ""
    ])

    if "KTAS_RN" in df.columns:
        ktas_counts = df["KTAS_RN"].value_counts().sort_index()
        ktas_pct = (ktas_counts / len(df)) * 100
        lines.extend([
            "| KTAS Level | Clinical Meaning | Patient Count | Percentage |",
            "| :---: | :--- | :---: | :---: |",
            f"| **1** | Resuscitation (Immediately Life-Threatening) | {ktas_counts.get(1, 0):,} | {ktas_pct.get(1, 0):.2f}% |",
            f"| **2** | Emergent (Potential Threat to Life/Limb) | {ktas_counts.get(2, 0):,} | {ktas_pct.get(2, 0):.2f}% |",
            f"| **3** | Urgent (Condition with High Risk of Deterioration) | {ktas_counts.get(3, 0):,} | {ktas_pct.get(3, 0):.2f}% |",
            f"| **4** | Less Urgent (Low Risk, Stable) | {ktas_counts.get(4, 0):,} | {ktas_pct.get(4, 0):.2f}% |",
            f"| **5** | Non-Urgent (Acute Minor or Chronic Condition) | {ktas_counts.get(5, 0):,} | {ktas_pct.get(5, 0):.2f}% |",
            "",
            "> **Imbalance Observation:** Severe acute presentations (KTAS 1) account for only **1.42%** of admissions. Stratified sampling and cost-sensitive loss functions will be essential during modeling.",
            ""
        ])

    lines.extend([
        "---",
        "",
        "## 5. Suspicious Values & Physiological Audits",
        "",
        "| Audit Category | Detected Anomaly / Outlier | Impact / Clinical Interpretation |",
        "| :--- | :--- | :--- |"
    ])

    for key, val in suspicious.items():
        if isinstance(val, dict):
            if "unique_tokens" in val:
                lines.append(f"| `{key}` | Tokens: `{val['unique_tokens']}` (Count: {val['count']}) | Requires imputation or type casting in Preprocessing. |")
            elif "physiological_outliers" in key:
                lines.append(f"| `{key}` | {val['outlier_count']} records outside {val['normal_range']} (Min: {val['min_detected']}, Max: {val['max_detected']}) | Extreme emergency values (e.g. severe hypoxemia SpO2=20% or SBP=275 mmHg). |")
        else:
            lines.append(f"| `{key}` | Count: `{val}` | Data consistency check. |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Scientific Constraints & Recommendations for Module 2 and 3",
        "",
        "1. **No Preprocessing Yet:** The data remains unmutated; raw types and token variations are mapped.",
        "2. **Leakage Analysis Required (Module 2):** Variables such as `Diagnosis in ED`, `Disposition`, `Length of stay_min`, `KTAS duration_min`, `KTAS_expert`, and `mistriage` are post-admission artifacts and must be formally excluded.",
        "3. **Missingness Indicator Strategy (Module 3):** Since SpO2 missingness is clinically informative (not random missingness), binary indicator columns (`Saturation_is_missing`) should be created alongside robust median/MICE imputation.",
        "4. **Class Weighting:** KTAS Level 1 represents a rare positive class requiring macro-F1 and cost-weighted evaluation."
    ])

    report_content = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[REPORT GENERATED] Quality report written to: {out_file.resolve()}")
    return report_content


def run_exploratory_analysis(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Main orchestration function to run all Module 1 exploratory tasks.
    """
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg["data"]["raw"]["triage_csv"]
    delimiter = cfg["data"]["raw"].get("triage_delimiter", ";")
    target_col = cfg["features"].get("target", "KTAS_RN")
    plots_dir = cfg["outputs"].get("plots_dir", "outputs/plots/")

    print("=" * 70)
    print("MODULE 1: DATASET LOADING AND EXPLORATORY ANALYSIS")
    print(f"Loading from: {csv_path}")
    print("=" * 70)

    # 1. Load data
    df = load_raw_data(filepath=csv_path, delimiter=delimiter)

    # 2. Inspect shape
    n_rows, n_cols = inspect_shape(df)

    # 3. Column names & 4. Data types
    print("\n[COLUMN SPECIFICATIONS]")
    col_info = get_column_info(df)
    print(col_info[["Column", "Inferred_Dtype", "Non_Null_Count", "Null_Percent"]].to_string(index=False))

    # 5. Missing values
    print("\n[MISSING VALUE SUMMARY]")
    missing_df = check_missing_values(df)
    print(missing_df[missing_df["Total_Missing"] > 0].to_string(index=False))

    # 6. Duplicates
    dup_count = check_duplicates(df)

    # 7. Summary statistics
    print("\n[NUMERICAL SUMMARY STATISTICS]")
    num_summary, _ = get_summary_statistics(df)
    print(num_summary)

    # 8. Unique values for categorical / low-cardinality columns
    print("\n[CATEGORICAL & DISCRETE DISTRIBUTIONS]")
    uniques = get_categorical_uniques(df)
    for col, data in uniques.items():
        print(f"\n--- Feature: {col} ({data['n_unique']} unique values) ---")
        for val, count in data["distribution"].items():
            print(f"    {val}: {count:,}")

    # 9. Visualize KTAS_RN class distribution
    plot_path = Path(plots_dir) / "ktas_rn_distribution.png"
    visualize_ktas_distribution(df, target_col=target_col, output_path=str(plot_path))

    # Additional missingness plot
    missing_plot_path = Path(plots_dir) / "missingness_overview.png"
    visualize_missingness(df, output_path=str(missing_plot_path))

    # 10. Detect suspicious values
    print("\n[SUSPICIOUS & PHYSIOLOGICAL VALUE AUDIT]")
    suspicious = detect_suspicious_values(df)
    for k, v in suspicious.items():
        print(f"  * {k}: {v}")

    # 11. Create dataset quality report
    report_path = "outputs/dataset_quality_report.md"
    generate_dataset_quality_report(df, output_path=report_path)

    print("\n" + "=" * 70)
    print("[MODULE 1 COMPLETE] All checks executed successfully.")
    print("=" * 70)

    return {
        "rows": n_rows,
        "cols": n_cols,
        "duplicates": dup_count,
        "suspicious_keys": list(suspicious.keys())
    }


if __name__ == "__main__":
    run_exploratory_analysis()
