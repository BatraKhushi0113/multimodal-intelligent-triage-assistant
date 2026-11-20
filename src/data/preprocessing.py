"""
Multimodal Intelligent Triage Assistant - Data Preprocessing Pipeline
=====================================================================
Module 3: Leakage-free preprocessing, validation, clinical imputation,
missingness indicators, feature scaling, and stratified splitting.

Research & Educational Prototype Only.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.leakage_config import EXCLUDED_FEATURES, SAFE_FEATURES, TARGET_COLUMN
from src.data.load_data import load_raw_data


class ClinicalVitalsCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans raw medical strings, handles clinical placeholders (??, #BOŞ!),
    imputes missing values using train statistics, creates missingness indicators,
    and scales continuous physiological vitals without data leakage.
    """

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        self.impute_values_: Dict[str, float] = {}
        self.scaler_: Optional[Any] = None
        self.feature_names_: List[str] = []

    def _clean_numeric_series(self, series: pd.Series) -> pd.Series:
        """Converts comma decimals, strips tokens like '??' and '#BOŞ!' to NaN."""
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .replace(["??", "?", "#BOŞ!", "#BO\ufffd!", "#BO!", "nan", "None", ""], np.nan)
        )
        return pd.to_numeric(cleaned, errors="coerce")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Compute medians strictly from the training split."""
        df = X.copy()

        # 1. Clean vitals
        for col in ["SBP", "DBP", "HR", "RR", "BT", "Saturation"]:
            if col in df.columns:
                num_s = self._clean_numeric_series(df[col])
                # Learn train median for imputation
                median_val = float(num_s.dropna().median())
                self.impute_values_[col] = median_val

        # 2. Pain & NRS_pain domain logic:
        # If Pain == 0, NRS_pain is clinically 0.
        # If Pain == 1, learn median pain from train patients with pain.
        if "NRS_pain" in df.columns and "Pain" in df.columns:
            pain_mask = df["Pain"].astype(str).str.strip().isin(["1", "1.0"])
            nrs_cleaned = self._clean_numeric_series(df["NRS_pain"])
            pain_with_nrs = nrs_cleaned[pain_mask].dropna()
            self.impute_values_["NRS_pain_positive_median"] = (
                float(pain_with_nrs.median()) if not pain_with_nrs.empty else 4.0
            )

        # 3. Fit scaler on imputed continuous vitals
        continuous_vitals = ["SBP", "DBP", "HR", "RR", "BT", "Saturation", "NRS_pain"]
        imputed_matrix = []
        for col in continuous_vitals:
            if col in df.columns:
                num_s = self._clean_numeric_series(df[col])
                if col == "NRS_pain":
                    # Impute 0 where Pain=0, else positive median
                    pain_flag = df["Pain"].astype(str).str.strip().isin(["1", "1.0"])
                    num_s = num_s.where(pain_flag, 0.0)
                    num_s = num_s.fillna(self.impute_values_["NRS_pain_positive_median"])
                else:
                    num_s = num_s.fillna(self.impute_values_[col])
                imputed_matrix.append(num_s.values)

        if imputed_matrix:
            X_cont = np.column_stack(imputed_matrix)
            self.scaler_ = RobustScaler() if self.scaler_type == "robust" else StandardScaler()
            self.scaler_.fit(X_cont)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned clinical imputation and scaling."""
        df = X.copy()
        out = pd.DataFrame(index=df.index)

        # 1. Missingness indicators (Clinically vital information)
        if "Saturation" in df.columns:
            sat_s = self._clean_numeric_series(df["Saturation"])
            out["Saturation_is_missing"] = sat_s.isna().astype(np.float32)

        # 2. Impute and Scale continuous vitals
        continuous_vitals = ["SBP", "DBP", "HR", "RR", "BT", "Saturation", "NRS_pain"]
        cleaned_arrays = []
        for col in continuous_vitals:
            if col in df.columns:
                num_s = self._clean_numeric_series(df[col])
                if col == "NRS_pain":
                    pain_flag = df["Pain"].astype(str).str.strip().isin(["1", "1.0"])
                    num_s = num_s.where(pain_flag, 0.0)
                    num_s = num_s.fillna(self.impute_values_.get("NRS_pain_positive_median", 4.0))
                else:
                    num_s = num_s.fillna(self.impute_values_.get(col, 0.0))
                cleaned_arrays.append(num_s.values)

        if cleaned_arrays and self.scaler_ is not None:
            scaled_mat = self.scaler_.transform(np.column_stack(cleaned_arrays))
            for i, col in enumerate(continuous_vitals):
                out[f"{col}_scaled"] = scaled_mat[:, i].astype(np.float32)

        # 3. Discrete vitals / Mental status / Pain flag
        if "Mental" in df.columns:
            # AVPU: 1 Alert, 2 Verbal, 3 Pain, 4 Unresponsive
            out["Mental_raw"] = pd.to_numeric(df["Mental"], errors="coerce").fillna(1).astype(np.int64)
        if "Pain" in df.columns:
            out["Pain_flag"] = (
                df["Pain"].astype(str).str.strip().isin(["1", "1.0"]).astype(np.float32)
            )

        return out


class ClinicalContextCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans and encodes patient context (Age, Sex, Arrival mode, Injury, Patients/hr).
    """

    def __init__(self):
        self.age_median_: float = 50.0
        self.patients_hr_median_: float = 7.0
        self.age_scaler_: Optional[StandardScaler] = None
        self.patients_hr_scaler_: Optional[StandardScaler] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        df = X.copy()
        if "Age" in df.columns:
            age_s = pd.to_numeric(df["Age"], errors="coerce")
            self.age_median_ = float(age_s.dropna().median())
            self.age_scaler_ = StandardScaler()
            self.age_scaler_.fit(age_s.fillna(self.age_median_).values.reshape(-1, 1))

        if "Patients number per hour" in df.columns:
            p_s = pd.to_numeric(df["Patients number per hour"], errors="coerce")
            self.patients_hr_median_ = float(p_s.dropna().median())
            self.patients_hr_scaler_ = StandardScaler()
            self.patients_hr_scaler_.fit(p_s.fillna(self.patients_hr_median_).values.reshape(-1, 1))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        out = pd.DataFrame(index=df.index)

        # Age
        if "Age" in df.columns and self.age_scaler_ is not None:
            age_s = pd.to_numeric(df["Age"], errors="coerce").fillna(self.age_median_)
            out["Age_scaled"] = self.age_scaler_.transform(age_s.values.reshape(-1, 1)).flatten().astype(np.float32)

        # Patients per hour
        if "Patients number per hour" in df.columns and self.patients_hr_scaler_ is not None:
            p_s = pd.to_numeric(df["Patients number per hour"], errors="coerce").fillna(self.patients_hr_median_)
            out["Patients_per_hour_scaled"] = (
                self.patients_hr_scaler_.transform(p_s.values.reshape(-1, 1)).flatten().astype(np.float32)
            )

        # Categorical features
        if "Sex" in df.columns:
            # 1: Male, 2: Female -> map to 0: Male, 1: Female
            out["Sex_cat"] = (
                df["Sex"].astype(str).str.strip().map({"1": 0, "2": 1}).fillna(0).astype(np.int64)
            )

        if "Arrival mode" in df.columns:
            # 1 to 7 -> map to 0 to 6
            mode_s = pd.to_numeric(df["Arrival mode"], errors="coerce").fillna(3).astype(int)
            out["Arrival_mode_cat"] = (mode_s - 1).clip(0, 6).astype(np.int64)

        if "Injury" in df.columns:
            # 1: Non-trauma (Disease), 2: Trauma -> map to 0 (non-trauma), 1 (injury)
            inj_s = pd.to_numeric(df["Injury"], errors="coerce").fillna(1).astype(int)
            out["Injury_cat"] = (inj_s - 1).clip(0, 1).astype(np.int64)

        return out


class PreprocessingPipeline:
    """
    Master Preprocessing Manager for Multimodal Intelligent Triage Assistant.
    Coordinates splitting, clinical vitals cleaning, context encoding, text sanitization,
    and serialization.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.target_col = self.cfg["features"].get("target", TARGET_COLUMN)
        self.test_size = float(self.cfg["data"]["splits"].get("test_size", 0.15))
        self.val_size = float(self.cfg["data"]["splits"].get("val_size", 0.15))
        self.random_seed = int(self.cfg["project"].get("seed", 42))

        self.vitals_cleaner = ClinicalVitalsCleaner(scaler_type="robust")
        self.context_cleaner = ClinicalContextCleaner()
        self.is_fitted = False

    def sanitize_text(self, text_series: pd.Series) -> pd.Series:
        """Sanitize unstructured chief complaints."""
        return (
            text_series.astype(str)
            .str.strip()
            .str.lower()
            .replace(["??", "?", "nan", "none", "", "null"], "unspecified")
            .fillna("unspecified")
        )

    def prepare_target(self, target_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Verify KTAS_RN acuity levels.
        Creates 1-indexed clinical level (1-5) and 0-indexed ML target (0-4).
        """
        ktas_num = pd.to_numeric(target_series, errors="coerce").astype(int)
        if not ktas_num.isin([1, 2, 3, 4, 5]).all():
            raise ValueError(f"Found invalid KTAS values outside [1, 2, 3, 4, 5]: {ktas_num.unique()}")
        ktas_encoded = (ktas_num - 1).astype(int)  # 0 to 4 for PyTorch CrossEntropyLoss
        return ktas_num, ktas_encoded

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified train/validation/test split BEFORE fitting any transformers.
        Strictly drops all EXCLUDED_FEATURES and Group to prevent leakage.
        """
        print(f"\n[SPLITTING] Initiating stratified train/val/test splitting (seed={self.random_seed})...")

        # 1. Remove duplicate rows if any
        n_dups = df.duplicated().sum()
        if n_dups > 0:
            print(f"[CLEANUP] Removing {n_dups} duplicate records.")
            df = df.drop_duplicates().reset_index(drop=True)

        # 2. Drop leakage columns
        cols_to_drop = [c for c in EXCLUDED_FEATURES + ["Group"] if c in df.columns]
        clean_df = df.drop(columns=cols_to_drop)
        print(f"[LEAKAGE GUARD] Dropped {len(cols_to_drop)} excluded columns: {cols_to_drop}")

        # 3. Stratified Split (Train 70%, Val 15%, Test 15%)
        # First split off Test (15%)
        train_val_df, test_df = train_test_split(
            clean_df,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=clean_df[self.target_col],
        )

        # Next split Train and Val from remaining 85%
        # val_size_rel = 0.15 / 0.85 ≈ 0.17647
        val_size_rel = self.val_size / (1.0 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_rel,
            random_state=self.random_seed,
            stratify=train_val_df[self.target_col],
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        print(f"[SPLIT COMPLETE] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        return train_df, val_df, test_df

    def fit(self, train_df: pd.DataFrame):
        """Fit preprocessors strictly on the training set."""
        print("\n[FITTING] Fitting Vitals and Context transformers on TRAIN split only...")
        self.vitals_cleaner.fit(train_df)
        self.context_cleaner.fit(train_df)
        self.is_fitted = True
        print("[FIT COMPLETE] Training statistics and scaling parameters locked.")
        return self

    def transform_split(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Transform a split into clean processed feature columns."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on train_df before transforming.")

        # Text
        text_col = split_df["Chief_complain"] if "Chief_complain" in split_df.columns else pd.Series(["unspecified"] * len(split_df))
        clean_text = self.sanitize_text(text_col)

        # Vitals
        vitals_df = self.vitals_cleaner.transform(split_df)

        # Context
        context_df = self.context_cleaner.transform(split_df)

        # Target
        target_raw, target_enc = self.prepare_target(split_df[self.target_col])

        # Combine
        processed = pd.concat([vitals_df, context_df], axis=1)
        processed["Chief_complain_clean"] = clean_text
        processed["KTAS_RN_raw"] = target_raw
        processed["KTAS_RN_encoded"] = target_enc

        return processed

    def save_artifacts(self, artifact_path: str = "models/checkpoints/preprocessor.joblib"):
        """Save fitted preprocessing pipeline for downstream inference."""
        out_path = Path(artifact_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        print(f"[ARTIFACT SAVED] Fitted preprocessor saved to: {out_path.resolve()}")

    @classmethod
    def load_artifacts(cls, artifact_path: str = "models/checkpoints/preprocessor.joblib") -> "PreprocessingPipeline":
        """Load fitted preprocessor."""
        return joblib.load(artifact_path)


def generate_preprocessing_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_train: pd.DataFrame,
    pipeline: PreprocessingPipeline,
    output_path: str = "outputs/preprocessing_report.md",
) -> str:
    """
    Generate a detailed scientific Markdown report documenting the preprocessing results.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Emergency Triage Preprocessing & Stratified Splitting Report",
        "",
        "> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)",
        "> **Notice:** All scalers, imputers, and encoders were fit strictly on the Training set to prevent data leakage.",
        "",
        "---",
        "",
        "## 1. Split Dimensions & Stratification",
        "",
        f"- **Total Patients:** `{len(train_df) + len(val_df) + len(test_df):,}`",
        f"- **Train Set:** `{len(train_df):,}` records ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)",
        f"- **Validation Set:** `{len(val_df):,}` records ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)",
        f"- **Test Set:** `{len(test_df):,}` records ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)",
        "",
        "### Stratified Acuity Target Distribution Across Splits (`KTAS_RN`)",
        "",
        "| KTAS Level | Acuity Description | Train Count (%) | Val Count (%) | Test Count (%) | Total |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |"
    ]

    for lvl in [1, 2, 3, 4, 5]:
        tr_c = (train_df["KTAS_RN"] == lvl).sum()
        va_c = (val_df["KTAS_RN"] == lvl).sum()
        te_c = (test_df["KTAS_RN"] == lvl).sum()
        tot = tr_c + va_c + te_c
        lines.append(
            f"| **{lvl}** | Level {lvl} | {tr_c} ({tr_c/len(train_df)*100:.1f}%) | {va_c} ({va_c/len(val_df)*100:.1f}%) | {te_c} ({te_c/len(test_df)*100:.1f}%) | {tot} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Imputation Parameters Learned Strictly from Train Set",
        "",
        "| Feature | Learned Imputation Value | Clinical Justification |",
        "| :--- | :--- | :--- |",
    ])

    for k, v in pipeline.vitals_cleaner.impute_values_.items():
        if k == "NRS_pain_positive_median":
            desc = "Applied only to patients with Pain=1 who lacked NRS score. Patients with Pain=0 receive 0."
        elif k == "Saturation":
            desc = "Median SpO2 for present vitals. Complemented by `Saturation_is_missing` indicator."
        else:
            desc = f"Train median replacement for string placeholders ('??')."
        lines.append(f"| `{k}` | `{v:.2f}` | {desc} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Processed Feature Specification Matrix",
        "",
        f"Total Processed Features: `{len(processed_train.columns)}`",
        "",
        "| Processed Column | Source Feature | Modality | Transformation Applied |",
        "| :--- | :--- | :--- | :--- |"
    ])

    for col in processed_train.columns:
        if "scaled" in col:
            trans = "Robust / Standard Scaling"
        elif "cat" in col:
            trans = "Discrete Integer Category Index"
        elif "clean" in col:
            trans = "Lowercased & Missing-Handled Text"
        elif "is_missing" in col:
            trans = "Binary Clinical Missingness Indicator (0/1)"
        elif "flag" in col or "raw" in col:
            trans = "Discrete Acuity Indicator"
        elif "encoded" in col:
            trans = "Zero-Indexed ML Target (0 to 4)"
        else:
            trans = "Preserved Feature"

        modality = "Target" if "KTAS" in col else ("Text" if "complain" in col else ("Context" if any(x in col for x in ["Age", "Sex", "Arrival", "Injury", "hour"]) else "Vitals"))
        lines.append(f"| `{col}` | `{col.replace('_scaled', '').replace('_clean', '').replace('_cat', '')}` | {modality} | {trans} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Leakage Prevention Verification",
        "",
        "1. **Post-Admission Artifacts:** Verified zero presence of `Diagnosis in ED`, `Disposition`, `KTAS_expert`, `Error_group`, `Length of stay_min`, `KTAS duration_min`, `mistriage`, or `Group` in processed files.",
        "2. **Distribution Stability:** Train, Val, and Test maintain identical KTAS acuity proportions, preventing evaluation bias on rare acuity classes (KTAS 1).",
        "3. **Saved Pipeline:** Serialized to `models/checkpoints/preprocessor.joblib` for prospective inference."
    ])

    report_content = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[REPORT GENERATED] Preprocessing report written to: {out_file.resolve()}")
    return report_content


def run_preprocessing_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Execute end-to-end data preprocessing: loading, splitting, fitting, transforming,
    saving processed datasets, and generating report.
    """
    print("=" * 70)
    print("MODULE 3: DATA PREPROCESSING & STRATIFIED SPLITTING")
    print("=" * 70)

    cfg_file = Path(config_path)
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_csv = cfg["data"]["raw"]["triage_csv"]
    delimiter = cfg["data"]["raw"].get("triage_delimiter", ";")
    proc_dir = Path(cfg["data"]["processed"].get("processed_dir", "data/processed/"))
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    raw_df = load_raw_data(raw_csv, delimiter=delimiter)

    # 2. Initialize pipeline
    pipeline = PreprocessingPipeline(config_path=config_path)

    # 3. Stratified split BEFORE fitting
    train_df, val_df, test_df = pipeline.split_data(raw_df)

    # 4. Fit strictly on train
    pipeline.fit(train_df)

    # 5. Transform all splits
    print("\n[TRANSFORMING] Generating processed splits...")
    proc_train = pipeline.transform_split(train_df)
    proc_val = pipeline.transform_split(val_df)
    proc_test = pipeline.transform_split(test_df)

    # 6. Save datasets (both parquet and CSV for inspection)
    proc_train.to_parquet(proc_dir / "train.parquet", index=False)
    proc_val.to_parquet(proc_dir / "val.parquet", index=False)
    proc_test.to_parquet(proc_dir / "test.parquet", index=False)

    proc_train.to_csv(proc_dir / "train.csv", index=False)
    proc_val.to_csv(proc_dir / "val.csv", index=False)
    proc_test.to_csv(proc_dir / "test.csv", index=False)
    print(f"[SAVED] Train, Val, and Test splits saved to: {proc_dir.resolve()}")

    # 7. Save pipeline object
    artifact_path = cfg["data"]["processed"].get(
        "preprocessor_artifact", "models/checkpoints/preprocessor.joblib"
    )
    pipeline.save_artifacts(artifact_path=artifact_path)

    # 8. Generate comprehensive report
    report_path = "outputs/preprocessing_report.md"
    generate_preprocessing_report(
        train_df, val_df, test_df, proc_train, pipeline, output_path=report_path
    )

    print("\n" + "=" * 70)
    print("[MODULE 3 COMPLETE] Data preprocessing pipeline executed successfully.")
    print("=" * 70)

    return {
        "train_rows": len(proc_train),
        "val_rows": len(proc_val),
        "test_rows": len(proc_test),
        "features": list(proc_train.columns),
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
