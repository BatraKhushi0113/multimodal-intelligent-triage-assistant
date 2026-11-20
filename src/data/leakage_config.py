"""
Multimodal Intelligent Triage Assistant - Data Leakage Taxonomy & Configuration
==============================================================================
Module 2: Formal temporal classification of features into:
- SAFE_FEATURES (Available at initial triage arrival)
- POTENTIALLY_UNSAFE_FEATURES (Operational/protocol context requiring caution)
- EXCLUDED_FEATURES (Severe target leakage, post-admission, or retrospective audits)

Research & Educational Prototype Only.
"""

from typing import Dict, List, Set, Tuple

# ==============================================================================
# TARGET DEFINITION
# ==============================================================================
TARGET_COLUMN: str = "KTAS_RN"

# ==============================================================================
# A. SAFE INPUT FEATURES
# Physiological vitals, chief complaint text, and patient demographics
# collected strictly at the initial triage booth upon arrival.
# ==============================================================================
SAFE_FEATURES: List[str] = [
    # Text Modality
    "Chief_complain",             # Unstructured verbal/transcribed reason for presentation
    
    # Vitals Modality (Immediate physiological assessment)
    "SBP",                        # Systolic Blood Pressure (mmHg)
    "DBP",                        # Diastolic Blood Pressure (mmHg)
    "HR",                         # Heart Rate (beats per minute)
    "RR",                         # Respiratory Rate (breaths per minute)
    "BT",                         # Body Temperature (°C)
    "Saturation",                 # Peripheral Oxygen Saturation (SpO2 %)
    "Mental",                     # AVPU / Glasgow Coma mental status (1: Alert, 2: Verbal, 3: Pain, 4: Unresponsive)
    "Pain",                       # Binary acute pain presence (1: Present, 0: Absent)
    "NRS_pain",                   # Numerical Rating Scale pain intensity (0 to 10)
    
    # Context Modality (Demographics and baseline intake survey)
    "Sex",                        # Biological sex / gender
    "Age",                        # Patient chronological age
    "Arrival mode",               # Mode of arrival (e.g. walk-in, ambulance, 119 emergency rescue, private vehicle)
    "Injury"                      # Binary presentation type (1: Non-traumatic disease, 2: Traumatic injury)
]

# ==============================================================================
# B. POTENTIALLY UNSAFE FEATURES
# Operational, crowd-level, or protocol-specific variables that do not reflect
# the patient's clinical physiology and may introduce institutional bias or shift.
# ==============================================================================
POTENTIALLY_UNSAFE_FEATURES: List[str] = [
    "Patients number per hour",   # Real-time ED crowd throughput. Available at intake via registration systems,
                                  # but reflects hospital operational backlog rather than patient acuity.
    "Group"                       # Clinical trial / study cohort protocol group (Group 1 vs Group 2).
                                  # Not a biological feature; introduces synthetic cohort shifts.
]

# ==============================================================================
# C. EXCLUDED FEATURES (DATA LEAKAGE & POST-ADMISSION ARTIFACTS)
# Features captured during or after full clinical workup, disposition decisions,
# or retrospective physician audit consensus. Using these creates catastrophic
# artificial performance (data leakage) that cannot exist in real-time triage.
# ==============================================================================
EXCLUDED_FEATURES: List[str] = [
    "Diagnosis in ED",            # Final/working diagnosis reached AFTER doctor evaluation, blood work, & imaging.
    "Disposition",                # Final patient disposition (discharge, ward admission, ICU, transfer, death).
    "KTAS_expert",                # Retrospective review panel triage score (consensus comparator).
    "Error_group",                # Retrospective disagreement classification between nurse and expert.
    "Length of stay_min",         # Total elapsed duration of the ED visit (only known at discharge).
    "KTAS duration_min",          # Duration of the nurse's triage interview (process artifact, unknown at initiation).
    "mistriage"                   # Retrospective label flag comparing KTAS_RN vs KTAS_expert (direct target derivative).
]

# Detailed clinical rationale dictionary mapping every excluded feature to its temporal violation
EXCLUSION_RATIONALE: Dict[str, Dict[str, str]] = {
    "Diagnosis in ED": {
        "timeline": "Post-Workup / Discharge",
        "category": "Definitive Medical Diagnosis",
        "clinical_violation": (
            "The emergency physician's diagnosis is established only after comprehensive history-taking, "
            "physical examination, point-of-care laboratory tests (troponin, arterial blood gas, electrolytes), "
            "and medical imaging (CT, X-ray, ultrasound). Using this feature in an intake triage model represents "
            "severe circular reasoning: the goal of triage is to prioritize who receives a diagnostic workup first."
        )
    },
    "Disposition": {
        "timeline": "Post-Treatment / Departure",
        "category": "Outcome Metric",
        "clinical_violation": (
            "Patient disposition (discharge home, admission to general ward, emergency surgery, or ICU transfer) "
            "is determined hours after triage when response to acute treatment has been observed. "
            "Conditioning triage decisions on future hospitalization outcomes is an impossible temporal leak."
        )
    },
    "KTAS_expert": {
        "timeline": "Retrospective Audit",
        "category": "Secondary Ground Truth / Expert Consensus",
        "clinical_violation": (
            "Assigned retrospectively by a panel of expert emergency physicians who reviewed the entire complete "
            "patient medical record. Serving this to the model as an input would bypass the entire triage task "
            "by feeding an alternative answer key."
        )
    },
    "Error_group": {
        "timeline": "Retrospective Audit",
        "category": "Audit Discrepancy Code",
        "clinical_violation": (
            "Categorizes whether and why the registered nurse's triage score differed from the expert review panel. "
            "It is mathematically and logically dependent on both KTAS_RN and KTAS_expert, creating direct target leakage."
        )
    },
    "Length of stay_min": {
        "timeline": "Post-Discharge",
        "category": "Operational Duration",
        "clinical_violation": (
            "The total number of minutes the patient occupied an ED bed before departure. This metric can only be "
            "computed at discharge time. Furthermore, length of stay varies widely based on hospital bed availability, "
            "not merely acute triage acuity."
        )
    },
    "KTAS duration_min": {
        "timeline": "End of Triage Consultation",
        "category": "Clinical Process Artifact",
        "clinical_violation": (
            "The time taken by the nurse to evaluate the patient and enter the triage rating. At the onset of automated "
            "or AI-assisted triage, this duration does not exist. While longer triage times correlate with ambiguous "
            "or difficult presentations, it is a meta-process artifact rather than an intrinsic patient attribute."
        )
    },
    "mistriage": {
        "timeline": "Retrospective Audit",
        "category": "Ground-Truth Mistriage Indicator",
        "clinical_violation": (
            "A categorical flag (0: Correct, 1: Under-triage, 2: Over-triage) generated post-hoc by comparing KTAS_RN "
            "with KTAS_expert. Incorporating this into the feature matrix would directly expose the target's validity."
        )
    }
}


def get_all_dataset_columns() -> Set[str]:
    """Return all columns expected in the raw triage dataset."""
    return set(SAFE_FEATURES) | set(POTENTIALLY_UNSAFE_FEATURES) | set(EXCLUDED_FEATURES) | {TARGET_COLUMN}


def validate_feature_subset(selected_features: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that a proposed list of feature names does not contain any forbidden
    data leakage columns.

    Parameters
    ----------
    selected_features : List[str]
        Features proposed for model training or inference.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_violating_features)
    """
    violations = [f for f in selected_features if f in EXCLUDED_FEATURES]
    is_valid = len(violations) == 0
    return is_valid, violations
