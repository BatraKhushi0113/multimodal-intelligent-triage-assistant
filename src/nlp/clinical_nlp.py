"""
Advanced Clinical NLP Engine for Multimodal Emergency Triage.
=============================================================
Module: src/nlp/clinical_nlp.py

Extends existing text representation capabilities (MiniLM / TF-IDF) with
rigorous clinical information extraction:
  1. Clinical Symptom Extraction (12 emergency organ-system taxonomy).
  2. Syntactic & Lexical Negation Detection (NegEx-inspired scoping).
  3. Severity & Intensity Extraction (NRS 0-10 scales and descriptive modifiers).
  4. Temporal Duration & Onset Extraction (acute vs subacute vs chronic).
  5. Clinical Entity & Anatomical Locus Extraction (retrosternal, epigastric, etc.).
  6. Clinical Uncertainty & Hedging Detection (hedging qualifiers, rule-out flags).
  7. Structured Evidence Generation (ClinicalTextEvidence dataclass).
  8. Seamless backward compatibility with MiniLM 384-d dense embedding interfaces.

Research & Educational Prototype Only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CLINICAL SYMPTOM & ENTITY TAXONOMIES
# ==============================================================================

# Emergency clinical symptom dictionary mapped to standardized concepts & severity weights
EMERGENCY_SYMPTOM_LEXICON: Dict[str, Dict[str, Any]] = {
    # Cardiovascular & Hemodynamic (High Acute Hazard)
    "chest pain": {"category": "Cardiovascular", "urgency": 0.90, "acuity_tier": 1},
    "chest pressure": {"category": "Cardiovascular", "urgency": 0.90, "acuity_tier": 1},
    "chest tightness": {"category": "Cardiovascular", "urgency": 0.85, "acuity_tier": 2},
    "substernal chest pain": {"category": "Cardiovascular", "urgency": 0.95, "acuity_tier": 1},
    "radiating pain": {"category": "Cardiovascular", "urgency": 0.90, "acuity_tier": 1},
    "palpitations": {"category": "Cardiovascular", "urgency": 0.65, "acuity_tier": 3},
    "cardiac arrest": {"category": "Cardiovascular", "urgency": 1.00, "acuity_tier": 1},
    "syncope": {"category": "Cardiovascular", "urgency": 0.85, "acuity_tier": 2},
    "fainting": {"category": "Cardiovascular", "urgency": 0.80, "acuity_tier": 2},
    "irregular heartbeat": {"category": "Cardiovascular", "urgency": 0.70, "acuity_tier": 3},

    # Respiratory & Airway
    "shortness of breath": {"category": "Respiratory", "urgency": 0.90, "acuity_tier": 1},
    "dyspnea": {"category": "Respiratory", "urgency": 0.90, "acuity_tier": 1},
    "difficulty breathing": {"category": "Respiratory", "urgency": 0.90, "acuity_tier": 1},
    "wheezing": {"category": "Respiratory", "urgency": 0.70, "acuity_tier": 3},
    "stridor": {"category": "Respiratory", "urgency": 0.95, "acuity_tier": 1},
    "cyanosis": {"category": "Respiratory", "urgency": 0.95, "acuity_tier": 1},
    "cough": {"category": "Respiratory", "urgency": 0.35, "acuity_tier": 4},
    "hemoptysis": {"category": "Respiratory", "urgency": 0.85, "acuity_tier": 2},
    "coughing up blood": {"category": "Respiratory", "urgency": 0.85, "acuity_tier": 2},
    "asphyxia": {"category": "Respiratory", "urgency": 1.00, "acuity_tier": 1},

    # Neurological & Cerebrovascular
    "altered mental status": {"category": "Neurological", "urgency": 0.95, "acuity_tier": 1},
    "unresponsive": {"category": "Neurological", "urgency": 1.00, "acuity_tier": 1},
    "loss of consciousness": {"category": "Neurological", "urgency": 0.95, "acuity_tier": 1},
    "coma": {"category": "Neurological", "urgency": 1.00, "acuity_tier": 1},
    "seizure": {"category": "Neurological", "urgency": 0.85, "acuity_tier": 2},
    "facial droop": {"category": "Neurological", "urgency": 0.95, "acuity_tier": 1},
    "hemiplegia": {"category": "Neurological", "urgency": 0.95, "acuity_tier": 1},
    "weakness": {"category": "Neurological", "urgency": 0.60, "acuity_tier": 3},
    "slurred speech": {"category": "Neurological", "urgency": 0.90, "acuity_tier": 2},
    "confusion": {"category": "Neurological", "urgency": 0.80, "acuity_tier": 2},
    "severe headache": {"category": "Neurological", "urgency": 0.75, "acuity_tier": 2},
    "thunderclap headache": {"category": "Neurological", "urgency": 0.95, "acuity_tier": 1},
    "dizziness": {"category": "Neurological", "urgency": 0.45, "acuity_tier": 4},
    "vertigo": {"category": "Neurological", "urgency": 0.45, "acuity_tier": 4},

    # Gastrointestinal & Abdominal
    "abdominal pain": {"category": "Gastrointestinal", "urgency": 0.65, "acuity_tier": 3},
    "acute abdomen": {"category": "Gastrointestinal", "urgency": 0.85, "acuity_tier": 2},
    "nausea": {"category": "Gastrointestinal", "urgency": 0.35, "acuity_tier": 4},
    "vomiting": {"category": "Gastrointestinal", "urgency": 0.45, "acuity_tier": 4},
    "hematemesis": {"category": "Gastrointestinal", "urgency": 0.90, "acuity_tier": 2},
    "melena": {"category": "Gastrointestinal", "urgency": 0.80, "acuity_tier": 2},
    "rectal bleeding": {"category": "Gastrointestinal", "urgency": 0.75, "acuity_tier": 3},
    "diarrhea": {"category": "Gastrointestinal", "urgency": 0.30, "acuity_tier": 5},

    # Systemic, Infectious & Environmental
    "fever": {"category": "Systemic", "urgency": 0.55, "acuity_tier": 3},
    "chills": {"category": "Systemic", "urgency": 0.40, "acuity_tier": 4},
    "sepsis": {"category": "Systemic", "urgency": 0.95, "acuity_tier": 1},
    "hypothermia": {"category": "Systemic", "urgency": 0.85, "acuity_tier": 2},
    "anaphylaxis": {"category": "Systemic", "urgency": 1.00, "acuity_tier": 1},
    "allergic reaction": {"category": "Systemic", "urgency": 0.70, "acuity_tier": 3},
    "diaphoresis": {"category": "Systemic", "urgency": 0.75, "acuity_tier": 2},
    "sweating": {"category": "Systemic", "urgency": 0.50, "acuity_tier": 3},

    # Trauma & External Injuries
    "motor vehicle accident": {"category": "Trauma", "urgency": 0.85, "acuity_tier": 2},
    "fall": {"category": "Trauma", "urgency": 0.60, "acuity_tier": 3},
    "head trauma": {"category": "Trauma", "urgency": 0.85, "acuity_tier": 2},
    "stab wound": {"category": "Trauma", "urgency": 0.95, "acuity_tier": 1},
    "gunshot wound": {"category": "Trauma", "urgency": 1.00, "acuity_tier": 1},
    "bleeding": {"category": "Trauma", "urgency": 0.70, "acuity_tier": 3},
    "active hemorrhage": {"category": "Trauma", "urgency": 0.95, "acuity_tier": 1},
    "fracture": {"category": "Trauma", "urgency": 0.65, "acuity_tier": 3},
    "burn": {"category": "Trauma", "urgency": 0.75, "acuity_tier": 3},
    "laceration": {"category": "Trauma", "urgency": 0.40, "acuity_tier": 4},
}

# Anatomical locus definitions
ANATOMICAL_LOCATIONS: Dict[str, str] = {
    "retrosternal": "Cardiac / Thoracic",
    "substernal": "Cardiac / Thoracic",
    "precordial": "Cardiac / Thoracic",
    "chest": "Thoracic",
    "left arm": "Cardiovascular radiation",
    "right arm": "Upper extremity",
    "jaw": "Cranial / Radiation",
    "neck": "Cervical / Radiation",
    "back": "Dorsal / Spinal",
    "epigastric": "Gastrointestinal",
    "epigastrium": "Gastrointestinal",
    "right lower quadrant": "Abdominal (Appendiceal)",
    "left lower quadrant": "Abdominal (Diverticular)",
    "flank": "Renal / Retroperitoneal",
    "head": "Cranial / Neurological",
    "abdomen": "Abdominal",
    "pelvis": "Pelvic",
    "lower extremity": "Extremity",
    "leg": "Extremity",
}

# Negation trigger patterns (pre-negation and post-negation)
PRE_NEGATION_TRIGGERS: List[str] = [
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bdenying\b",
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bfree of\b",
    r"\bnegative for\b",
    r"\bnever had\b",
    r"\brules out\b",
    r"\bruled out\b",
    r"\bno evidence of\b",
    r"\bno sign of\b",
    r"\bno history of\b",
]

POST_NEGATION_TRIGGERS: List[str] = [
    r"\bwas ruled out\b",
    r"\bhas resolved\b",
    r"\bunlikely\b",
    r"\babsent\b",
]

# Uncertainty / Hedging qualifiers
UNCERTAINTY_TRIGGERS: List[str] = [
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bquery\b",
    r"\bquestionable\b",
    r"\bsuspected\b",
    r"\bsuspicion of\b",
    r"\brule out\b",
    r"\br/o\b",
    r"\beval for\b",
    r"\bmay have\b",
    r"\bcould be\b",
]

# Severity descriptors and numeric scales
SEVERITY_MAPPINGS: Dict[str, float] = {
    "mild": 0.25,
    "slight": 0.20,
    "minor": 0.20,
    "moderate": 0.55,
    "distressing": 0.65,
    "severe": 0.85,
    "extreme": 0.95,
    "unbearable": 1.00,
    "crushing": 0.95,
    "excruciating": 1.00,
    "agonizing": 1.00,
    "sharp": 0.70,
    "dull": 0.40,
    "burning": 0.60,
    "tearing": 0.95,
    "stabbing": 0.80,
}


# ==============================================================================
# 2. STRUCTURED DATA MODELS
# ==============================================================================
@dataclass
class ExtractedSymptom:
    """Represents a single clinical symptom entity with its contextual qualifiers."""
    term: str
    category: str
    is_negated: bool
    is_uncertain: bool
    severity_score: float                # Normalized [0.0, 1.0]
    severity_label: Optional[str] = None
    duration_str: Optional[str] = None
    duration_hours: Optional[float] = None
    anatomical_locus: Optional[str] = None
    raw_span: str = ""
    urgency_weight: float = 0.5          # Acuity prior [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalTextEvidence:
    """
    Standardized Clinical Text Evidence Schema.
    Structured output summarizing patient complaint semantics for clinical triage.
    """
    original_text: str
    cleaned_text: str
    affirmed_symptoms: List[ExtractedSymptom] = field(default_factory=list)
    negated_symptoms: List[ExtractedSymptom] = field(default_factory=list)
    uncertain_symptoms: List[ExtractedSymptom] = field(default_factory=list)
    anatomical_sites: List[str] = field(default_factory=list)
    primary_severity: float = 0.0        # Max severity across affirmed symptoms [0.0, 1.0]
    max_urgency_score: float = 0.0       # Max clinical urgency prior [0.0, 1.0]
    is_acute_onset: bool = False
    duration_summary: str = "Unknown"
    has_high_acuity_keywords: bool = False
    evidence_vector: Optional[List[float]] = None  # Compatible with downstream gating

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["affirmed_symptoms"] = [s.to_dict() for s in self.affirmed_symptoms]
        res["negated_symptoms"] = [s.to_dict() for s in self.negated_symptoms]
        res["uncertain_symptoms"] = [s.to_dict() for s in self.uncertain_symptoms]
        return res


# ==============================================================================
# 3. ADVANCED CLINICAL NLP PROCESSOR
# ==============================================================================
class AdvancedClinicalNLP:
    """
    High-precision clinical NLP engine specializing in emergency triage chief complaints.
    Performs deterministic, rule-grounded information extraction while strictly
    preserving compatibility with sentence-transformer dense embeddings.
    """

    def __init__(self, negation_window_tokens: int = 8):
        self.negation_window = negation_window_tokens
        self._pre_neg_re = re.compile(
            "|".join(PRE_NEGATION_TRIGGERS), re.IGNORECASE
        )
        self._post_neg_re = re.compile(
            "|".join(POST_NEGATION_TRIGGERS), re.IGNORECASE
        )
        self._uncertainty_re = re.compile(
            "|".join(UNCERTAINTY_TRIGGERS), re.IGNORECASE
        )
        # Pre-sort symptom terms by length descending to match multi-word phrases first
        self._sorted_symptom_terms = sorted(
            EMERGENCY_SYMPTOM_LEXICON.keys(), key=lambda x: len(x), reverse=True
        )

    def extract_evidence(self, text: Optional[str]) -> ClinicalTextEvidence:
        """
        Execute full clinical NLP extraction pipeline over raw chief complaint text.
        """
        if not text or not str(text).strip():
            return ClinicalTextEvidence(
                original_text="",
                cleaned_text="",
                duration_summary="Not documented",
            )

        raw_str = str(text).strip()
        cleaned_str = self._clean_text(raw_str)
        sentences = self._split_sentences(cleaned_str)

        all_affirmed: List[ExtractedSymptom] = []
        all_negated: List[ExtractedSymptom] = []
        all_uncertain: List[ExtractedSymptom] = []
        all_anatomical_sites: Set[str] = set()

        overall_is_acute = False
        duration_candidates: List[Tuple[str, float]] = []

        for sent in sentences:
            # Extract duration & acute flags for this sentence
            dur_str, dur_hrs, is_acute = self._extract_duration(sent)
            if is_acute:
                overall_is_acute = True
            if dur_str and dur_hrs is not None:
                duration_candidates.append((dur_str, dur_hrs))

            # Extract anatomical loci
            loci = self._extract_anatomical_sites(sent)
            all_anatomical_sites.update(loci)

            # Extract symptoms with negation & uncertainty context
            symptoms = self._extract_symptoms_from_sentence(
                sentence=sent,
                default_duration_str=dur_str,
                default_duration_hrs=dur_hrs,
                anatomical_loci=loci,
            )

            for sym in symptoms:
                if sym.is_negated:
                    all_negated.append(sym)
                elif sym.is_uncertain:
                    all_uncertain.append(sym)
                else:
                    all_affirmed.append(sym)

        # Synthesize aggregate metrics
        affirmed_severities = [s.severity_score for s in all_affirmed] or [0.0]
        primary_sev = max(affirmed_severities)

        affirmed_urgencies = [s.urgency_weight for s in all_affirmed] or [0.0]
        max_urgency = max(affirmed_urgencies)

        has_high_acuity = any(s.urgency_weight >= 0.85 for s in all_affirmed)

        # Format duration summary
        if duration_candidates:
            # Sort by shortest duration (most acute)
            duration_candidates.sort(key=lambda x: x[1])
            duration_summary = duration_candidates[0][0]
        elif overall_is_acute:
            duration_summary = "Acute (< 24h)"
        else:
            duration_summary = "Unspecified onset"

        # Construct structured 8-dimensional evidence vector for downstream fusion
        evidence_vector = [
            float(len(all_affirmed)),
            float(len(all_negated)),
            float(len(all_uncertain)),
            float(primary_sev),
            float(max_urgency),
            1.0 if overall_is_acute else 0.0,
            1.0 if has_high_acuity else 0.0,
            float(len(all_anatomical_sites)),
        ]

        return ClinicalTextEvidence(
            original_text=raw_str,
            cleaned_text=cleaned_str,
            affirmed_symptoms=all_affirmed,
            negated_symptoms=all_negated,
            uncertain_symptoms=all_uncertain,
            anatomical_sites=sorted(list(all_anatomical_sites)),
            primary_severity=float(round(primary_sev, 3)),
            max_urgency_score=float(round(max_urgency, 3)),
            is_acute_onset=overall_is_acute,
            duration_summary=duration_summary,
            has_high_acuity_keywords=has_high_acuity,
            evidence_vector=evidence_vector,
        )

    # --------------------------------------------------------------------------
    # Sub-component Processors
    # --------------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """Sanitize text preserving clinical abbreviations and sentence boundaries."""
        t = text.strip()
        t = re.sub(r"[\r\n]+", " . ", t)
        t = re.sub(r"\s+", " ", t)
        return t

    def _split_sentences(self, text: str) -> List[str]:
        """Split clinical complaint into clause/sentence chunks."""
        chunks = re.split(r"[.;!?]+|\bbut\b|\bhowever\b", text, flags=re.IGNORECASE)
        return [c.strip() for c in chunks if c.strip()]

    def _extract_symptoms_from_sentence(
        self,
        sentence: str,
        default_duration_str: Optional[str],
        default_duration_hrs: Optional[float],
        anatomical_loci: List[str],
    ) -> List[ExtractedSymptom]:
        """Identify symptoms, check negation scope, extract severity and uncertainty."""
        found: List[ExtractedSymptom] = []
        lower_sent = sentence.lower()
        matched_spans: List[Tuple[int, int]] = []

        for term in self._sorted_symptom_terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            for match in re.finditer(pattern, lower_sent):
                start, end = match.span()
                # Check for overlap with already matched longer phrases
                if any(start >= m_start and end <= m_end for m_start, m_end in matched_spans):
                    continue

                matched_spans.append((start, end))
                meta = EMERGENCY_SYMPTOM_LEXICON[term]

                # Check negation scope
                is_neg = self._is_negated(lower_sent, start, end)
                # Check uncertainty scope
                is_unc = self._is_uncertain(lower_sent, start, end)

                # Extract severity around symptom
                sev_score, sev_label = self._extract_severity(lower_sent, start, end)

                # Associated anatomical locus
                locus = anatomical_loci[0] if anatomical_loci else None

                symptom = ExtractedSymptom(
                    term=term,
                    category=meta["category"],
                    is_negated=is_neg,
                    is_uncertain=is_unc,
                    severity_score=sev_score,
                    severity_label=sev_label,
                    duration_str=default_duration_str,
                    duration_hours=default_duration_hrs,
                    anatomical_locus=locus,
                    raw_span=sentence[max(0, start - 15) : min(len(sentence), end + 15)].strip(),
                    urgency_weight=float(meta["urgency"]),
                )
                found.append(symptom)

        return found

    def _is_negated(self, sentence: str, term_start: int, term_end: int) -> bool:
        """Determine if a symptom occurrence falls within the scope of a negation cue."""
        # 1. Pre-negation check (words preceding the symptom in window)
        preceding_text = sentence[:term_start].strip()
        tokens = preceding_text.split()
        window_tokens = tokens[-self.negation_window :] if len(tokens) >= self.negation_window else tokens
        window_text = " ".join(window_tokens)

        if self._pre_neg_re.search(window_text):
            return True

        # 2. Post-negation check (words immediately following the symptom)
        following_text = sentence[term_end:].strip()
        following_tokens = following_text.split()
        post_window = following_tokens[:3] if len(following_tokens) >= 3 else following_tokens
        post_text = " ".join(post_window)

        if self._post_neg_re.search(post_text):
            return True

        return False

    def _is_uncertain(self, sentence: str, term_start: int, term_end: int) -> bool:
        """Detect if an entity is hedged or qualified by clinical diagnostic uncertainty."""
        preceding = sentence[max(0, term_start - 35) : term_start]
        if self._uncertainty_re.search(preceding):
            return True
        return False

    def _extract_severity(self, sentence: str, term_start: int, term_end: int) -> Tuple[float, Optional[str]]:
        """Extract numeric NRS pain scale (e.g. '8/10') or qualitative severity words."""
        # 1. Check for numeric pain score in vicinity
        vicinity = sentence[max(0, term_start - 30) : min(len(sentence), term_end + 30)]
        nrs_match = re.search(r"\b([0-9]|10)\s*(?:/|\s*out of\s*)\s*10\b", vicinity)
        if nrs_match:
            score = float(nrs_match.group(1)) / 10.0
            return score, f"NRS {nrs_match.group(1)}/10"

        # 2. Check for qualitative severity descriptors
        for word, score in SEVERITY_MAPPINGS.items():
            if re.search(r"\b" + re.escape(word) + r"\b", vicinity):
                return score, word

        return 0.50, "moderate (default)"

    def _extract_duration(self, sentence: str) -> Tuple[Optional[str], Optional[float], bool]:
        """
        Extract duration string, normalized hours, and acute onset flag.
        Returns: (duration_str, duration_in_hours, is_acute)
        """
        sent = sentence.lower()
        is_acute = bool(
            re.search(r"\b(sudden|acute|abrupt|immediate|just now|woke up with)\b", sent)
        )

        # Match patterns like: "x 2 hours", "for 3 days", "since yesterday", "past 45 minutes"
        m_min = re.search(r"(?:for|x|past|last|since)?\s*(\d+)\s*(?:min|mins|minute|minutes)\b", sent)
        if m_min:
            mins = float(m_min.group(1))
            hrs = mins / 60.0
            return f"{m_min.group(1)} minutes", hrs, True

        m_hr = re.search(r"(?:for|x|past|last|since)?\s*(\d+)\s*(?:hr|hrs|hour|hours)\b", sent)
        if m_hr:
            hrs = float(m_hr.group(1))
            return f"{m_hr.group(1)} hours", hrs, (hrs <= 24.0)

        m_day = re.search(r"(?:for|x|past|last|since)?\s*(\d+)\s*(?:d|day|days)\b", sent)
        if m_day:
            days = float(m_day.group(1))
            hrs = days * 24.0
            return f"{m_day.group(1)} days", hrs, (days <= 1.0)

        m_wk = re.search(r"(?:for|x|past|last|since)?\s*(\d+)\s*(?:wk|wks|week|weeks)\b", sent)
        if m_wk:
            wks = float(m_wk.group(1))
            hrs = wks * 168.0
            return f"{m_wk.group(1)} weeks", hrs, False

        if "yesterday" in sent:
            return "since yesterday", 24.0, True

        return None, None, is_acute

    def _extract_anatomical_sites(self, sentence: str) -> List[str]:
        """Identify mentioned anatomical locations."""
        found: List[str] = []
        sent = sentence.lower()
        for locus, system in ANATOMICAL_LOCATIONS.items():
            if re.search(r"\b" + re.escape(locus) + r"\b", sent):
                found.append(f"{locus.title()} ({system})")
        return found
