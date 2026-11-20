"""
Multimodal Intelligent Triage Assistant - Clinical Decision Support Command Center.
===================================================================================
Module 17: Interactive Streamlit Web Application uniting:
  - Tab 1: Live Multimodal Patient Intake & Real-Time Decision Support.
  - Tab 2: Clinical Case Study Benchmark Gallery.
  - Tab 3: Deep Learning Architecture & Scientific Governance.

Research & Educational Prototype Only.
"""

from __future__ import annotations

from dataclasses import asdict
import io
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure unpickling compatibility for preprocessor.joblib
import src.data.preprocessing as prep
for attr in ["ClinicalVitalsCleaner", "ClinicalContextCleaner", "PreprocessingPipeline"]:
    if hasattr(prep, attr):
        setattr(sys.modules.setdefault("__main__", sys.modules[__name__]), attr, getattr(prep, attr))

from src.integration.triage_orchestrator import (
    KTAS_DESCRIPTIONS,
    KTAS_LEVEL_NAMES,
    PatientIntakeRecord,
    TriageOrchestrator,
    TriageResult,
)
from src.vision.xray_model import generate_synthetic_chest_xray
from src.voice.speech_to_text import generate_synthetic_emergency_call

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & HIGH-END MEDICAL CSS THEME
# ==============================================================================
st.set_page_config(
    page_title="AI Emergency Triage Command Center | Multimodal Clinical CDS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom High-End Modern Medical Dark/Glassmorphic Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 98% !important;
    }

    /* Command Center Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #090D16 0%, #0F172A 45%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 28px;
        color: white;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 30%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.94rem;
        color: #94A3B8;
        margin-top: 6px;
        font-weight: 400;
        letter-spacing: -0.01em;
    }

    /* Glowing Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #E2E8F0;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }
    .status-pill:hover {
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.3);
    }
    .status-dot-active {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Animated ECG Line */
    .ecg-container {
        width: 100%;
        height: 28px;
        overflow: hidden;
        margin-top: 8px;
        opacity: 0.85;
    }
    .ecg-path {
        stroke: #06B6D4;
        stroke-width: 2;
        fill: none;
        stroke-dasharray: 600;
        stroke-dashoffset: 600;
        animation: ecg-sweep 3s linear infinite;
        filter: drop-shadow(0 0 4px #06B6D4);
    }
    @keyframes ecg-sweep {
        0% { stroke-dashoffset: 600; }
        100% { stroke-dashoffset: 0; }
    }

    /* Compact Disclaimer Strip */
    .disclaimer-strip {
        background: linear-gradient(90deg, rgba(254, 243, 199, 0.95), rgba(254, 249, 195, 0.95));
        border: 1px solid #FDE68A;
        border-left: 5px solid #F59E0B;
        border-radius: 10px;
        padding: 8px 16px;
        margin-bottom: 16px;
        font-size: 0.82rem;
        color: #92400E;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.08);
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.08);
    }
    .glass-card-header {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 8px;
    }

    /* ICU Telemetry Vital Tile */
    .icu-tile {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 12px;
        transition: all 0.2s ease;
    }
    .icu-tile:hover {
        background: #FFFFFF;
        border-color: #CBD5E1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.04);
    }
    .icu-tile-critical {
        background: #FEF2F2 !important;
        border-color: #FECACA !important;
        border-left: 4px solid #EF4444 !important;
    }
    .icu-tile-warning {
        background: #FFFBEB !important;
        border-color: #FDE68A !important;
        border-left: 4px solid #F59E0B !important;
    }
    .icu-tile-normal {
        border-left: 4px solid #10B981 !important;
    }
    .icu-tile-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.04em;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .icu-tile-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2px;
    }
    .icu-tile-unit {
        font-size: 0.76rem;
        font-weight: 500;
        color: #94A3B8;
        margin-left: 4px;
    }

    /* Decision Hero Banner */
    .decision-hero {
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 14px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
    }
    .decision-hero-1 {
        background: linear-gradient(135deg, #7F1D1D 0%, #DC2626 100%);
        border: 2px solid #EF4444;
    }
    .decision-hero-2 {
        background: linear-gradient(135deg, #7C2D12 0%, #EA580C 100%);
        border: 2px solid #F97316;
    }
    .decision-hero-3 {
        background: linear-gradient(135deg, #78350F 0%, #D97706 100%);
        border: 2px solid #FBBF24;
    }
    .decision-hero-4 {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        border: 2px solid #3B82F6;
    }
    .decision-hero-5 {
        background: linear-gradient(135deg, #064E3B 0%, #059669 100%);
        border: 2px solid #10B981;
    }

    /* KTAS Badges */
    .badge-pill-1 { background: #EF4444; color: white; padding: 6px 14px; border-radius: 9999px; font-weight: 800; font-size: 0.92rem; box-shadow: 0 0 12px rgba(239, 68, 68, 0.6); }
    .badge-pill-2 { background: #F97316; color: white; padding: 6px 14px; border-radius: 9999px; font-weight: 800; font-size: 0.92rem; box-shadow: 0 0 12px rgba(249, 115, 22, 0.6); }
    .badge-pill-3 { background: #FBBF24; color: #1E293B; padding: 6px 14px; border-radius: 9999px; font-weight: 800; font-size: 0.92rem; box-shadow: 0 0 12px rgba(251, 191, 36, 0.6); }
    .badge-pill-4 { background: #3B82F6; color: white; padding: 6px 14px; border-radius: 9999px; font-weight: 800; font-size: 0.92rem; box-shadow: 0 0 12px rgba(59, 130, 246, 0.6); }
    .badge-pill-5 { background: #10B981; color: white; padding: 6px 14px; border-radius: 9999px; font-weight: 800; font-size: 0.92rem; box-shadow: 0 0 12px rgba(16, 185, 129, 0.6); }

    /* Radiological Film Lightbox Frame */
    .xray-lightbox {
        background: #090D16;
        border: 2px solid #1E293B;
        border-radius: 12px;
        padding: 12px;
        color: #E2E8F0;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
    }
    .xray-meta-tag {
        font-size: 0.72rem;
        color: #06B6D4;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* Red Flag Chip */
    .redflag-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.84rem;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }

    /* Primary CTA Button Polish */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.55) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing Deep Learning Encoders & Calibration Models...")
def get_triage_orchestrator() -> TriageOrchestrator:
    """Load and cache the master orchestrator in memory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TriageOrchestrator(
        config_path="config.yaml",
        device=device,
        load_whisper=True,
        load_vision=True,
    )


# ==============================================================================
# HERO HEADER & DISCLOSURE
# ==============================================================================
device_label = "CUDA (GPU)" if torch.cuda.is_available() else "CPU (Standard)"

st.markdown(
    f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
            <div>
                <h1 class="hero-title">🏥 Multimodal Intelligent Triage Assistant</h1>
                <div class="hero-subtitle">
                    Autonomous Emergency Decision Support &bull; Audio Whisper ASR &bull; MiniLM Semantic Embeddings &bull; Vitals MLP &bull; Monte Carlo Calibration &bull; Decoupled X-ray Grad-CAM
                </div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="status-pill"><span class="status-dot-active"></span>{device_label}</span>
                <span class="status-pill">🎙️ Whisper Online</span>
                <span class="status-pill">🩻 DenseNet-121</span>
                <span class="status-pill">🎲 MC Dropout (T=20)</span>
            </div>
        </div>
        <div class="ecg-container">
            <svg class="ecg-line" viewBox="0 0 600 30" preserveAspectRatio="none">
                <path class="ecg-path" d="M0,15 L140,15 L150,8 L160,22 L170,15 L190,15 L198,3 L206,27 L214,15 L320,15 L328,10 L336,20 L344,15 L600,15" />
            </svg>
        </div>
    </div>
    <div class="disclaimer-strip">
        <span style="font-size: 1.15rem;">⚠️</span>
        <div>
            <strong>SCIENTIFIC RESEARCH & EDUCATIONAL PROTOTYPE ONLY:</strong>
            This software is an investigational deep learning research platform. It is <strong>NOT</strong> an FDA/CE-cleared medical device, does not provide medical diagnoses, and cannot substitute for the clinical judgment of a licensed physician or triage nurse. All outputs must be verified by clinical personnel.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Orchestrator
try:
    orchestrator = get_triage_orchestrator()
except Exception as e:
    st.error(f"Failed to initialize Triage Orchestrator: {e}")
    st.stop()


# ==============================================================================
# HELPER FUNCTIONS FOR PLOTLY VISUALIZATIONS
# ==============================================================================
def create_ktas_spectrum_chart(probabilities: List[float], final_ktas: int) -> go.Figure:
    """Create a sleek modern Plotly horizontal probability spectrum bar chart."""
    levels = ["KTAS 5 (Non)", "KTAS 4 (Less)", "KTAS 3 (Urgent)", "KTAS 2 (Emerg)", "KTAS 1 (Resus)"]
    probs = [probabilities[4] * 100, probabilities[3] * 100, probabilities[2] * 100, probabilities[1] * 100, probabilities[0] * 100]
    palette = ["#10B981", "#3B82F6", "#FBBF24", "#F97316", "#EF4444"]
    border_widths = [3 if (5 - i) == final_ktas else 1 for i in range(5)]
    border_colors = ["#FFFFFF" if (5 - i) == final_ktas else "rgba(255,255,255,0.2)" for i in range(5)]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=levels,
            orientation="h",
            marker=dict(
                color=palette,
                line=dict(color=border_colors, width=border_widths),
                opacity=0.92,
            ),
            text=[f"  {p:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=12, color="#0F172A", weight="bold"),
            hoverinfo="x+y",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        height=185,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="#E2E8F0",
            range=[0, max(max(probs) * 1.25, 30)],
            ticksuffix="%",
            title_text="",
            tickfont=dict(size=11, color="#64748B"),
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#0F172A", family="Plus Jakarta Sans", weight="bold"),
        ),
    )
    return fig


def create_dual_uncertainty_gauge(confidence: float, entropy: float) -> go.Figure:
    """Create dual circular indicator dials for Model Confidence and Calibrated Entropy."""
    fig = go.Figure()

    # Gauge 1: Model Confidence
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={"x": [0, 0.48], "y": [0, 1]},
            title={"text": "<b>MODEL CONFIDENCE</b>", "font": {"size": 13, "color": "#0F172A"}},
            number={"suffix": "%", "font": {"size": 26, "color": "#2563EB", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": "#2563EB"},
                "bgcolor": "#F1F5F9",
                "steps": [
                    {"range": [0, 40], "color": "#FEE2E2"},
                    {"range": [40, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#DCFCE7"},
                ],
            },
        )
    )

    # Gauge 2: Predictive Entropy (Shannon H / ln 5)
    entropy_color = "#DC2626" if entropy > 0.82 else ("#D97706" if entropy > 0.65 else "#059669")
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=entropy,
            domain={"x": [0.52, 1], "y": [0, 1]},
            title={"text": "<b>PREDICTIVE ENTROPY</b>", "font": {"size": 13, "color": "#0F172A"}},
            number={"valueformat": ".2f", "font": {"size": 26, "color": entropy_color, "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 1.0], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": entropy_color},
                "bgcolor": "#F1F5F9",
                "threshold": {
                    "line": {"color": "#DC2626", "width": 3},
                    "thickness": 0.8,
                    "value": 0.82,
                },
                "steps": [
                    {"range": [0, 0.65], "color": "#DCFCE7"},
                    {"range": [0.65, 0.82], "color": "#FEF3C7"},
                    {"range": [0.82, 1.0], "color": "#FEE2E2"},
                ],
            },
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=25, b=5),
        height=145,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_modality_reliance_donut(weights: Dict[str, float]) -> go.Figure:
    """Create a sleek modern Plotly Donut Chart for Adaptive Modality Weights."""
    labels = ["Vitals Stream", "Text Stream", "Context Stream"]
    vals = [weights.get("Vitals", 0.33), weights.get("Text", 0.33), weights.get("Context", 0.33)]
    colors = ["#3B82F6", "#06B6D4", "#8B5CF6"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=vals,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                textposition="outside",
                textfont=dict(size=11, family="Plus Jakarta Sans", color="#1E293B"),
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=180,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text="<b>RELIANCE</b>",
                x=0.5,
                y=0.5,
                font_size=12,
                font_family="Plus Jakarta Sans",
                font_color="#475569",
                showarrow=False,
            )
        ],
    )
    return fig


# ==============================================================================
# MAIN TABS INTERFACE
# ==============================================================================
tab_live, tab_benchmarks, tab_architecture = st.tabs(
    ["🩺 Live Patient Triage", "📋 Case Studies & Benchmarks", "🧠 Architecture & Transparency"]
)


# ==============================================================================
# TAB 1: LIVE PATIENT TRIAGE (SYMMETRICAL MEDICAL COCKPIT)
# ==============================================================================
with tab_live:
    # --------------------------------------------------------------------------
    # 1. Quick Clinical Presets Interactive Bar
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #475569; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;'>⚡ 1-Click Clinical Intake Presets:</div>", unsafe_allow_html=True)

    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)

    presets = {
        "p1": {
            "name": "🚨 Shock (KTAS 1)",
            "pid": "PRESET-SHOCK-01", "age": 72, "sex": "Male", "arr": "119", "inj": "Non-trauma", "avpu": "Unresponsive", "pph": 4.0,
            "sbp": 68, "dbp": 38, "hr": 142, "rr": 36, "bt": 35.2, "spo2": 82, "pain": False, "nrs": 0,
            "complaint": "cardiac collapse, agonal respirations, unresponsive on arrival", "xray": "None",
        },
        "p2": {
            "name": "⚠️ Cardiac (KTAS 2)",
            "pid": "PRESET-ACS-02", "age": 58, "sex": "Male", "arr": "119", "inj": "Non-trauma", "avpu": "Alert", "pph": 3.5,
            "sbp": 156, "dbp": 98, "hr": 108, "rr": 22, "bt": 36.9, "spo2": 94, "pain": True, "nrs": 9,
            "complaint": "crushing retrosternal chest pressure radiating to left arm with diaphoresis", "xray": "None",
        },
        "p3": {
            "name": "🟡 Asthma (KTAS 3)",
            "pid": "PRESET-ASTHMA-03", "age": 28, "sex": "Female", "arr": "Walking", "inj": "Non-trauma", "avpu": "Alert", "pph": 3.0,
            "sbp": 128, "dbp": 82, "hr": 102, "rr": 26, "bt": 37.1, "spo2": 93, "pain": False, "nrs": 0,
            "complaint": "acute asthma exacerbation, expiratory wheezing, speaks in short sentences", "xray": "None",
        },
        "p4": {
            "name": "🔵 Sprain (KTAS 4)",
            "pid": "PRESET-TRAUMA-04", "age": 24, "sex": "Female", "arr": "Walking", "inj": "Trauma", "avpu": "Alert", "pph": 2.5,
            "sbp": 118, "dbp": 76, "hr": 78, "rr": 16, "bt": 36.6, "spo2": 99, "pain": True, "nrs": 4,
            "complaint": "inversion injury right ankle while playing basketball, swelling without deformity", "xray": "None",
        },
        "p5": {
            "name": "🩻 Pneumothorax (X-ray)",
            "pid": "PRESET-PNEUMO-05", "age": 22, "sex": "Male", "arr": "Private car", "inj": "Non-trauma", "avpu": "Alert", "pph": 3.0,
            "sbp": 114, "dbp": 72, "hr": 92, "rr": 20, "bt": 36.6, "spo2": 96, "pain": True, "nrs": 6,
            "complaint": "sharp sudden right-sided pleuritic chest pain on inspiration", "xray": "Acute Tension Pneumothorax",
        },
    }

    with col_p1:
        if st.button("🚨 Shock (KTAS 1)", use_container_width=True, help="Profound cardiogenic shock with hypoxia"):
            st.session_state["active_preset"] = presets["p1"]
            st.session_state["auto_trigger"] = True
    with col_p2:
        if st.button("⚠️ Cardiac (KTAS 2)", use_container_width=True, help="Acute coronary syndrome / severe chest pain"):
            st.session_state["active_preset"] = presets["p2"]
            st.session_state["auto_trigger"] = True
    with col_p3:
        if st.button("🟡 Asthma (KTAS 3)", use_container_width=True, help="Moderate bronchospasm and tachypnea"):
            st.session_state["active_preset"] = presets["p3"]
            st.session_state["auto_trigger"] = True
    with col_p4:
        if st.button("🔵 Sprain (KTAS 4)", use_container_width=True, help="Stable musculoskeletal trauma"):
            st.session_state["active_preset"] = presets["p4"]
            st.session_state["auto_trigger"] = True
    with col_p5:
        if st.button("🩻 Pneumothorax", use_container_width=True, help="Auxiliary radiograph tension pneumothorax escalation"):
            st.session_state["active_preset"] = presets["p5"]
            st.session_state["auto_trigger"] = True

    preset_data = st.session_state.get("active_preset", None)

    st.markdown("<hr style='margin: 12px 0 18px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Symmetrical 2-Column Clinical Layout
    # --------------------------------------------------------------------------
    col_input, col_output = st.columns([1.0, 1.15], gap="large")

    # ==========================================================================
    # LEFT COLUMN: INTAKE TELEMETRY & CONSOLE
    # ==========================================================================
    with col_input:
        st.markdown("<div style='font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-bottom: 12px;'>📥 1. Patient Intake Record</div>", unsafe_allow_html=True)

        # 1. Chief Complaint & Audio Stream
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-header">
                    <span>🗣️ Chief Complaint & Voice Intake</span>
                    <span style="font-size: 0.78rem; font-weight: 600; color: #3B82F6;">NLP MiniLM / Whisper</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        input_mode = st.radio(
            "Intake Channel",
            ["Text Entry", "Spoken Voice Dispatch (Whisper ASR)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        chief_text_val = preset_data["complaint"] if preset_data else "crushing substernal chest pressure radiating to left jaw with diaphoresis"
        audio_source = None

        if input_mode == "Text Entry":
            chief_text = st.text_area(
                "Triage Nurse Chief Complaint",
                value=chief_text_val,
                height=72,
                placeholder="Enter patient primary symptoms and presentation...",
                label_visibility="collapsed",
            )
        else:
            sample_voice = st.selectbox(
                "Select Dispatch Audio Recording",
                ["Emergency Cardiac Call (Chest Pain)", "Generate Procedural Radio Transmission"],
            )
            if sample_voice == "Emergency Cardiac Call (Chest Pain)":
                audio_source = PROJECT_ROOT / "data" / "raw" / "voice" / "case2_emergency_dispatch.wav"
                if audio_source.exists():
                    st.audio(str(audio_source))
                chief_text = "crushing substernal chest pain radiating to left arm with diaphoresis"
            else:
                audio_arr = generate_synthetic_emergency_call(duration_sec=2.5, sample_rate=16000)
                audio_source = audio_arr
                st.info("🎙️ Procedural 16kHz radio audio synthesized in memory buffer.")
                chief_text = "dispatch incoming patient in severe distress"

        # 2. Physiological Vitals Grid
        st.markdown(
            """
            <div class="glass-card" style="margin-top: 14px;">
                <div class="glass-card-header">
                    <span>💓 Physiological Vital Signs</span>
                    <span style="font-size: 0.78rem; font-weight: 600; color: #10B981;">ICU Telemetry</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cv1, cv2, cv3 = st.columns(3)
        with cv1:
            sbp_val = preset_data["sbp"] if preset_data else 145
            dbp_val = preset_data["dbp"] if preset_data else 92
            sbp = st.number_input("Systolic BP (mmHg)", min_value=40, max_value=280, value=sbp_val, step=5)
            dbp = st.number_input("Diastolic BP (mmHg)", min_value=20, max_value=180, value=dbp_val, step=5)
        with cv2:
            hr_val = preset_data["hr"] if preset_data else 104
            rr_val = preset_data["rr"] if preset_data else 22
            hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=hr_val, step=2)
            rr = st.number_input("Resp Rate (bpm)", min_value=6, max_value=60, value=rr_val, step=1)
        with cv3:
            bt_val = preset_data["bt"] if preset_data else 36.8
            spo2_val = preset_data["spo2"] if preset_data else 94
            bt = st.number_input("Temp (°C)", min_value=30.0, max_value=43.0, value=bt_val, step=0.1)
            spo2_unmeasured = st.checkbox("SpO2 Unmeasured", value=False)
            spo2 = None if spo2_unmeasured else st.number_input("SpO2 (%)", min_value=50, max_value=100, value=spo2_val, step=1)

        cp1, cp2 = st.columns([1, 2])
        with cp1:
            pain_present_val = preset_data["pain"] if preset_data else True
            pain_present = st.checkbox("Pain Reported", value=pain_present_val)
        with cp2:
            nrs_val = preset_data["nrs"] if preset_data else 8
            nrs_pain = st.slider("Pain Score (NRS 0-10)", 0, 10, nrs_val, label_visibility="collapsed") if pain_present else 0.0

        # 3. Demographics & Presentation Context
        st.markdown(
            """
            <div class="glass-card" style="margin-top: 14px;">
                <div class="glass-card-header">
                    <span>👤 Demographics & Presentation Context</span>
                    <span style="font-size: 0.78rem; font-weight: 600; color: #8B5CF6;">Context Embeddings</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            pid_val = preset_data["pid"] if preset_data else "EM-LIVE-001"
            pid = st.text_input("Patient ID", value=pid_val)
            age_val = preset_data["age"] if preset_data else 56
            age = st.number_input("Age", min_value=1, max_value=105, value=age_val)
        with cd2:
            sex = st.selectbox("Sex", ["Male", "Female"], index=0)
            arr_list = ["119 (Ambulance)", "Walking", "Private car", "Other"]
            arr_idx = 0 if not preset_data else (0 if preset_data["arr"] == "119" else (1 if preset_data["arr"] == "Walking" else 2))
            arrival_mode = st.selectbox("Arrival Mode", arr_list, index=arr_idx)
        with cd3:
            inj_list = ["Non-trauma (Medical)", "Trauma (Injury)"]
            inj_idx = 1 if (preset_data and preset_data["inj"] == "Trauma") else 0
            injury = st.selectbox("Type", inj_list, index=inj_idx)
            avpu_list = ["Alert", "Verbal", "Pain", "Unresponsive"]
            avpu_idx = 0 if not preset_data else (3 if preset_data["avpu"] == "Unresponsive" else 0)
            mental = st.selectbox("Mental Status", avpu_list, index=avpu_idx)

        # 4. Decoupled Radiograph
        with st.expander("🩻 Auxiliary Decoupled Chest Radiograph (Optional)", expanded=bool(preset_data and preset_data["xray"] != "None")):
            st.caption("Decoupled from initial $t=0$ triage vector to eliminate post-admission data leakage.")
            xray_options = [
                "None (Intake Only)",
                "Acute Tension Pneumothorax",
                "Congestive Cardiomegaly",
                "Lobar Consolidation (Pneumonia)",
                "Acute Pulmonary Edema",
                "Normal Clear Radiograph",
            ]
            default_xray_idx = 1 if (preset_data and "Pneumo" in preset_data["xray"]) else 0
            xray_choice = st.selectbox("Select Radiograph Study", xray_options, index=default_xray_idx)

            xray_input = None
            if xray_choice == "Acute Tension Pneumothorax":
                xray_input = generate_synthetic_chest_xray("pneumothorax", image_size=224, random_seed=902)
                st.image(xray_input, caption="Synthetic: Tension Pneumothorax", width=170)
            elif xray_choice == "Congestive Cardiomegaly":
                xray_input = generate_synthetic_chest_xray("cardiomegaly", image_size=224, random_seed=903)
                st.image(xray_input, caption="Synthetic: Cardiomegaly", width=170)
            elif xray_choice == "Lobar Consolidation (Pneumonia)":
                xray_input = generate_synthetic_chest_xray("consolidation", image_size=224, random_seed=904)
                st.image(xray_input, caption="Synthetic: Lobar Consolidation", width=170)
            elif xray_choice == "Acute Pulmonary Edema":
                xray_input = generate_synthetic_chest_xray("edema", image_size=224, random_seed=905)
                st.image(xray_input, caption="Synthetic: Pulmonary Edema", width=170)
            elif xray_choice == "Normal Clear Radiograph":
                xray_input = generate_synthetic_chest_xray("normal", image_size=224, random_seed=901)
                st.image(xray_input, caption="Synthetic: Normal Clear", width=170)

        # Action Trigger
        run_button = st.button("🚀 Run Multimodal Triage Assessment", type="primary", use_container_width=True)
        auto_run = st.session_state.pop("auto_trigger", False)

    # ==========================================================================
    # RIGHT COLUMN: CLINICAL DECISION SUPPORT & LIVE TELEMETRY
    # ==========================================================================
    with col_output:
        st.markdown("<div style='font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-bottom: 12px;'>📊 2. Clinical Decision Support & Transparency</div>", unsafe_allow_html=True)

        should_evaluate = run_button or auto_run or ("last_result" in st.session_state and not preset_data)

        if should_evaluate:
            clean_arr = arrival_mode.split(" (")[0]
            clean_inj = "Trauma" if "Trauma" in injury and "Non" not in injury else "Non-trauma"

            record = PatientIntakeRecord(
                patient_id=pid,
                age=float(age),
                sex=sex,
                arrival_mode=clean_arr,
                injury=clean_inj,
                mental_status=mental,
                patients_per_hour=3.5,
                sbp=float(sbp),
                dbp=float(dbp),
                hr=float(hr),
                rr=float(rr),
                bt=float(bt),
                saturation=float(spo2) if spo2 is not None else None,
                pain=1 if pain_present else 0,
                nrs_pain=float(nrs_pain) if pain_present else 0.0,
                chief_complaint_text=chief_text,
                voice_audio_path=audio_source if isinstance(audio_source, (str, Path)) else None,
                voice_audio_array=audio_source if isinstance(audio_source, np.ndarray) else None,
                chest_xray_input=xray_input,
            )

            with st.spinner("Executing Multimodal Encoders, Cross-Modal Gating, and 20 MC Dropout Passes..."):
                result: TriageResult = orchestrator.triage_patient(record, generate_plots=True)
                st.session_state["last_result"] = result

            # ------------------------------------------------------------------
            # 1. Decision Hero Banner
            # ------------------------------------------------------------------
            lvl = result.final_ktas_level
            st.markdown(
                f"""
                <div class="decision-hero decision-hero-{lvl}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.9;">
                                Recommended KTAS Triage Acuity
                            </div>
                            <h2 style="margin: 4px 0 0 0; font-size: 1.85rem; font-weight: 800; color: white;">
                                {result.final_ktas_name}
                            </h2>
                        </div>
                        <span class="badge-pill-{lvl}">{result.triage_status_badge}</span>
                    </div>
                    <div style="margin-top: 14px; font-size: 0.96rem; line-height: 1.5; background: rgba(0,0,0,0.25); padding: 12px 16px; border-radius: 10px; border-left: 4px solid rgba(255,255,255,0.6);">
                        <strong>Clinical Directive:</strong> {result.clinical_recommendation}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------------------------
            # 2. Interactive Dual Gauge: Confidence & Calibrated Entropy
            # ------------------------------------------------------------------
            st.plotly_chart(create_dual_uncertainty_gauge(result.confidence, result.predictive_entropy), use_container_width=True)

            if result.requires_human_review:
                st.error(f"🚨 **CLINICAL SAFETY ALERT:** {result.safety_alert_message}")

            # ------------------------------------------------------------------
            # 3. Interactive Probability Spectrum Bar Chart
            # ------------------------------------------------------------------
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #1E293B; margin: 10px 0 2px 0;'>KTAS Posterior Probability Spectrum (20 Stochastic Passes)</div>", unsafe_allow_html=True)
            st.plotly_chart(create_ktas_spectrum_chart(result.intake_probabilities, result.final_ktas_level), use_container_width=True)

            # ------------------------------------------------------------------
            # 4. Modality Reliance Donut & Primary Clinical Red Flags
            # ------------------------------------------------------------------
            c_mod, c_drvs = st.columns([1, 1.25])
            with c_mod:
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #1E293B; margin-bottom: 2px;'>Adaptive Attention Reliance</div>", unsafe_allow_html=True)
                st.plotly_chart(create_modality_reliance_donut(result.modality_weights), use_container_width=True)
            with c_drvs:
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #1E293B; margin-bottom: 6px;'>Primary Diagnostic Red Flags</div>", unsafe_allow_html=True)
                chips_html = []
                for drv in result.top_clinical_drivers:
                    chips_html.append(f"<span class='redflag-chip'>🚨 <b>{drv['factor']}</b>: {drv['detail']}</span>")
                st.markdown("<div>" + "".join(chips_html) + "</div>", unsafe_allow_html=True)

            # ------------------------------------------------------------------
            # 5. Radiological Lightbox (Grad-CAM Saliency)
            # ------------------------------------------------------------------
            if result.xray_evaluated:
                st.markdown("<hr style='margin: 14px 0 10px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="xray-lightbox">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E293B; padding-bottom: 6px; margin-bottom: 8px;">
                            <span class="xray-meta-tag">🩻 RADIOLOGICAL FILM WORKSTATION (NIH-14 CHEXNET)</span>
                            <span style="font-size: 0.74rem; color: #10B981;">CALIBRATED DENSENET-121</span>
                        </div>
                        <div style="font-size: 0.86rem; margin-bottom: 6px;">
                            <strong>Primary Pathology:</strong> <span style="color: #38BDF8;">{result.primary_xray_finding}</span> ({result.xray_confidence*100:.1f}%) &bull; 
                            <strong>Severity Tier:</strong> <span style="color: #FBBF24;">{result.xray_acuity_tier}</span> (RSI: {result.radiographic_severity_index:.2f}) &bull;
                            <strong>Zone:</strong> <span style="color: #A78BFA;">{result.xray_anatomical_zone}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if result.escalation_reason:
                    st.warning(f"🚨 **IMAGING MODIFIER:** {result.escalation_reason}")

                if result.xray_overlay_path and Path(result.xray_overlay_path).exists():
                    st.image(result.xray_overlay_path, caption=f"Grad-CAM Heatmap: {result.primary_xray_finding} ({result.xray_anatomical_zone})", use_container_width=True)

            # ------------------------------------------------------------------
            # 6. EHR Narrative Export
            # ------------------------------------------------------------------
            with st.expander("📄 Electronic Health Record (EHR) Clinical Summary", expanded=False):
                st.code(result.clinical_narrative, language="text")

        else:
            # ==================================================================
            # TELEMETRY MONITOR & LIVE VITALS RADAR (BEFORE SUBMISSION)
            # ==================================================================
            st.markdown(
                """
                <div class="glass-card" style="border-top: 4px solid #06B6D4;">
                    <div class="glass-card-header">
                        <span>⚡ Live Telemetry & Clinical Parameter Monitor</span>
                        <span class="status-pill"><span class="status-dot-active"></span>Telemetry Online</span>
                    </div>
                    <p style="font-size: 0.86rem; color: #64748B; margin-top: 2px;">
                        Continuous physiological boundary monitoring for active console settings:
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 4 ICU Telemetry Tiles
            t1, t2 = st.columns(2)
            with t1:
                bp_class = "icu-tile-critical" if (sbp < 90 or sbp > 180 or dbp < 50 or dbp > 110) else "icu-tile-normal"
                bp_tag = "CRITICAL HYPOTENSION" if sbp < 90 else ("HYPERTENSIVE CRISIS" if sbp > 180 else "NORMAL RANGE")
                st.markdown(
                    f"""
                    <div class="icu-tile {bp_class}">
                        <div class="icu-tile-label"><span>🩺 Blood Pressure</span><span style="font-size: 0.68rem;">{bp_tag}</span></div>
                        <div class="icu-tile-val">{sbp}/{dbp}<span class="icu-tile-unit">mmHg</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                hr_class = "icu-tile-critical" if (hr > 120 or hr < 45) else ("icu-tile-warning" if hr > 100 else "icu-tile-normal")
                hr_tag = "TACHYCARDIA" if hr > 100 else ("BRADYCARDIA" if hr < 50 else "NORMAL RHYTHM")
                st.markdown(
                    f"""
                    <div class="icu-tile {hr_class}" style="margin-top: 10px;">
                        <div class="icu-tile-label"><span>❤️ Heart Rate</span><span style="font-size: 0.68rem;">{hr_tag}</span></div>
                        <div class="icu-tile-val">{hr}<span class="icu-tile-unit">bpm</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with t2:
                spo2_class = "icu-tile-critical" if (spo2 is not None and spo2 < 90) else ("icu-tile-warning" if (spo2 is None or spo2 < 95) else "icu-tile-normal")
                spo2_tag = "SEVERE HYPOXIA" if (spo2 is not None and spo2 < 90) else ("UNMEASURED" if spo2 is None else "ADEQUATE SAT")
                spo2_disp = f"{spo2}%" if spo2 is not None else "SENSOR DROP"
                st.markdown(
                    f"""
                    <div class="icu-tile {spo2_class}">
                        <div class="icu-tile-label"><span>🫧 Oxygen Saturation</span><span style="font-size: 0.68rem;">{spo2_tag}</span></div>
                        <div class="icu-tile-val">{spo2_disp}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                temp_class = "icu-tile-critical" if (bt > 38.5 or bt < 35.0) else "icu-tile-normal"
                temp_tag = "FEBRILE" if bt > 38.0 else ("HYPOTHERMIC" if bt < 35.5 else "NORMOTHERMIC")
                st.markdown(
                    f"""
                    <div class="icu-tile {temp_class}" style="margin-top: 10px;">
                        <div class="icu-tile-label"><span>🌡️ Body Temperature</span><span style="font-size: 0.68rem;">{temp_tag}</span></div>
                        <div class="icu-tile-val">{bt}<span class="icu-tile-unit">°C</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # KTAS Reference Guide Matrix
            st.markdown(
                """
                <div class="glass-card" style="margin-top: 14px;">
                    <div class="glass-card-header">
                        <span>📋 Emergency KTAS Acuity Quick Reference Matrix</span>
                        <span style="font-size: 0.78rem; font-weight: 600; color: #475569;">Korean Triage Scale</span>
                    </div>
                    <table style="width: 100%; font-size: 0.82rem; border-collapse: collapse; text-align: left;">
                        <tr style="border-bottom: 2px solid #E2E8F0; color: #475569;">
                            <th style="padding: 6px 8px;">Level</th>
                            <th style="padding: 6px 8px;">Classification</th>
                            <th style="padding: 6px 8px;">Max Wait Time</th>
                            <th style="padding: 6px 8px;">Clinical Presentation</th>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding: 6px 8px;"><span class="badge-pill-1" style="font-size: 0.72rem; padding: 2px 8px;">KTAS 1</span></td>
                            <td style="padding: 6px 8px; font-weight: 700; color: #991B1B;">Resuscitation</td>
                            <td style="padding: 6px 8px; color: #DC2626; font-weight: 700;">Immediate</td>
                            <td style="padding: 6px 8px; color: #64748B;">Cardiac arrest, severe shock, coma</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding: 6px 8px;"><span class="badge-pill-2" style="font-size: 0.72rem; padding: 2px 8px;">KTAS 2</span></td>
                            <td style="padding: 6px 8px; font-weight: 700; color: #C2410C;">Emergent</td>
                            <td style="padding: 6px 8px; color: #EA580C; font-weight: 700;">&le; 15 min</td>
                            <td style="padding: 6px 8px; color: #64748B;">Acute coronary syndrome, stroke, severe hypoxia</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding: 6px 8px;"><span class="badge-pill-3" style="font-size: 0.72rem; padding: 2px 8px;">KTAS 3</span></td>
                            <td style="padding: 6px 8px; font-weight: 700; color: #B45309;">Urgent</td>
                            <td style="padding: 6px 8px; color: #D97706; font-weight: 700;">&le; 30 min</td>
                            <td style="padding: 6px 8px; color: #64748B;">Moderate asthma, severe pain (NRS &ge; 7)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding: 6px 8px;"><span class="badge-pill-4" style="font-size: 0.72rem; padding: 2px 8px;">KTAS 4</span></td>
                            <td style="padding: 6px 8px; font-weight: 700; color: #1D4ED8;">Less Urgent</td>
                            <td style="padding: 6px 8px; color: #2563EB; font-weight: 700;">&le; 60 min</td>
                            <td style="padding: 6px 8px; color: #64748B;">Simple fracture, mild laceration, UTI</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 8px;"><span class="badge-pill-5" style="font-size: 0.72rem; padding: 2px 8px;">KTAS 5</span></td>
                            <td style="padding: 6px 8px; font-weight: 700; color: #047857;">Non-Urgent</td>
                            <td style="padding: 6px 8px; color: #059669; font-weight: 700;">&le; 120 min</td>
                            <td style="padding: 6px 8px; color: #64748B;">Suture removal, medication refill, mild cold</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info("👈 **Ready for Inference:** Click **'Run Multimodal Triage Assessment'** or select any **1-Click Clinical Preset** above to initiate deep learning evaluation.")


# ==============================================================================
# TAB 2: CASE STUDIES & BENCHMARKS
# ==============================================================================
with tab_benchmarks:
    st.markdown("<div style='font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;'>📋 Clinical Case Study Benchmark Gallery</div>", unsafe_allow_html=True)
    st.markdown("Explore pre-validated multi-pipeline emergency scenarios exhibiting sensor missingness, speech dictation, ambiguous constitutional symptoms, and independent radiographic escalation.")

    cases = [
        {
            "title": "Case 1: Severe Hemodynamic Shock (KTAS 1 Resuscitation)",
            "summary": "Elderly patient in profound shock (BP 72/42, HR 138, SpO2 84%, Unresponsive). Vitals dominance leads to immediate KTAS 1 assignment.",
            "record": PatientIntakeRecord(
                patient_id="CASE-SHOCK-01",
                age=68.0,
                sex="Male",
                arrival_mode="119",
                injury="Non-trauma",
                mental_status="Unresponsive",
                sbp=72.0,
                dbp=42.0,
                hr=138.0,
                rr=34.0,
                bt=35.4,
                saturation=84.0,
                pain=0,
                chief_complaint_text="cardiac arrest, agonal gasping, unresponsive on arrival",
            ),
        },
        {
            "title": "Case 2: Emergency Dispatch Audio Call (Whisper KTAS 2)",
            "summary": "Audio recording from emergency ambulance dispatch. Whisper speech-to-text transcribes acute coronary symptoms and assigns KTAS 2 Emergent.",
            "record": PatientIntakeRecord(
                patient_id="CASE-VOICE-02",
                age=54.0,
                sex="Male",
                arrival_mode="119",
                mental_status="Alert",
                sbp=152.0,
                dbp=96.0,
                hr=104.0,
                rr=22.0,
                saturation=94.0,
                pain=1,
                nrs_pain=9.0,
                chief_complaint_text="crushing substernal chest pain radiating to left arm with diaphoresis",
            ),
        },
        {
            "title": "Case 3: Missing Sensor / Pulse Oximeter Failure (Adaptive Fallback)",
            "summary": "Patient presenting with migraine; pulse oximeter failed. Adaptive cross-modal gate reweights away from missing vitals to context and chief complaint.",
            "record": PatientIntakeRecord(
                patient_id="CASE-MISSING-03",
                age=42.0,
                sex="Female",
                arrival_mode="Walking",
                mental_status="Alert",
                sbp=122.0,
                dbp=78.0,
                hr=76.0,
                rr=16.0,
                saturation=None,
                pain=1,
                nrs_pain=5.0,
                chief_complaint_text="severe migraine headache and nausea",
            ),
        },
        {
            "title": "Case 4: Ambiguous Symptoms (Safety Gate & Human Review Alert)",
            "summary": "Vague generalized malaise in young adult. Elevated predictive entropy (H > 0.82) triggers safety alert requiring human nurse review.",
            "record": PatientIntakeRecord(
                patient_id="CASE-AMBIGUOUS-04",
                age=35.0,
                sex="Female",
                arrival_mode="Walking",
                mental_status="Alert",
                sbp=118.0,
                dbp=76.0,
                hr=82.0,
                rr=17.0,
                saturation=98.0,
                pain=1,
                nrs_pain=4.0,
                chief_complaint_text="generalized fatigue, mild non-specific abdominal discomfort, malaise",
            ),
        },
        {
            "title": "Case 5: Tension Pneumothorax on Imaging (Escalation Protocol)",
            "summary": "Pleuritic chest pain assigned KTAS 3 at intake. Subsequent chest radiograph reveals Acute Tension Pneumothorax, automatically escalating triage to KTAS 2.",
            "record": PatientIntakeRecord(
                patient_id="CASE-PNEUMO-05",
                age=24.0,
                sex="Male",
                arrival_mode="Private car",
                mental_status="Alert",
                sbp=112.0,
                dbp=74.0,
                hr=92.0,
                rr=20.0,
                saturation=95.0,
                pain=1,
                nrs_pain=6.0,
                chief_complaint_text="sudden sharp right-sided chest pain while resting",
                chest_xray_input=generate_synthetic_chest_xray("pneumothorax", image_size=224, random_seed=902),
            ),
        },
    ]

    col_bench_sel, col_bench_view = st.columns([1, 1.4], gap="large")

    with col_bench_sel:
        selected_idx = st.radio(
            "Select Benchmark Case Study",
            range(len(cases)),
            format_func=lambda i: cases[i]["title"],
        )
        selected_case = cases[selected_idx]
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid #3B82F6;">
                <strong>Case Summary:</strong><br>
                <span style="font-size: 0.9rem; color: #475569;">{selected_case['summary']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        run_bench = st.button("⚡ Evaluate Selected Benchmark", type="primary", use_container_width=True)

    with col_bench_view:
        if run_bench:
            with st.spinner("Processing benchmark scenario through orchestrator..."):
                b_res = orchestrator.triage_patient(selected_case["record"], generate_plots=True)

            b_num = b_res.final_ktas_level
            st.markdown(
                f"""
                <div class="decision-hero decision-hero-{b_num}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: white;">{b_res.final_ktas_name}</h3>
                        <span class="badge-pill-{b_num}">{b_res.triage_status_badge}</span>
                    </div>
                    <p style="margin-top: 8px; color: rgba(255,255,255,0.9); font-size: 0.9rem;"><strong>Directive:</strong> {b_res.clinical_recommendation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            bm1, bm2, bm3 = st.columns(3)
            with bm1:
                st.metric("Confidence", f"{b_res.confidence*100:.1f}%")
            with bm2:
                st.metric("Predictive Entropy", f"{b_res.predictive_entropy:.2f}")
            with bm3:
                st.metric("Safety Alert", "⚠️ Review Required" if b_res.requires_human_review else "✅ Cleared")

            if b_res.requires_human_review:
                st.warning(f"⚠️ {b_res.safety_alert_message}")

            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #1E293B; margin: 10px 0 2px 0;'>KTAS Posterior Probability Spectrum</div>", unsafe_allow_html=True)
            st.plotly_chart(create_ktas_spectrum_chart(b_res.intake_probabilities, b_res.final_ktas_level), use_container_width=True)

            if b_res.xray_evaluated:
                st.markdown(f"**Auxiliary Radiograph Finding:** `{b_res.primary_xray_finding}` ({b_res.xray_confidence*100:.1f}%) | RSI: {b_res.radiographic_severity_index:.2f}")
                if b_res.xray_overlay_path and Path(b_res.xray_overlay_path).exists():
                    st.image(b_res.xray_overlay_path, caption="Grad-CAM Saliency Overlay", width=280)
        else:
            st.info("Select a clinical presentation on the left and click **'Evaluate Selected Benchmark'**.")


# ==============================================================================
# TAB 3: ARCHITECTURE & SCIENTIFIC TRANSPARENCY
# ==============================================================================
with tab_architecture:
    st.markdown("<div style='font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;'>🧠 System Architecture & Scientific Governance</div>", unsafe_allow_html=True)

    st.markdown(
        """
        ### Multi-Pipeline Deep Learning Architecture
        
        The **Multimodal Intelligent Triage Assistant** is structured into decoupled, scientifically validated modules:
        
        1. **Chief Complaint Text Stream**:
           - Architecture: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense semantic embedding).
           - Performance: Macro F1: 0.443 on short free-text complaints, outperforming traditional TF-IDF bag-of-words.
           
        2. **Physiological Vitals Stream**:
           - Architecture: PyTorch MLP (10-d input $\\to$ 64-d hidden $\\to$ BatchNorm/GELU $\\to$ 128-d latent representation).
           - Missingness Encoding: Explicit indicator flag for unmeasured SpO2 with clinical robust scaling.
           
        3. **Demographic Context Stream**:
           - Architecture: Categorical Entity Embeddings (Sex: 8-d, Arrival Mode: 16-d, Injury: 8-d) concatenated with continuous variables (Age, ED volume) $\\to$ 64-d latent vector.
           
        4. **Adaptive Reliability Fusion**:
           - Learned Cross-Modal Attention Gating dynamically allocates channel weights ($w_{text}, w_{vitals}, w_{context}$).
           - Missingness Penalty ensures robust fallback when physiological vitals or text are unrecorded.
           
        5. **Monte Carlo Dropout Uncertainty**:
           - Executes $T = 20$ stochastic forward passes per patient with frozen normalization layers.
           - Quantifies predictive entropy and decomposes uncertainty into aleatoric (data noise) vs epistemic (model ambiguity).
           - Flags cases with entropy $> 0.82$ for mandatory human clinician intervention.
           
        6. **Decoupled Chest X-ray Auxiliary Stream**:
           - Vision Backbone: DenseNet-121 (CheXNet configuration) calibrated for 14 NIH thoracic findings.
           - Saliency XAI: Grad-CAM gradient backpropagation to `features.denseblock4.denselayer16.conv2`.
           - **Scientific Integrity**: Radiographs are treated as post-admission auxiliary diagnostic modifiers, completely avoiding patient-level data leakage into initial arrival triage ($t=0$).
        """
    )

    st.markdown("---")
    st.markdown(
        """
        ### KTAS Acuity Level Definitions
        
        | KTAS Level | Classification | Max Wait Time | Typical Presentation |
        | :---: | :--- | :---: | :--- |
        | **KTAS 1** | **Resuscitation** | Immediate | Cardiac arrest, severe shock, unresponsive, major trauma |
        | **KTAS 2** | **Emergent** | $\\le$ 15 minutes | Acute coronary syndrome, stroke signs, severe respiratory distress |
        | **KTAS 3** | **Urgent** | $\\le$ 30 minutes | Moderate asthma, severe pain, localized infection with fever |
        | **KTAS 4** | **Less Urgent** | $\\le$ 60 minutes | Mild trauma, simple laceration, uncomplicated urinary infection |
        | **KTAS 5** | **Non-Urgent** | $\\le$ 120 minutes | Suture removal, mild cold symptoms, prescription refill |
        """
    )
