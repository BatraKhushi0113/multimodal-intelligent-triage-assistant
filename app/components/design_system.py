"""
MITA Design System — Clean White Clinical Intelligence.
======================================================
Exact styling tokens and CSS matching the MITA commercial reference design.
Features:
  - Clean white (#FFFFFF) and soft off-white (#F8FAFC) canvas.
  - MITA royal blue primary (#174EA6 / #1D4ED8).
  - Deep navy typography (#142033) and slate gray (#64748B).
  - Left navigation sidebar styling.
  - Horizontal progress stepper without indentation (prevents Markdown pre/code block leaks).
  - Vitals 6-cell grid, audio waveform, and radiology 3-card layout.
  - KTAS acuity cards and multimodal explanation checklist.
"""

from __future__ import annotations

# Color Palette Constants
COLOR_BG_PRIMARY = "#F8FAFC"
COLOR_SURFACE = "#FFFFFF"
COLOR_TEXT_PRIMARY = "#142033"
COLOR_TEXT_SECONDARY = "#64748B"
COLOR_PRIMARY_BLUE = "#174EA6"
COLOR_ACTION_BLUE = "#2563EB"
COLOR_CYAN_ACCENT = "#0EA5E9"
COLOR_BORDER = "#E2E8F0"
COLOR_SUCCESS = "#059669"
COLOR_WARNING = "#D97706"
COLOR_CRITICAL = "#DC2626"

# KTAS Clinical Palette
KTAS_PALETTE = {
    1: {"color": "#DC2626", "bg": "#FEE2E2", "border": "#EF4444", "name": "KTAS 1 (Resuscitation)", "badge": "RESUSCITATION"},
    2: {"color": "#DC2626", "bg": "#FEF2F2", "border": "#F87171", "name": "KTAS 2 (Emergent)", "badge": "EMERGENT"},
    3: {"color": "#D97706", "bg": "#FEF3C7", "border": "#FBBF24", "name": "KTAS 3 (Urgent)", "badge": "URGENT"},
    4: {"color": "#2563EB", "bg": "#DBEAFE", "border": "#60A5FA", "name": "KTAS 4 (Less Urgent)", "badge": "LESS URGENT"},
    5: {"color": "#059669", "bg": "#D1FAE5", "border": "#34D399", "name": "KTAS 5 (Non-Urgent)", "badge": "NON-URGENT"},
}

MASTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* Global Reset & Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #142033 !important;
    background-color: #F8FAFC !important;
}

/* Remove default blank Streamlit header */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
}
footer {
    visibility: hidden;
}

/* Main Container Spacing */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1300px !important;
}

/* Sidebar Styling matching MITA */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    padding-top: 1.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem !important;
}

/* MITA Brand Header */
.mita-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.35rem;
    font-weight: 800;
    color: #142033;
    letter-spacing: -0.02em;
    margin-bottom: 24px;
}
.mita-pulse-icon {
    color: #2563EB;
    font-size: 1.45rem;
    font-weight: 900;
}

/* Clinical Card */
.mita-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
}
.mita-card-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #142033;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.mita-card-sub {
    font-size: 0.82rem;
    color: #64748B;
    margin-bottom: 16px;
}

/* Page Title & Subtitle */
.mita-page-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #142033;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.mita-page-sub {
    font-size: 0.86rem;
    color: #64748B;
    margin-bottom: 20px;
}

/* Stepper Bar */
.mita-stepper-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 10px 0 20px 0;
    margin-bottom: 12px;
}
.mita-step-item {
    display: flex;
    align-items: center;
    gap: 8px;
}
.mita-step-circle {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.76rem;
    font-weight: 700;
}
.mita-step-circle-active {
    background: #174EA6;
    color: #FFFFFF;
}
.mita-step-circle-done {
    background: #ECFDF5;
    border: 1.5px solid #10B981;
    color: #059669;
}
.mita-step-circle-upcoming {
    background: #FFFFFF;
    border: 1.5px solid #CBD5E1;
    color: #94A3B8;
}
.mita-step-name {
    font-size: 0.84rem;
    font-weight: 500;
    color: #64748B;
}
.mita-step-name-active {
    color: #142033;
    font-weight: 700;
}
.mita-step-line {
    width: 48px;
    height: 1.5px;
    background: #E2E8F0;
}

/* Vital Signs 6-Tile Grid */
.vitals-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}
.vital-tile {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 14px;
    text-align: center;
}
.vital-tile-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748B;
    margin-bottom: 4px;
}
.vital-tile-val {
    font-size: 1.25rem;
    font-weight: 800;
    color: #142033;
    font-family: 'JetBrains Mono', monospace;
}

/* Audio Waveform Player Strip */
.waveform-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 10px;
}

/* Streamlit Button Overrides to Match MITA */
div.stButton > button[data-testid*="primary"],
div.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid*="primary"] {
    background-color: #174EA6 !important;
    background: #174EA6 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 2px 8px rgba(23, 78, 166, 0.2) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[data-testid*="primary"]:hover,
div.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid*="primary"]:hover {
    background-color: #1d5fc7 !important;
    background: #1d5fc7 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(23, 78, 166, 0.3) !important;
}
div.stButton > button[data-testid*="secondary"],
div.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"],
button[data-testid*="secondary"] {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #142033 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[data-testid*="secondary"]:hover,
div.stButton > button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover,
button[data-testid*="secondary"]:hover {
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
}

/* Input Fields Clean Styling */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
    border-color: #CBD5E1 !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #174EA6 !important;
    box-shadow: 0 0 0 2px rgba(23, 78, 166, 0.12) !important;
}
</style>
"""


def get_ktas_badge_html(ktas_level: int) -> str:
    """Generate consistent HTML pill badge for KTAS level."""
    meta = KTAS_PALETTE.get(ktas_level, KTAS_PALETTE[3])
    return (
        f'<span style="background-color: {meta["bg"]}; color: {meta["color"]}; '
        f'border: 1px solid {meta["border"]}; padding: 4px 12px; border-radius: 9999px; '
        f'font-size: 0.82rem; font-weight: 700; letter-spacing: 0.04em;">'
        f'{meta["badge"]} (LEVEL {ktas_level})</span>'
    )


def get_conflict_callout_html(rule_name: str, message: str, severity: str = "critical") -> str:
    """Generate styled CECE conflict callout HTML."""
    if severity == "critical":
        bg, border, text_color = "#FEF2F2", "#EF4444", "#991B1B"
    elif severity == "warning":
        bg, border, text_color = "#FFFBEB", "#F59E0B", "#92400E"
    else:
        bg, border, text_color = "#EFF6FF", "#3B82F6", "#1E40AF"
    return (
        f'<div style="background: {bg}; border: 1px solid {border}; border-left: 4px solid {border}; '
        f'border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 0.84rem; color: {text_color};">'
        f'<strong>⚠️ [{rule_name}]:</strong> {message}</div>'
    )


def get_telemetry_tile_html(label: str, value: str, unit: str = "", status: str = "normal") -> str:
    """Generate telemetry vital sign tile HTML."""
    border_color = "#E2E8F0"
    if status == "critical":
        border_color = "#EF4444"
    elif status == "warning":
        border_color = "#F59E0B"
    elif status == "success":
        border_color = "#10B981"
    return (
        f'<div class="vital-tile" style="border-left: 3px solid {border_color};">'
        f'<div class="vital-tile-label">{label}</div>'
        f'<div class="vital-tile-val">{value} <span style="font-size: 0.72rem; font-weight: 500; color: #64748B;">{unit}</span></div>'
        f'</div>'
    )


def get_iqa_badge_html(passed: bool, label: str) -> str:
    """Generate IQA pass/fail status badge HTML."""
    if passed:
        return f'<span style="background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; padding: 2px 8px; border-radius: 4px; font-size: 0.74rem; font-weight: 600;">✓ {label}: PASS</span>'
    return f'<span style="background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; padding: 2px 8px; border-radius: 4px; font-size: 0.74rem; font-weight: 600;">✕ {label}: WARN</span>'
