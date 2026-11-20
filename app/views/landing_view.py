"""
Screen 1: MITA Landing Page.
============================
Clean, elegant medical intelligence landing page matching Screen 1 of the reference mockup.
Features:
  - Top Navbar with MITA brand, Navigation links, and Get Started button.
  - Left Hero: H1 title, subtitle, and primary/secondary CTAs.
  - Right Hero: Futuristic 3D Holographic Medical Laboratory display with CXR scan,
    Grad-CAM focal target, animated ECG telemetry, and interactive mouse parallax.
"""

from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_sample_xray_b64() -> str:
    """Loads sample radiograph as base64 string for the 3D holographic display."""
    xray_path = PROJECT_ROOT / "data" / "raw" / "xray" / "sample_edema.png"
    if xray_path.exists():
        try:
            return base64.b64encode(xray_path.read_bytes()).decode("utf-8")
        except Exception:
            pass
    return ""


def render_landing_view() -> None:
    """Renders the clean white MITA landing experience."""
    # 1. Top MITA Navbar
    c_brand, c_links, c_btn = st.columns([1.5, 3.0, 1.2])

    with c_brand:
        st.markdown(
            '<div class="mita-brand" style="margin-bottom: 0;"><span class="mita-pulse-icon">∿</span> MITA</div>',
            unsafe_allow_html=True,
        )

    with c_links:
        st.markdown(
            '<div style="display: flex; justify-content: center; align-items: center; gap: 28px; height: 100%; font-size: 0.88rem; font-weight: 500; color: #475569;">'
            '<span style="color: #174EA6; font-weight: 700; cursor: pointer;">Home</span>'
            '<span style="cursor: pointer;">About</span>'
            '<span style="cursor: pointer;">Workflow</span>'
            '<span style="cursor: pointer;">Model</span>'
            '<span style="cursor: pointer;">Contact</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    with c_btn:
        if st.button("Get Started", key="btn_top_get_started", type="primary", width="stretch"):
            st.session_state["current_view"] = "assessment"
            st.session_state["assessment_stage"] = 1
            st.rerun()

    st.markdown("<hr style='margin: 12px 0 32px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

    # 2. Main Hero Section (Two Columns)
    col_hero_text, col_hero_visual = st.columns([1.05, 1.35], gap="large")

    with col_hero_text:
        st.markdown(
            '<div style="padding-top: 28px;">'
            '<h1 style="font-size: 2.9rem; font-weight: 800; color: #142033; line-height: 1.15; letter-spacing: -0.025em; margin-bottom: 16px;">'
            'Multimodal Intelligent<br/>'
            '<span style="color: #174EA6;">Triage Assistant</span>'
            '</h1>'
            '<p style="font-size: 1.05rem; color: #475569; line-height: 1.55; margin-bottom: 32px; max-width: 480px;">'
            'AI-powered emergency triage using text, voice, vitals and medical imaging.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        c_cta1, c_cta2 = st.columns([1.4, 1.0])
        with c_cta1:
            if st.button(
                "Launch Triage Assistant →",
                key="btn_hero_launch",
                type="primary",
                width="stretch",
            ):
                st.session_state["current_view"] = "assessment"
                st.session_state["assessment_stage"] = 1
                st.rerun()

        with c_cta2:
            if st.button(
                "Learn More",
                key="btn_hero_learn",
                type="secondary",
                width="stretch",
            ):
                st.session_state["current_view"] = "architecture"
                st.rerun()

    with col_hero_visual:
        # Futuristic 3D Holographic Medical Workspace Component (Rendered via Components HTML iframe)
        b64_xray = _get_sample_xray_b64()

        holo_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: transparent;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    height: 400px;
    display: flex;
    align-items: center;
    justify-content: center;
    perspective: 1200px;
  }}
  .lab-frame {{
    position: relative;
    width: 100%;
    height: 380px;
    border-radius: 20px;
    background: radial-gradient(circle at 75% 25%, #152E4D 0%, #0D1C30 50%, #060E18 100%);
    border: 1.5px solid rgba(56, 189, 248, 0.4);
    box-shadow: 0 20px 45px -10px rgba(6, 14, 24, 0.65), 0 0 30px rgba(56, 189, 248, 0.2);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .lab-grid {{
    position: absolute;
    inset: 0;
    background-image: 
      radial-gradient(ellipse at 80% 20%, rgba(56, 189, 248, 0.25) 0%, transparent 60%),
      radial-gradient(ellipse at 20% 80%, rgba(14, 165, 233, 0.15) 0%, transparent 60%),
      linear-gradient(rgba(56, 189, 248, 0.08) 1px, transparent 1px),
      linear-gradient(90deg, rgba(56, 189, 248, 0.08) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 24px 24px, 24px 24px;
    pointer-events: none;
  }}
  .holo-rig {{
    position: relative;
    z-index: 5;
    display: flex;
    align-items: center;
    gap: 16px;
    transform: rotateY(-8deg) rotateX(4deg);
    transform-style: preserve-3d;
    transition: transform 0.15s ease-out;
  }}
  .holo-xray-card {{
    position: relative;
    width: 205px;
    background: rgba(13, 27, 46, 0.88);
    border: 1.5px solid #38BDF8;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 0 26px rgba(56, 189, 248, 0.32), inset 0 0 16px rgba(56, 189, 248, 0.15);
    backdrop-filter: blur(10px);
    text-align: center;
  }}
  .xray-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.66rem;
    font-weight: 700;
    color: #38BDF8;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    text-transform: uppercase;
  }}
  .xray-img-wrapper {{
    position: relative;
    width: 100%;
    height: 180px;
    border-radius: 8px;
    overflow: hidden;
    background: #020617;
    border: 1px solid rgba(56, 189, 248, 0.35);
  }}
  .xray-img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(1.15) contrast(1.2) hue-rotate(185deg);
    opacity: 0.92;
  }}
  .scanline {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #38BDF8, transparent);
    box-shadow: 0 0 12px #38BDF8;
    animation: scan 2.6s linear infinite;
  }}
  @keyframes scan {{
    0% {{ top: 0; opacity: 0; }}
    15% {{ opacity: 1; }}
    85% {{ opacity: 1; }}
    100% {{ top: 100%; opacity: 0; }}
  }}
  .focal-target {{
    position: absolute;
    top: 48%;
    left: 55%;
    transform: translate(-50%, -50%);
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(239, 68, 68, 0.85) 0%, rgba(245, 158, 11, 0.45) 50%, transparent 75%);
    filter: blur(4px);
    pointer-events: none;
    animation: pulse-heat 2.2s infinite ease-in-out;
  }}
  @keyframes pulse-heat {{
    0%, 100% {{ transform: translate(-50%, -50%) scale(0.92); opacity: 0.75; }}
    50% {{ transform: translate(-50%, -50%) scale(1.15); opacity: 1.0; }}
  }}
  .xray-footer {{
    font-size: 0.65rem;
    color: #94A3B8;
    margin-top: 8px;
    font-family: monospace;
    display: flex;
    justify-content: space-between;
  }}
  .holo-side-col {{
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .telemetry-card {{
    background: rgba(13, 27, 46, 0.88);
    border: 1.5px solid #10B981;
    border-radius: 12px;
    padding: 12px 14px;
    width: 220px;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.22);
    backdrop-filter: blur(8px);
  }}
  .telemetry-header {{
    display: flex;
    justify-content: space-between;
    font-size: 0.66rem;
    font-weight: 700;
    color: #34D399;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .ecg-svg {{
    width: 100%;
    height: 36px;
    display: block;
  }}
  .ecg-pulse-path {{
    stroke-dasharray: 200;
    stroke-dashoffset: 200;
    animation: ecg-draw 2.4s linear infinite;
  }}
  @keyframes ecg-draw {{
    0% {{ stroke-dashoffset: 200; }}
    100% {{ stroke-dashoffset: 0; }}
  }}
  .telemetry-footer {{
    display: flex;
    justify-content: space-between;
    font-size: 0.70rem;
    color: #CBD5E1;
    font-family: monospace;
    margin-top: 4px;
  }}
  .fusion-badge-card {{
    background: rgba(13, 27, 46, 0.88);
    border: 1.5px solid #60A5FA;
    border-radius: 12px;
    padding: 12px 14px;
    width: 220px;
    box-shadow: 0 0 20px rgba(96, 165, 250, 0.22);
    backdrop-filter: blur(8px);
  }}
  .fusion-title {{
    font-size: 0.66rem;
    font-weight: 700;
    color: #93C5FD;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}
  .fusion-name {{
    font-size: 1.1rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.01em;
  }}
  .fusion-desc {{
    font-size: 0.68rem;
    color: #94A3B8;
    margin-top: 3px;
  }}
</style>
</head>
<body>
  <div class="lab-frame" id="labFrame">
    <div class="lab-grid"></div>
    <div class="holo-rig" id="holoRig">
      <!-- CXR Unit -->
      <div class="holo-xray-card">
        <div class="xray-header">
          <span>CXR SCAN [NIH-14]</span>
          <span style="color: #4ADE80;">LIVE</span>
        </div>
        <div class="xray-img-wrapper">
          <img class="xray-img" src="data:image/png;base64,{b64_xray}" alt="Thoracic Radiograph"/>
          <div class="scanline"></div>
          <div class="focal-target"></div>
        </div>
        <div class="xray-footer">
          <span>GRAD-CAM++</span>
          <span style="color: #38BDF8;">PULMONARY EDEMA</span>
        </div>
      </div>
      
      <!-- Telemetry Column -->
      <div class="holo-side-col">
        <div class="telemetry-card">
          <div class="telemetry-header">
            <span>ECG LEAD II</span>
            <span>104 BPM</span>
          </div>
          <svg class="ecg-svg" viewBox="0 0 160 35">
            <path class="ecg-pulse-path" d="M0,18 L30,18 L35,8 L40,28 L45,2 L50,30 L55,18 L80,18 L85,8 L90,28 L95,2 L100,30 L105,18 L160,18" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round"/>
          </svg>
          <div class="telemetry-footer">
            <span>NSR RHYTHM</span>
            <span style="color: #34D399;">TELEMETRY SYNC</span>
          </div>
        </div>
        
        <div class="fusion-badge-card">
          <div class="fusion-title">MULTIMODAL FUSION</div>
          <div class="fusion-name">RCCF-Net v2</div>
          <div class="fusion-desc">Cross-Attention &bull; Calibrated (T*=1.05)</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const frame = document.getElementById('labFrame');
    const rig = document.getElementById('holoRig');
    if (frame && rig) {{
      frame.addEventListener('mousemove', (e) => {{
        const rect = frame.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width - 0.5;
        const y = (e.clientY - rect.top) / rect.height - 0.5;
        rig.style.transform = `rotateY(${{x * 16 - 8}}deg) rotateX(${{-y * 12 + 4}}deg)`;
      }});
      frame.addEventListener('mouseleave', () => {{
        rig.style.transform = `rotateY(-8deg) rotateX(4deg)`;
      }});
    }}
  </script>
</body>
</html>"""
        components.html(holo_html, height=400, scrolling=False)
