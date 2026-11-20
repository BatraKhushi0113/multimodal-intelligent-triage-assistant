"""
Architecture & Scientific Governance View.
=========================================
Presents the multimodal neural architecture, CECE neuro-symbolic rule tables,
experimental comparative benchmarks (E0-E3), model registry, and clinical safety protocols.
"""

from __future__ import annotations

import streamlit as st
from app.components.design_system import get_ktas_badge_html


def render_architecture_view() -> None:
    """
    Renders Architecture, Research Benchmarks, and Scientific Governance page.
    """
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #0891B2;"></span>
                <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #0891B2;">System Blueprint & Research Evidence</span>
            </div>
            <h2 style="font-size: 1.65rem; font-weight: 800; color: #142033; letter-spacing: -0.02em; margin-bottom: 6px;">
                Multimodal Architecture & Scientific Governance
            </h2>
            <p style="font-size: 0.90rem; color: #64748B; margin: 0; line-height: 1.5;">
                Comprehensive documentation of the 5-pillar multimodal pipeline, comparative experimental evaluations ($E_0 \to E_3$), neuro-symbolic conflict rules, and zero-leakage safety boundaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_arch, tab_experiments, tab_cece_rules, tab_registry, tab_ktas = st.tabs([
        "System Architecture",
        "Benchmark Experiments (E0-E3)",
        "CECE Neuro-Symbolic Engine",
        "Model Registry & Checkpoints",
        "KTAS Acuity & Safety Protocol",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: SYSTEM ARCHITECTURE
    # -------------------------------------------------------------------------
    with tab_arch:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <div>
                        <span class="card-header-title">Five-Pillar Multimodal Architecture</span>
                        <div class="card-header-desc">End-to-end clinical workflow connecting patient intake, feature extraction, cross-attention fusion, and calibrated inference</div>
                    </div>
                    <span style="background: #EFF6FF; color: #1D4ED8; font-weight: 700; font-size: 0.75rem; padding: 4px 10px; border-radius: 9999px; border: 1px solid #BFDBFE;">v2.4 Production</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="clinical-card" style="height: 100%;">
                    <div style="font-size: 0.92rem; font-weight: 700; color: #174EA6; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <span style="background: #E8F0FE; color: #174EA6; width: 24px; height: 24px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.78rem;">A</span>
                        Pillar A: Advanced Clinical NLP
                    </div>
                    <ul style="font-size: 0.83rem; color: #334155; line-height: 1.6; margin: 0; padding-left: 18px;">
                        <li><strong>NegEx Scoping Engine</strong>: Deterministic negation parser resolving pseudo-negations within bounded ±8-token windows.</li>
                        <li><strong>12-Organ System Taxonomy</strong>: Automated ICD-aligned classification (Cardiovascular, Respiratory, Neuro, GI, Trauma, etc.).</li>
                        <li><strong>Temporal & Severity Extraction</strong>: Identifies acute onset (&lt;24h) vs chronic progression and pain ratings (NRS 0–10).</li>
                        <li><strong>MiniLM Semantic Embedding</strong>: 384-dimensional dense context representation.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="clinical-card" style="height: 100%; margin-top: 16px;">
                    <div style="font-size: 0.92rem; font-weight: 700; color: #0891B2; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <span style="background: #ECFEFF; color: #0891B2; width: 24px; height: 24px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.78rem;">C</span>
                        Pillar C: Cross-Modal Evidence & Conflict Engine (CECE)
                    </div>
                    <ul style="font-size: 0.83rem; color: #334155; line-height: 1.6; margin: 0; padding-left: 18px;">
                        <li><strong>Physiological Bounds Verification</strong>: Detects inverted blood pressure (SBP &lt; DBP) and narrow pulse pressure (&lt;18 mmHg).</li>
                        <li><strong>Occult Shock Screening</strong>: Computes bedside Shock Index (HR / SBP &ge; 1.3).</li>
                        <li><strong>Cross-Modal Discordance Traps</strong>: Intercepts silent hypoxia (SpO2 &le; 90% with mild complaint) and fever denial with core pyrexia.</li>
                        <li><strong>Conditioning Output</strong>: Generates an 8-dim evidence tensor modulating downstream fusion.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="clinical-card" style="height: 100%;">
                    <div style="font-size: 0.92rem; font-weight: 700; color: #7C3AED; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <span style="background: #F3E8FF; color: #7C3AED; width: 24px; height: 24px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.78rem;">B</span>
                        Pillar B: Advanced Computer Vision & IQA
                    </div>
                    <ul style="font-size: 0.83rem; color: #334155; line-height: 1.6; margin: 0; padding-left: 18px;">
                        <li><strong>Pre-Inference IQA</strong>: Evaluates Laplacian blur variance (&ge;35.0), RMS contrast (&ge;0.08), exposure clipping, and non-CXR color checks.</li>
                        <li><strong>DenseNet-121 CheXNet</strong>: 14 multi-label thoracic pathology detection heads.</li>
                        <li><strong>Grad-CAM++ Multi-Focal Saliency</strong>: Visualizes high-order gradient heatmaps across 6 anatomical thoracic zones.</li>
                        <li><strong>Decoupled Safety Architecture</strong>: Operates as an auxiliary escalation module without contaminating non-imaging triage splits.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="clinical-card" style="height: 100%; margin-top: 16px;">
                    <div style="font-size: 0.92rem; font-weight: 700; color: #0284C7; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <span style="background: #E0F2FE; color: #0284C7; width: 24px; height: 24px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.78rem;">D & E</span>
                        Pillars D & E: RCCF-Net v2 & Safety Calibration
                    </div>
                    <ul style="font-size: 0.83rem; color: #334155; line-height: 1.6; margin: 0; padding-left: 18px;">
                        <li><strong>Residual Cross-Attention</strong>: Multi-head cross-attention (4 heads, dk=32) linking Text, Vitals, and Context tokens.</li>
                        <li><strong>Missing-Modality Tokens</strong>: Learned imputation tokens maintaining robustness during partial telemetry loss.</li>
                        <li><strong>Post-Hoc Calibration</strong>: Temperature scaling (T* = 1.05) yielding minimal calibration error (ECE = 0.0673).</li>
                        <li><strong>Selective Risk Abstention</strong>: Entropy-guided rejection for uncertain edge cases requiring senior physician review.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # TAB 2: BENCHMARK EXPERIMENTS
    # -------------------------------------------------------------------------
    with tab_experiments:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <div>
                        <span class="card-header-title">Comparative Research Evaluation (N=191 Held-Out Test Set)</span>
                        <div class="card-header-desc">Empirical comparison of baseline and advanced architectures across classification, calibration, and emergency recall</div>
                    </div>
                </div>
                <div style="overflow-x: auto; margin-top: 12px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.84rem; text-align: left;">
                        <thead>
                            <tr style="border-bottom: 2px solid #CBD5E1; color: #475569;">
                                <th style="padding: 10px;">ID</th>
                                <th style="padding: 10px;">Architecture Configuration</th>
                                <th style="padding: 10px; text-align: right;">Accuracy</th>
                                <th style="padding: 10px; text-align: right;">Macro F1</th>
                                <th style="padding: 10px; text-align: right;">ECE (Error)</th>
                                <th style="padding: 10px; text-align: right;">Brier Score</th>
                                <th style="padding: 10px; text-align: right;">KTAS 1 Recall</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #E2E8F0; background: #FFFFFF;">
                                <td style="padding: 10px; font-weight: 700; color: #1E293B;">E0</td>
                                <td style="padding: 10px;">Baseline Adaptive Reliability Fusion</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">60.21%</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace; font-weight: 700; color: #174EA6;">0.5387</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.0757</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.5308</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">33.3%</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0; background: #F8FAFC;">
                                <td style="padding: 10px; font-weight: 700; color: #1E293B;">E1</td>
                                <td style="padding: 10px;">Baseline + CECE Neuro-Symbolic Audit</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">60.21%</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.5285</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.0757</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.5308</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace; font-weight: 700; color: #059669;">66.7% (+33.4%)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0; background: #FFFFFF;">
                                <td style="padding: 10px; font-weight: 700; color: #1E293B;">E2</td>
                                <td style="padding: 10px;">RCCF-Net v2 (Cross-Attention Network)</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">59.16%</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.4870</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.1952</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.5951</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">33.3%</td>
                            </tr>
                            <tr style="border-bottom: 2px solid #CBD5E1; background: #EFF6FF;">
                                <td style="padding: 10px; font-weight: 700; color: #1D4ED8;">E3</td>
                                <td style="padding: 10px; font-weight: 600; color: #1D4ED8;">RCCF-Net v2 + CECE + Calibrated (T*=1.05)</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace; font-weight: 700; color: #1D4ED8;">60.21%</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.4974</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace; font-weight: 700; color: #059669;">0.0673 (Best)</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.5800</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">33.3%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="clinical-card" style="margin-top: 16px;">
                <div class="clinical-card-header">
                    <div>
                        <span class="card-header-title">Missing-Modality Robustness Evaluation</span>
                        <div class="card-header-desc">Simulated telemetry sensor dropouts demonstrating RCCF-Net v2 cross-attention superiority under data sparsity</div>
                    </div>
                </div>
                <div style="overflow-x: auto; margin-top: 12px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.84rem; text-align: left;">
                        <thead>
                            <tr style="border-bottom: 2px solid #CBD5E1; color: #475569;">
                                <th style="padding: 10px;">Missing Channel Scenario</th>
                                <th style="padding: 10px; text-align: right;">Baseline E0 Macro F1</th>
                                <th style="padding: 10px; text-align: right;">RCCF-Net v2 Macro F1</th>
                                <th style="padding: 10px; text-align: right;">Net Advantage</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight: 600; color: #1E293B;">Complete Data (All Channels Present)</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace; font-weight: 700; color: #174EA6;">0.5387</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.4870</td>
                                <td style="padding: 10px; text-align: right; color: #64748B;">Baseline (+0.0517)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0; background: #F0FDF4;">
                                <td style="padding: 10px; font-weight: 600; color: #1E293B;">Vitals Missing (Telemetry Disconnection)</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.3875</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace; font-weight: 700; color: #059669;">0.4143</td>
                                <td style="padding: 10px; text-align: right; font-weight: 700; color: #059669;">+0.0268 (RCCF-Net v2 wins)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0; background: #F0FDF4;">
                                <td style="padding: 10px; font-weight: 600; color: #1E293B;">Context Missing (Trauma / Unconscious)</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace;">0.4982</td>
                                <td style="padding: 10px; text-align: right; font-family: monospace; font-weight: 700; color: #059669;">0.5061</td>
                                <td style="padding: 10px; text-align: right; font-weight: 700; color: #059669;">+0.0079 (RCCF-Net v2 wins)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # TAB 3: CECE NEURO-SYMBOLIC ENGINE
    # -------------------------------------------------------------------------
    with tab_cece_rules:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <div>
                        <span class="card-header-title">CECE Cross-Modal Neuro-Symbolic Audit Rules</span>
                        <div class="card-header-desc">Hard physiological sanity assertions and cross-modal discordance traps preventing fatal under-triage</div>
                    </div>
                </div>
                <div style="overflow-x: auto; margin-top: 12px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.84rem; text-align: left;">
                        <thead>
                            <tr style="border-bottom: 2px solid #CBD5E1; color: #475569;">
                                <th style="padding: 10px;">Rule Identifier</th>
                                <th style="padding: 10px;">Trigger Condition</th>
                                <th style="padding: 10px;">Clinical Rationale</th>
                                <th style="padding: 10px;">Action / Override</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-family: monospace; font-weight: 700; color: #DC2626;">RULE-HYPOXIA</td>
                                <td style="padding: 10px; font-size: 0.80rem;">SpO2 &le; 90% AND text lacks dyspnea/sob</td>
                                <td style="padding: 10px; font-size: 0.82rem; color: #475569;">Silent / happy hypoxia detection common in viral pneumonias</td>
                                <td style="padding: 10px;"><span style="background: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">Flag Discordance + Escalate KTAS &le; 2</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-family: monospace; font-weight: 700; color: #DC2626;">RULE-SHOCK</td>
                                <td style="padding: 10px; font-size: 0.80rem;">Shock Index (HR/SBP) &ge; 1.3</td>
                                <td style="padding: 10px; font-size: 0.82rem; color: #475569;">Occult circulatory shock preceding overt hemodynamic collapse</td>
                                <td style="padding: 10px;"><span style="background: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">Force KTAS 1 (Resuscitation)</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-family: monospace; font-weight: 700; color: #D97706;">RULE-PYREXIA</td>
                                <td style="padding: 10px; font-size: 0.80rem;">Temp &ge; 38.3°C AND text negates fever</td>
                                <td style="padding: 10px; font-size: 0.82rem; color: #475569;">Subjective symptom denial contradicted by objective core temperature</td>
                                <td style="padding: 10px;"><span style="background: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">Evidence discordance penalty</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-family: monospace; font-weight: 700; color: #DC2626;">RULE-INVERTED-BP</td>
                                <td style="padding: 10px; font-size: 0.80rem;">SBP &le; DBP</td>
                                <td style="padding: 10px; font-size: 0.82rem; color: #475569;">Physiological impossibility / telemetry sensor malfunction</td>
                                <td style="padding: 10px;"><span style="background: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">Flag Sensor Invalidation</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-family: monospace; font-weight: 700; color: #D97706;">RULE-PULSE-PRESSURE</td>
                                <td style="padding: 10px; font-size: 0.80rem;">SBP - DBP &lt; 18 mmHg</td>
                                <td style="padding: 10px; font-size: 0.82rem; color: #475569;">Critically narrow pulse pressure indicating cardiac tamponade or vasoconstriction</td>
                                <td style="padding: 10px;"><span style="background: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">Flag Hemodynamic Hazard</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-family: monospace; font-weight: 700; color: #2563EB;">RULE-CXR-ESCALATION</td>
                                <td style="padding: 10px; font-size: 0.80rem;">Pneumothorax conf &ge; 0.85</td>
                                <td style="padding: 10px; font-size: 0.82rem; color: #475569;">Auxiliary imaging detection of tension pneumothorax overriding intake</td>
                                <td style="padding: 10px;"><span style="background: #EFF6FF; color: #1E40AF; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">Force KTAS 1 Override</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # TAB 4: MODEL REGISTRY & CHECKPOINTS
    # -------------------------------------------------------------------------
    with tab_registry:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <div>
                        <span class="card-header-title">Production Model Registry & Checkpoints Manifest</span>
                        <div class="card-header-desc">Cryptographically verified weights preserved without modification in models/checkpoints/</div>
                    </div>
                </div>
                <div style="overflow-x: auto; margin-top: 12px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.84rem; text-align: left;">
                        <thead>
                            <tr style="border-bottom: 2px solid #CBD5E1; color: #475569;">
                                <th style="padding: 10px;">Model Component</th>
                                <th style="padding: 10px;">Checkpoint Path</th>
                                <th style="padding: 10px;">Architecture Type</th>
                                <th style="padding: 10px;">Input / Latent Dim</th>
                                <th style="padding: 10px;">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight: 600; color: #1E293B;">MiniLM-L6-v2 Encoder</td>
                                <td style="padding: 10px; font-family: monospace; font-size: 0.78rem;">all-MiniLM-L6-v2</td>
                                <td style="padding: 10px;">Transformer Bi-Encoder</td>
                                <td style="padding: 10px; font-family: monospace;">384-d semantic</td>
                                <td style="padding: 10px;"><span style="color: #059669; font-weight: 600;">Active Online</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight: 600; color: #1E293B;">Vitals MLP Encoder</td>
                                <td style="padding: 10px; font-family: monospace; font-size: 0.78rem;">vitals_encoder.pt</td>
                                <td style="padding: 10px;">Multi-Layer Perceptron</td>
                                <td style="padding: 10px; font-family: monospace;">7-d &rarr; 128-d</td>
                                <td style="padding: 10px;"><span style="color: #059669; font-weight: 600;">Verified (Intact)</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight: 600; color: #1E293B;">Context MLP Encoder</td>
                                <td style="padding: 10px; font-family: monospace; font-size: 0.78rem;">context_encoder.pt</td>
                                <td style="padding: 10px;">Multi-Layer Perceptron</td>
                                <td style="padding: 10px; font-family: monospace;">Tabular &rarr; 64-d</td>
                                <td style="padding: 10px;"><span style="color: #059669; font-weight: 600;">Verified (Intact)</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight: 600; color: #1E293B;">DenseNet-121 CheXNet</td>
                                <td style="padding: 10px; font-family: monospace; font-size: 0.78rem;">chest_xray_densenet121.pt</td>
                                <td style="padding: 10px;">Densely Connected CNN</td>
                                <td style="padding: 10px; font-family: monospace;">(3, 224, 224) &rarr; 14 findings</td>
                                <td style="padding: 10px;"><span style="color: #059669; font-weight: 600;">Verified (Intact)</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight: 600; color: #1E293B;">Adaptive Reliability Fusion (E0)</td>
                                <td style="padding: 10px; font-family: monospace; font-size: 0.78rem;">adaptive_fusion_model.pt</td>
                                <td style="padding: 10px;">Reliability-Gated Fusion</td>
                                <td style="padding: 10px; font-family: monospace;">576-d &rarr; 5 KTAS classes</td>
                                <td style="padding: 10px;"><span style="color: #059669; font-weight: 600;">Verified (Intact)</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0; background: #EFF6FF;">
                                <td style="padding: 10px; font-weight: 600; color: #1D4ED8;">RCCF-Net v2 (E2 / E3)</td>
                                <td style="padding: 10px; font-family: monospace; font-size: 0.78rem;">rccf_net_v2_model.pt</td>
                                <td style="padding: 10px;">Residual Cross-Attention</td>
                                <td style="padding: 10px; font-family: monospace;">Cross-Modal &rarr; 5 KTAS classes</td>
                                <td style="padding: 10px;"><span style="color: #1D4ED8; font-weight: 700;">Trained & Calibrated</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # TAB 5: KTAS ACUITY & SAFETY PROTOCOL
    # -------------------------------------------------------------------------
    with tab_ktas:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <div>
                        <span class="card-header-title">Korean Triage and Acuity Scale (KTAS 1-5) Standard</span>
                        <div class="card-header-desc">National emergency medical triage classification and mandated clinical assessment latency</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_k1, col_k2 = st.columns([1, 1])
        with col_k1:
            st.markdown(
                """
                <div class="clinical-card" style="border-left: 4px solid #DC2626;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 800; font-size: 0.95rem; color: #DC2626;">KTAS Level 1: Resuscitation</span>
                        <span style="font-size: 0.75rem; background: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-weight: 700;">Immediate (0 min)</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.5;">
                        Threat to life or limb or imminent risk of deterioration. Cardiac arrest, severe circulatory shock, major trauma with respiratory arrest, coma (GCS &lt; 8).
                    </p>
                </div>
                <div class="clinical-card" style="border-left: 4px solid #EA580C; margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 800; font-size: 0.95rem; color: #EA580C;">KTAS Level 2: Emergent</span>
                        <span style="font-size: 0.75rem; background: #FFEDD5; color: #9A3412; padding: 2px 8px; border-radius: 4px; font-weight: 700;">&le; 15 minutes</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.5;">
                        Potential threat to life, limb, or organ function. Acute myocardial infarction, severe dyspnea, stroke symptoms &lt;4.5h, acute severe pain &ge;8/10.
                    </p>
                </div>
                <div class="clinical-card" style="border-left: 4px solid #CA8A04; margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 800; font-size: 0.95rem; color: #CA8A04;">KTAS Level 3: Urgent</span>
                        <span style="font-size: 0.75rem; background: #FEF9C3; color: #854D0E; padding: 2px 8px; border-radius: 4px; font-weight: 700;">&le; 30 minutes</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.5;">
                        Condition related to serious illness, work, or distress. Moderate asthma, mild respiratory distress, moderate pain (4-7/10), persistent vomiting without dehydration.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_k2:
            st.markdown(
                """
                <div class="clinical-card" style="border-left: 4px solid #16A34A;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 800; font-size: 0.95rem; color: #16A34A;">KTAS Level 4: Less Urgent</span>
                        <span style="font-size: 0.75rem; background: #DCFCE7; color: #166534; padding: 2px 8px; border-radius: 4px; font-weight: 700;">&le; 60 minutes</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.5;">
                        Condition related to age, distress, or potential for complications. Minor trauma, simple laceration, mild sprains, urinary tract infection without fever.
                    </p>
                </div>
                <div class="clinical-card" style="border-left: 4px solid #2563EB; margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 800; font-size: 0.95rem; color: #2563EB;">KTAS Level 5: Non-Urgent</span>
                        <span style="font-size: 0.75rem; background: #DBEAFE; color: #1E40AF; padding: 2px 8px; border-radius: 4px; font-weight: 700;">&le; 120 minutes</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.5;">
                        Conditions that may be acute but non-urgent, or part of a chronic problem. Suture removal, medication refill, chronic minor skin rash, mild sore throat.
                    </p>
                </div>
                <div class="clinical-card" style="background: #F8FAFC; border: 1px dashed #CBD5E1; margin-top: 12px;">
                    <div style="font-weight: 700; font-size: 0.88rem; color: #1E293B; margin-bottom: 6px;">
                        Decoupled Radiographic Governance Protocol
                    </div>
                    <p style="font-size: 0.81rem; color: #64748B; margin: 0; line-height: 1.5;">
                        Chest X-rays are auxiliary diagnostic tests ordered subsequent to initial triage. In strict adherence to scientific zero-leakage protocols, chest radiographs are not mixed into non-radiographic emergency intake splits, preventing clinical data leakage and distribution shift.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

