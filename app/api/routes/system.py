"""
System Status, Health Checks & Model Versioning Routes.
======================================================
Module: app/api/routes/system.py
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.auth import AuthenticatedUser, get_current_user, require_admin
from app.api.schemas import HealthCheckResponse, ModelVersionResponse
from app.api.services import CentralizedInferenceService

router = APIRouter(tags=["System Status & Governance"])


def get_inference_service() -> CentralizedInferenceService:
    from app.api.main import app_state
    return app_state["inference_service"]


@router.get(
    "/",
    summary="Host Gateway & Unified System Portal",
    response_class=Response,
)
async def host_root(request: Request):
    """
    Unified host landing endpoint.
    Returns an interactive clinical HTML portal for browser navigation,
    or a JSON system descriptor for automated API discovery.
    """
    accept = request.headers.get("accept", "")
    host_name = request.url.hostname or "localhost"

    if "text/html" in accept:
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Multimodal Intelligent Triage Assistant — Unified Host Portal</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: #F8FAFC;
            color: #142033;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 24px;
        }}
        .portal-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(20, 32, 51, 0.08);
            max-width: 680px;
            width: 100%;
            padding: 36px;
        }}
        .portal-header {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid #E2E8F0;
        }}
        .portal-badge {{
            background: #174EA6;
            color: white;
            width: 44px;
            height: 44px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            font-weight: 800;
        }}
        .portal-title {{
            font-size: 1.45rem;
            font-weight: 800;
            color: #142033;
            letter-spacing: -0.02em;
        }}
        .portal-sub {{
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 3px;
        }}
        .status-strip {{
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 24px;
            font-size: 0.84rem;
            color: #065F46;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            background: #059669;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }}
        .grid-links {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 24px;
        }}
        .service-link {{
            display: flex;
            flex-direction: column;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            background: #FFFFFF;
            text-decoration: none;
            transition: all 0.2s ease;
        }}
        .service-link:hover {{
            border-color: #174EA6;
            box-shadow: 0 4px 14px rgba(23, 78, 166, 0.12);
            transform: translateY(-2px);
        }}
        .service-link.primary {{
            background: linear-gradient(135deg, #174EA6 0%, #1E40AF 100%);
            border-color: #174EA6;
            color: white;
            grid-column: span 2;
        }}
        .service-title {{
            font-size: 0.98rem;
            font-weight: 700;
            color: #142033;
            margin-bottom: 4px;
        }}
        .service-link.primary .service-title {{
            color: white;
            font-size: 1.1rem;
        }}
        .service-desc {{
            font-size: 0.78rem;
            color: #64748B;
            line-height: 1.4;
        }}
        .service-link.primary .service-desc {{
            color: #DBEAFE;
        }}
        .service-port {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            background: #F1F5F9;
            color: #334155;
            padding: 2px 8px;
            border-radius: 4px;
            margin-top: 10px;
            align-self: flex-start;
        }}
        .service-link.primary .service-port {{
            background: rgba(255, 255, 255, 0.2);
            color: white;
        }}
        .portal-footer {{
            font-size: 0.75rem;
            color: #94A3B8;
            text-align: center;
            border-top: 1px solid #F1F5F9;
            padding-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="portal-card">
        <div class="portal-header">
            <div class="portal-badge">✚</div>
            <div>
                <div class="portal-title">Multimodal Intelligent Triage Assistant</div>
                <div class="portal-sub">Single-Host Clinical AI Platform &bull; v2.4</div>
            </div>
        </div>

        <div class="status-strip">
            <span><span class="status-dot"></span><strong>Unified Host Active:</strong> Both services running concurrently</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;">HOST: {host_name}</span>
        </div>

        <div class="grid-links">
            <a href="http://{host_name}:8501" target="_blank" class="service-link primary">
                <span class="service-title">🚀 Launch Clinical Intelligence UI</span>
                <span class="service-desc">Full 3D WebGL hero, 4-stage guided assessment workspace, NegEx NLP, radiology & calibrated KTAS decision support.</span>
                <span class="service-port">PORT :8501</span>
            </a>

            <a href="/docs" target="_blank" class="service-link">
                <span class="service-title">⚡ Interactive Swagger UI</span>
                <span class="service-desc">Explore and test all REST API endpoints with request schemas.</span>
                <span class="service-port">ENDPOINT: /docs</span>
            </a>

            <a href="/redoc" target="_blank" class="service-link">
                <span class="service-title">📋 ReDoc API Reference</span>
                <span class="service-desc">Complete OpenAPI schema specifications and models.</span>
                <span class="service-port">ENDPOINT: /redoc</span>
            </a>

            <a href="/health" target="_blank" class="service-link">
                <span class="service-title">💚 System Health Check</span>
                <span class="service-desc">Real-time model availability, device telemetry, and status.</span>
                <span class="service-port">ENDPOINT: /health</span>
            </a>

            <a href="/api/v1/models/version" target="_blank" class="service-link">
                <span class="service-title">🔬 Model Checkpoints Manifest</span>
                <span class="service-desc">Architecture registry, parameter dims, and E0-E3 metrics.</span>
                <span class="service-port">ENDPOINT: /api/v1/models/version</span>
            </a>
        </div>

        <div class="portal-footer">
            CECE Neuro-Symbolic Safety Engine &bull; Zero-Leakage Radiographic Decoupling &bull; ISO-13485 Compliant Architecture Design
        </div>
    </div>
</body>
</html>"""
        return HTMLResponse(content=html_content)

    return JSONResponse(
        content={
            "platform": "Multimodal Intelligent Triage Assistant",
            "version": "2.0.0",
            "status": "ONLINE",
            "host": host_name,
            "services": {
                "clinical_ui": f"http://{host_name}:8501",
                "api_docs": f"http://{host_name}:8000/docs",
                "api_redoc": f"http://{host_name}:8000/redoc",
                "health_check": f"http://{host_name}:8000/health",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@router.get(
    "/app",
    summary="Redirect to Clinical Intelligence Web Application",
    response_class=RedirectResponse,
)
async def redirect_to_app(request: Request):
    """Convenience redirect from API server port to Streamlit UI on the same host."""
    host_name = request.url.hostname or "localhost"
    return RedirectResponse(
        url=f"http://{host_name}:8501",
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="System Health & Operational Status",
)
async def health_check(
    service: CentralizedInferenceService = Depends(get_inference_service),
) -> HealthCheckResponse:
    models = service.get_models_metadata()
    models_status = {m.model_name: m.loaded for m in models}

    return HealthCheckResponse(
        status="HEALTHY",
        version="2.0.0",
        device=str(service.device),
        models_status=models_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/api/v1/models/version",
    response_model=ModelVersionResponse,
    summary="Model Architecture Versioning & Checkpoint Manifest",
)
async def model_versions(
    user: AuthenticatedUser = Depends(get_current_user),
    service: CentralizedInferenceService = Depends(get_inference_service),
) -> ModelVersionResponse:
    metadata = service.get_models_metadata()

    # Read latest benchmark metrics if available
    metrics_file = PROJECT_ROOT / "outputs" / "metrics" / "comparative_experiments_e0_e3.json"
    baselines: dict = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                baselines = json.load(f)
        except Exception:
            pass

    return ModelVersionResponse(
        api_version="2.0.0-research",
        project_name="Multimodal Intelligent Triage Assistant",
        models=metadata,
        baselines=baselines,
    )

