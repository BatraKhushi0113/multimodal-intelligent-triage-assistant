"""
Integration & Unit Tests for FastAPI Backend.
=============================================
Tests /health, /api/v1/models/version, /api/v1/triage/predict (Adaptive & RCCF-Net v2),
/api/v1/cece/audit, and /api/v1/vision/analyze using TestClient.
"""

import io
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.main import app
from src.vision.xray_model import generate_synthetic_chest_xray


@pytest.fixture(scope="module")
def client():
    """Module-level FastAPI test client with managed lifespan."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client: TestClient):
    """Verify /health returns operational status and model checkpoint checklist."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["version"] == "2.0.0"
    assert "models_status" in data
    assert data["models_status"]["AdaptiveReliabilityFusion"] is True


def test_model_versioning_endpoint(client: TestClient):
    """Verify /api/v1/models/version returns metadata on models and baselines."""
    response = client.get("/api/v1/models/version", headers={"X-API-Key": "triage-secret-key-2026"})
    assert response.status_code == 200
    data = response.json()
    assert data["api_version"] == "2.0.0-research"
    assert len(data["models"]) >= 4
    model_names = [m["model_name"] for m in data["models"]]
    assert "RCCFNetV2" in model_names
    assert "AdaptiveReliabilityFusion" in model_names


def test_triage_predict_adaptive_baseline(client: TestClient):
    """Verify /api/v1/triage/predict executes baseline adaptive fusion."""
    payload = {
        "patient_id": "PT-TEST-001",
        "age": 55.0,
        "sex": "Male",
        "arrival_mode": "119",
        "injury": "Non-trauma",
        "mental_status": "Alert",
        "chief_complaint": "Acute severe crushing chest pain radiating to left jaw for 1 hour.",
        "sbp": 135.0,
        "dbp": 85.0,
        "hr": 95.0,
        "rr": 20.0,
        "bt": 37.0,
        "saturation": 97.0,
        "pain": 1,
        "nrs_pain": 8.0,
        "model_architecture": "adaptive",
    }
    response = client.post("/api/v1/triage/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert 1 <= res["predicted_ktas_level"] <= 5
    assert res["predicted_ktas_level"] in [1, 2, 3]  # Acute chest pain presentation
    assert "class_probabilities" in res
    assert len(res["class_probabilities"]) == 5
    assert "modality_weights" in res
    assert "safety_verdict" in res
    assert res["safety_verdict"]["verdict"] in [
        "AUTOMATED_DECISION_CLEARED",
        "DEFERRED_FOR_NURSE_TRIAGE",
        "IMMEDIATE_PHYSICIAN_OVERRIDE",
    ]


def test_triage_predict_rccf_v2(client: TestClient):
    """Verify /api/v1/triage/predict executes RCCF-Net v2 architecture."""
    payload = {
        "patient_id": "PT-TEST-RCCF",
        "age": 68.0,
        "sex": "Female",
        "arrival_mode": "Walking",
        "injury": "Non-trauma",
        "mental_status": "Alert",
        "chief_complaint": "Shortness of breath and productive cough for 3 days.",
        "sbp": 122.0,
        "dbp": 78.0,
        "hr": 88.0,
        "rr": 22.0,
        "bt": 38.2,
        "saturation": 93.0,
        "pain": 0,
        "nrs_pain": 2.0,
        "model_architecture": "rccf_v2",
    }
    response = client.post("/api/v1/triage/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert 1 <= res["predicted_ktas_level"] <= 5
    assert "RCCF-Net v2" in res["architecture_used"]
    assert res["modality_weights"]["Text"] > 0.0
    assert res["modality_weights"]["Vitals"] > 0.0


def test_cece_audit_endpoint(client: TestClient):
    """Verify /api/v1/cece/audit returns concordance score and conflict diagnostics."""
    payload = {
        "patient_id": "PT-CECE-TEST",
        "age": 42.0,
        "chief_complaint": "Denies fever, denies pain.",
        "sbp": 120.0,
        "dbp": 80.0,
        "hr": 105.0,
        "rr": 18.0,
        "bt": 39.5,  # Pyrexia discordance
        "saturation": 98.0,
        "nrs_pain": 8.0,  # Pain discordance
        "pain": 1,
    }
    response = client.post("/api/v1/cece/audit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["conflict_severity"] in ["MODERATE", "CRITICAL"]
    assert data["requires_human_review"] is True
    assert len(data["conflicts"]) >= 1


def test_triage_validation_error_handling(client: TestClient):
    """Verify invalid payloads return HTTP 422 Unprocessable Entity."""
    # Age out of bounds (> 125)
    invalid_payload = {
        "patient_id": "PT-INVALID",
        "age": 999.0,
    }
    response = client.post("/api/v1/triage/predict", json=invalid_payload)
    assert response.status_code == 422


def test_vision_analyze_endpoint(client: TestClient):
    """Verify /api/v1/vision/analyze accepts uploaded image bytes and returns full vision diagnostic."""
    pil_img = generate_synthetic_chest_xray(pathology="pneumothorax", image_size=224, random_seed=42)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    files = {"file": ("xray.png", img_bytes, "image/png")}
    response = client.post("/api/v1/vision/analyze", files=files, data={"target_finding": "Pneumothorax"})
    assert response.status_code == 200
    res = response.json()
    assert res["evaluated"] is True
    assert len(res["predictions"]) == 14
    assert res["quality_report"]["is_valid"] is True
    assert res["cam_overlay_base64"] is not None


def test_unified_host_portal_endpoint(client: TestClient):
    """Verify GET / returns interactive HTML portal or JSON depending on Accept header."""
    # HTML request
    resp_html = client.get("/", headers={"Accept": "text/html"})
    assert resp_html.status_code == 200
    assert "text/html" in resp_html.headers["content-type"]
    assert "Unified Host Active" in resp_html.text

    # JSON request
    resp_json = client.get("/", headers={"Accept": "application/json"})
    assert resp_json.status_code == 200
    data = resp_json.json()
    assert data["status"] == "ONLINE"
    assert "8501" in data["services"]["clinical_ui"]
    assert "8000" in data["services"]["api_docs"]


def test_app_redirect_endpoint(client: TestClient):
    """Verify GET /app issues HTTP 307 temporary redirect to Streamlit UI on :8501."""
    resp = client.get("/app", follow_redirects=False)
    assert resp.status_code == 307
    assert ":8501" in resp.headers["location"]


