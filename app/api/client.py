"""
HTTP API Client for Multimodal Intelligent Triage Assistant.
============================================================
Communicates with the FastAPI backend over HTTP (CORS enabled) on port 8000.
Provides client methods for triage prediction, CECE audit, and vision analysis,
with automatic fallback to in-process inference if the backend is unreachable.
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Dict, Optional
import httpx

logger = logging.getLogger("triage_client")

API_DEFAULT_BASE_URL = os.environ.get("TRIAGE_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("TRIAGE_API_KEY", "triage-secret-key-2026")


class TriageAPIClient:
    """HTTP client communicating with FastAPI backend via REST / CORS."""

    def __init__(self, base_url: str = API_DEFAULT_BASE_URL, api_key: str = API_KEY, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "X-API-Key": self.api_key,
            "User-Agent": "MultimodalTriageClient/2.0",
            "Accept": "application/json",
        }

    def is_backend_online(self) -> bool:
        """Pings the FastAPI /health endpoint."""
        try:
            with httpx.Client(timeout=1.5) as client:
                resp = client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    def predict_triage(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sends patient intake to POST /api/v1/triage/predict via HTTP CORS."""
        url = f"{self.base_url}/api/v1/triage/predict"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload, headers=self.headers)
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(f"FastAPI /predict returned status {resp.status_code}: {resp.text}")
        except Exception as exc:
            logger.warning(f"Could not connect to FastAPI backend at {url}: {exc}")
        return None

    def audit_cece(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sends patient intake to POST /api/v1/cece/audit via HTTP CORS."""
        url = f"{self.base_url}/api/v1/cece/audit"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload, headers=self.headers)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as exc:
            logger.warning(f"Could not connect to FastAPI /cece/audit: {exc}")
        return None

    def analyze_vision(self, image_bytes: bytes, target_finding: str = "Pneumothorax") -> Optional[Dict[str, Any]]:
        """Sends chest radiograph to POST /api/v1/vision/analyze via HTTP multipart."""
        url = f"{self.base_url}/api/v1/vision/analyze"
        try:
            files = {"file": ("radiograph.png", image_bytes, "image/png")}
            data = {"target_finding": target_finding}
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, files=files, data=data, headers=self.headers)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as exc:
            logger.warning(f"Could not connect to FastAPI /vision/analyze: {exc}")
        return None


# Module-level singleton
api_client = TriageAPIClient()

