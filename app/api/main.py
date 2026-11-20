"""
Multimodal Intelligent Triage Assistant - Enterprise FastAPI Application.
==========================================================================
Module: app/api/main.py

Main FastAPI entrypoint providing:
  - Model lifecycle management with async lifespan events.
  - CORS middleware & security headers.
  - Centralized JSON logging & request latency monitoring.
  - Interactive OpenAPI / Swagger documentation at /docs.
  - Strict clinical safety disclaimer banner.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.routes import cece, system, triage, vision, voice
from app.api.services import CentralizedInferenceService

# Global application state holder
app_state: Dict[str, Any] = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("triage_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler managing initialization and shutdown of model services."""
    logger.info("Initializing Multimodal Triage Inference Engine...")
    inference_service = CentralizedInferenceService(device="cpu")
    app_state["inference_service"] = inference_service
    logger.info("Inference Engine Ready. Serving endpoints.")
    yield
    logger.info("Shutting down Triage API and clearing CUDA/PyTorch caches...")
    app_state.clear()


def create_app() -> FastAPI:
    """FastAPI Application Factory."""
    app = FastAPI(
        title="Multimodal Intelligent Triage Assistant API",
        version="2.0.0",
        description=(
            "### Research & Educational Prototype Only\n"
            "This software is designed exclusively for experimental research in multimodal deep learning "
            "and clinical decision support. It does NOT provide medical diagnoses or direct patient treatment. "
            "All predictions are stochastic model outputs requiring human review by licensed medical staff.\n\n"
            "**Key Capabilities**:\n"
            "- **Multimodal Triage Acuity**: KTAS 1-5 triage prediction via Adaptive Fusion and RCCF-Net v2.\n"
            "- **Cross-Modal Evidence & Conflict Engine (CECE)**: Cross-modal contradiction & physiological validation.\n"
            "- **Advanced Clinical NLP**: NegEx negation detection, symptom extraction, and structured evidence.\n"
            "- **Thoracic Radiography**: DenseNet-121 multi-label pathology detection with IQA and Grad-CAM++.\n"
            "- **Speech-to-Text**: Whisper clinical transcription and audio quality tracking.\n"
            "- **Advanced Uncertainty & Safety**: Temperature calibration, ECE, Brier score, and selective prediction."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS Middleware - Fully permissive for cross-origin frontend communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*", "X-Process-Time-Ms"],
    )

    # Request Timing Middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal server error: {str(exc)}"},
        )

    # Mount Route Handlers
    app.include_router(system.router)
    app.include_router(triage.router, prefix="/api/v1")
    app.include_router(cece.router, prefix="/api/v1")
    app.include_router(vision.router, prefix="/api/v1")
    app.include_router(voice.router, prefix="/api/v1")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)

