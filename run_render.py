"""
Production Cloud Launcher for Render Deployment.
================================================
Binds dynamically to Render's assigned $PORT and 0.0.0.0.
Optimized for single-process memory efficiency within Render's Free tier (512MB RAM).

Supports:
  - SERVICE_TYPE="ui" (default): Runs the full Streamlit Clinical Portal on $PORT.
  - SERVICE_TYPE="api": Runs the FastAPI REST API Service on $PORT.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure offline execution and redirect cache paths to /tmp
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)


def main():
    port = os.environ.get("PORT", "8501")
    host = "0.0.0.0"
    service_type = os.environ.get("SERVICE_TYPE", "ui").lower().strip()

    print(f"[*] Booting MITA Cloud Service on {host}:{port} [Mode: {service_type.upper()}]...")

    if service_type == "api":
        # Launch FastAPI Backend
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "info",
        ]
        print(f"[*] Executing FastAPI backend on port {port}...")
        sys.exit(subprocess.call(cmd))

    else:
        # Default: Clinical UI on public $PORT with reverse-proxy WebSocket & memory flags
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
            "--server.fileWatcherType",
            "none",
            "--browser.gatherUsageStats",
            "false",
        ]
        print(f"[*] Executing Streamlit Clinical Portal on port {port}...")
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
