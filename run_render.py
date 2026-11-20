"""
Production Cloud Launcher for Render Deployment.
================================================
Binds dynamically to Render's assigned $PORT and 0.0.0.0.
Supports:
  - SERVICE_TYPE="ui" (default): Runs the full Streamlit Clinical Portal on $PORT.
  - SERVICE_TYPE="api": Runs the FastAPI REST API Service on $PORT.
  - SERVICE_TYPE="unified": Runs both FastAPI (internal) and Streamlit on $PORT.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure offline execution and redirect cache paths to /tmp for non-root containers
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

    elif service_type == "unified":
        # Launch FastAPI on internal port 8000 and Streamlit on public $PORT
        internal_api_port = "8000"
        api_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            internal_api_port,
            "--log-level",
            "warning",
        ]
        ui_cmd = [
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
            "--browser.gatherUsageStats",
            "false",
        ]

        env = os.environ.copy()
        env["TRIAGE_API_URL"] = f"http://127.0.0.1:{internal_api_port}"

        print(f"[*] Launching internal API on port {internal_api_port} and clinical UI on public port {port}...")
        api_proc = subprocess.Popen(api_cmd, env=env)
        time.sleep(2.0)
        ui_proc = subprocess.Popen(ui_cmd, env=env)

        try:
            while True:
                time.sleep(1.0)
                if ui_proc.poll() is not None:
                    break
                if api_proc.poll() is not None:
                    print("[!] Notice: Internal API service exited, UI continuing in fallback mode.")
        except KeyboardInterrupt:
            pass
        finally:
            for p in [ui_proc, api_proc]:
                if p and p.poll() is None:
                    p.terminate()
        sys.exit(0)

    else:
        # Default: Clinical UI on public $PORT
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
            "--browser.gatherUsageStats",
            "false",
        ]
        print(f"[*] Executing Streamlit Clinical Portal on port {port}...")
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

