"""
Multimodal Intelligent Clinical Triage Assistant — Unified Single-Host Launcher.
=================================================================================
Launches both the FastAPI Backend Service and the Streamlit Clinical Frontend
concurrently on a single host with process supervision, automated health checks,
and clean multi-process signal lifecycle management.

Usage:
    python run.py
    python run.py --host 0.0.0.0 --api-port 8000 --ui-port 8501
    python run.py --no-browser
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure offline operation and prevent font cache / hub latency on Windows
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "logs" / ".mpl"))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a network port is already open on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (socket.error, ConnectionRefusedError):
            return False


def wait_for_service(url: str, timeout_sec: float = 40.0, step_sec: float = 0.5) -> bool:
    """Polls a URL until it responds with HTTP 200/307 or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthChecker/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status in (200, 307):
                    return True
        except Exception:
            pass
        time.sleep(step_sec)
    return False


def print_banner(host: str, api_port: int, ui_port: int, api_pid: int, ui_pid: int) -> None:
    """Prints a styled clinical operational banner."""
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print("\n" + "=" * 82)
    print("  ✚  MULTIMODAL INTELLIGENT CLINICAL TRIAGE ASSISTANT — UNIFIED HOST")
    print("=" * 82)
    print(f"  ● Host Domain:                 http://{display_host}")
    print(f"  ● Clinical UI (Streamlit):     http://{display_host}:{ui_port}")
    print(f"  ● Backend API (FastAPI):       http://{display_host}:{api_port}")
    print(f"  ● Swagger Documentation:       http://{display_host}:{api_port}/docs")
    print(f"  ● ReDoc Technical Reference:   http://{display_host}:{api_port}/redoc")
    print(f"  ● System Health Endpoint:      http://{display_host}:{api_port}/health")
    print(f"  ● Unified Host Portal:         http://{display_host}:{api_port}/")
    print("-" * 82)
    print(f"  Processes: [Streamlit PID: {ui_pid}]  |  [FastAPI PID: {api_pid}]")
    print("  Status:    BOTH SERVICES ACTIVE & CONCURRENT ON ONE HOST")
    print("  Press Ctrl+C to stop all services simultaneously.")
    print("=" * 82 + "\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified Single-Host Runner for Multimodal Intelligent Triage Assistant."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind both services (default: 127.0.0.1)")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for FastAPI backend (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8501, help="Port for Streamlit clinical UI (default: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    parser.add_argument("--reload", action="store_true", help="Enable live code reloading for development")

    args = parser.parse_args()

    # Pre-flight Port Checks
    if is_port_in_use(args.api_port, args.host):
        print(f"[!] Warning: Port {args.api_port} is already in use. Please terminate existing processes or select another port via --api-port.")
    if is_port_in_use(args.ui_port, args.host):
        print(f"[!] Warning: Port {args.ui_port} is already in use. Please terminate existing processes or select another port via --ui-port.")

    print(f"[*] Booting Multimodal Intelligent Triage Assistant on unified host '{args.host}'...")

    # 1. Build FastAPI Uvicorn Command
    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api.main:app",
        "--host",
        args.host,
        "--port",
        str(args.api_port),
        "--log-level",
        "warning",
    ]
    if args.reload:
        api_cmd.append("--reload")

    # 2. Build Streamlit Frontend Command
    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.address",
        args.host,
        "--server.port",
        str(args.ui_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["TRIAGE_API_URL"] = f"http://{display_host}:{args.api_port}"

    proc_api: subprocess.Popen | None = None
    proc_ui: subprocess.Popen | None = None

    def shutdown(signum=None, frame=None):
        print("\n[*] Shutting down unified host services...")
        for p, name in [(proc_ui, "Streamlit UI"), (proc_api, "FastAPI API")]:
            if p and p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=3.0)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        print("[✓] All services cleanly terminated. Goodbye.\n")
        sys.exit(0)

    # Register OS signals for graceful shutdown
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        # Start FastAPI backend process
        print(f"[*] Starting FastAPI service (Port {args.api_port})...")
        proc_api = subprocess.Popen(api_cmd, env=env)

        # Start Streamlit UI process
        print(f"[*] Starting Streamlit clinical interface (Port {args.ui_port})...")
        proc_ui = subprocess.Popen(ui_cmd, env=env)

        # Wait for both services to become ready
        display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
        api_health_url = f"http://{display_host}:{args.api_port}/health"
        ui_root_url = f"http://{display_host}:{args.ui_port}"

        print(f"[*] Verifying services readiness on {display_host}...")
        api_ready = wait_for_service(api_health_url, timeout_sec=35.0)
        ui_ready = wait_for_service(ui_root_url, timeout_sec=35.0)

        if api_ready and ui_ready:
            print_banner(args.host, args.api_port, args.ui_port, proc_api.pid, proc_ui.pid)
            if not args.no_browser:
                try:
                    webbrowser.open(ui_root_url)
                except Exception:
                    pass
        else:
            print("[!] Notice: Startup took longer than expected; services are continuing in background.")
            print(f"    Check http://{display_host}:{args.ui_port} (UI) and http://{display_host}:{args.api_port} (API)")

        # Supervisor loop: Monitor both processes
        while True:
            time.sleep(1.0)
            if proc_api.poll() is not None:
                print(f"[!] FastAPI service exited prematurely with code {proc_api.returncode}")
                break
            if proc_ui.poll() is not None:
                print(f"[!] Streamlit UI exited prematurely with code {proc_ui.returncode}")
                break

    except KeyboardInterrupt:
        shutdown()
    except Exception as exc:
        print(f"[!] Unexpected error in host supervisor: {exc}")
    finally:
        shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())

