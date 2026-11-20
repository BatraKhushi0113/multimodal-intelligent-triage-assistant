# 🚀 Cloud Deployment Guide — Render & Docker

This document describes how to deploy the **Multimodal Intelligent Triage Assistant (MITA)** to **Render** using Docker.

---

## 🏗️ Architecture Overview on Render

MITA can be deployed on Render in two modes:

| Mode | `SERVICE_TYPE` | Port Binding | Description |
| :--- | :---: | :---: | :--- |
| **Unified Portal (Recommended)** | `unified` | `$PORT` | Runs FastAPI backend on internal port `8000` and Streamlit Clinical UI on public Render `$PORT`. |
| **Clinical UI Only** | `ui` | `$PORT` | Streamlit clinical portal running on `$PORT` with in-process ML inference fallback. |
| **REST API Only** | `api` | `$PORT` | FastAPI backend running on `$PORT` with interactive Swagger docs at `/docs` and health at `/health`. |

---

## 🛠️ Option 1: One-Click Render Blueprint (Recommended)

1. Log into your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** > **Blueprint**.
3. Connect your GitHub repository: `https://github.com/BatraKhushi0113/multimodal-intelligent-triage-assistant`.
4. Render will automatically detect `render.yaml` and configure:
   - **Service Name**: `mita-clinical-assistant`
   - **Runtime**: `Docker`
   - **DockerfilePath**: `./Dockerfile`
   - **Health Check Path**: `/_stcore/health`
5. Click **Apply**. Render will build the container and deploy your service!

---

## ⚙️ Option 2: Manual Web Service Setup on Render

If creating a Web Service manually in Render:

1. Click **New +** > **Web Service**.
2. Select your repository: `BatraKhushi0113/multimodal-intelligent-triage-assistant`.
3. Choose the `main` branch.
4. **Environment**: `Docker` (Render will use `Dockerfile`).
5. **Region**: Oregon (US West) or Frankfurt (EU).
6. **Plan**: `Free` or `Starter`.
7. **Health Check Path**: `/_stcore/health` (or `/health` if in API mode).
8. **Environment Variables**:
   * `SERVICE_TYPE`: `unified` (or `ui`)
   * `PYTHONUNBUFFERED`: `1`
   * `MPLCONFIGDIR`: `/tmp/mpl`
   * `TRIAGE_API_KEY`: *(Auto-generated or custom secure secret)*
9. Click **Deploy Web Service**.

---

## 🐳 Local Docker Testing

To verify the Docker container locally before cloud deployment:

```bash
# 1. Build the Docker image
docker build -t mita-clinical-assistant .

# 2. Run the container locally (mapped to port 8501)
docker run -p 8501:8501 -e PORT=8501 -e SERVICE_TYPE=unified mita-clinical-assistant

# 3. Open your browser
# Clinical UI: http://localhost:8501
```

---

## 🔒 Security & Privacy Notice
* All patient identifiers are synthetic or de-identified.
* Real credentials and API keys are read from environment variables; none are stored in source code.
* The system is for educational and research prototype demonstration only.
