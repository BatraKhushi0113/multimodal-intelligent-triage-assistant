"""
FastAPI Backend Application Subpackage.
"""

import os
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
mpl_dir = os.path.join(tempfile.gettempdir(), "mpl")
os.makedirs(mpl_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_dir)

from app.api.main import app, create_app

__all__ = ["app", "create_app"]
