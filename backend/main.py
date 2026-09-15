# -*- coding: utf-8 -*-
"""Good-Badminton FastAPI application.

Only responsible for: app setup, CORS, router registration, static mounts and
startup bootstrap. All business logic lives in ``backend/api`` (routes) and
``backend/services`` (workflow).
"""
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make the project root importable (backend, badminton_analysis) and
# anchor the working directory so the pipeline's relative paths resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backend.core.env import load_local_env  # noqa: E402

load_local_env()

from backend.api import analysis, batches, chat, court, jobs, system, videos  # noqa: E402
from backend.core.config import API_PREFIX, APP_NAME, CORS_ORIGINS  # noqa: E402
from backend.core.paths import OUTPUTS_DIR, UPLOADS_DIR, ensure_dirs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("goodbadminton.backend")

ensure_dirs()

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="AI badminton match analysis — React frontend + FastAPI backend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for uploads, analysis outputs and bundled sample media.
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

_VIDEOS_DIR = PROJECT_ROOT / "videos"
if _VIDEOS_DIR.is_dir():
    app.mount("/videos", StaticFiles(directory=str(_VIDEOS_DIR)), name="videos")

# Routers
app.include_router(videos.router, prefix=API_PREFIX)
app.include_router(court.router, prefix=API_PREFIX)
app.include_router(analysis.router, prefix=API_PREFIX)
app.include_router(batches.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(jobs.router, prefix=API_PREFIX)
app.include_router(system.router, prefix=API_PREFIX)


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "app": APP_NAME}


@app.on_event("startup")
def startup() -> None:
    logger.info("Good-Badminton API started (root=%s)", PROJECT_ROOT)
