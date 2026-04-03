from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger

from src.api.routes import router, get_pipeline
from src.pipeline.scheduler import PipelineScheduler
from src.utils.config import cfg
from src.db.mongo import setup_indexes

# ── Startup / shutdown ────────────────────────────────────────────────────────

scheduler: PipelineScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_indexes()
    global scheduler

    logger.info("ARGUS API: starting up...")

    # Pre-warm the pipeline on first request
    get_pipeline()

    # Start background scheduler (non-blocking)
    scheduler = PipelineScheduler(interval_seconds=300)
    scheduler.start()

    logger.info("ARGUS API: ready ✓")
    yield

    # Shutdown
    if scheduler:
        scheduler.stop()
    logger.info("ARGUS API: shut down cleanly")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ARGUS — Autonomous Real-time Greenhouse Gas Unified Surveillance",
    description=(
        "REST API for methane super-emitter detection, flux quantification, "
        "source attribution, and automated regulatory enforcement. "
        "Accepts geographic bounding boxes and returns plume detections "
        "with confidence-bounded flux estimates and facility-level attribution."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        workers=1,          # 1 worker — pipeline holds GPU memory
        reload=False,
        log_level="info",
    )