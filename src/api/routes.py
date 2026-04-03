from __future__ import annotations

import time
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from loguru import logger
from src.db.mongo import get_async_db


from src.pipeline.orchestrator import ARGUSPipeline, RunStore
from src.pipeline.scheduler import PipelineScheduler
from src.agents.active_learning import ActiveLearningQueue
from src.utils.config import cfg

router   = APIRouter()
store    = RunStore()
al_queue = ActiveLearningQueue()

# Single shared pipeline instance (loaded once at startup)
_pipeline: ARGUSPipeline | None = None


def get_pipeline() -> ARGUSPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ARGUSPipeline()
    return _pipeline


# ── Request / Response schemas ────────────────────────────────────────────────

class DetectRequest(BaseModel):
    lat_min: float = Field(..., ge=-90,  le=90,  description="Southern latitude bound")
    lat_max: float = Field(..., ge=-90,  le=90,  description="Northern latitude bound")
    lon_min: float = Field(..., ge=-180, le=180, description="Western longitude bound")
    lon_max: float = Field(..., ge=-180, le=180, description="Eastern longitude bound")
    date:    str | None = Field(None, description="ISO date string e.g. 2024-06-01")

    @field_validator("lat_max")
    @classmethod
    def lat_max_gt_min(cls, v, info):
        if "lat_min" in info.data and v <= info.data["lat_min"]:
            raise ValueError("lat_max must be greater than lat_min")
        return v

    @field_validator("lon_max")
    @classmethod
    def lon_max_gt_min(cls, v, info):
        if "lon_min" in info.data and v <= info.data["lon_min"]:
            raise ValueError("lon_max must be greater than lon_min")
        return v


class ReviewLabelRequest(BaseModel):
    detection_id: int
    run_id:       str
    is_plume:     bool
    reviewer:     str = "human_expert"
    notes:        str = ""


# ── Core detection endpoint (required by problem statement) ───────────────────

@router.post("/detect", tags=["Detection"])
async def detect(
    req:              DetectRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    **Primary endpoint** — accepts a geographic bounding box and returns
    all detected methane plumes with flux estimates and source attribution.

    This is the exact endpoint format required by the EN02 problem statement.
    """
    t0 = time.perf_counter()

    try:
        date_obj = datetime.fromisoformat(req.date) if req.date else None
        pipeline = get_pipeline()

        result = pipeline.run(
            lat_min=req.lat_min,
            lat_max=req.lat_max,
            lon_min=req.lon_min,
            lon_max=req.lon_max,
            date=date_obj,
        )

        # Persist result in background so response is not blocked
        background_tasks.add_task(store.save, result)

        logger.info(
            f"API /detect: {result.n_super_emitters} super-emitters | "
            f"{time.perf_counter()-t0:.2f}s"
        )
        return result.to_api_dict()

    except Exception as e:
        logger.error(f"API /detect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Historical heatmap data ───────────────────────────────────────────────────

@router.get("/heatmap")
async def heatmap(n_runs: int = 100):
    db = get_async_db()
    detections = await db.detections.find(
        {}, {"_id": 0}
    ).sort("timestamp", -1).limit(500).to_list(500)

    runs = await db.runs.find(
        {}, {"_id": 0}
    ).sort("timestamp", -1).limit(n_runs).to_list(n_runs)

    return {
        "total_detections": len(detections),
        "detections":       detections,
        "runs_summary": [
            {
                "run_id":           r["run_id"],
                "timestamp":        r["timestamp"],
                "n_super_emitters": r["summary"]["n_super_emitters"],
                "total_flux_kg_hr": r["summary"]["total_flux_kg_hr"],
                "total_impact_usd": r["summary"]["total_impact_usd"],
            }
            for r in runs
        ],
    }



# ── Facility risk scorecard ───────────────────────────────────────────────────

@router.get("/scorecard")
async def scorecard(limit: int = 50):
    db    = get_async_db()
    cards = await db.scorecard.find(
        {}, {"_id": 0}
    ).sort("compliance_score", 1).limit(limit).to_list(limit)

    return {
        "total_facilities": len(cards),
        "scorecard":        cards,
        "generated_at":     datetime.utcnow().isoformat(),
    }


# ── Per-facility detail ───────────────────────────────────────────────────────

@router.get("/facility/{facility_id}", tags=["Facilities"])
async def facility_detail(facility_id: str) -> dict:
    """Full profile + detection history for a single facility."""
    from src.data.facility_db import get_facility_by_id

    record = get_facility_by_id(facility_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Facility {facility_id} not found")

    # Attach detection history from run store
    all_dets = store.load_all_detections()
    fac_dets = [
        d for d in all_dets
        if d.get("attribution", {}).get("facility_id") == facility_id
    ]

    record["detection_history"] = fac_dets
    record["n_detections"]      = len(fac_dets)
    record["total_flux_kg_hr"]  = sum(
        d.get("flux_kg_hr", 0) for d in fac_dets
    )
    return record


# ── Active learning review queue ──────────────────────────────────────────────

@router.get("/review-queue", tags=["Active Learning"])
async def review_queue() -> dict:
    """Returns all uncertain detections awaiting human expert review."""
    queue = al_queue.load_queue()
    curve = al_queue.learning_curve()
    return {
        "queue_size":     len(queue),
        "items":          queue,
        "learning_curve": curve,
    }


@router.post("/review-queue/label", tags=["Active Learning"])
async def submit_label(req: ReviewLabelRequest) -> dict:
    """
    Submit a human expert label for a queued detection.
    Labels are written to the training data directory for Stage 1 retraining.
    """
    import json
    from pathlib import Path

    label_path = Path("data/active_learning/labels.jsonl")
    label_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "detection_id": req.detection_id,
        "run_id":       req.run_id,
        "is_plume":     req.is_plume,
        "reviewer":     req.reviewer,
        "notes":        req.notes,
        "labeled_at":   datetime.utcnow().isoformat(),
    }

    with open(label_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        f"API /review-queue/label: "
        f"det={req.detection_id} | is_plume={req.is_plume} | by={req.reviewer}"
    )
    return {"status": "accepted", "record": record}


# ── Economic impact summary ───────────────────────────────────────────────────

@router.get("/economic-summary", tags=["Intelligence"])
async def economic_summary() -> dict:
    """
    Aggregates total economic impact across all recent runs.
    This feeds the live economic ticker on the dashboard.
    """
    runs = store.load_recent(n=100)

    total_usd = sum(r["summary"]["total_impact_usd"] for r in runs)
    total_inr = total_usd * 83.5
    total_flux = sum(r["summary"]["total_flux_kg_hr"] for r in runs)
    total_super = sum(r["summary"]["n_super_emitters"] for r in runs)

    return {
        "total_runs":           len(runs),
        "total_super_emitters": total_super,
        "total_flux_kg_hr":     round(total_flux, 2),
        "total_impact_usd":     round(total_usd, 2),
        "total_impact_inr":     round(total_inr, 2),
        "total_impact_inr_cr":  round(total_inr / 1e7, 3),
        "as_of":                datetime.utcnow().isoformat(),
    }


# ── Health check ──────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
async def health() -> dict:
    return {
        "status":   "ok",
        "version":  "1.0.0",
        "model":    "ARGUS",
        "time":     datetime.utcnow().isoformat(),
    }