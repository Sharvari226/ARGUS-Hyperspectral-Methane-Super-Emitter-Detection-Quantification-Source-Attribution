from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from src.db.mongo import get_async_db, get_sync_db
from src.pipeline.orchestrator import ARGUSPipeline, RunStore
from src.pipeline.scheduler import PipelineScheduler
from src.agents.active_learning import ActiveLearningQueue
from src.utils.config import cfg

router    = APIRouter()
ws_router = APIRouter()
store     = RunStore()
al_queue  = ActiveLearningQueue()

_pipeline: ARGUSPipeline | None = None


# ── Pipeline ────────────────────────────────────────────────────────────────

def get_pipeline() -> ARGUSPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ARGUSPipeline()
    return _pipeline


# ── Schemas ─────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    lat_min: float = Field(..., ge=-90, le=90)
    lat_max: float = Field(..., ge=-90, le=90)
    lon_min: float = Field(..., ge=-180, le=180)
    lon_max: float = Field(..., ge=-180, le=180)
    date: str | None = None

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
    run_id: str
    is_plume: bool
    reviewer: str = "human_expert"
    notes: str = ""


# ── Helpers ─────────────────────────────────────────────────────────────────

PROJECTION = {"_id": 0, "cls_embedding": 0, "raw_spectra": 0, "pixel_mask": 0}

def _flux_to_risk(flux: float) -> str:
    if flux >= 300: return "CRITICAL"
    if flux >= 100: return "HIGH"
    if flux >= 40: return "MEDIUM"
    return "LOW"


def _normalise_detection(d: dict) -> dict:
    if "attribution" in d and isinstance(d["attribution"], dict):
        return d

    return {
        "detection_id": d.get("detection_id") or d.get("label_id"),
        "centroid_lat": d.get("centroid_lat"),
        "centroid_lon": d.get("centroid_lon"),
        "flux_kg_hr": d.get("flux_kg_hr", 0),
        "co2e_kg_hr": d.get("co2e_kg_hr", d.get("flux_kg_hr", 0) * 80),
        "confidence": d.get("mean_probability", d.get("confidence", 0)),
        "epistemic_variance": d.get("epistemic_variance", 0),
        "high_confidence": d.get("high_confidence", False),
        "run_id": d.get("run_id"),
        "timestamp": d.get("timestamp"),
        "bbox": d.get("bbox", {}),
        "attribution": {
            "facility_id": d.get("facility_id", ""),
            "facility_name": d.get("facility_name", ""),
            "operator": d.get("operator", ""),
            "facility_type": d.get("facility_type", ""),
            "distance_km": d.get("distance_km", 0),
            "confidence": d.get("attribution_confidence", 0),
        },
        "enforcement": {
            "notice_id": d.get("notice_id", ""),
            "risk_level": d.get("risk_level", ""),
        },
        "economics": d.get("economics", {}),
    }


def _ensure_indexes():
    try:
        db = get_sync_db()
        db.detections.create_index([("timestamp", -1)], background=True)
        db.runs.create_index([("timestamp", -1)], background=True)
        logger.info("MongoDB indexes ensured")
    except Exception as e:
        logger.warning(f"Index creation failed: {e}")


_ensure_indexes()


# ── SAFE Mongo Fetch ─────────────────────────────────────────────────────────

async def _async_fetch(collection: str, limit: int) -> list:
    try:
        db = get_async_db()

        cursor = (
            getattr(db, collection)
            .find({}, PROJECTION)
            .sort("timestamp", -1)
            .limit(limit)
        )

        return await cursor.to_list(length=limit)

    except Exception as e:
        logger.warning(f"Mongo fetch failed ({collection}): {e}")
        return []   # ✅ NEVER CRASH


async def _async_fetch_heatmap(n_runs: int):
    db = get_async_db()

    try:
        dets = await asyncio.wait_for(
            db.detections.find({}, PROJECTION).limit(100).to_list(100), timeout=10
        )
        runs = await asyncio.wait_for(
            db.runs.find({}, {"_id": 0}).limit(n_runs).to_list(n_runs), timeout=10
        )
        return dets, runs

    except Exception as e:
        logger.warning(f"Heatmap fetch failed: {e}")
        return [], []


# ── Detect ───────────────────────────────────────────────────────────────────

@router.post("/detect")
async def detect(req: DetectRequest, background_tasks: BackgroundTasks):
    try:
        pipeline = get_pipeline()
        date_obj = datetime.fromisoformat(req.date) if req.date else None

        result = pipeline.run(
            lat_min=req.lat_min,
            lat_max=req.lat_max,
            lon_min=req.lon_min,
            lon_max=req.lon_max,
            date=date_obj,
        )

        background_tasks.add_task(store.save, result)

        return result.to_api_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Detections ───────────────────────────────────────────────────────────────

@router.get("/detections")
async def detections(limit: int = 50):
    dets = await _async_fetch("detections", limit)

    if not dets:
        dets = store.load_all_detections()[-limit:]

    return {
        "detections": [_normalise_detection(d) for d in dets],
        "total": len(dets),
    }


# ── Alerts ───────────────────────────────────────────────────────────────────

@router.get("/alerts")
async def alerts(limit: int = 50):
    raw = await _async_fetch("detections", 50)

    if not raw:
        raw = store.load_all_detections()[-50:]

    normalised = [_normalise_detection(d) for d in raw]

    alerts = []
    for d in normalised:
        flux = d.get("flux_kg_hr", 0)
        risk = d["enforcement"]["risk_level"] or _flux_to_risk(flux)

        alerts.append({
            "id": f"ALERT-{str(d.get('detection_id', 0)).zfill(4)}",
            "facility_name": d["attribution"]["facility_name"],
            "risk": risk,
            "flux_kg_hr": flux,
        })

    return {"alerts": alerts[:limit]}


# ── Stream ───────────────────────────────────────────────────────────────────

@router.get("/stream")
async def stream():
    async def generator():
        while True:
            dets = await _async_fetch("detections", 50)
            if not dets:
                dets = store.load_all_detections()[-50:]

            yield f"data: {json.dumps(dets)}\n\n"
            await asyncio.sleep(60)

    return StreamingResponse(generator(), media_type="text/event-stream")


# ── WebSocket ────────────────────────────────────────────────────────────────

@ws_router.websocket("/ws/detections")
async def ws(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            dets = await _async_fetch("detections", 50)
            if not dets:
                dets = store.load_all_detections()[-50:]

            await websocket.send_json(dets)
            await asyncio.sleep(60)

    except WebSocketDisconnect:
        pass


# ── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok"}