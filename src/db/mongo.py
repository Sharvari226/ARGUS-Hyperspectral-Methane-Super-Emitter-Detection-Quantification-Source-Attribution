from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient, GEOSPHERE, ASCENDING, DESCENDING
from loguru import logger
from src.utils.config import cfg

# ── Async client (used by FastAPI routes) ─────────────────────────
_async_client: AsyncIOMotorClient | None = None


def get_async_db() -> AsyncIOMotorDatabase:
    global _async_client
    if _async_client is None:
        _async_client = AsyncIOMotorClient(cfg["env"]["mongodb_url"])
        logger.info("MongoDB: async client connected")
    return _async_client["argus"]


# ── Sync client (used by pipeline + scripts) ──────────────────────
_sync_client: MongoClient | None = None


def get_sync_db():
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(cfg["env"]["mongodb_url"])
        logger.info("MongoDB: sync client connected")
    return _sync_client["argus"]


# ── Index setup (run once at startup) ─────────────────────────────

def setup_indexes() -> None:
    db = get_sync_db()

    # Runs collection
    db.runs.create_index([("timestamp", DESCENDING)])
    db.runs.create_index([("run_id", ASCENDING)], unique=True)
    db.runs.create_index([("summary.n_super_emitters", DESCENDING)])

    # Detections collection (flattened for fast heatmap queries)
    db.detections.create_index([("centroid_lat", ASCENDING),
                                  ("centroid_lon", ASCENDING)])
    db.detections.create_index([("flux_kg_hr", DESCENDING)])
    db.detections.create_index([("timestamp", DESCENDING)])
    db.detections.create_index(
        [("location", GEOSPHERE)]  # 2dsphere for geo queries
    )

    # Facilities collection
    db.facilities.create_index([("facility_id", ASCENDING)], unique=True)
    db.facilities.create_index([("location", GEOSPHERE)])
    db.facilities.create_index([("operator", ASCENDING)])

    # Review queue
    db.review_queue.create_index([("queued_at", DESCENDING)])
    db.review_queue.create_index([("run_id", ASCENDING)])

    # Scorecard
    db.scorecard.create_index([("compliance_score", ASCENDING)])
    db.scorecard.create_index([("facility_id", ASCENDING)])

    # Enforcement notices
    db.notices.create_index([("notice_id", ASCENDING)], unique=True)
    db.notices.create_index([("facility_id", ASCENDING)])
    db.notices.create_index([("issued_at", DESCENDING)])

    logger.info("MongoDB: indexes created ✓")