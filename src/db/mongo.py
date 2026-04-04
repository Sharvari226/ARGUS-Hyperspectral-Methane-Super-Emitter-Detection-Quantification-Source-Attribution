from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient, GEOSPHERE, ASCENDING, DESCENDING
from loguru import logger

from src.utils.config import cfg

# ── Mongo Options ───────────────────────────────────────────

_MONGO_OPTS = dict(
    socketTimeoutMS=5000,          # ⚡ LOWER = faster fail
    connectTimeoutMS=5000,
    serverSelectionTimeoutMS=5000,
    retryReads=False,              # ⚡ stop retry delays
    retryWrites=False,
    maxPoolSize=5,
)

# ── Safe Config Getter (FIXES YOUR ERROR) ───────────────────

def _get_mongo_url():
    try:
        # supports BOTH formats
        if isinstance(cfg, dict):
            if "env" in cfg and "mongodb_url" in cfg["env"]:
                return cfg["env"]["mongodb_url"]
            if "mongo_uri" in cfg:
                return cfg["mongo_uri"]
        else:
            return getattr(cfg, "mongo_uri", None)

    except Exception:
        pass

    # fallback (NEVER FAIL)
    logger.warning("MongoDB URL not found in config, using localhost")
    return "mongodb://localhost:27017"


def _get_db_name():
    if isinstance(cfg, dict):
        return cfg.get("mongo_db", "argus")
    return getattr(cfg, "mongo_db", "argus")


# ── Async Client ────────────────────────────────────────────

_async_client: AsyncIOMotorClient | None = None


def get_async_db() -> AsyncIOMotorDatabase:
    global _async_client

    if _async_client is None:
        try:
            url = _get_mongo_url()
            logger.info(f"MongoDB async: connecting to {url[:40]}…")

            _async_client = AsyncIOMotorClient(url, **_MONGO_OPTS)

        except Exception as e:
            logger.error(f"Mongo async connection failed: {e}")
            raise

    return _async_client[_get_db_name()]


# ── Sync Client ─────────────────────────────────────────────

_sync_client: MongoClient | None = None


def get_sync_db():
    global _sync_client

    if _sync_client is None:
        try:
            url = _get_mongo_url()
            logger.info(f"MongoDB sync: connecting to {url[:40]}…")

            _sync_client = MongoClient(url, **_MONGO_OPTS)

            # force connection
            _sync_client.admin.command("ping")
            logger.info("MongoDB sync: ping OK ✓")

        except Exception as e:
            logger.error(f"Mongo sync connection failed: {e}")
            raise

    return _sync_client[_get_db_name()]


# ── Index Setup ─────────────────────────────────────────────

def setup_indexes() -> None:
    try:
        db = get_sync_db()

        db.runs.create_index([("timestamp", DESCENDING)])
        db.runs.create_index([("run_id", ASCENDING)], unique=True)
        db.runs.create_index([("summary.n_super_emitters", DESCENDING)])

        db.detections.create_index([
            ("centroid_lat", ASCENDING),
            ("centroid_lon", ASCENDING)
        ])
        db.detections.create_index([("flux_kg_hr", DESCENDING)])
        db.detections.create_index([("timestamp", DESCENDING)])
        db.detections.create_index([("location", GEOSPHERE)])

        db.facilities.create_index([("facility_id", ASCENDING)], unique=True)
        db.facilities.create_index([("location", GEOSPHERE)])
        db.facilities.create_index([("operator", ASCENDING)])

        db.review_queue.create_index([("queued_at", DESCENDING)])
        db.review_queue.create_index([("run_id", ASCENDING)])

        db.scorecard.create_index([("compliance_score", ASCENDING)])
        db.scorecard.create_index([("facility_id", ASCENDING)])

        db.notices.create_index([("notice_id", ASCENDING)], unique=True)
        db.notices.create_index([("facility_id", ASCENDING)])
        db.notices.create_index([("issued_at", DESCENDING)])

        logger.info("MongoDB: indexes created ✓")

    except Exception as e:
        logger.error(f"MongoDB setup_indexes failed: {e}")
        logger.warning("Continuing without indexes (safe mode)")