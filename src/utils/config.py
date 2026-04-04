from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
import yaml
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────
load_dotenv()

# ── Project Root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]


# ── Config Loader ─────────────────────────────────────────

@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Loads config.yaml + environment variables.
    Always returns a safe config (never crashes).
    """

    cfg_path = ROOT / "configs" / "config.yaml"

    # 1️⃣ Load YAML (safe)
    try:
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
    except Exception:
        cfg = {}

    # 2️⃣ Environment overrides (MOST IMPORTANT)
    cfg["env"] = {
        # ── DATABASE ─────────────────────────────
        "mongodb_url": os.getenv(
            "MONGODB_URL",
            "mongodb://localhost:27017"   # safe fallback
        ),
        "mongo_db": os.getenv(
            "MONGO_DB",
            "argus"
        ),

        # ── EARTH / SATELLITE ───────────────────
        "gee_project": os.getenv("GEE_PROJECT", ""),
        "earthdata_token": os.getenv("EARTHDATA_TOKEN", ""),
        "copernicus_user": os.getenv("COPERNICUS_USER", ""),
        "copernicus_password": os.getenv("COPERNICUS_PASSWORD", ""),

        # ── WEATHER / ECMWF ─────────────────────
        "ecmwf_api_key": os.getenv("ECMWF_API_KEY", ""),
        "ecmwf_api_url": os.getenv("ECMWF_API_URL", ""),

        # ── LLM ─────────────────────────────────
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),

        # ── ECONOMICS ───────────────────────────
        "gas_price_usd": float(os.getenv("GAS_PRICE_USD_PER_MMBTU", 2.80)),
        "carbon_price_usd": float(os.getenv("CARBON_PRICE_USD_PER_TON", 15.0)),
    }

    # 3️⃣ Convenience shortcuts (prevents your earlier errors)
    cfg["mongo_uri"] = cfg["env"]["mongodb_url"]
    cfg["mongo_db"]  = cfg["env"]["mongo_db"]

    # 4️⃣ API defaults (safe)
    cfg["api"] = cfg.get("api", {})
    cfg["api"]["host"] = cfg["api"].get("host", "127.0.0.1")
    cfg["api"]["port"] = cfg["api"].get("port", 8000)

    return cfg


# ── Global Config ─────────────────────────────────────────
cfg = get_config()