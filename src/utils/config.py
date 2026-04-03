from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_config() -> dict:
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["env"] = {
        # MongoDB Atlas
        "mongodb_url": os.environ.get(
            "MONGODB_URL",
            "mongodb://argus:argus_secret@localhost:27017/argus?authSource=admin",
        ),
        # Google Earth Engine
        "gee_project": os.environ.get("GEE_PROJECT", ""),
        # Groq (free LLM)
        "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
        # Gas & carbon prices
        "gas_price_usd":    float(os.environ.get("GAS_PRICE_USD_PER_MMBTU", 2.80)),
        "carbon_price_usd": float(os.environ.get("CARBON_PRICE_USD_PER_TON", 15.0)),
    }
    return cfg


cfg = get_config()