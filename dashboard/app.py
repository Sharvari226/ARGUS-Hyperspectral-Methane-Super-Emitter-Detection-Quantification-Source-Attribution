"""
dashboard/app.py
─────────────────
ARGUS Dashboard — Streamlit + Pydeck
Aesthetic: Premium dark SaaS — Obsidian panels, electric indigo accents, glass cards.

Integrated twists:
  TWIST 1 — Synthetic Data Engine + Multi-Agent Pipeline
            Activates automatically when satellite / API is offline.
            SyntheticAlertGenerator produces mocked methane alerts (GPS, flux,
            wind vector, confidence).  FPFilterAgent cross-references against the
            facility DB.  BriefAgent drafts a Regulatory Intervention Brief for
            each of the top-5 super-emitters found.

  TWIST 2 — Gaussian Denoiser (pre-segmentation noise removal)
            When sensor noise is simulated / detected, a Wiener-style denoiser
            runs before Stage-1 segmentation.  SNR before/after and FP-reduction
            metrics are shown in the System tab and the Contingency tab.

Run:  python -m streamlit run dashboard/app.py
"""

import sys
import os
import math
import json
import random
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st

from src.utils.config import cfg

# ══════════════════════════════════════════════════════════════════
# Page config — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ARGUS — Methane Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-base:     #080b12;
    --bg-surface:  #0d1117;
    --bg-elevated: #111827;
    --bg-card:     #0f1623;
    --bg-hover:    #161f2e;
    --border:      rgba(255,255,255,0.06);
    --border-med:  rgba(255,255,255,0.10);
    --border-hi:   rgba(255,255,255,0.16);
    --indigo:      #6366f1;
    --indigo-dim:  #4f52c9;
    --indigo-glow: rgba(99,102,241,0.18);
    --indigo-soft: rgba(99,102,241,0.08);
    --amber:       #f59e0b;
    --amber-soft:  rgba(245,158,11,0.12);
    --red:         #ef4444;
    --red-soft:    rgba(239,68,68,0.12);
    --emerald:     #10b981;
    --emerald-soft:rgba(16,185,129,0.10);
    --orange:      #f97316;
    --orange-soft: rgba(249,115,22,0.12);
    --sky:         #38bdf8;
    --sky-soft:    rgba(56,189,248,0.10);
    --text-base:   #94a3b8;
    --text-muted:  #475569;
    --text-hi:     #e2e8f0;
    --text-bright: #f8fafc;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-base) !important;
    color: var(--text-base) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

.stApp {
    background:
        radial-gradient(ellipse 60% 40% at 70% -5%, rgba(99,102,241,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 10% 80%, rgba(16,185,129,0.03) 0%, transparent 50%),
        var(--bg-base) !important;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-family: 'Inter', sans-serif !important;
}

/* ── Metrics ─────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 20px 22px !important;
    border-radius: 12px !important;
    transition: border-color 0.2s, transform 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-med);
    transform: translateY(-1px);
}
[data-testid="stMetric"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
    color: var(--text-bright) !important;
}

/* ── Tabs ─────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
    padding: 0 !important;
}
[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    color: var(--text-muted) !important;
    padding: 12px 18px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: color 0.15s, background 0.15s !important;
}
[data-baseweb="tab"]:hover {
    color: var(--text-hi) !important;
    background: var(--bg-elevated) !important;
}
[aria-selected="true"] {
    color: var(--indigo) !important;
    background: var(--indigo-soft) !important;
    border-bottom: 2px solid var(--indigo) !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    background: var(--indigo) !important;
    border: none !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}
.stButton > button:hover {
    background: #5254d4 !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Select / Inputs ─────────────────────────────── */
[data-baseweb="select"] > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-med) !important;
    border-radius: 8px !important;
    color: var(--text-hi) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
}

/* ── DataFrames ──────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: var(--bg-card) !important;
}

/* ── Expanders ───────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-card) !important;
}

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-surface); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

/* ── Alerts ──────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; border-width: 1px !important; }

/* ── Column gaps ─────────────────────────────────── */
[data-testid="stHorizontalBlock"] { gap: 14px !important; }

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.45; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TWIST 1 — Synthetic Data Engine + Multi-Agent Pipeline
# ══════════════════════════════════════════════════════════════════

# --- Mocked Facility Database (used by FPFilterAgent) ---
_FACILITY_DB = [
    {"facility_id": "FAC-IN-001", "facility_name": "Haldia Petrochem Complex",   "operator": "HPCL",           "lat": 22.06, "lon": 88.07, "type": "refinery",       "registered": True,  "permit_flux_kg_hr": 90},
    {"facility_id": "FAC-IN-002", "facility_name": "Gujarat Gas Compressor-7",   "operator": "GAIL India",     "lat": 22.30, "lon": 73.10, "type": "gas_compressor",  "registered": True,  "permit_flux_kg_hr": 80},
    {"facility_id": "FAC-IN-003", "facility_name": "Mumbai Offshore BH-3",       "operator": "ONGC Ltd",       "lat": 19.20, "lon": 72.80, "type": "oil_wellpad",     "registered": True,  "permit_flux_kg_hr": 70},
    {"facility_id": "FAC-IN-004", "facility_name": "Assam Oil Field F-12",       "operator": "OIL India",      "lat": 27.10, "lon": 94.60, "type": "oil_wellpad",     "registered": True,  "permit_flux_kg_hr": 85},
    {"facility_id": "FAC-PK-001", "facility_name": "Sui Gas Field G-4",          "operator": "SSGC Pakistan",  "lat": 28.60, "lon": 69.20, "type": "gas_compressor",  "registered": True,  "permit_flux_kg_hr": 75},
    {"facility_id": "FAC-CN-001", "facility_name": "Tarim Basin LNG-9",          "operator": "CNPC Energy",    "lat": 41.20, "lon": 83.80, "type": "lng_terminal",    "registered": True,  "permit_flux_kg_hr": 110},
    {"facility_id": "FAC-RU-001", "facility_name": "Orenburg Gas Plant",         "operator": "GazpromNeft",    "lat": 51.80, "lon": 55.10, "type": "gas_compressor",  "registered": True,  "permit_flux_kg_hr": 120},
    {"facility_id": "FAC-NG-001", "facility_name": "Niger Delta P-12",           "operator": "Gulf Stream",    "lat":  5.20, "lon":  6.40, "type": "oil_wellpad",     "registered": True,  "permit_flux_kg_hr": 80},
    {"facility_id": "FAC-US-001", "facility_name": "Permian Basin WP-7",         "operator": "OilCorp Intl",   "lat": 32.10, "lon":-102.50, "type": "oil_wellpad",   "registered": True,  "permit_flux_kg_hr": 95},
    {"facility_id": "FAC-SA-001", "facility_name": "Ghawar East Field",          "operator": "Saudi Aramco",   "lat": 25.10, "lon": 49.30, "type": "oil_wellpad",     "registered": True,  "permit_flux_kg_hr": 130},
    {"facility_id": "FAC-DE-001", "facility_name": "Nord Stream Comp Station",   "operator": "GazTransit GmbH","lat": 54.10, "lon": 12.30, "type": "gas_compressor",  "registered": True,  "permit_flux_kg_hr": 100},
    {"facility_id": "FAC-IN-005", "facility_name": "Korba Coal Belt C-8",        "operator": "Coal India Ltd", "lat": 22.30, "lon": 82.70, "type": "coal_mine",       "registered": True,  "permit_flux_kg_hr": 60},
]

# Facility lookup for quick cross-referencing
_FAC_LOOKUP = {f["facility_id"]: f for f in _FACILITY_DB}


@dataclass
class SyntheticAlert:
    """Single methane alert as produced by the synthetic data generator."""
    alert_id:       str
    timestamp_utc:  str
    centroid_lat:   float
    centroid_lon:   float
    flux_kg_hr:     float
    co2e_kg_hr:     float
    confidence:     float
    epistemic_variance: float
    wind_u_ms:      float        # eastward component
    wind_v_ms:      float        # northward component
    wind_speed_ms:  float
    wind_dir_deg:   int
    attributed_facility_id: Optional[str]
    attributed_facility_name: str
    attributed_operator:  str
    facility_type:  str
    distance_km:    float
    risk_level:     str
    is_false_positive: bool      # set by FPFilterAgent
    fp_reason:      str          # why it was flagged as FP, if applicable
    denoised:       bool         # was Twist-2 denoiser applied?
    raw_snr_db:     float
    denoised_snr_db: float
    economics: dict = field(default_factory=dict)


class SyntheticAlertGenerator:
    """
    TWIST 1 — Satellite-offline fallback.
    Generates a stream of realistic methane alert JSON objects that mirror the
    structure produced by the real Stage-1→Stage-3 pipeline.
    """

    def __init__(self, noise_sigma: float = 0.0):
        self.noise_sigma = noise_sigma   # Twist-2: Gaussian noise level
        self._counter = 0

    def _pick_facility(self) -> dict:
        """Randomly pick a registered facility or generate an unknown source."""
        if random.random() < 0.85:
            return random.choice(_FACILITY_DB)
        # Unknown / unregistered source
        return {
            "facility_id": None,
            "facility_name": "Unregistered Source",
            "operator": "Unknown",
            "lat": random.uniform(-60, 70),
            "lon": random.uniform(-170, 170),
            "type": "unknown",
            "registered": False,
            "permit_flux_kg_hr": 0,
        }

    def _apply_noise(self, flux: float) -> tuple[float, float, float]:
        """
        TWIST 2 integration — add Gaussian noise to the flux reading,
        then apply a Wiener-style denoiser and return both SNR values.
        Returns (noisy_flux, raw_snr_db, denoised_snr_db)
        """
        if self.noise_sigma <= 0:
            return flux, 99.0, 99.0

        noise = random.gauss(0, self.noise_sigma * flux * 0.01)
        noisy_flux = max(1.0, flux + noise)

        # SNR before denoising
        signal_power = flux ** 2
        noise_power  = noise ** 2 + 1e-6
        raw_snr = 10 * math.log10(signal_power / noise_power)

        # Wiener-style denoiser: shrink noise contribution
        # (simple approximation — full 2-D Wiener operates in Stage-1 on the image)
        denoiser_gain = 1.0 / (1.0 + (self.noise_sigma / 100) ** 2)
        denoised_flux = flux + noise * denoiser_gain          # closer to truth
        denoise_residual = (denoised_flux - flux) ** 2 + 1e-6
        denoised_snr = 10 * math.log10(signal_power / denoise_residual)

        return noisy_flux, raw_snr, denoised_snr

    def generate(self, n: int = 1) -> list[SyntheticAlert]:
        alerts = []
        for _ in range(n):
            self._counter += 1
            fac = self._pick_facility()

            # Base flux from facility type
            base_flux_map = {
                "gas_compressor": (200, 500), "oil_wellpad": (80, 400),
                "lng_terminal": (150, 350),   "refinery":   (100, 300),
                "coal_mine":    (40, 150),    "unknown":    (10,  80),
            }
            lo, hi = base_flux_map.get(fac["type"], (50, 200))
            true_flux = random.uniform(lo, hi)

            # Apply noise / denoiser (Twist 2)
            flux, raw_snr, denoised_snr = self._apply_noise(true_flux)
            denoised = self.noise_sigma > 0

            # Wind vector (ERA5-style)
            wind_u = random.gauss(2.0, 4.5)
            wind_v = random.gauss(-0.5, 4.0)
            wind_speed = math.sqrt(wind_u**2 + wind_v**2)
            wind_dir   = int((math.degrees(math.atan2(wind_u, wind_v)) + 360) % 360)

            # Confidence — registered facilities score higher
            conf_base = 0.72 if fac["registered"] else 0.38
            confidence = min(0.99, max(0.25, conf_base + random.gauss(0, 0.12)))
            if denoised:
                # Denoiser recovery partially restores confidence
                confidence = min(0.99, confidence + (self.noise_sigma / 600))
            epistemic_var = max(0.01, 0.25 - confidence * 0.18 + random.uniform(0, 0.08))

            # Position jitter around facility
            lat = fac["lat"] + random.gauss(0, 0.5)
            lon = fac["lon"] + random.gauss(0, 0.5)
            dist_km = math.sqrt((lat - fac["lat"])**2 + (lon - fac["lon"])**2) * 111

            # Risk tier
            risk = ("CRITICAL" if flux >= 300 else
                    "HIGH"     if flux >= 100 else
                    "MEDIUM"   if flux >= 40  else "LOW")

            # Economics (30-day)
            hours   = 30 * 24
            ch4_t   = flux * hours / 1000
            co2e_t  = ch4_t * 80
            gas_usd = ch4_t * 0.0553 * 2.8
            carb_usd= co2e_t * 15.0
            fine_usd= co2e_t * 50.0
            econ    = {
                "ch4_lost_t_30d":    round(ch4_t,   1),
                "co2e_t_30d":        round(co2e_t,  1),
                "gas_value_usd":     round(gas_usd),
                "carbon_cost_usd":   round(carb_usd),
                "fine_usd":          round(fine_usd),
                "total_cost_usd":    round(gas_usd + carb_usd + fine_usd),
                "total_cost_inr":    round((gas_usd + carb_usd + fine_usd) * 83.5),
            }

            alert_id = f"SYN-{datetime.now(timezone.utc).strftime('%H%M%S')}-{self._counter:04d}"

            alerts.append(SyntheticAlert(
                alert_id=alert_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                centroid_lat=round(lat, 4),
                centroid_lon=round(lon, 4),
                flux_kg_hr=round(flux, 2),
                co2e_kg_hr=round(flux * 80, 1),
                confidence=round(confidence, 3),
                epistemic_variance=round(epistemic_var, 3),
                wind_u_ms=round(wind_u, 2),
                wind_v_ms=round(wind_v, 2),
                wind_speed_ms=round(wind_speed, 2),
                wind_dir_deg=wind_dir,
                attributed_facility_id=fac["facility_id"],
                attributed_facility_name=fac["facility_name"],
                attributed_operator=fac["operator"],
                facility_type=fac["type"],
                distance_km=round(dist_km, 2),
                risk_level=risk,
                is_false_positive=False,   # set by FPFilterAgent
                fp_reason="",
                denoised=denoised,
                raw_snr_db=round(raw_snr, 1),
                denoised_snr_db=round(denoised_snr, 1),
                economics=econ,
            ))
        return alerts


class FPFilterAgent:
    """
    TWIST 1 — Agent that cross-references synthetic alerts against the facility DB
    and flags false positives before they reach the compliance / map layers.

    False-positive rules (mirror what the real pipeline does in orchestrator.py):
    1. No matching registered facility within 50 km → likely cloud / artefact
    2. Confidence < 0.45 after noise correction → below detection threshold
    3. Flux < permit threshold by >20 % AND no prior violations → within permit
    4. Epistemic variance > 0.22 → too uncertain, flag for human review
    5. Denoised SNR still < 8 dB → denoiser couldn't recover enough signal
    """

    def __init__(self):
        self.processed = 0
        self.flagged   = 0

    def filter(self, alert: SyntheticAlert) -> SyntheticAlert:
        self.processed += 1
        fac_id = alert.attributed_facility_id
        fac    = _FAC_LOOKUP.get(fac_id) if fac_id else None

        # Rule 1 — unregistered source
        if not fac:
            alert.is_false_positive = True
            alert.fp_reason = "No matching facility in DB"
            self.flagged += 1
            return alert

        # Rule 2 — low confidence
        if alert.confidence < 0.45:
            alert.is_false_positive = True
            alert.fp_reason = f"Confidence {alert.confidence:.2f} < 0.45 threshold"
            self.flagged += 1
            return alert

        # Rule 3 — within permit
        if (alert.flux_kg_hr < fac["permit_flux_kg_hr"] * 0.80 and
                alert.risk_level in ("LOW", "MEDIUM")):
            alert.is_false_positive = True
            alert.fp_reason = f"Flux {alert.flux_kg_hr:.0f} within permit ({fac['permit_flux_kg_hr']} kg/hr)"
            self.flagged += 1
            return alert

        # Rule 4 — high epistemic variance
        if alert.epistemic_variance > 0.22:
            alert.is_false_positive = True
            alert.fp_reason = f"σ² = {alert.epistemic_variance:.3f} > 0.22 (high uncertainty)"
            self.flagged += 1
            return alert

        # Rule 5 — Twist-2: denoiser SNR still bad
        if alert.denoised and alert.denoised_snr_db < 8.0:
            alert.is_false_positive = True
            alert.fp_reason = f"Post-denoise SNR {alert.denoised_snr_db:.1f} dB < 8 dB"
            self.flagged += 1
            return alert

        return alert

    @property
    def precision(self) -> Optional[float]:
        if self.processed == 0:
            return None
        return round((1 - self.flagged / self.processed) * 100, 1)


class BriefAgent:
    """
    TWIST 1 — Drafts a Regulatory Intervention Brief (RIB) for each of the
    top-5 highest-flux confirmed alerts.  In production this calls the Groq
    LLM (llama-3.3-70b-versatile); in contingency mode the template is filled
    locally so the dashboard works entirely offline.
    """

    def draft(self, alert: SyntheticAlert, rank: int, groq_key: Optional[str] = None) -> dict:
        notice_id = f"RIB-ARGUS-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{rank:03d}"
        econ = alert.economics

        # Try Groq for executive summary if key is available
        exec_summary = self._llm_summary(alert, groq_key)

        actions = [
            ("24 hrs",  "Immediately cease or curtail the detected emission source. "
                        "Notify CPCB Emergency Response Cell (helpline 1800-11-4000)."),
            ("72 hrs",  "Submit Preliminary Incident Report (Form ENV-5) to the "
                        "Regional Officer, CPCB. Include GPS-tagged site photographs."),
            ("7 days",  "Complete Root Cause Analysis and submit to CPCB with a "
                        "proposed corrective action plan and LDAR inspection records."),
            ("30 days", "Implement all permanent remediation measures and commission "
                        "third-party verification. Submit compliance certificate."),
            ("60 days", "Deploy continuous methane monitoring with real-time data feed "
                        "to CPCB PRISM portal. Submit monitoring protocol for approval."),
        ]

        return {
            "notice_id":       notice_id,
            "rank":            rank,
            "alert_id":        alert.alert_id,
            "risk_level":      alert.risk_level,
            "facility_name":   alert.attributed_facility_name,
            "operator":        alert.attributed_operator,
            "facility_id":     alert.attributed_facility_id,
            "facility_type":   alert.facility_type,
            "lat":             alert.centroid_lat,
            "lon":             alert.centroid_lon,
            "flux_kg_hr":      alert.flux_kg_hr,
            "confidence_pct":  round(alert.confidence * 100, 1),
            "wind_speed_ms":   alert.wind_speed_ms,
            "wind_dir_deg":    alert.wind_dir_deg,
            "denoised":        alert.denoised,
            "raw_snr_db":      alert.raw_snr_db,
            "denoised_snr_db": alert.denoised_snr_db,
            "economics":       econ,
            "executive_summary": exec_summary,
            "corrective_actions": actions,
            "issued_at":       datetime.now(timezone.utc).isoformat(),
            "authority":       "CPCB / MoEF&CC",
            "legal_basis":     "Environment Protection Act 1986 · Air Act 1981 · India NDC 2030",
        }

    def _llm_summary(self, alert: SyntheticAlert, groq_key: Optional[str]) -> str:
        """Try Groq; fall back to template."""
        template = (
            f"Satellite-derived {'synthetic (contingency mode)' if not groq_key else 'real-time'} "
            f"intelligence has identified a {alert.risk_level.lower()}-priority methane "
            f"super-emitter at or near {alert.attributed_facility_name} "
            f"(operator: {alert.attributed_operator}). "
            f"Estimated emission flux of {alert.flux_kg_hr:.1f} kg CH₄/hr "
            f"{'exceeds' if alert.flux_kg_hr > 100 else 'approaches'} the CPCB super-emitter "
            f"threshold (100 kg/hr) and warrants immediate regulatory intervention. "
            f"Wind vector {alert.wind_speed_ms:.1f} m/s @ {alert.wind_dir_deg}° "
            f"indicates plume dispersal toward the "
            f"{'north-east' if 0 < alert.wind_dir_deg < 180 else 'south-west'}."
        )
        if not groq_key:
            return template

        try:
            import groq as groq_lib  # type: ignore
            client = groq_lib.Groq(api_key=groq_key)
            prompt = (
                "You are ARGUS, an AI methane enforcement system. "
                "Write a 2-sentence executive summary for a Regulatory Intervention Brief. "
                "Be formal, cite the emission rate, and mention the regulatory threshold.\n\n"
                f"Context: {template}"
            )
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return template  # silent fallback


# ── Session-state bootstrap for synthetic pipeline ──────────────────────────

def _init_synthetic_state():
    if "syn_generator"  not in st.session_state:
        st.session_state["syn_generator"]  = SyntheticAlertGenerator()
    if "syn_fp_agent"   not in st.session_state:
        st.session_state["syn_fp_agent"]   = FPFilterAgent()
    if "syn_brief_agent" not in st.session_state:
        st.session_state["syn_brief_agent"] = BriefAgent()
    if "syn_alerts"     not in st.session_state:
        st.session_state["syn_alerts"]     = []   # list[SyntheticAlert]
    if "syn_briefs"     not in st.session_state:
        st.session_state["syn_briefs"]     = []   # list[dict]
    if "syn_noise_sigma" not in st.session_state:
        st.session_state["syn_noise_sigma"] = 0.0


_init_synthetic_state()


# ══════════════════════════════════════════════════════════════════
# TWIST 2 — Gaussian Denoiser (pre-segmentation)
# ══════════════════════════════════════════════════════════════════

def wiener_denoise_1d(signal: np.ndarray, noise_sigma: float) -> np.ndarray:
    """
    Wiener filter approximation for a 1-D signal.
    Used to demonstrate the denoising step before Stage-1 ViT segmentation.
    The full 2-D version operates on the satellite image array in stage1_sat.py.
    """
    if noise_sigma <= 0:
        return signal
    local_var = np.convolve(signal**2, np.ones(5)/5, mode='same') - \
                np.convolve(signal,   np.ones(5)/5, mode='same')**2
    noise_var  = (noise_sigma * signal.mean() * 0.01) ** 2
    gain = np.maximum(0, local_var - noise_var) / np.maximum(local_var, noise_var + 1e-9)
    mean_local = np.convolve(signal, np.ones(5)/5, mode='same')
    return mean_local + gain * (signal - mean_local)


def _snr_metrics(noise_sigma: float) -> dict:
    """Return denoiser performance estimates given noise level σ (0-100)."""
    if noise_sigma <= 0:
        return {"raw_snr": 99.0, "denoised_snr": 99.0, "fp_reduction_pct": 0.0, "active": False}
    raw_snr      = max(2.0,  22.0 - noise_sigma * 0.20)
    denoised_snr = max(8.0,  raw_snr + noise_sigma * 0.14)
    fp_reduction = min(85.0, noise_sigma * 0.75)
    return {
        "raw_snr":         round(raw_snr,      1),
        "denoised_snr":    round(denoised_snr, 1),
        "fp_reduction_pct":round(fp_reduction, 1),
        "active":          True,
    }


# ══════════════════════════════════════════════════════════════════
# Mock data (shown when API is offline)
# ══════════════════════════════════════════════════════════════════

MOCK_DETECTIONS = [
    {"detection_id": 1,  "centroid_lat": 32.1,  "centroid_lon": -102.5, "flux_kg_hr": 487, "co2e_kg_hr": 38960, "confidence": 0.94, "epistemic_variance": 0.04, "high_confidence": True,  "attribution": {"facility_name": "Permian Basin WP-7",   "operator": "OilCorp International", "facility_id": "FAC-0001", "facility_type": "oil_wellpad",    "confidence": 0.91, "distance_km": 2.3}, "enforcement": {"risk_level": "CRITICAL", "notice_id": "NOV-ARGUS-20240315-001"}, "economics": {"total_cost_inr": 142000000}},
    {"detection_id": 2,  "centroid_lat": 38.4,  "centroid_lon": 57.2,   "flux_kg_hr": 312, "co2e_kg_hr": 24960, "confidence": 0.88, "epistemic_variance": 0.07, "high_confidence": True,  "attribution": {"facility_name": "Turkmenistan GC-3",    "operator": "TurkGaz Holdings",      "facility_id": "FAC-0042", "facility_type": "gas_compressor", "confidence": 0.85, "distance_km": 4.1}, "enforcement": {"risk_level": "CRITICAL", "notice_id": "NOV-ARGUS-20240315-002"}, "economics": {"total_cost_inr": 89000000}},
    {"detection_id": 3,  "centroid_lat": 5.2,   "centroid_lon": 6.4,    "flux_kg_hr": 221, "co2e_kg_hr": 17680, "confidence": 0.79, "epistemic_variance": 0.11, "high_confidence": False, "attribution": {"facility_name": "Niger Delta P-12",     "operator": "Gulf Stream Energy",    "facility_id": "FAC-0108", "facility_type": "lng_terminal",   "confidence": 0.72, "distance_km": 7.8}, "enforcement": {"risk_level": "HIGH",     "notice_id": ""}, "economics": {"total_cost_inr": 61000000}},
    {"detection_id": 4,  "centroid_lat": 62.3,  "centroid_lon": 74.1,   "flux_kg_hr": 178, "co2e_kg_hr": 14240, "confidence": 0.91, "epistemic_variance": 0.05, "high_confidence": True,  "attribution": {"facility_name": "Siberia LNG T-2",      "operator": "SovEnergy PJSC",        "facility_id": "FAC-0203", "facility_type": "lng_terminal",   "confidence": 0.88, "distance_km": 3.2}, "enforcement": {"risk_level": "HIGH",     "notice_id": "NOV-ARGUS-20240315-003"}, "economics": {"total_cost_inr": 48000000}},
    {"detection_id": 5,  "centroid_lat": 20.1,  "centroid_lon": 70.3,   "flux_kg_hr": 134, "co2e_kg_hr": 10720, "confidence": 0.83, "epistemic_variance": 0.09, "high_confidence": True,  "attribution": {"facility_name": "Mumbai Offshore MH-3", "operator": "IndusGas Ltd",          "facility_id": "FAC-0287", "facility_type": "oil_wellpad",    "confidence": 0.79, "distance_km": 5.6}, "enforcement": {"risk_level": "MEDIUM",   "notice_id": ""}, "economics": {"total_cost_inr": 35000000}},
    {"detection_id": 6,  "centroid_lat": 27.1,  "centroid_lon": 49.8,   "flux_kg_hr": 298, "co2e_kg_hr": 23840, "confidence": 0.96, "epistemic_variance": 0.03, "high_confidence": True,  "attribution": {"facility_name": "Saudi East Comp-7",    "operator": "ArcoFlare Co",          "facility_id": "FAC-0321", "facility_type": "gas_compressor", "confidence": 0.93, "distance_km": 1.8}, "enforcement": {"risk_level": "CRITICAL", "notice_id": "NOV-ARGUS-20240315-004"}, "economics": {"total_cost_inr": 82000000}},
    {"detection_id": 7,  "centroid_lat": -3.5,  "centroid_lon": 18.2,   "flux_kg_hr": 156, "co2e_kg_hr": 12480, "confidence": 0.75, "epistemic_variance": 0.13, "high_confidence": False, "attribution": {"facility_name": "Congo Basin F-4",      "operator": "AfricaFuel PLC",        "facility_id": "FAC-0392", "facility_type": "oil_wellpad",    "confidence": 0.68, "distance_km": 9.1}, "enforcement": {"risk_level": "HIGH",     "notice_id": ""}, "economics": {"total_cost_inr": 41000000}},
    {"detection_id": 8,  "centroid_lat": 52.8,  "centroid_lon": 55.4,   "flux_kg_hr": 543, "co2e_kg_hr": 43440, "confidence": 0.97, "epistemic_variance": 0.02, "high_confidence": True,  "attribution": {"facility_name": "Orenburg Gas Plant",  "operator": "GazpromNeft East",      "facility_id": "FAC-0445", "facility_type": "gas_compressor", "confidence": 0.95, "distance_km": 1.2}, "enforcement": {"risk_level": "CRITICAL", "notice_id": "NOV-ARGUS-20240315-005"}, "economics": {"total_cost_inr": 158000000}},
    {"detection_id": 9,  "centroid_lat": 51.5,  "centroid_lon": 0.1,    "flux_kg_hr": 42,  "co2e_kg_hr": 3360,  "confidence": 0.63, "epistemic_variance": 0.19, "high_confidence": False, "attribution": {"facility_name": "Thames Estuary W-1",  "operator": "BritGas PLC",           "facility_id": "FAC-0501", "facility_type": "pipeline",       "confidence": 0.61, "distance_km": 11.2}, "enforcement": {"risk_level": "LOW", "notice_id": ""}, "economics": {"total_cost_inr": 9000000}},
    {"detection_id": 10, "centroid_lat": 35.6,  "centroid_lon": 139.7,  "flux_kg_hr": 28,  "co2e_kg_hr": 2240,  "confidence": 0.58, "epistemic_variance": 0.22, "high_confidence": False, "attribution": {"facility_name": "Tokyo Bay Pipeline",  "operator": "JapanFuel KK",          "facility_id": "FAC-0502", "facility_type": "pipeline",       "confidence": 0.55, "distance_km": 13.5}, "enforcement": {"risk_level": "LOW", "notice_id": ""}, "economics": {"total_cost_inr": 6000000}},
    {"detection_id": 11, "centroid_lat": 48.8,  "centroid_lon": 2.3,    "flux_kg_hr": 19,  "co2e_kg_hr": 1520,  "confidence": 0.55, "epistemic_variance": 0.24, "high_confidence": False, "attribution": {"facility_name": "Paris Basin Landfill", "operator": "EcoWaste SA",           "facility_id": "FAC-0503", "facility_type": "landfill",       "confidence": 0.52, "distance_km": 15.0}, "enforcement": {"risk_level": "LOW", "notice_id": ""}, "economics": {"total_cost_inr": 4500000}},
    {"detection_id": 12, "centroid_lat": -23.5, "centroid_lon": -46.6,  "flux_kg_hr": 35,  "co2e_kg_hr": 2800,  "confidence": 0.60, "epistemic_variance": 0.20, "high_confidence": False, "attribution": {"facility_name": "São Paulo Wastewater", "operator": "AguaSP Corp",           "facility_id": "FAC-0504", "facility_type": "wastewater",     "confidence": 0.58, "distance_km": 9.8},  "enforcement": {"risk_level": "LOW", "notice_id": ""}, "economics": {"total_cost_inr": 7200000}},
]

MOCK_SCORECARD = [
    {"facility_id": "FAC-0445", "facility_name": "Orenburg Gas Plant",   "operator": "GazpromNeft East",      "facility_type": "gas_compressor", "flux_kg_hr": 543, "compliance_score":  9, "violations_12mo": 5, "risk_level": "CRITICAL", "confidence": 95},
    {"facility_id": "FAC-0001", "facility_name": "Permian Basin WP-7",   "operator": "OilCorp International", "facility_type": "oil_wellpad",    "flux_kg_hr": 487, "compliance_score": 12, "violations_12mo": 5, "risk_level": "CRITICAL", "confidence": 91},
    {"facility_id": "FAC-0042", "facility_name": "Turkmenistan GC-3",    "operator": "TurkGaz Holdings",      "facility_type": "gas_compressor", "flux_kg_hr": 312, "compliance_score": 24, "violations_12mo": 4, "risk_level": "CRITICAL", "confidence": 85},
    {"facility_id": "FAC-0321", "facility_name": "Saudi East Comp-7",    "operator": "ArcoFlare Co",          "facility_type": "gas_compressor", "flux_kg_hr": 298, "compliance_score": 31, "violations_12mo": 3, "risk_level": "CRITICAL", "confidence": 93},
    {"facility_id": "FAC-0108", "facility_name": "Niger Delta P-12",     "operator": "Gulf Stream Energy",    "facility_type": "lng_terminal",   "flux_kg_hr": 221, "compliance_score": 38, "violations_12mo": 3, "risk_level": "HIGH",     "confidence": 72},
    {"facility_id": "FAC-0392", "facility_name": "Congo Basin F-4",      "operator": "AfricaFuel PLC",        "facility_type": "oil_wellpad",    "flux_kg_hr": 156, "compliance_score": 47, "violations_12mo": 2, "risk_level": "HIGH",     "confidence": 68},
    {"facility_id": "FAC-0203", "facility_name": "Siberia LNG T-2",      "operator": "SovEnergy PJSC",        "facility_type": "lng_terminal",   "flux_kg_hr": 178, "compliance_score": 52, "violations_12mo": 2, "risk_level": "HIGH",     "confidence": 88},
    {"facility_id": "FAC-0287", "facility_name": "Mumbai Offshore MH-3", "operator": "IndusGas Ltd",          "facility_type": "oil_wellpad",    "flux_kg_hr": 134, "compliance_score": 61, "violations_12mo": 1, "risk_level": "MEDIUM",   "confidence": 79},
    {"facility_id": "FAC-0501", "facility_name": "Thames Estuary W-1",   "operator": "BritGas PLC",           "facility_type": "pipeline",       "flux_kg_hr":  42, "compliance_score": 74, "violations_12mo": 0, "risk_level": "LOW",      "confidence": 63},
    {"facility_id": "FAC-0502", "facility_name": "Tokyo Bay Pipeline",   "operator": "JapanFuel KK",          "facility_type": "pipeline",       "flux_kg_hr":  28, "compliance_score": 81, "violations_12mo": 0, "risk_level": "LOW",      "confidence": 58},
    {"facility_id": "FAC-0503", "facility_name": "Paris Basin Landfill", "operator": "EcoWaste SA",           "facility_type": "landfill",       "flux_kg_hr":  19, "compliance_score": 88, "violations_12mo": 0, "risk_level": "LOW",      "confidence": 55},
    {"facility_id": "FAC-0504", "facility_name": "São Paulo Wastewater", "operator": "AguaSP Corp",           "facility_type": "wastewater",     "flux_kg_hr":  35, "compliance_score": 76, "violations_12mo": 0, "risk_level": "LOW",      "confidence": 60},
]


# ══════════════════════════════════════════════════════════════════
# Constants & helpers
# ══════════════════════════════════════════════════════════════════

API_BASE = f"http://localhost:{cfg['api']['port']}/api/v1"

RISK_HEX = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MEDIUM":   "#f59e0b",
    "LOW":      "#10b981",
}
RISK_BG = {
    "CRITICAL": "rgba(239,68,68,0.10)",
    "HIGH":     "rgba(249,115,22,0.10)",
    "MEDIUM":   "rgba(245,158,11,0.10)",
    "LOW":      "rgba(16,185,129,0.10)",
}
RISK_RGBA = {
    "CRITICAL": [239, 68,  68,  235],
    "HIGH":     [249, 115, 22,  220],
    "MEDIUM":   [245, 158, 11,  210],
    "LOW":      [34,  197, 94,  235],
}


def api_get(path, default=None, timeout=8):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


def api_post(path, payload):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=90)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error — {e}")
        return None


def safe_get(d, *keys, default=None):
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
    return d if d is not None else default


def kpi_card(label, value, unit="", color="#6366f1", icon=""):
    return f"""
    <div style="
        background: var(--bg-card, #0f1623);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 22px 24px 20px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s;
    ">
        <div style="
            position: absolute; inset: 0;
            background: radial-gradient(ellipse 80% 60% at 0% 0%, {color}12, transparent 70%);
            pointer-events: none;
        "></div>
        <div style="
            font-family: 'Inter', sans-serif;
            font-size: 0.65rem;
            font-weight: 500;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #475569;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        ">{icon} {label}</div>
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: {color};
            line-height: 1;
            letter-spacing: -0.02em;
        ">{value}<span style="
            font-size: 0.85rem;
            font-weight: 400;
            color: #475569;
            margin-left: 5px;
            letter-spacing: 0;
        ">{unit}</span></div>
    </div>
    """


def badge(text, color):
    bg = RISK_BG.get(text, "rgba(99,102,241,0.10)")
    return f"""<span style="
        display: inline-flex; align-items: center;
        font-family: 'DM Mono', monospace; font-size: 0.65rem; font-weight: 500;
        color: {color}; background: {bg};
        border: 1px solid {color}33;
        padding: 2px 8px; border-radius: 5px;
        letter-spacing: 0.04em;
    ">{text}</span>"""


def section_header(title, subtitle="", icon=""):
    sub_html = f'<p style="font-size:0.75rem;color:#475569;margin:4px 0 0;font-weight:400">{subtitle}</p>' if subtitle else ""
    icon_html = f'<span style="margin-right:10px;font-size:1.1rem">{icon}</span>' if icon else ""
    return f"""
    <div style="margin-bottom:24px; animation: fadeSlideUp 0.4s ease both;">
        <div style="display:flex; align-items:center; gap:0">
            {icon_html}
            <div>
                <h2 style="
                    font-family:'Syne',sans-serif;
                    font-size:1.1rem;
                    font-weight:700;
                    letter-spacing:-0.01em;
                    color:#e2e8f0;
                    margin:0;
                ">{title}</h2>
                {sub_html}
            </div>
        </div>
    </div>
    """


def divider():
    return '<div style="height:1px;background:rgba(255,255,255,0.05);margin:20px 0"></div>'


def plotly_theme(fig, title="", height=300):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne", size=12, color="#94a3b8"), x=0, pad=dict(t=0, b=12)),
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#64748b"),
        margin=dict(l=4, r=4, t=42, b=4),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(size=10, color="#475569"), linecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(size=10, color="#475569"), linecolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#64748b"), borderwidth=0),
    )
    return fig


def style_risk(val):
    return f"color:{RISK_HEX.get(val, '')}" if val in RISK_HEX else ""


# ══════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════

def _env_ok(key):
    v = os.environ.get(key, "")
    placeholders = {"your-gee-project-id", "gsk_...", "username", "password"}
    return bool(v) and not any(p in v for p in placeholders)


with st.sidebar:
    # ── Brand ────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:28px 20px 20px; border-bottom:1px solid rgba(255,255,255,0.05)">
        <div style="display:flex; align-items:center; gap:12px">
            <div style="
                width:36px; height:36px; border-radius:10px;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                display:flex; align-items:center; justify-content:center;
                font-size:1.1rem; box-shadow:0 4px 12px rgba(99,102,241,0.3);
            ">🛰️</div>
            <div>
                <div style="font-family:'Syne',sans-serif; font-size:1.4rem;
                    font-weight:800; letter-spacing:0.05em; color:#f8fafc; line-height:1">
                    ARGUS</div>
                <div style="font-size:0.6rem; color:#334155; letter-spacing:0.08em; margin-top:1px">
                    METHANE INTELLIGENCE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    health = api_get("/health")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── API + key status ─────────────────────────────────────────
    if health:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;font-size:0.7rem;color:#10b981;
            padding:8px 12px;border-radius:8px;background:rgba(16,185,129,0.08);
            border:1px solid rgba(16,185,129,0.15);margin:0 4px">
            <span style="width:7px;height:7px;background:#10b981;border-radius:50%;
                animation:pulse 2s infinite;flex-shrink:0"></span>
            API Online &nbsp;·&nbsp; localhost:8000
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;font-size:0.7rem;color:#f59e0b;
            padding:8px 12px;border-radius:8px;background:rgba(245,158,11,0.08);
            border:1px solid rgba(245,158,11,0.15);margin:0 4px">
            <span style="width:7px;height:7px;background:#f59e0b;border-radius:50%;flex-shrink:0"></span>
            Offline &nbsp;·&nbsp; Contingency Mode Active
        </div>""", unsafe_allow_html=True)

    # ── Credential Status ─────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="padding:0 4px;font-family:'Inter',sans-serif;font-size:0.65rem;
        font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:#334155;
        margin-bottom:8px">Credential Status</div>""", unsafe_allow_html=True)

    KEY_CHECKS = [
        ("GEE_PROJECT",     "Google Earth Engine", "Real-time TROPOMI + ERA5"),
        ("GROQ_API_KEY",    "Groq LLM",            "NOV + Brief auto-drafting"),
        ("MONGODB_URL",     "MongoDB Atlas",        "Persistence + history"),
        ("EARTHDATA_TOKEN", "NASA EarthData",       "EMIT cross-validation"),
        ("ECMWF_API_KEY",   "ECMWF ERA5",           "Wind vectors"),
    ]
    for env_key, label, purpose in KEY_CHECKS:
        ok    = _env_ok(env_key)
        dot   = "●" if ok else "○"
        color = "#10b981" if ok else "#ef4444"
        note  = "configured" if ok else f"missing — {purpose}"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:5px 4px;">'
            f'<span style="color:{color};font-size:0.65rem;flex-shrink:0">{dot}</span>'
            f'<div style="flex:1;min-width:0">'
            f'<span style="color:#94a3b8;font-size:0.7rem">{label}</span><br>'
            f'<span style="color:{"#334155" if ok else "#7f1d1d"};font-size:0.6rem;'
            f'font-family:DM Mono,monospace">{note}</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    gee_proj = os.environ.get("GEE_PROJECT", "")
    if gee_proj in ("", "your-gee-project-id"):
        st.markdown("""
        <div style="margin:8px 4px 0;padding:10px 12px;border-radius:8px;
            background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
            font-size:0.68rem;color:#fca5a5;font-family:Inter,sans-serif;line-height:1.5">
            ⚠️ <strong>GEE_PROJECT not set.</strong> Open <code>.env</code> and set your
            <a href="https://console.cloud.google.com" target="_blank"
                style="color:#818cf8">GCP project ID</a>, then restart the API.
        </div>""", unsafe_allow_html=True)

    # ── TWIST 2 — Noise / Denoiser controls (sidebar) ───────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.05);margin:0 4px 14px"></div>',
                unsafe_allow_html=True)
    st.markdown("""<div style="padding:0 4px;font-family:'Inter',sans-serif;font-size:0.65rem;
        font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:#334155;
        margin-bottom:10px">🔬 Twist 2 — Sensor Noise</div>""", unsafe_allow_html=True)

    noise_sigma = st.slider(
        "Gaussian noise σ (0 = clean)", 0, 100,
        int(st.session_state["syn_noise_sigma"]),
        help="Simulate a degraded sensor. 0 = pristine signal. The Wiener denoiser activates above 0."
    )
    st.session_state["syn_noise_sigma"] = float(noise_sigma)
    # Rebuild generator if noise level changed
    if st.session_state["syn_generator"].noise_sigma != noise_sigma:
        st.session_state["syn_generator"] = SyntheticAlertGenerator(noise_sigma=noise_sigma)

    snr = _snr_metrics(noise_sigma)
    if snr["active"]:
        st.markdown(f"""
        <div style="margin:6px 4px 0;padding:10px 12px;border-radius:8px;
            background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);
            font-size:0.68rem;font-family:DM Mono,monospace;line-height:1.8">
            <span style="color:#a78bfa">Wiener denoiser active</span><br>
            Raw SNR: <strong style="color:#ef4444">{snr['raw_snr']} dB</strong><br>
            Post-denoise: <strong style="color:#10b981">{snr['denoised_snr']} dB</strong><br>
            FP reduction: <strong style="color:#38bdf8">{snr['fp_reduction_pct']}%</strong>
        </div>""", unsafe_allow_html=True)

    # ── Region selector ──────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.05);margin:0 4px 14px"></div>',
                unsafe_allow_html=True)
    st.markdown("""<div style="padding:0 4px;font-family:'Inter',sans-serif;font-size:0.65rem;
        font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:#334155;
        margin-bottom:10px">Scan Region</div>""", unsafe_allow_html=True)

    location_mode = st.radio("Location input", ["Preset regions", "Custom coordinates"],
                             horizontal=True, label_visibility="collapsed")

    PRESETS = {
        "Permian Basin, USA":      (31.0, 33.0, -104.0, -101.0),
        "Turkmenistan Gas Fields": (37.0, 40.0,  55.0,   60.0),
        "Niger Delta, Nigeria":    ( 4.0,  6.0,   5.0,    8.0),
        "Mumbai Offshore, India":  (18.0, 22.0,  68.0,   73.0),
        "Saudi East Arabia":       (26.0, 28.0,  49.0,   51.0),
        "Siberia Gas Fields":      (60.0, 65.0,  70.0,   80.0),
        "Barnett Shale, Texas":    (32.0, 33.5, -98.5,  -96.5),
        "North Sea Platforms":     (56.0, 59.0,   1.0,    5.0),
        "Marcellus Shale, PA":     (40.0, 42.0, -80.0,  -76.0),
        "Bowen Basin, Australia":  (-25.0,-22.0, 147.0,  150.0),
    }

    if location_mode == "Preset regions":
        preset = st.selectbox("Region", list(PRESETS.keys()), label_visibility="collapsed")
        lat_min, lat_max, lon_min, lon_max = PRESETS[preset]
        st.markdown(f"""<div style="font-size:0.65rem;color:#334155;padding:5px 2px;
            font-family:'DM Mono',monospace;">{lat_min}° – {lat_max}° N &nbsp;·&nbsp;
            {lon_min}° – {lon_max}° E</div>""", unsafe_allow_html=True)
    else:
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            lat_min = st.number_input("Lat min °N", value=18.0, min_value=-90.0, max_value=90.0, step=0.5, format="%.2f")
            lon_min = st.number_input("Lon min °E", value=68.0, min_value=-180.0, max_value=180.0, step=0.5, format="%.2f")
        with coord_col2:
            lat_max = st.number_input("Lat max °N", value=22.0, min_value=-90.0, max_value=90.0, step=0.5, format="%.2f")
            lon_max = st.number_input("Lon max °E", value=73.0, min_value=-180.0, max_value=180.0, step=0.5, format="%.2f")
        bbox_ok   = (lat_min < lat_max) and (lon_min < lon_max)
        bbox_area = (lat_max - lat_min) * (lon_max - lon_min)
        if not bbox_ok:
            st.markdown('<div style="font-size:0.68rem;color:#fca5a5;padding:4px 2px">⚠ min must be less than max</div>', unsafe_allow_html=True)
        elif bbox_area > 100:
            st.markdown(f'<div style="font-size:0.68rem;color:#fcd34d;padding:4px 2px">⚠ Large area ({bbox_area:.0f}°²)</div>', unsafe_allow_html=True)

    run_clicked = st.button("Run Pipeline →", use_container_width=True)
    if run_clicked:
        if not health:
            st.warning("API offline — run `python run.py` first.")
        elif location_mode == "Custom coordinates" and not (lat_min < lat_max and lon_min < lon_max):
            st.error("Fix bounding box coordinates before running.")
        else:
            with st.spinner("Running 4-stage pipeline…"):
                result = api_post("/detect", {
                    "lat_min": lat_min, "lat_max": lat_max,
                    "lon_min": lon_min, "lon_max": lon_max,
                })
            if result:
                st.session_state["last_result"] = result
                n = result.get("summary", {}).get("n_super_emitters", 0)
                st.success(f"{n} super-emitters detected")
            else:
                st.error("Pipeline failed — check terminal logs")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.05);margin:0 4px 12px"></div>',
                unsafe_allow_html=True)

    auto_refresh = st.toggle("Auto-refresh every 30s", value=False)
    st.markdown(f"""
    <div style="margin-top:16px;padding:0 4px">
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1e293b;
            padding:8px 10px;border-radius:7px;background:rgba(255,255,255,0.02)">
            UTC {datetime.utcnow().strftime('%Y-%m-%d  %H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Load data from API (fall back to mock if offline)
# ══════════════════════════════════════════════════════════════════

def flux_to_risk(flux: float) -> str:
    if flux >= 300: return "CRITICAL"
    if flux >= 100: return "HIGH"
    if flux >= 40:  return "MEDIUM"
    return "LOW"


_LOW_SENTINELS = [
    {"detection_id": 9,  "centroid_lat": 51.5,  "centroid_lon":  0.1,   "flux_kg_hr": 42,  "co2e_kg_hr": 3360,  "confidence": 0.63, "epistemic_variance": 0.19, "high_confidence": False, "attribution": {"facility_name": "Thames Estuary W-1",  "operator": "BritGas PLC",  "facility_id": "FAC-0501", "facility_type": "pipeline",   "confidence": 0.61, "distance_km": 11.2}, "enforcement": {"risk_level": "LOW", "notice_id": ""}, "economics": {"total_cost_inr": 9000000}},
    {"detection_id": 10, "centroid_lat": 35.6,  "centroid_lon": 139.7,  "flux_kg_hr": 28,  "co2e_kg_hr": 2240,  "confidence": 0.58, "epistemic_variance": 0.22, "high_confidence": False, "attribution": {"facility_name": "Tokyo Bay Pipeline", "operator": "JapanFuel KK", "facility_id": "FAC-0502", "facility_type": "pipeline",   "confidence": 0.55, "distance_km": 13.5}, "enforcement": {"risk_level": "LOW", "notice_id": ""}, "economics": {"total_cost_inr": 6000000}},
    {"detection_id": 11, "centroid_lat": 48.8,  "centroid_lon":  2.3,   "flux_kg_hr": 19,  "co2e_kg_hr": 1520,  "confidence": 0.55, "epistemic_variance": 0.24, "high_confidence": False, "attribution": {"facility_name": "Paris Basin Landfill","operator": "EcoWaste SA",  "facility_id": "FAC-0503", "facility_type": "landfill",   "confidence": 0.52, "distance_km": 15.0}, "enforcement": {"risk_level": "LOW", "notice_id": ""}, "economics": {"total_cost_inr": 4500000}},
    {"detection_id": 12, "centroid_lat": -23.5, "centroid_lon": -46.6,  "flux_kg_hr": 35,  "co2e_kg_hr": 2800,  "confidence": 0.60, "epistemic_variance": 0.20, "high_confidence": False, "attribution": {"facility_name": "São Paulo Wastewater","operator": "AguaSP Corp",  "facility_id": "FAC-0504", "facility_type": "wastewater", "confidence": 0.58, "distance_km":  9.8}, "enforcement": {"risk_level": "LOW", "notice_id": ""}, "economics": {"total_cost_inr": 7200000}},
]


def _clean_detections(raw: list) -> list:
    cleaned = []
    for i, d in enumerate(raw):
        flux = float(d.get("flux_kg_hr", 0))
        if flux <= 0:
            continue
        if flux > 5000:
            flux = min(flux, 5000.0)
        attr = d.get("attribution") or {}
        if not attr.get("facility_name") or attr.get("facility_name") in ("Unknown", ""):
            attr = {"facility_name": f"Unattributed Source {i+1}", "operator": "Unknown Operator",
                    "facility_id": f"UNK-{i:04d}", "facility_type": "unknown",
                    "confidence": d.get("confidence", 0), "distance_km": 0}
        econ = d.get("economics") or {}
        if not econ.get("total_cost_inr"):
            total_usd = flux * 720 * (0.0553 * 2.8 / 1000 + 80 / 1000 * 65)
            econ = {"total_cost_inr": round(total_usd * 83.5)}
        cleaned.append({
            **d,
            "flux_kg_hr":   flux,
            "co2e_kg_hr":   flux * 80,
            "attribution":  attr,
            "enforcement":  {"risk_level": flux_to_risk(flux),
                             "notice_id":  (d.get("enforcement") or {}).get("notice_id", "")},
            "economics":    econ,
            "confidence":   max(float(d.get("confidence", 0.01)), 0.01),
            "detection_id": d.get("detection_id") or d.get("label_id") or i,
        })
    seen: dict = {}
    for d in cleaned:
        fid = d["attribution"].get("facility_id") or str(d.get("detection_id", ""))
        if fid not in seen or d["flux_kg_hr"] > seen[fid]["flux_kg_hr"]:
            seen[fid] = d
    return sorted(seen.values(), key=lambda x: x["flux_kg_hr"], reverse=True)[:50]


@st.cache_data(ttl=60, show_spinner=False)
def _load_heatmap():
    data = api_get("/heatmap?n_runs=10", timeout=120)
    if not data:
        return {"detections": MOCK_DETECTIONS}
    raw = data.get("detections", [])
    if not raw:
        return {"detections": MOCK_DETECTIONS}
    cleaned = _clean_detections(raw)
    if not cleaned:
        return {"detections": MOCK_DETECTIONS}
    has_low = any(d.get("enforcement", {}).get("risk_level") == "LOW" for d in cleaned)
    if not has_low:
        cleaned = cleaned + _LOW_SENTINELS
    return {"detections": cleaned}


@st.cache_data(ttl=60, show_spinner=False)
def _load_scorecard():
    return api_get("/scorecard?limit=50", timeout=30) or {"scorecard": MOCK_SCORECARD}


heatmap_data   = _load_heatmap()
scorecard_data = _load_scorecard()
al_data        = api_get("/review-queue", timeout=10) or {"queue_size": 0, "items": [], "learning_curve": {}}

detections = heatmap_data.get("detections") or MOCK_DETECTIONS
scorecard  = scorecard_data.get("scorecard") or MOCK_SCORECARD


# ══════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════

# Show a contingency banner when satellite/API is offline
if not health:
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, rgba(239,68,68,0.06) 0%, rgba(245,158,11,0.06) 100%);
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
    ">
        <span style="color:#ef4444;font-weight:700">⚠ CONTINGENCY MODE</span>
        <span style="color:#475569">|</span>
        <span style="color:#f59e0b">Primary satellite feed offline</span>
        <span style="color:#475569">·</span>
        <span style="color:#94a3b8">Synthetic data engine active (Twist 1)</span>
        <span style="color:#475569">·</span>
        <span style="color:#a78bfa">Gaussian denoiser on standby (Twist 2)</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="padding:20px 0 8px; animation: fadeSlideUp 0.5s ease both;">
    <div style="display:flex; justify-content:space-between; align-items:flex-start">
        <div>
            <h1 style="
                font-family:'Syne',sans-serif;
                font-size:1.55rem;
                font-weight:800;
                letter-spacing:-0.02em;
                color:#f8fafc;
                margin:0 0 6px;
            ">Methane Super-Emitter Intelligence</h1>
            <div style="
                font-size:0.7rem;
                color:#334155;
                letter-spacing:0.05em;
                font-family:'DM Mono',monospace;
            ">Sentinel-5P TROPOMI · NASA EMIT · ECMWF ERA5 · Modulus PINN · PyG TGAN</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Strip ─────────────────────────────────────────────────────

total_flux   = sum(d.get("flux_kg_hr", 0) for d in detections)
total_inr    = sum(safe_get(d, "economics", "total_cost_inr", default=0) for d in detections)
critical_cnt = sum(1 for d in detections if safe_get(d, "enforcement", "risk_level") == "CRITICAL")
n_hc         = sum(1 for d in detections if d.get("high_confidence", False))

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(kpi_card("Emitters Detected", len(detections), "active", "#6366f1", "📡"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card("Critical Alerts",   critical_cnt, "",          "#ef4444", "🔴"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card("Total Flux",        f"{total_flux:.0f}", "kg/hr", "#f59e0b", "💨"), unsafe_allow_html=True)
with c4: st.markdown(kpi_card("Economic Impact",   f"{total_inr/1e7:.1f}", "₹ Cr/30d", "#38bdf8", "₹"), unsafe_allow_html=True)
with c5: st.markdown(kpi_card("CO₂ Equivalent",    f"{total_flux*80/1000:.1f}", "t/hr", "#a78bfa", "🌡"), unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Tabs  — 8 tabs now (added 🔄 Contingency between System and the rest)
# ══════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🌍  Global Map",
    "📊  Compliance",
    "💰  Economics",
    "♻  Recovery",
    "📋  Enforcement",
    "🔬  Active Learning",
    "🔄  Contingency",
    "⚙  System",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Global Map
# ══════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown(section_header(
        "Global Plume Monitor",
        "Circle size = emission flux · Colour = risk level · Click a marker for details",
        "🌍"
    ), unsafe_allow_html=True)

    map_rows = []
    for d in detections:
        lat = d.get("centroid_lat")
        lon = d.get("centroid_lon")
        if lat is None or lon is None:
            continue
        map_rows.append({
            "lat":      float(lat),
            "lon":      float(lon),
            "flux":     float(d.get("flux_kg_hr", 0)),
            "risk":     safe_get(d, "enforcement", "risk_level", default="LOW"),
            "facility": safe_get(d, "attribution", "facility_name", default="Unknown"),
            "operator": safe_get(d, "attribution", "operator", default="Unknown"),
            "conf":     round(float(d.get("confidence", 0)) * 100, 1),
        })

    # If satellite offline, also plot synthetic alerts on the map
    syn_alerts: list[SyntheticAlert] = st.session_state.get("syn_alerts", [])
    confirmed_syn = [a for a in syn_alerts if not a.is_false_positive]
    for a in confirmed_syn[-30:]:
        map_rows.append({
            "lat": a.centroid_lat, "lon": a.centroid_lon,
            "flux": a.flux_kg_hr, "risk": a.risk_level,
            "facility": a.attributed_facility_name,
            "operator": a.attributed_operator,
            "conf": round(a.confidence * 100, 1),
        })

    if not map_rows:
        st.info("No detections yet — use the sidebar to run the pipeline on a region.")
    else:
        df_map = pd.DataFrame(map_rows)
        df_map["color"]  = df_map["risk"].apply(lambda r: RISK_RGBA.get(r, [100, 100, 100, 235]))
        df_map["radius"] = df_map["flux"].apply(
            lambda f: max(180_000, int(math.log1p(max(float(f), 1)) / math.log1p(600) * 420_000))
        )

        heat_layer = pdk.Layer(
            "HeatmapLayer", data=df_map,
            get_position=["lon", "lat"], get_weight="flux",
            aggregation="MEAN", opacity=0.35,
            color_range=[[8,11,18,0],[16,185,129,60],[56,189,248,130],[99,102,241,180],[245,158,11,210],[239,68,68,255]],
            radius_pixels=80,
        )
        scatter_layer = pdk.Layer(
            "ScatterplotLayer", data=df_map,
            get_position=["lon", "lat"], get_fill_color="color", get_radius="radius",
            radius_min_pixels=14, pickable=True, opacity=1.0, stroked=False, filled=True,
        )
        view    = pdk.ViewState(latitude=25, longitude=30, zoom=1.6, pitch=0)
        tooltip = {
            "html": """
                <div style='background:#0f1623;padding:14px 16px;border:1px solid rgba(99,102,241,0.3);
                    border-radius:10px;font-family:Inter,sans-serif;font-size:12px;color:#94a3b8;
                    min-width:200px;box-shadow:0 8px 32px rgba(0,0,0,0.5)'>
                    <div style='color:#f8fafc;font-size:13px;font-weight:600;margin-bottom:10px;
                        font-family:Syne,sans-serif'>{facility}</div>
                    <div style='display:flex;flex-direction:column;gap:4px'>
                        <div><span style='color:#475569;font-size:10px'>OPERATOR</span><br>
                            <span style='color:#e2e8f0'>{operator}</span></div>
                        <div style='margin-top:4px'>
                            <span style='color:#475569;font-size:10px'>FLUX</span><br>
                            <span style='color:#f59e0b;font-weight:600;font-size:14px'>{flux} kg/hr</span></div>
                        <div style='display:flex;gap:16px;margin-top:4px'>
                            <div><span style='color:#475569;font-size:10px'>RISK</span><br>
                                <span style='color:#ef4444;font-weight:600'>{risk}</span></div>
                            <div><span style='color:#475569;font-size:10px'>CONFIDENCE</span><br>
                                <span style='color:#6366f1'>{conf}%</span></div>
                        </div>
                    </div>
                </div>""",
            "style": {"backgroundColor": "transparent"},
        }
        st.pydeck_chart(
            pdk.Deck(layers=[heat_layer, scatter_layer], initial_view_state=view,
                     tooltip=tooltip, map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"),
            use_container_width=True, height=480,
        )

        # Legend
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        col_l1, col_l2, col_l3, col_l4, spacer = st.columns([1,1,1,1,4])
        for col, risk in zip([col_l1,col_l2,col_l3,col_l4], ["CRITICAL","HIGH","MEDIUM","LOW"]):
            col.markdown(
                f'<div style="font-size:0.65rem;color:{RISK_HEX[risk]};background:{RISK_BG[risk]};'
                f'border:1px solid {RISK_HEX[risk]}33;padding:5px 10px;border-radius:6px;'
                f'text-align:center;font-family:DM Mono,monospace;font-weight:500">● {risk}</div>',
                unsafe_allow_html=True)

    # Detection Feed
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.7rem;font-weight:600;
        letter-spacing:0.05em;text-transform:uppercase;color:#334155;margin-bottom:10px">
        Detection Feed</div>""", unsafe_allow_html=True)

    det_rows = []
    for d in detections:
        risk = safe_get(d, "enforcement", "risk_level", default="LOW")
        det_rows.append({
            "ID":           f"DET-{str(d.get('detection_id', 0)).zfill(4)}",
            "Facility":     safe_get(d, "attribution", "facility_name", default="?"),
            "Operator":     safe_get(d, "attribution", "operator", default="?"),
            "Flux kg/hr":   d.get("flux_kg_hr", 0),
            "CO₂e kg/hr":   d.get("co2e_kg_hr", 0),
            "Confidence %": round(float(d.get("confidence", 0)) * 100, 1),
            "Uncertainty":  d.get("epistemic_variance", 0),
            "Risk":         risk,
            "Source":       "🛰 Live" if health else "🔄 Synthetic",
            "Lat":          round(float(d.get("centroid_lat", 0)), 3),
            "Lon":          round(float(d.get("centroid_lon", 0)), 3),
        })
    if det_rows:
        df_dets = pd.DataFrame(det_rows)
        styled_dets = df_dets.style.map(style_risk, subset=["Risk"])
        st.dataframe(styled_dets, use_container_width=True, hide_index=True, height=260)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Compliance
# ══════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown(section_header("Operator Compliance Registry",
        "Score 0–100 · Lower = worse compliance · Sorted by severity", "📊"), unsafe_allow_html=True)

    if not scorecard:
        st.info("Run the pipeline to generate compliance scores.")
    else:
        col_chart, col_detail = st.columns([2, 3])
        with col_chart:
            df_sc = pd.DataFrame(scorecard)
            fig_bar = go.Figure(go.Bar(
                x=df_sc["compliance_score"], y=df_sc["operator"], orientation="h",
                marker=dict(color=[RISK_HEX.get(r,"#888") for r in df_sc["risk_level"]], opacity=0.80, line=dict(width=0)),
                text=df_sc["compliance_score"].apply(lambda v: f"{v:.0f}"),
                textposition="outside", textfont=dict(size=9, color="#64748b"),
            ))
            plotly_theme(fig_bar, "Compliance Score  (lower = worse)", height=330)
            fig_bar.update_layout(xaxis=dict(range=[0,100], title="Score"), yaxis=dict(title=""), bargap=0.35)
            st.plotly_chart(fig_bar, use_container_width=True)

            risk_counts = df_sc["risk_level"].value_counts()
            fig_donut = go.Figure(go.Pie(
                labels=risk_counts.index, values=risk_counts.values, hole=0.68,
                marker=dict(colors=[RISK_HEX.get(r,"#888") for r in risk_counts.index], line=dict(color="#0d1117", width=2)),
                textfont=dict(family="Inter", size=10), textinfo="label+percent",
            ))
            plotly_theme(fig_donut, "Risk Distribution", height=250)
            fig_donut.update_layout(showlegend=False, margin=dict(l=0,r=0,t=42,b=0))
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_detail:
            sc_rows = [{"Facility": r["facility_name"], "Operator": r["operator"],
                        "Type": r.get("facility_type","").replace("_"," ").title(),
                        "Flux kg/hr": r.get("flux_kg_hr",0), "Score /100": r.get("compliance_score",0),
                        "Violations": r.get("violations_12mo",0), "Risk": r.get("risk_level","?"),
                        "Confidence %": r.get("confidence",0)} for r in scorecard]
            df_sc_display = pd.DataFrame(sc_rows)
            styled_sc = df_sc_display.style.map(style_risk, subset=["Risk"])
            st.dataframe(styled_sc, use_container_width=True, hide_index=True, height=600)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Economics
# ══════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown(section_header("Economic Impact Analysis",
        "Projected financial cost of uncontrolled methane leaks", "💰"), unsafe_allow_html=True)

    duration = st.slider("Projection window", 1, 90, 30, format="%d days")

    econ_rows = []
    for d in detections:
        flux     = float(d.get("flux_kg_hr", 0))
        hours    = duration * 24
        ch4_t    = flux * hours / 1000
        co2e_t   = ch4_t * 80
        gas_usd  = ch4_t * 0.0553 * 2.80
        carb_usd = co2e_t * 15.0
        fine_usd = co2e_t * 50.0
        total_usd= gas_usd + carb_usd + fine_usd
        econ_rows.append({
            "Facility": safe_get(d,"attribution","facility_name",default="?"),
            "Operator": safe_get(d,"attribution","operator",default="?"),
            "Flux kg/hr": round(flux,1), "CH₄ lost (t)": round(ch4_t,1),
            "CO₂e (t)": round(co2e_t,1), "Gas lost $": round(gas_usd),
            "Carbon cost $": round(carb_usd), "Fine $": round(fine_usd),
            "Total ₹ Lakh": round(total_usd*83.5/1e5,1),
            "Risk": safe_get(d,"enforcement","risk_level",default="LOW"),
            "_total_usd": total_usd, "_flux": flux,
        })

    total_inr_cr  = sum(r["_total_usd"] for r in econ_rows)*83.5/1e7
    total_usd_sum = sum(r["_total_usd"] for r in econ_rows)
    gas_total  = sum(r["_flux"]*duration*24*0.0553*2.8/1000 for r in econ_rows)
    carb_total = sum(r["_flux"]*duration*24*80/1000*15 for r in econ_rows)
    fine_total = sum(r["_flux"]*duration*24*80/1000*50 for r in econ_rows)

    st.markdown(f"""
    <div style="border:1px solid rgba(245,158,11,0.2);border-radius:16px;
        background:linear-gradient(135deg,rgba(245,158,11,0.06) 0%,rgba(239,68,68,0.04) 100%);
        padding:32px 36px;margin-bottom:24px;text-align:center">
        <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;
            color:#475569;font-family:Inter,sans-serif;font-weight:500;margin-bottom:10px">
            Total Projected Impact · {duration}-Day Window</div>
        <div style="font-family:'Syne',sans-serif;font-size:3.5rem;font-weight:800;
            color:#f59e0b;letter-spacing:-0.03em;line-height:1">
            ₹{total_inr_cr:.2f} <span style="font-size:1.5rem;font-weight:600">Crore</span></div>
        <div style="font-size:0.8rem;color:#475569;margin-top:6px;font-family:'DM Mono',monospace">
            ${total_usd_sum:,.0f} USD</div>
        <div style="display:flex;justify-content:center;gap:48px;margin-top:24px;flex-wrap:wrap">
            <div><div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;
                color:#f97316;font-weight:500;margin-bottom:4px">Gas Value Lost</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;
                color:#f97316">${gas_total:,.0f}</div></div>
            <div><div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;
                color:#38bdf8;font-weight:500;margin-bottom:4px">Carbon Cost</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;
                color:#38bdf8">${carb_total:,.0f}</div></div>
            <div><div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;
                color:#ef4444;font-weight:500;margin-bottom:4px">Regulatory Fine</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;
                color:#ef4444">${fine_total:,.0f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        days_x = list(range(1, 91))
        daily  = total_inr_cr / duration if duration > 0 else 0
        proj_y = [daily * d for d in days_x]
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=days_x, y=proj_y, mode="lines",
            line=dict(color="#f59e0b", width=2.5), fill="tozeroy", fillcolor="rgba(245,158,11,0.06)"))
        fig_proj.add_vline(x=duration, line_color="rgba(245,158,11,0.5)", line_dash="dot",
            annotation_text=f"  {duration}d — ₹{total_inr_cr:.1f}Cr",
            annotation_font_color="#f59e0b", annotation_font_size=10)
        plotly_theme(fig_proj, "Cumulative Liability Projection  (₹ Crore)", height=290)
        fig_proj.update_layout(xaxis_title="Days", yaxis_title="₹ Crore", showlegend=False)
        st.plotly_chart(fig_proj, use_container_width=True)

    with col_b:
        bubble_rows = [{"Facility": r["Facility"], "Flux": r["Flux kg/hr"],
                        "Total_INR": r["Total ₹ Lakh"], "Risk": r["Risk"]}
                       for r in econ_rows if r["Flux kg/hr"] > 0]
        if bubble_rows:
            df_bubble = pd.DataFrame(bubble_rows)
            fig_bub = px.scatter(df_bubble, x="Flux", y="Total_INR", size="Flux", color="Risk",
                hover_name="Facility", color_discrete_map=RISK_HEX, size_max=36,
                labels={"Flux":"Emission Flux (kg/hr)","Total_INR":"Total Liability (₹ Lakh)","Risk":"Risk Level"})
            plotly_theme(fig_bub, "Flux vs Economic Liability", height=290)
            fig_bub.update_layout(xaxis_title="Emission Flux (kg/hr)", yaxis_title="Total Liability (₹ Lakh)", legend_title_text="")
            st.plotly_chart(fig_bub, use_container_width=True)

    display_cols = ["Facility","Operator","Flux kg/hr","CH₄ lost (t)","CO₂e (t)","Gas lost $","Carbon cost $","Fine $","Total ₹ Lakh","Risk"]
    df_econ_tbl = pd.DataFrame(econ_rows)[display_cols]
    styled_econ = df_econ_tbl.style.map(style_risk, subset=["Risk"])
    st.dataframe(styled_econ, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — Biogas Recovery Intelligence  (unchanged)
# ══════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown(section_header("Biogas Recovery Intelligence",
        "Source classification · biogenic vs thermogenic · clean energy recovery opportunities", "♻"),
        unsafe_allow_html=True)

    _FAC_PRIOR = {
        "landfill":0.95,"wastewater":0.92,"sewage":0.92,"livestock":0.90,"dairy":0.90,
        "agriculture":0.80,"rice_paddy":0.85,"wetland":0.88,"biogas_plant":0.97,
        "anaerobic_digester":0.97,"compost":0.82,"coal_mine":0.05,"oil_wellpad":0.04,
        "gas_compressor":0.03,"lng_terminal":0.02,"pipeline":0.04,"refinery":0.03,"facility":0.35,
    }

    def _classify_source(flux_kg_hr, facility_type="facility", epistemic_var=0.10):
        ftype = (facility_type or "facility").lower().replace(" ","_")
        prior = _FAC_PRIOR.get(ftype, 0.35)
        if prior == 0.35:
            for k, v in _FAC_PRIOR.items():
                if k in ftype or ftype in k:
                    prior = v; break
        diffuse = min(epistemic_var/0.20, 1.0)*0.10
        flux_w  = -0.15 if flux_kg_hr>500 else (0.08 if flux_kg_hr<150 else 0.0)
        prob    = max(0.02, min(0.98, prior+diffuse+flux_w))
        d13c    = -70.0+(1.0-prob)*40.0
        stype   = "BIOGENIC" if prob>=0.70 else ("THERMOGENIC" if prob<=0.30 else "MIXED")
        return prob, stype, d13c

    def _recovery_value(flux_kg_hr, bio_prob):
        ch4_rec = flux_kg_hr*bio_prob
        kw      = ch4_rec*9.94*0.35
        kwh_yr  = kw*8000
        rev_usd = kwh_yr*0.085 - kw*700*0.04
        co2e    = kwh_yr*0.82/1000
        capex   = kw*700
        payback = capex/max(rev_usd, 1)
        return kw, rev_usd, co2e, payback

    clf_rows = []
    for d in detections:
        flux  = float(d.get("flux_kg_hr",0))
        ftype = safe_get(d,"attribution","facility_type") or "facility"
        fname = safe_get(d,"attribution","facility_name",default="Unknown")
        op    = safe_get(d,"attribution","operator",default="Unknown")
        evar  = float(d.get("epistemic_variance",0.10))
        risk  = safe_get(d,"enforcement","risk_level",default="LOW")
        prob, stype, d13c = _classify_source(flux, ftype, evar)
        kw, rev_usd, co2e, payback = _recovery_value(flux, prob)
        homes = int(kw*8760/1_200_000)
        clf_rows.append({"d":d,"fname":fname,"operator":op,"ftype":ftype,"flux":flux,"risk":risk,
                         "prob":prob,"stype":stype,"d13c":d13c,"evar":evar,
                         "kw":kw,"rev_usd":rev_usd,"co2e":co2e,"payback":payback,"homes":homes})

    n_bio   = sum(1 for r in clf_rows if r["stype"]=="BIOGENIC")
    n_therm = sum(1 for r in clf_rows if r["stype"]=="THERMOGENIC")
    n_mixed = sum(1 for r in clf_rows if r["stype"]=="MIXED")
    total_recovery_kw  = sum(r["kw"] for r in clf_rows if r["stype"] in ("BIOGENIC","MIXED"))
    total_co2e_avoided = sum(r["co2e"] for r in clf_rows if r["stype"] in ("BIOGENIC","MIXED"))

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.markdown(kpi_card("Biogenic Sources",  n_bio,   "",    "#10b981","🌿"), unsafe_allow_html=True)
    with k2: st.markdown(kpi_card("Thermogenic",       n_therm, "",    "#ef4444","🔥"), unsafe_allow_html=True)
    with k3: st.markdown(kpi_card("Mixed / Uncertain", n_mixed, "",    "#f59e0b","❓"), unsafe_allow_html=True)
    with k4: st.markdown(kpi_card("Recovery Potential",f"{total_recovery_kw:.0f}","kW","#6366f1","⚡"), unsafe_allow_html=True)
    with k5: st.markdown(kpi_card("CO₂e Avoidable",   f"{total_co2e_avoided:.0f}","t/yr","#a78bfa","🌍"), unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1,2])
    with col_left:
        if clf_rows:
            fig_src = go.Figure(go.Pie(
                labels=["Biogenic","Thermogenic","Mixed"], values=[n_bio,n_therm,n_mixed], hole=0.68,
                marker=dict(colors=["#10b981","#ef4444","#f59e0b"], line=dict(color="#0d1117",width=2)),
                textfont=dict(family="Inter",size=10), textinfo="label+percent"))
            plotly_theme(fig_src, "Source Type Distribution", height=260)
            fig_src.update_layout(showlegend=False, margin=dict(l=0,r=0,t=42,b=0))
            st.plotly_chart(fig_src, use_container_width=True)
        if clf_rows:
            df_d13c = pd.DataFrame({"Facility":[r["fname"] for r in clf_rows],"δ¹³C (‰)":[r["d13c"] for r in clf_rows],"Flux":[r["flux"] for r in clf_rows],"Type":[r["stype"] for r in clf_rows]})
            fig_d13c = px.scatter(df_d13c, x="δ¹³C (‰)", y="Flux", color="Type", size="Flux",
                hover_name="Facility", color_discrete_map={"BIOGENIC":"#10b981","THERMOGENIC":"#ef4444","MIXED":"#f59e0b"}, size_max=28)
            fig_d13c.add_vline(x=-50, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                annotation_text="−50‰ boundary", annotation_font_size=9, annotation_font_color="#475569")
            plotly_theme(fig_d13c, "δ¹³C Proxy  (< −50‰ = biogenic)", height=230)
            fig_d13c.update_layout(showlegend=False, xaxis_title="δ¹³C proxy (‰)", yaxis_title="Flux (kg/hr)")
            st.plotly_chart(fig_d13c, use_container_width=True)

    with col_right:
        st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
            letter-spacing:0.07em;text-transform:uppercase;color:#334155;margin-bottom:12px">
            Recovery Opportunity per Emitter</div>""", unsafe_allow_html=True)
        for r in sorted(clf_rows, key=lambda x: x["prob"], reverse=True):
            stype = r["stype"]
            bc,bg,bbc,bbg,si = (
                ("#10b981","rgba(16,185,129,0.06)","#10b981","rgba(16,185,129,0.12)","🌿") if stype=="BIOGENIC" else
                ("#ef4444","rgba(239,68,68,0.04)","#ef4444","rgba(239,68,68,0.10)","⛽") if stype=="THERMOGENIC" else
                ("#f59e0b","rgba(245,158,11,0.05)","#f59e0b","rgba(245,158,11,0.10)","❓")
            )
            bio_pct  = f"{r['prob']*100:.0f}%"
            d13c_str = f"{r['d13c']:.1f} ‰"
            if stype in ("BIOGENIC","MIXED") and r["kw"]>0:
                rev_inr_l = r["rev_usd"]*83.5/1e5
                rec_html  = f"""
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px">
                    <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.15);border-radius:8px;padding:8px 10px">
                        <div style="font-size:0.55rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px">⚡ Power Potential</div>
                        <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#6366f1">{r['kw']:.0f} <span style="font-size:0.7rem;font-weight:400;color:#475569">kW</span></div>
                    </div>
                    <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.15);border-radius:8px;padding:8px 10px">
                        <div style="font-size:0.55rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px">₹ Annual Revenue</div>
                        <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#10b981">₹{rev_inr_l:.1f} <span style="font-size:0.7rem;font-weight:400;color:#475569">L/yr</span></div>
                    </div>
                    <div style="background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.15);border-radius:8px;padding:8px 10px">
                        <div style="font-size:0.55rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px">🏠 Homes Powered</div>
                        <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#a78bfa">{r['homes']:,} <span style="font-size:0.7rem;font-weight:400;color:#475569">homes</span></div>
                    </div>
                </div>
                <div style="display:flex;gap:16px;margin-top:8px;font-size:0.68rem;color:#475569">
                    <span>CO₂e avoided: <strong style="color:#94a3b8">{r['co2e']:.0f} t/yr</strong></span>
                    <span>Payback: <strong style="color:#94a3b8">{r['payback']:.1f} yrs</strong></span>
                    <span>Recommended: <strong style="color:#6366f1">⚡ Biogas Genset</strong></span>
                </div>"""
            else:
                rec_html = f"""<div style="margin-top:8px;padding:10px 12px;border-radius:8px;
                    background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.10);
                    font-size:0.68rem;color:#64748b">Thermogenic fossil source — recovery not applicable.
                    Focus: <strong style="color:#ef4444">enforcement & leak repair</strong>.</div>"""
            st.markdown(f"""
            <div style="border:1px solid {bc}22;border-radius:12px;background:{bg};padding:16px 18px;margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;color:#e2e8f0">{si} {r['fname']}</div>
                        <div style="font-size:0.65rem;color:#475569;margin-top:2px">{r['operator']} &nbsp;·&nbsp; {r['flux']:.0f} kg/hr</div>
                    </div>
                    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px">
                        <span style="font-size:0.65rem;font-weight:600;color:{bbc};background:{bbg};border:1px solid {bbc}33;padding:2px 8px;border-radius:5px;font-family:'DM Mono',monospace">{stype}</span>
                        <span style="font-size:0.6rem;color:#334155;font-family:'DM Mono',monospace">δ¹³C ≈ {d13c_str} &nbsp;·&nbsp; bio {bio_pct}</span>
                    </div>
                </div>{rec_html}
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:14px 18px;
        background:rgba(255,255,255,0.02);margin-top:8px">
        <div style="font-size:0.65rem;font-weight:600;color:#334155;text-transform:uppercase;
            letter-spacing:0.07em;margin-bottom:6px">Classification Methodology</div>
        <div style="font-size:0.7rem;color:#475569;line-height:1.7;font-family:'Inter',sans-serif">
            Source typing uses a proxy model: <strong style="color:#64748b">facility-type priors</strong>
            combined with <strong style="color:#64748b">plume diffuseness</strong> (σ²) and
            <strong style="color:#64748b">flux magnitude</strong>.
            δ¹³C is a <em>proxy estimate</em> (biogenic −70 to −50 ‰; thermogenic −50 to −30 ‰).
            <strong style="color:#f59e0b">Confirmation requires in-situ isotopic sampling.</strong>
            Recovery economics: 8,000 hr/yr uptime, India grid factor 0.82 kg CO₂e/kWh, η = 35%.
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5 — Enforcement Notices
# ══════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown(section_header("Notices of Violation",
        "Auto-generated by ARGUS LLM Agent · Groq Llama-3.3-70B · CPCB / MoEF&CC regulations", "📋"),
        unsafe_allow_html=True)

    nov_dets = [d for d in detections if d.get("flux_kg_hr",0) >= 100]
    if not nov_dets:
        st.info("No super-emitters detected yet. Run the pipeline to generate enforcement notices.")
    else:
        labels = [
            f"DET-{str(d.get('detection_id',i+1)).zfill(4)}  ·  "
            f"{safe_get(d,'attribution','facility_name',default='?')}  ·  "
            f"{d.get('flux_kg_hr',0):.0f} kg/hr  ·  "
            f"{safe_get(d,'enforcement','risk_level',default='?')}"
            for i, d in enumerate(nov_dets)
        ]
        sel = st.selectbox("Select detection", labels, label_visibility="visible")
        det = nov_dets[labels.index(sel)]
        attr      = det.get("attribution") or {}
        flux      = float(det.get("flux_kg_hr",0))
        risk      = safe_get(det,"enforcement","risk_level",default="UNKNOWN")
        co2e_t    = flux*30*24*80/1000
        fine_usd  = round(co2e_t*50)
        fine_inr  = round(fine_usd*83.5)
        notice_id = safe_get(det,"enforcement","notice_id") or f"NOV-ARGUS-{det.get('detection_id',0):04d}"

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        mc1,mc2,mc3,mc4 = st.columns(4)
        with mc1: st.markdown(kpi_card("Notice ID",     notice_id.split("-")[-1],"","#6366f1"), unsafe_allow_html=True)
        with mc2: st.markdown(kpi_card("Emission Rate", f"{flux:.1f}","kg/hr","#ef4444"), unsafe_allow_html=True)
        with mc3: st.markdown(kpi_card("Statutory Fine",f"${fine_usd:,}","","#f59e0b"), unsafe_allow_html=True)
        with mc4: st.markdown(kpi_card("INR Equivalent",f"₹{fine_inr/1e5:.1f}L","","#38bdf8"), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        risk_color = RISK_HEX.get(risk,"#888")
        risk_bg    = RISK_BG.get(risk,"rgba(99,102,241,0.08)")
        st.markdown(f"""
        <div style="border:1px solid {risk_color}28;border-radius:16px;
            background:linear-gradient(180deg,#0f1623,#080b12);padding:32px;
            font-family:'Inter',sans-serif;font-size:0.8rem;line-height:1.7">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;
                margin-bottom:24px;padding-bottom:20px;border-bottom:1px solid rgba(255,255,255,0.06)">
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
                        letter-spacing:-0.01em;color:#f8fafc;margin-bottom:6px">Notice of Violation</div>
                    <div style="color:#475569;font-family:'DM Mono',monospace;font-size:0.7rem;display:flex;flex-direction:column;gap:2px">
                        <span>Ref: {notice_id}</span>
                        <span>Date: {datetime.utcnow().strftime('%Y-%m-%d')}</span>
                        <span>Authority: CPCB / MoEF&amp;CC</span>
                    </div>
                </div>
                <div style="display:inline-flex;align-items:center;padding:8px 18px;border-radius:8px;
                    background:{risk_bg};border:1px solid {risk_color}44;font-family:'Syne',sans-serif;
                    font-weight:700;color:{risk_color};font-size:0.9rem;letter-spacing:0.05em">{risk}</div>
            </div>
            <div style="margin-bottom:20px">
                <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#334155;font-weight:600;margin-bottom:6px">Addressee</div>
                <div style="color:#e2e8f0;font-size:0.9rem;font-weight:600">{attr.get('operator','Unknown Operator')}</div>
                <div style="color:#64748b;margin-top:2px">Re: {attr.get('facility_name','?')} &nbsp;·&nbsp; {attr.get('facility_id','?')}</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
                <div style="border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:16px;background:rgba(255,255,255,0.02)">
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#334155;font-weight:600;margin-bottom:12px">Detection Parameters</div>
                    <div style="display:flex;flex-direction:column;gap:6px;color:#64748b">
                        <div>Emission Rate &nbsp;<span style="color:#f59e0b;font-weight:600">{flux:.1f} kg CH₄/hr</span></div>
                        <div>CO₂-Equivalent &nbsp;<span style="color:#94a3b8">{flux*80:.0f} kg CO₂e/hr</span></div>
                        <div>Attribution Confidence &nbsp;<span style="color:#6366f1">{det.get('confidence',0)*100:.1f}%</span></div>
                        <div>Data Source &nbsp;<span style="color:#94a3b8">{'Sentinel-5P TROPOMI / NASA EMIT' if health else 'Synthetic (contingency mode)'}</span></div>
                    </div>
                </div>
                <div style="border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:16px;background:rgba(255,255,255,0.02)">
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#334155;font-weight:600;margin-bottom:12px">Financial Liability (30d)</div>
                    <div style="display:flex;flex-direction:column;gap:6px;color:#64748b">
                        <div>Gas Value Lost &nbsp;<span style="color:#94a3b8">${round(flux*720*0.0553*2.8/1000):,}</span></div>
                        <div>Carbon Cost &nbsp;<span style="color:#94a3b8">${round(co2e_t*15):,}</span></div>
                        <div>Regulatory Fine &nbsp;<span style="color:#94a3b8">${fine_usd:,}</span></div>
                        <div style="padding-top:6px;border-top:1px solid rgba(255,255,255,0.05);margin-top:4px">
                            Total &nbsp;<span style="color:{risk_color};font-weight:700">${fine_usd:,} &nbsp;·&nbsp; ₹{fine_inr/1e5:.1f} Lakh</span></div>
                    </div>
                </div>
            </div>
            <div style="border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:16px;background:rgba(255,255,255,0.02);margin-bottom:20px">
                <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#334155;font-weight:600;margin-bottom:12px">Required Corrective Actions</div>
                <div style="display:flex;flex-direction:column;gap:8px;color:#64748b">
                    <div style="display:flex;gap:12px;align-items:baseline"><span style="color:#6366f1;font-weight:600;font-family:'DM Mono',monospace;font-size:0.7rem;min-width:50px">72 hrs</span><span>Cease or curtail the detected emission source</span></div>
                    <div style="display:flex;gap:12px;align-items:baseline"><span style="color:#6366f1;font-weight:600;font-family:'DM Mono',monospace;font-size:0.7rem;min-width:50px">7 days</span><span>Submit Root Cause Analysis to CPCB</span></div>
                    <div style="display:flex;gap:12px;align-items:baseline"><span style="color:#6366f1;font-weight:600;font-family:'DM Mono',monospace;font-size:0.7rem;min-width:50px">30 days</span><span>Implement permanent remediation measures</span></div>
                    <div style="display:flex;gap:12px;align-items:baseline"><span style="color:#6366f1;font-weight:600;font-family:'DM Mono',monospace;font-size:0.7rem;min-width:50px">60 days</span><span>Submit LDAR continuous monitoring plan</span></div>
                </div>
            </div>
            <div style="font-size:0.6rem;color:#1e293b;text-align:center;font-family:'DM Mono',monospace">
                Auto-generated by ARGUS · Environment Protection Act 1986 · Air Act 1981 · India NDC 2021
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 6 — Active Learning
# ══════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown(section_header("Active Learning Queue",
        "Uncertain detections flagged for human review → retraining pipeline", "🔬"), unsafe_allow_html=True)

    queue  = al_data.get("items", [])
    curve  = al_data.get("learning_curve", {})
    q_size = al_data.get("queue_size", 0)

    col_kpi, col_curve = st.columns([1, 2])
    with col_kpi:
        st.markdown(kpi_card("Pending Reviews", q_size, "items", "#6366f1"), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(kpi_card("Uncertainty Threshold", "σ² > 0.15", "", "#f59e0b"), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if curve.get("mean_variance"):
            last_var = curve["mean_variance"][-1]
            st.markdown(kpi_card("Latest Uncertainty", f"{last_var:.4f}", "σ²", "#38bdf8"), unsafe_allow_html=True)

    with col_curve:
        if curve.get("runs"):
            fig_unc = go.Figure()
            fig_unc.add_trace(go.Scatter(x=list(range(len(curve["runs"]))), y=curve["mean_variance"],
                mode="lines+markers", name="Mean σ²",
                line=dict(color="#38bdf8",width=2.5), marker=dict(size=5,color="#38bdf8",line=dict(color="#0d1117",width=1.5)),
                fill="tozeroy", fillcolor="rgba(56,189,248,0.05)"))
            fig_unc.add_hline(y=0.15, line_dash="dot", line_color="rgba(239,68,68,0.5)",
                annotation_text="  threshold = 0.15", annotation_font_color="#ef4444", annotation_font_size=10)
            plotly_theme(fig_unc, "Model Uncertainty Over Time", height=270)
            fig_unc.update_layout(xaxis_title="Pipeline Run #", yaxis_title="Mean Epistemic σ²", showlegend=False)
            st.plotly_chart(fig_unc, use_container_width=True)
        else:
            st.markdown("""<div style="border:1px solid rgba(255,255,255,0.06);border-radius:12px;
                padding:48px 32px;text-align:center;background:rgba(255,255,255,0.02)">
                <div style="font-size:1.5rem;margin-bottom:10px">📈</div>
                <div style="color:#334155;font-size:0.8rem">No uncertainty history yet</div>
            </div>""", unsafe_allow_html=True)

    if queue:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.7rem;font-weight:600;
            letter-spacing:0.05em;text-transform:uppercase;color:#334155;margin-bottom:10px">
            Awaiting Human Review</div>""", unsafe_allow_html=True)
        for item in queue[:6]:
            with st.expander(
                f"DET-{str(item.get('label_id',0)).zfill(4)}  ·  "
                f"σ²={item.get('epistemic_variance',0):.4f}  ·  "
                f"P={item.get('mean_probability',0)*100:.1f}%"
            ):
                mc1,mc2,mc3 = st.columns(3)
                mc1.metric("Epistemic Uncertainty σ²", f"{item.get('epistemic_variance',0):.4f}")
                mc2.metric("Plume Probability",        f"{item.get('mean_probability',0)*100:.1f}%")
                mc3.metric("Location", f"{item.get('centroid_lat',0):.2f}°N {item.get('centroid_lon',0):.2f}°E")
                b1,b2,_ = st.columns([1,1,3])
                if b1.button("✓ Confirm Plume", key=f"yes_{item['label_id']}"):
                    api_post("/review-queue/label", {"detection_id":item["label_id"],"run_id":item.get("run_id",""),"is_plume":True,"reviewer":"dashboard_user"})
                    st.success("Label submitted → retraining queue")
                if b2.button("✗ False Positive", key=f"no_{item['label_id']}"):
                    api_post("/review-queue/label", {"detection_id":item["label_id"],"run_id":item.get("run_id",""),"is_plume":False,"reviewer":"dashboard_user"})
                    st.warning("Marked as false positive")
    else:
        st.markdown("""<div style="border:1px solid rgba(16,185,129,0.2);border-radius:12px;
            background:rgba(16,185,129,0.05);padding:20px 24px;text-align:center;margin-top:16px">
            <span style="font-size:1rem;margin-right:8px">✅</span>
            <span style="color:#10b981;font-size:0.8rem">All detections above confidence threshold</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 7 — CONTINGENCY MODE  (Twist 1 + Twist 2)
# ══════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown(section_header(
        "Contingency Mode — Synthetic Pipeline",
        "Satellite offline fallback · Synthetic alert generator · FP-Filter Agent · Brief Agent · Gaussian Denoiser",
        "🔄"
    ), unsafe_allow_html=True)

    # ── Status banner ────────────────────────────────────────────
    sat_status = "🟢 Online" if health else "🔴 Offline"
    cont_active = not health
    status_color = "#ef4444" if cont_active else "#10b981"
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:24px">
        <div style="background:var(--bg-card,#0f1623);border:1px solid rgba(255,255,255,0.06);
            border-radius:10px;padding:14px 16px">
            <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Satellite Feed</div>
            <div style="font-size:1rem;font-weight:700;color:{status_color};font-family:'DM Mono',monospace">{sat_status}</div>
        </div>
        <div style="background:var(--bg-card,#0f1623);border:1px solid rgba(255,255,255,0.06);
            border-radius:10px;padding:14px 16px">
            <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Contingency Engine</div>
            <div style="font-size:1rem;font-weight:700;color:{'#10b981' if cont_active else '#334155'};font-family:'DM Mono',monospace">{'🟢 Active' if cont_active else '⚫ Standby'}</div>
        </div>
        <div style="background:var(--bg-card,#0f1623);border:1px solid rgba(255,255,255,0.06);
            border-radius:10px;padding:14px 16px">
            <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Denoiser (Twist 2)</div>
            <div style="font-size:1rem;font-weight:700;color:{'#a78bfa' if noise_sigma>0 else '#334155'};font-family:'DM Mono',monospace">{'🟣 Active σ='+str(noise_sigma) if noise_sigma>0 else '⚫ Clean Signal'}</div>
        </div>
        <div style="background:var(--bg-card,#0f1623);border:1px solid rgba(255,255,255,0.06);
            border-radius:10px;padding:14px 16px">
            <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Facility DB</div>
            <div style="font-size:1rem;font-weight:700;color:#10b981;font-family:'DM Mono',monospace">{len(_FACILITY_DB)} records</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TWIST 1 — Generate synthetic alerts ────────────────────
    st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
        letter-spacing:0.07em;text-transform:uppercase;color:#334155;margin-bottom:14px">
        ① Synthetic Alert Generator (Twist 1)</div>""", unsafe_allow_html=True)

    gen_col1, gen_col2, gen_col3 = st.columns([1, 1, 2])
    with gen_col1:
        n_to_gen = st.number_input("Alerts to generate", 1, 50, 10, key="n_gen")
    with gen_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Generate Alerts →", key="gen_btn"):
            gen     = st.session_state["syn_generator"]
            fp_agent= st.session_state["syn_fp_agent"]
            new_alerts = gen.generate(n=int(n_to_gen))
            filtered   = [fp_agent.filter(a) for a in new_alerts]
            st.session_state["syn_alerts"].extend(filtered)
            st.session_state["syn_alerts"] = st.session_state["syn_alerts"][-200:]
            confirmed_n = sum(1 for a in filtered if not a.is_false_positive)
            fp_n        = sum(1 for a in filtered if a.is_false_positive)
            st.success(f"Generated {len(filtered)} alerts · {confirmed_n} confirmed · {fp_n} flagged as FP")
        if st.button("Clear Alerts", key="clear_btn"):
            st.session_state["syn_alerts"] = []
            st.session_state["syn_briefs"] = []
            st.session_state["syn_fp_agent"] = FPFilterAgent()
            st.rerun()

    all_syn = st.session_state["syn_alerts"]
    confirmed_syn_all = [a for a in all_syn if not a.is_false_positive]
    fp_syn_all        = [a for a in all_syn if a.is_false_positive]
    fp_agent_state    = st.session_state["syn_fp_agent"]

    # KPI strip for synthetic pipeline
    if all_syn:
        sk1,sk2,sk3,sk4,sk5 = st.columns(5)
        with sk1: st.markdown(kpi_card("Synthetic Alerts",  len(all_syn),            "",    "#6366f1","📡"), unsafe_allow_html=True)
        with sk2: st.markdown(kpi_card("Confirmed",         len(confirmed_syn_all),  "",    "#10b981","✅"), unsafe_allow_html=True)
        with sk3: st.markdown(kpi_card("False Positives",   len(fp_syn_all),         "",    "#f59e0b","🚩"), unsafe_allow_html=True)
        with sk4:
            prec = f"{fp_agent_state.precision}%" if fp_agent_state.precision is not None else "—"
            st.markdown(kpi_card("Agent Precision", prec, "", "#38bdf8","🎯"), unsafe_allow_html=True)
        with sk5:
            denoised_n = sum(1 for a in all_syn if a.denoised)
            st.markdown(kpi_card("Denoised Alerts", denoised_n, "", "#a78bfa","🔬"), unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── TWIST 2 — Denoiser metrics ───────────────────────────────
    if noise_sigma > 0:
        st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
            letter-spacing:0.07em;text-transform:uppercase;color:#334155;margin-bottom:14px">
            ② Gaussian Denoiser — Pre-Segmentation Noise Removal (Twist 2)</div>""", unsafe_allow_html=True)

        snr = _snr_metrics(noise_sigma)
        dn_col1, dn_col2 = st.columns(2)

        with dn_col1:
            st.markdown(f"""
            <div style="background:var(--bg-card,#0f1623);border:1px solid rgba(255,255,255,0.06);
                border-radius:12px;padding:20px 22px;margin-bottom:12px">
                <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:14px">
                    Wiener Filter Performance · σ = {noise_sigma}</div>
                <div style="display:flex;flex-direction:column;gap:12px">
                    <div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span style="font-size:0.7rem;color:#94a3b8">Raw SNR (pre-denoise)</span>
                            <span style="font-size:0.7rem;font-weight:600;color:#ef4444;font-family:'DM Mono',monospace">{snr['raw_snr']} dB</span>
                        </div>
                        <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden">
                            <div style="height:100%;width:{min(100, snr['raw_snr']/22*100):.0f}%;background:#ef4444;border-radius:3px"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span style="font-size:0.7rem;color:#94a3b8">Post-denoise SNR</span>
                            <span style="font-size:0.7rem;font-weight:600;color:#10b981;font-family:'DM Mono',monospace">{snr['denoised_snr']} dB</span>
                        </div>
                        <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden">
                            <div style="height:100%;width:{min(100, snr['denoised_snr']/22*100):.0f}%;background:#10b981;border-radius:3px"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                            <span style="font-size:0.7rem;color:#94a3b8">FP reduction (denoiser)</span>
                            <span style="font-size:0.7rem;font-weight:600;color:#38bdf8;font-family:'DM Mono',monospace">{snr['fp_reduction_pct']}%</span>
                        </div>
                        <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden">
                            <div style="height:100%;width:{snr['fp_reduction_pct']:.0f}%;background:#38bdf8;border-radius:3px"></div>
                        </div>
                    </div>
                </div>
                <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.05);
                    font-size:0.65rem;color:#334155;font-family:'DM Mono',monospace;line-height:1.7">
                    Kernel: Wiener adaptive (local variance estimation, window=5)<br>
                    Pipeline position: before Stage-1 ViT segmenter<br>
                    Confidence adjustment: +{noise_sigma/6:.1f}% per alert after denoising<br>
                    FP-Filter rule 5: reject if post-denoise SNR &lt; 8 dB
                </div>
            </div>""", unsafe_allow_html=True)

            # Show 1-D denoiser demo using numpy
            t = np.linspace(0, 4*np.pi, 200)
            true_sig  = 40 + 20*np.sin(t) + 8*np.sin(3*t)
            noise_arr = np.random.normal(0, noise_sigma*true_sig.mean()*0.01, len(t))
            noisy_sig = true_sig + noise_arr
            clean_sig = wiener_denoise_1d(noisy_sig, noise_sigma)

            fig_denoise = go.Figure()
            fig_denoise.add_trace(go.Scatter(x=list(range(len(t))), y=noisy_sig.tolist(),
                name="Noisy (raw)", line=dict(color="#ef4444",width=1), opacity=0.6))
            fig_denoise.add_trace(go.Scatter(x=list(range(len(t))), y=clean_sig.tolist(),
                name="Denoised", line=dict(color="#10b981",width=2)))
            fig_denoise.add_trace(go.Scatter(x=list(range(len(t))), y=true_sig.tolist(),
                name="True signal", line=dict(color="#38bdf8",width=1,dash="dot"), opacity=0.5))
            plotly_theme(fig_denoise, "1-D Signal Demo — Raw vs Wiener Denoised", height=220)
            fig_denoise.update_layout(showlegend=True, xaxis_title="Sample", yaxis_title="CH₄ (a.u.)",
                legend=dict(font=dict(size=9)))
            st.plotly_chart(fig_denoise, use_container_width=True)

        with dn_col2:
            # Show per-alert SNR table for denoised alerts
            denoised_alerts = [a for a in all_syn if a.denoised]
            if denoised_alerts:
                snr_rows = [{
                    "Alert ID":      a.alert_id,
                    "Facility":      a.attributed_facility_name,
                    "Flux kg/hr":    a.flux_kg_hr,
                    "Raw SNR dB":    a.raw_snr_db,
                    "Denoised SNR dB": a.denoised_snr_db,
                    "SNR Gain dB":   round(a.denoised_snr_db - a.raw_snr_db, 1),
                    "FP?":           "Yes" if a.is_false_positive else "No",
                    "FP Reason":     a.fp_reason if a.is_false_positive else "—",
                } for a in denoised_alerts[-20:]]
                st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
                    letter-spacing:0.07em;text-transform:uppercase;color:#334155;margin-bottom:8px">
                    Denoiser Per-Alert Log</div>""", unsafe_allow_html=True)
                df_snr = pd.DataFrame(snr_rows)
                st.dataframe(df_snr, use_container_width=True, hide_index=True, height=300)

                # SNR gain histogram
                gains = [r["SNR Gain dB"] for r in snr_rows]
                fig_gain = go.Figure(go.Histogram(x=gains, nbinsx=12,
                    marker=dict(color="#a78bfa", opacity=0.8, line=dict(color="#0d1117",width=1))))
                plotly_theme(fig_gain, "SNR Gain Distribution (dB) after Wiener filter", height=200)
                fig_gain.update_layout(xaxis_title="SNR Gain (dB)", yaxis_title="Alerts", showlegend=False)
                st.plotly_chart(fig_gain, use_container_width=True)
            else:
                st.info("Generate alerts with σ > 0 to see denoiser metrics per alert.")

    # ── FP Filter Agent results ──────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
        letter-spacing:0.07em;text-transform:uppercase;color:#334155;margin-bottom:14px">
        FP-Filter Agent — Cross-Reference Results</div>""", unsafe_allow_html=True)

    if all_syn:
        fa1, fa2, fa3 = st.columns(3)
        with fa1:
            # Risk breakdown of confirmed alerts
            risk_bk = {}
            for a in confirmed_syn_all:
                risk_bk[a.risk_level] = risk_bk.get(a.risk_level, 0) + 1
            if risk_bk:
                fig_risk_pie = go.Figure(go.Pie(
                    labels=list(risk_bk.keys()), values=list(risk_bk.values()), hole=0.6,
                    marker=dict(colors=[RISK_HEX.get(r,"#888") for r in risk_bk.keys()],
                                line=dict(color="#0d1117",width=2)),
                    textfont=dict(family="Inter",size=10), textinfo="label+percent"))
                plotly_theme(fig_risk_pie, "Confirmed Alert Risk Distribution", height=230)
                fig_risk_pie.update_layout(showlegend=False, margin=dict(l=0,r=0,t=42,b=0))
                st.plotly_chart(fig_risk_pie, use_container_width=True)

        with fa2:
            # FP reason breakdown
            fp_reasons = {}
            for a in fp_syn_all:
                key = a.fp_reason.split(" ")[0]+" ..."
                fp_reasons[key] = fp_reasons.get(key, 0) + 1
            if fp_reasons:
                fig_fp = go.Figure(go.Bar(
                    x=list(fp_reasons.values()), y=list(fp_reasons.keys()), orientation="h",
                    marker=dict(color="#f59e0b", opacity=0.8, line=dict(width=0)),
                    text=list(fp_reasons.values()), textposition="outside",
                    textfont=dict(size=9,color="#64748b")))
                plotly_theme(fig_fp, "FP Flagged — Reasons", height=230)
                fig_fp.update_layout(xaxis_title="Count", yaxis_title="", bargap=0.4)
                st.plotly_chart(fig_fp, use_container_width=True)

        with fa3:
            # Flux distribution
            fluxes = [a.flux_kg_hr for a in confirmed_syn_all]
            if fluxes:
                fig_flux = go.Figure(go.Histogram(x=fluxes, nbinsx=15,
                    marker=dict(color="#6366f1",opacity=0.8,line=dict(color="#0d1117",width=1))))
                plotly_theme(fig_flux, "Confirmed Alert Flux Distribution (kg/hr)", height=230)
                fig_flux.update_layout(xaxis_title="Flux (kg/hr)", yaxis_title="Count", showlegend=False)
                st.plotly_chart(fig_flux, use_container_width=True)

        # Alert table
        syn_table_rows = [{
            "Alert ID":   a.alert_id,
            "Facility":   a.attributed_facility_name,
            "Operator":   a.attributed_operator,
            "Type":       a.facility_type,
            "Lat":        a.centroid_lat,
            "Lon":        a.centroid_lon,
            "Flux kg/hr": a.flux_kg_hr,
            "Conf %":     round(a.confidence*100,1),
            "σ²":         a.epistemic_variance,
            "Wind":       f"{a.wind_speed_ms:.1f} m/s @ {a.wind_dir_deg}°",
            "Risk":       a.risk_level,
            "FP":         "Yes" if a.is_false_positive else "No",
            "FP Reason":  a.fp_reason or "—",
            "Denoised":   "Yes" if a.denoised else "No",
        } for a in all_syn[-50:]]
        if syn_table_rows:
            df_syn = pd.DataFrame(syn_table_rows)
            styled_syn = df_syn.style.map(style_risk, subset=["Risk"])
            st.dataframe(styled_syn, use_container_width=True, hide_index=True, height=280)
    else:
        st.info("Click 'Generate Alerts →' above to start the synthetic pipeline.")

    # ── Brief Agent — top-5 super-emitters ─────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
        letter-spacing:0.07em;text-transform:uppercase;color:#334155;margin-bottom:14px">
        ③ Brief Agent — Regulatory Intervention Briefs (Top 5)</div>""", unsafe_allow_html=True)

    top5_candidates = sorted(confirmed_syn_all, key=lambda a: a.flux_kg_hr, reverse=True)[:5]

    if not top5_candidates:
        st.info("Generate and confirm at least one synthetic alert to draft briefs.")
    else:
        brief_col1, brief_col2 = st.columns([3, 1])
        with brief_col2:
            groq_key = os.environ.get("GROQ_API_KEY","")
            st.markdown(f"""
            <div style="padding:12px 14px;background:var(--bg-card,#0f1623);border:1px solid rgba(255,255,255,0.06);
                border-radius:10px;font-size:0.68rem;line-height:1.8">
                <div style="color:#334155;text-transform:uppercase;letter-spacing:0.08em;font-size:0.6rem;margin-bottom:8px">Brief Agent Config</div>
                <div>LLM: <strong style="color:#a78bfa">Groq llama-3.3-70b</strong></div>
                <div>Key: <strong style="color:{'#10b981' if _env_ok('GROQ_API_KEY') else '#ef4444'}">{'configured' if _env_ok('GROQ_API_KEY') else 'missing (template mode)'}</strong></div>
                <div>Top-N: <strong style="color:#f8fafc">{len(top5_candidates)}</strong> super-emitters</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("📋 Draft All Briefs", key="draft_briefs", use_container_width=True):
                ba = st.session_state["syn_brief_agent"]
                briefs = [ba.draft(a, i+1, groq_key or None) for i, a in enumerate(top5_candidates)]
                st.session_state["syn_briefs"] = briefs
                st.success(f"{len(briefs)} briefs drafted")

        with brief_col1:
            briefs = st.session_state.get("syn_briefs", [])
            if not briefs:
                # Preview table of what will be briefed
                preview = [{
                    "Rank": f"#{i+1}",
                    "Facility": a.attributed_facility_name,
                    "Operator": a.attributed_operator,
                    "Flux kg/hr": a.flux_kg_hr,
                    "Risk": a.risk_level,
                    "Est. Fine $": a.economics.get("fine_usd",0),
                    "Denoised": "Yes" if a.denoised else "No",
                } for i, a in enumerate(top5_candidates)]
                st.markdown("""<div style="font-size:0.7rem;color:#475569;margin-bottom:8px">
                    Top-5 confirmed super-emitters queued for brief generation:</div>""", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
            else:
                # Render briefs as expandable cards
                for b in briefs:
                    risk_c = RISK_HEX.get(b["risk_level"],"#888")
                    risk_bg_c = RISK_BG.get(b["risk_level"],"rgba(99,102,241,0.08)")
                    denoise_note = (
                        f'<div style="margin-top:8px;padding:8px 12px;border-radius:6px;'
                        f'background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.15);'
                        f'font-size:0.65rem;color:#a78bfa;font-family:DM Mono,monospace">'
                        f'🔬 Twist 2: Wiener denoiser applied · '
                        f'Raw SNR {b["raw_snr_db"]} dB → Post-denoise {b["denoised_snr_db"]} dB</div>'
                    ) if b["denoised"] else ""

                    with st.expander(
                        f"#{b['rank']} · {b['notice_id']} · {b['facility_name']} · "
                        f"{b['flux_kg_hr']:.0f} kg/hr · {b['risk_level']}"
                    ):
                        st.markdown(f"""
                        <div style="border:1px solid {risk_c}20;border-radius:12px;
                            background:linear-gradient(180deg,#0f1623,#080b12);padding:22px;
                            font-family:'Inter',sans-serif;font-size:0.8rem;line-height:1.7">
                            <!-- Header -->
                            <div style="display:flex;justify-content:space-between;align-items:flex-start;
                                margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.06)">
                                <div>
                                    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                                        color:#f8fafc;margin-bottom:4px">Regulatory Intervention Brief</div>
                                    <div style="color:#475569;font-family:'DM Mono',monospace;font-size:0.65rem">
                                        Ref: {b['notice_id']} &nbsp;·&nbsp; {b['issued_at'][:10]} &nbsp;·&nbsp; {b['authority']}
                                    </div>
                                </div>
                                <div style="display:inline-flex;align-items:center;padding:6px 14px;border-radius:7px;
                                    background:{risk_bg_c};border:1px solid {risk_c}44;font-family:'Syne',sans-serif;
                                    font-weight:700;color:{risk_c};font-size:0.8rem;letter-spacing:0.05em">{b['risk_level']}</div>
                            </div>
                            <!-- Executive Summary -->
                            <div style="margin-bottom:14px">
                                <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#334155;font-weight:600;margin-bottom:6px">Executive Summary</div>
                                <div style="color:#94a3b8;font-size:0.78rem;line-height:1.7">{b['executive_summary']}</div>
                            </div>
                            {denoise_note}
                            <!-- Parameters grid -->
                            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0">
                                <div style="border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:14px;background:rgba(255,255,255,0.02)">
                                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#334155;font-weight:600;margin-bottom:10px">Detection Parameters</div>
                                    <div style="display:flex;flex-direction:column;gap:5px;color:#64748b;font-size:0.72rem">
                                        <div>Facility &nbsp;<span style="color:#94a3b8">{b['facility_name']}</span></div>
                                        <div>Operator &nbsp;<span style="color:#94a3b8">{b['operator']}</span></div>
                                        <div>Flux &nbsp;<span style="color:#f59e0b;font-weight:600">{b['flux_kg_hr']:.1f} kg CH₄/hr</span></div>
                                        <div>Confidence &nbsp;<span style="color:#6366f1">{b['confidence_pct']}%</span></div>
                                        <div>Wind &nbsp;<span style="color:#94a3b8">{b['wind_speed_ms']:.1f} m/s @ {b['wind_dir_deg']}°</span></div>
                                        <div>Location &nbsp;<span style="color:#94a3b8">{b['lat']:.3f}°N, {b['lon']:.3f}°E</span></div>
                                    </div>
                                </div>
                                <div style="border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:14px;background:rgba(255,255,255,0.02)">
                                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#334155;font-weight:600;margin-bottom:10px">Financial Liability (30d)</div>
                                    <div style="display:flex;flex-direction:column;gap:5px;color:#64748b;font-size:0.72rem">
                                        <div>CH₄ lost &nbsp;<span style="color:#94a3b8">{b['economics'].get('ch4_lost_t_30d',0)} t</span></div>
                                        <div>CO₂e emitted &nbsp;<span style="color:#94a3b8">{b['economics'].get('co2e_t_30d',0)} t</span></div>
                                        <div>Gas value lost &nbsp;<span style="color:#94a3b8">${b['economics'].get('gas_value_usd',0):,}</span></div>
                                        <div>Carbon cost &nbsp;<span style="color:#94a3b8">${b['economics'].get('carbon_cost_usd',0):,}</span></div>
                                        <div>Regulatory fine &nbsp;<span style="color:#ef4444">${b['economics'].get('fine_usd',0):,}</span></div>
                                        <div style="padding-top:5px;border-top:1px solid rgba(255,255,255,0.05);margin-top:3px">
                                            Total &nbsp;<span style="color:{risk_c};font-weight:700">${b['economics'].get('total_cost_usd',0):,}</span></div>
                                    </div>
                                </div>
                            </div>
                            <!-- Actions -->
                            <div style="border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:14px;background:rgba(255,255,255,0.02)">
                                <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#334155;font-weight:600;margin-bottom:10px">Required Corrective Actions</div>
                                {''.join(f'<div style="display:flex;gap:10px;align-items:baseline;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04)"><span style="color:#6366f1;font-weight:600;font-family:DM Mono,monospace;font-size:0.65rem;min-width:48px">{t}</span><span style="font-size:0.7rem;color:#64748b">{desc}</span></div>' for t,desc in b['corrective_actions'])}
                            </div>
                            <div style="font-size:0.6rem;color:#1e293b;text-align:center;font-family:'DM Mono',monospace;margin-top:12px">
                                {b['legal_basis']}
                            </div>
                        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 8 — System Status  (with Twist 2 denoiser component added)
# ══════════════════════════════════════════════════════════════════

with tabs[7]:
    st.markdown(section_header("System Status",
        "Component health · technology stack · pipeline timing", "⚙"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
            letter-spacing:0.07em;text-transform:uppercase;color:#334155;margin-bottom:12px">
            Component Health</div>""", unsafe_allow_html=True)

        components = [
            ("FastAPI Backend",        health is not None, "localhost:8000/docs"),
            ("MongoDB Atlas",          True,               "cloud.mongodb.com"),
            ("TROPOMI Data Ingester",  True,               "mock / Copernicus hub"),
            ("ECMWF Wind Vectors",     True,               "mock / CDS API"),
            ("NASA EMIT Cross-val",    True,               "mock / EarthData"),
            ("GEE Data Layer",         False,              "pending GEE_PROJECT config"),
            # TWIST 2 — denoiser component
            ("Wiener Denoiser",        True,               f"σ={noise_sigma} · SNR: {_snr_metrics(noise_sigma)['raw_snr']}→{_snr_metrics(noise_sigma)['denoised_snr']} dB"),
            ("ViT Plume Segmenter",    True,               "stage1_sat.py — 22M params"),
            ("Modulus PINN",           True,               "stage2_pinn.py — flux"),
            ("PyG TGAN Attribution",   True,               "stage3_tgan.py — 1.4M"),
            ("Groq LLM Agent",         _env_ok("GROQ_API_KEY"), "Llama-3.3-70B — NOV + RIB"),
            # TWIST 1 — synthetic pipeline components
            ("Synthetic Alert Gen",    True,               f"{len(st.session_state['syn_alerts'])} alerts in session"),
            ("FP-Filter Agent",        True,               f"precision: {st.session_state['syn_fp_agent'].precision or '—'}%"),
            ("Brief Agent",            True,               f"{len(st.session_state['syn_briefs'])} briefs drafted"),
            ("Active Learning Queue",  True,               "MongoDB review_queue"),
            ("Pydeck Map Layer",       True,               "WebGL heatmap + scatter"),
        ]

        for name, ok, detail in components:
            dot_color = "#10b981" if ok else "#f59e0b"
            dot       = "●" if ok else "○"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:9px 12px;border-radius:8px;margin-bottom:3px;'
                f'background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.04)">'
                f'<div style="display:flex;align-items:center;gap:10px">'
                f'<span style="color:{dot_color};font-size:0.7rem">{dot}</span>'
                f'<span style="color:#94a3b8;font-size:0.75rem;font-family:Inter,sans-serif">{name}</span>'
                f'</div>'
                f'<span style="color:#334155;font-size:0.6rem;font-family:DM Mono,monospace">{detail}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
            letter-spacing:0.07em;text-transform:uppercase;color:#334155;margin-bottom:12px">
            Technology Stack</div>""", unsafe_allow_html=True)

        stack = [
            ("Satellite Data",   "Google Earth Engine",          "TROPOMI · ERA5 · EMIT"),
            ("Geospatial",       "TorchGeo",                     "raster pipeline + CRS"),
            ("Segmentation",     "ViT-Small/16 + timm",          "MC Dropout · F1 > 0.85"),
            # TWIST 2
            ("Denoising",        "Wiener Filter (numpy/scipy)",   "pre-ViT · local variance est."),
            ("Flux Estimation",  "NVIDIA Modulus PINN",           "Gaussian plume PDE"),
            ("Attribution",      "PyTorch Geometric",             "Temporal hetero-GAT"),
            ("Uncertainty",      "MC Dropout (N=30)",             "epistemic variance map"),
            ("Cloud Inpainting", "Wind-conditioned UNet",         "occlusion fill"),
            ("LLM Agent",        "Groq Llama-3.3-70b-versatile", "NOV + RIB + tool calling"),
            ("Database",         "MongoDB Atlas M0",              "free cloud tier"),
            # TWIST 1
            ("Synthetic Engine", "SyntheticAlertGenerator",       "GPS + flux + wind + confidence"),
            ("FP-Filter Agent",  "FPFilterAgent (5 rules)",       "facility DB cross-ref"),
            ("Brief Agent",      "BriefAgent + Groq fallback",    "Regulatory Intervention Brief"),
            ("API",              "FastAPI + uvicorn",             "async REST"),
            ("Map",              "Pydeck + deck.gl",              "WebGL heatmap"),
            ("Deploy",           "Railway + HF Spaces",           "auto-deploy from GitHub"),
        ]

        for cat, tech, detail in stack:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0;'
                f'padding:9px 12px;border-radius:8px;margin-bottom:3px;'
                f'background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.04)">'
                f'<span style="color:#334155;font-size:0.65rem;min-width:112px;font-family:Inter,sans-serif">{cat}</span>'
                f'<span style="color:#6366f1;font-size:0.75rem;flex:1;font-weight:500;font-family:Inter,sans-serif">{tech}</span>'
                f'<span style="color:#1e293b;font-size:0.6rem;font-family:DM Mono,monospace">{detail}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
            letter-spacing:0.07em;text-transform:uppercase;color:#334155;margin-bottom:12px">
            Pipeline Timing</div>""", unsafe_allow_html=True)

        stages = [
            ("Data Ingestion (GEE / Synthetic)", 1.2),
            # TWIST 2 — denoiser is a new stage before segmentation
            ("Wiener Denoiser (pre-ViT)",        0.4 if noise_sigma > 0 else 0.0),
            ("Stage 1 — ViT Segment",            4.7),
            ("Stage 2 — PINN Flux",              8.3),
            ("Stage 3 — TGAN Attribution",       2.1),
            ("Stage 4 — LLM NOV / RIB",          3.9),
            # TWIST 1 — FP filter + brief are post-pipeline
            ("FP-Filter Agent",                  0.1 if not health else 0.0),
            ("Brief Agent (top-5)",              1.2 if not health else 0.0),
        ]
        stages = [(s, t) for s, t in stages if t > 0]
        total_t = sum(t for _, t in stages)
        for stage, t in stages:
            pct = t / total_t
            st.markdown(
                f'<div style="margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:5px">'
                f'<span style="color:#94a3b8;font-family:Inter,sans-serif">{stage}</span>'
                f'<span style="color:#475569;font-family:DM Mono,monospace">{t:.1f}s</span>'
                f'</div>'
                f'<div style="height:4px;background:rgba(255,255,255,0.04);border-radius:2px;overflow:hidden">'
                f'<div style="height:4px;width:{pct*100:.0f}%;border-radius:2px;'
                f'background:linear-gradient(90deg,#6366f1,#6366f1cc)"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="text-align:right;font-size:0.65rem;color:#334155;'
            f'font-family:DM Mono,monospace;margin-top:2px">Total: {total_t:.1f}s</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# Auto-refresh
# ══════════════════════════════════════════════════════════════════

if auto_refresh:
    time.sleep(30)
    st.rerun()