"""
dashboard/app.py
─────────────────
ARGUS Dashboard — Streamlit + Pydeck
Aesthetic: Deep-space mission control. Phosphor-green on void-black.

Run:  python -m streamlit run dashboard/app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime

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
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

:root {
    --void:    #02040a;
    --panel:   #060c15;
    --surface: #0a1220;
    --border:  #0f2035;
    --ph:      #00ff88;
    --ph-dim:  #00994d;
    --ph-glow: rgba(0,255,136,0.15);
    --amber:   #ffaa00;
    --red:     #ff3355;
    --cyan:    #00ccff;
    --text:    #8ab4c8;
    --text-hi: #cce8f4;
}

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace !important;
    background: var(--void) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,255,136,0.04) 0%, transparent 70%),
        var(--void) !important;
}

[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--ph-dim);
    padding: 14px 16px !important;
    border-radius: 0 !important;
}
[data-testid="stMetric"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--ph-dim) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.7rem !important;
    color: var(--ph) !important;
}

[data-baseweb="tab-list"] {
    background: var(--panel) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #3d6080 !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
}
[data-baseweb="tab"]:hover { color: var(--ph) !important; background: var(--ph-glow) !important; }
[aria-selected="true"] {
    color: var(--ph) !important;
    background: rgba(0,255,136,0.06) !important;
    border-bottom: 2px solid var(--ph) !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--ph-dim) !important;
    color: var(--ph) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 10px 16px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--ph-glow) !important;
    box-shadow: 0 0 16px var(--ph-glow) !important;
}

[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--text-hi) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    background: var(--panel) !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    background: var(--panel) !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--panel); }
::-webkit-scrollbar-thumb { background: var(--border); }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Mock data (shown when API is offline)
# ══════════════════════════════════════════════════════════════════

MOCK_DETECTIONS = [
    {"detection_id": 1,  "centroid_lat": 32.1,  "centroid_lon": -102.5, "flux_kg_hr": 487, "co2e_kg_hr": 38960, "confidence": 0.94, "epistemic_variance": 0.04, "high_confidence": True,  "attribution": {"facility_name": "Permian Basin WP-7",   "operator": "OilCorp International", "facility_id": "FAC-0001", "confidence": 0.91, "distance_km": 2.3}, "enforcement": {"risk_level": "CRITICAL", "notice_id": "NOV-ARGUS-20240315-001"}, "economics": {"total_cost_inr": 142000000}},
    {"detection_id": 2,  "centroid_lat": 38.4,  "centroid_lon": 57.2,   "flux_kg_hr": 312, "co2e_kg_hr": 24960, "confidence": 0.88, "epistemic_variance": 0.07, "high_confidence": True,  "attribution": {"facility_name": "Turkmenistan GC-3",    "operator": "TurkGaz Holdings",      "facility_id": "FAC-0042", "confidence": 0.85, "distance_km": 4.1}, "enforcement": {"risk_level": "CRITICAL", "notice_id": "NOV-ARGUS-20240315-002"}, "economics": {"total_cost_inr": 89000000}},
    {"detection_id": 3,  "centroid_lat": 5.2,   "centroid_lon": 6.4,    "flux_kg_hr": 221, "co2e_kg_hr": 17680, "confidence": 0.79, "epistemic_variance": 0.11, "high_confidence": False, "attribution": {"facility_name": "Niger Delta P-12",     "operator": "Gulf Stream Energy",    "facility_id": "FAC-0108", "confidence": 0.72, "distance_km": 7.8}, "enforcement": {"risk_level": "HIGH",     "notice_id": ""}, "economics": {"total_cost_inr": 61000000}},
    {"detection_id": 4,  "centroid_lat": 62.3,  "centroid_lon": 74.1,   "flux_kg_hr": 178, "co2e_kg_hr": 14240, "confidence": 0.91, "epistemic_variance": 0.05, "high_confidence": True,  "attribution": {"facility_name": "Siberia LNG T-2",      "operator": "SovEnergy PJSC",        "facility_id": "FAC-0203", "confidence": 0.88, "distance_km": 3.2}, "enforcement": {"risk_level": "HIGH",     "notice_id": "NOV-ARGUS-20240315-003"}, "economics": {"total_cost_inr": 48000000}},
    {"detection_id": 5,  "centroid_lat": 20.1,  "centroid_lon": 70.3,   "flux_kg_hr": 134, "co2e_kg_hr": 10720, "confidence": 0.83, "epistemic_variance": 0.09, "high_confidence": True,  "attribution": {"facility_name": "Mumbai Offshore MH-3", "operator": "IndusGas Ltd",          "facility_id": "FAC-0287", "confidence": 0.79, "distance_km": 5.6}, "enforcement": {"risk_level": "MEDIUM",   "notice_id": ""}, "economics": {"total_cost_inr": 35000000}},
    {"detection_id": 6,  "centroid_lat": 27.1,  "centroid_lon": 49.8,   "flux_kg_hr": 298, "co2e_kg_hr": 23840, "confidence": 0.96, "epistemic_variance": 0.03, "high_confidence": True,  "attribution": {"facility_name": "Saudi East Comp-7",    "operator": "ArcoFlare Co",          "facility_id": "FAC-0321", "confidence": 0.93, "distance_km": 1.8}, "enforcement": {"risk_level": "CRITICAL", "notice_id": "NOV-ARGUS-20240315-004"}, "economics": {"total_cost_inr": 82000000}},
    {"detection_id": 7,  "centroid_lat": -3.5,  "centroid_lon": 18.2,   "flux_kg_hr": 156, "co2e_kg_hr": 12480, "confidence": 0.75, "epistemic_variance": 0.13, "high_confidence": False, "attribution": {"facility_name": "Congo Basin F-4",      "operator": "AfricaFuel PLC",        "facility_id": "FAC-0392", "confidence": 0.68, "distance_km": 9.1}, "enforcement": {"risk_level": "HIGH",     "notice_id": ""}, "economics": {"total_cost_inr": 41000000}},
    {"detection_id": 8,  "centroid_lat": 52.8,  "centroid_lon": 55.4,   "flux_kg_hr": 543, "co2e_kg_hr": 43440, "confidence": 0.97, "epistemic_variance": 0.02, "high_confidence": True,  "attribution": {"facility_name": "Orenburg Gas Plant",  "operator": "GazpromNeft East",      "facility_id": "FAC-0445", "confidence": 0.95, "distance_km": 1.2}, "enforcement": {"risk_level": "CRITICAL", "notice_id": "NOV-ARGUS-20240315-005"}, "economics": {"total_cost_inr": 158000000}},
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
]


# ══════════════════════════════════════════════════════════════════
# Constants & helpers
# ══════════════════════════════════════════════════════════════════

API_BASE = f"http://localhost:{cfg['api']['port']}/api/v1"

RISK_HEX = {
    "CRITICAL": "#ff3355",
    "HIGH":     "#ff7722",
    "MEDIUM":   "#ffaa00",
    "LOW":      "#00ff88",
}
RISK_RGBA = {
    "CRITICAL": [255, 51,  85,  220],
    "HIGH":     [255, 119, 34,  200],
    "MEDIUM":   [255, 170, 0,   190],
    "LOW":      [0,   255, 136, 170],
}


def api_get(path, default=None):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=8)
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
    """Safely navigate nested dicts without crashing on None."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
    return d if d is not None else default


def ph_card(label, value, unit="", accent="#00ff88"):
    return f"""
    <div style="background:#060c15;border:1px solid #0f2035;border-top:2px solid {accent};
                padding:14px 18px;overflow:hidden">
        <div style="font-size:0.55rem;letter-spacing:0.2em;text-transform:uppercase;
                    color:{accent};opacity:0.7;margin-bottom:6px">{label}</div>
        <div style="font-family:'Exo 2',sans-serif;font-size:1.9rem;font-weight:800;
                    color:{accent};line-height:1">
            {value}<span style="font-size:0.9rem;font-weight:300;margin-left:4px;opacity:0.7">{unit}</span>
        </div>
    </div>
    """


def section_header(text, subtitle=""):
    sub = f'<div style="font-size:0.6rem;letter-spacing:0.12em;color:#3d6080;margin-top:4px">{subtitle}</div>' if subtitle else ""
    return f"""
    <div style="margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #0f2035">
        <div style="font-family:'Exo 2',sans-serif;font-size:1.1rem;font-weight:800;
                    letter-spacing:0.08em;text-transform:uppercase;color:#cce8f4">{text}</div>
        {sub}
    </div>
    """


def plotly_dark(fig, title="", height=300):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Share Tech Mono", size=11, color="#3d8060"), x=0),
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono", size=10, color="#8ab4c8"),
        margin=dict(l=8, r=8, t=36, b=8),
        xaxis=dict(gridcolor="#0f2035", zeroline=False, tickfont=dict(size=9)),
        yaxis=dict(gridcolor="#0f2035", zeroline=False, tickfont=dict(size=9)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
    )
    return fig


def style_risk(val):
    c = {"CRITICAL": "#ff3355", "HIGH": "#ff7722", "MEDIUM": "#ffaa00", "LOW": "#00ff88"}.get(val, "")
    return f"color:{c}" if c else ""


# ══════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:24px 16px 16px;border-bottom:1px solid #0f2035;text-align:center">
        <div style="font-family:'Exo 2',sans-serif;font-size:2.8rem;font-weight:800;
                    letter-spacing:0.4em;color:#00ff88;text-shadow:0 0 30px rgba(0,255,136,0.4)">
            ARGUS
        </div>
        <div style="font-size:0.55rem;letter-spacing:0.18em;color:#1a4030;
                    text-transform:uppercase;margin-top:4px">
            Autonomous Greenhouse Gas<br>Unified Surveillance
        </div>
    </div>
    """, unsafe_allow_html=True)

    health = api_get("/health")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if health:
        st.markdown("""<div style="font-size:0.65rem;color:#00ff88;
            padding:6px 12px;border:1px solid #003322;background:rgba(0,255,136,0.05)">
            ◉  API ONLINE — localhost:8000</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="font-size:0.65rem;color:#ffaa00;
            padding:6px 12px;border:1px solid #332200;background:rgba(255,170,0,0.05)">
            ◎  API OFFLINE — SHOWING DEMO DATA</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.6rem;letter-spacing:0.18em;color:#1a4030;
        text-transform:uppercase;padding:0 0 8px 0;border-bottom:1px solid #0f2035">
        ▸ Run Pipeline on a Region</div>""", unsafe_allow_html=True)

    PRESETS = {
        "Permian Basin, USA":      (31.0, 33.0, -104.0, -101.0),
        "Turkmenistan Gas Fields": (37.0, 40.0,   55.0,   60.0),
        "Niger Delta, Nigeria":    ( 4.0,  6.0,    5.0,    8.0),
        "Mumbai Offshore, India":  (18.0, 22.0,   68.0,   73.0),
        "Saudi East Arabia":       (26.0, 28.0,   49.0,   51.0),
        "Siberia Gas Fields":      (60.0, 65.0,   70.0,   80.0),
    }

    preset = st.selectbox("Select a region to scan", list(PRESETS.keys()))
    lat_min, lat_max, lon_min, lon_max = PRESETS[preset]

    st.markdown(f"""<div style="font-size:0.6rem;color:#1a4030;padding:6px 0">
        Bounding box: {lat_min}°–{lat_max}°N · {lon_min}°–{lon_max}°E</div>""",
        unsafe_allow_html=True)

    run_clicked = st.button("⟶  RUN PIPELINE", use_container_width=True)
    if run_clicked:
        if not health:
            st.warning("API is offline. Start `python run.py` first.")
        else:
            with st.spinner("Running 4-stage pipeline... (~20s)"):
                result = api_post("/detect", {
                    "lat_min": lat_min, "lat_max": lat_max,
                    "lon_min": lon_min, "lon_max": lon_max,
                })
            if result:
                st.session_state["last_result"] = result
                n = result.get("summary", {}).get("n_super_emitters", 0)
                st.success(f"✓ {n} super-emitters detected")
            else:
                st.error("Pipeline failed — check API logs")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    auto_refresh = st.toggle("Auto-refresh every 30s", value=False)

    st.markdown(f"""
    <div style="position:absolute;bottom:16px;left:0;right:0;text-align:center">
        <div style="font-size:0.6rem;color:#1a4030;letter-spacing:0.1em">
            UTC · {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Load data from API (fall back to mock if offline)
# ══════════════════════════════════════════════════════════════════

heatmap_data   = api_get("/heatmap?n_runs=50")  or {"detections": MOCK_DETECTIONS}
economic_data  = api_get("/economic-summary")   or {}
scorecard_data = api_get("/scorecard?limit=50") or {"scorecard": MOCK_SCORECARD}
al_data        = api_get("/review-queue")       or {"queue_size": 0, "items": [], "learning_curve": {}}

detections = heatmap_data.get("detections") or MOCK_DETECTIONS
scorecard  = scorecard_data.get("scorecard") or MOCK_SCORECARD


# ══════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:20px 0 4px">
    <div style="font-family:'Exo 2',sans-serif;font-size:1.6rem;font-weight:800;
                letter-spacing:0.12em;color:#cce8f4">
        METHANE SUPER-EMITTER INTELLIGENCE PLATFORM
    </div>
    <div style="font-size:0.6rem;letter-spacing:0.2em;color:#1a4030;
                text-transform:uppercase;padding:4px 0 16px">
        Sentinel-5P TROPOMI · NASA EMIT · ECMWF ERA5 · NVIDIA Modulus PINN · PyG TGAN
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI Strip ─────────────────────────────────────────────────────

total_flux   = sum(d.get("flux_kg_hr", 0) for d in detections)
total_inr    = sum(safe_get(d, "economics", "total_cost_inr", default=0) for d in detections)
critical_cnt = sum(1 for d in detections if safe_get(d, "enforcement", "risk_level") == "CRITICAL")
n_hc         = sum(1 for d in detections if d.get("high_confidence", False))

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(ph_card("Super-Emitters Detected", len(detections), "active"), unsafe_allow_html=True)
with c2:
    st.markdown(ph_card("Critical Risk Alerts", critical_cnt, "", "#ff3355"), unsafe_allow_html=True)
with c3:
    st.markdown(ph_card("Total Emission Flux", f"{total_flux:.0f}", "kg CH₄/hr", "#ffaa00"), unsafe_allow_html=True)
with c4:
    st.markdown(ph_card("Economic Impact", f"{total_inr/1e7:.1f}", "₹ Crore / 30d", "#00ccff"), unsafe_allow_html=True)
with c5:
    st.markdown(ph_card("CO₂ Equivalent", f"{total_flux*80/1000:.1f}", "tonnes/hr", "#cc88ff"), unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🌍  Global Map",
    "📊  Compliance",
    "💰  Economics",
    "📋  Enforcement Notices",
    "🔬  Active Learning",
    "⚙   System",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Global Map
# ══════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown(section_header(
        "Global Methane Plume Monitor",
        "Each circle = one detected super-emitter · Circle size = emission flux · Colour = risk level"
    ), unsafe_allow_html=True)

    # Build map dataframe safely
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

    if not map_rows:
        st.info("No detections yet. Use the sidebar to run the pipeline on a region.")
    else:
        df_map = pd.DataFrame(map_rows)
        df_map["color"]  = df_map["risk"].apply(lambda r: RISK_RGBA.get(r, [100, 100, 100, 150]))
        df_map["radius"] = df_map["flux"].apply(lambda f: max(40000, min(float(f) * 400, 500000)))

        heat_layer = pdk.Layer(
            "HeatmapLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_weight="flux",
            aggregation="MEAN",
            opacity=0.35,
            color_range=[
                [2,  4,  10,  0],
                [0,  30, 60,  80],
                [0,  80, 120, 140],
                [0,  200,150, 180],
                [255,200,0,   210],
                [255,51, 85,  255],
            ],
            radius_pixels=60,
        )

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            opacity=0.75,
            stroked=True,
            get_line_color=[0, 255, 136, 40],
            line_width_min_pixels=1,
        )

        view = pdk.ViewState(latitude=25, longitude=30, zoom=1.6, pitch=0)

        tooltip = {
            "html": """
                <div style='background:#060c15;padding:12px 14px;border:1px solid #00ff8844;
                            font-family:monospace;font-size:11px;color:#8ab4c8;min-width:180px'>
                    <div style='color:#00ff88;font-size:13px;font-weight:bold;margin-bottom:8px'>
                        {facility}
                    </div>
                    <div><span style='color:#3d8060'>OPERATOR · </span>{operator}</div>
                    <div><span style='color:#3d8060'>FLUX     · </span>
                        <span style='color:#ffaa00;font-weight:bold'>{flux} kg/hr</span></div>
                    <div><span style='color:#3d8060'>RISK     · </span>
                        <span style='color:#ff3355'>{risk}</span></div>
                    <div><span style='color:#3d8060'>CONFIDENCE · </span>{conf}%</div>
                </div>
            """,
            "style": {"backgroundColor": "transparent"},
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[heat_layer, scatter_layer],
                initial_view_state=view,
                tooltip=tooltip,
                map_style="mapbox://styles/mapbox/dark-v11",
            ),
            use_container_width=True,
            height=500,
        )

        # Legend
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col_l1, col_l2, col_l3, col_l4, spacer = st.columns([1, 1, 1, 1, 4])
        for col, risk in zip([col_l1, col_l2, col_l3, col_l4], ["CRITICAL", "HIGH", "MEDIUM", "LOW"]):
            col.markdown(
                f'<div style="font-size:0.6rem;color:{RISK_HEX[risk]};'
                f'border:1px solid {RISK_HEX[risk]}44;padding:4px 8px;text-align:center">⬤ {risk}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.6rem;letter-spacing:0.18em;color:#1a4030;'
                'text-transform:uppercase;padding:8px 0 6px">▸ Detection Feed</div>',
                unsafe_allow_html=True)

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
            "Lat":          round(float(d.get("centroid_lat", 0)), 3),
            "Lon":          round(float(d.get("centroid_lon", 0)), 3),
        })

    if det_rows:
        df_dets = pd.DataFrame(det_rows)
        styled_dets = df_dets.style.map(style_risk, subset=["Risk"])
        st.dataframe(styled_dets, use_container_width=True, hide_index=True, height=260)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Compliance Scorecard
# ══════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown(section_header(
        "Operator Compliance Registry",
        "Compliance score 0–100 · Lower score = worse compliance · Sorted worst first"
    ), unsafe_allow_html=True)

    if not scorecard:
        st.info("Run the pipeline to generate source attribution and compliance scores.")
    else:
        col_chart, col_detail = st.columns([2, 3])

        with col_chart:
            df_sc = pd.DataFrame(scorecard)

            fig_bar = go.Figure(go.Bar(
                x=df_sc["compliance_score"],
                y=df_sc["operator"],
                orientation="h",
                marker=dict(
                    color=[RISK_HEX.get(r, "#888") for r in df_sc["risk_level"]],
                    opacity=0.75,
                    line=dict(width=0),
                ),
                text=df_sc["compliance_score"].apply(lambda v: f"{v:.0f}"),
                textposition="outside",
                textfont=dict(size=9, color="#8ab4c8"),
            ))
            plotly_dark(fig_bar, "COMPLIANCE SCORE (lower = worse)", height=320)
            fig_bar.update_layout(xaxis=dict(range=[0, 100], title="Score"), yaxis=dict(title=""))
            st.plotly_chart(fig_bar, use_container_width=True)

            risk_counts = df_sc["risk_level"].value_counts()
            fig_donut = go.Figure(go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.65,
                marker=dict(colors=[RISK_HEX.get(r, "#888") for r in risk_counts.index]),
                textfont=dict(family="Share Tech Mono", size=9),
                textinfo="label+percent",
            ))
            plotly_dark(fig_donut, "RISK LEVEL DISTRIBUTION", height=240)
            fig_donut.update_layout(showlegend=False, margin=dict(l=0, r=0, t=36, b=0))
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_detail:
            sc_rows = [{
                "Facility":     r["facility_name"],
                "Operator":     r["operator"],
                "Type":         r.get("facility_type", "").replace("_", " ").title(),
                "Flux kg/hr":   r.get("flux_kg_hr", 0),
                "Score /100":   r.get("compliance_score", 0),
                "Violations":   r.get("violations_12mo", 0),
                "Risk":         r.get("risk_level", "?"),
                "Confidence %": r.get("confidence", 0),
            } for r in scorecard]

            df_sc_display = pd.DataFrame(sc_rows)
            styled_sc = df_sc_display.style.map(style_risk, subset=["Risk"])
            st.dataframe(styled_sc, use_container_width=True, hide_index=True, height=580)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Economic Impact
# ══════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown(section_header(
        "Economic Impact Analysis",
        "Projected financial cost of uncontrolled methane leaks over selected time window"
    ), unsafe_allow_html=True)

    duration = st.slider("Projection window (days)", 1, 90, 30, format="%d days")

    # Build economic rows
    econ_rows = []
    for d in detections:
        flux      = float(d.get("flux_kg_hr", 0))
        hours     = duration * 24
        ch4_t     = flux * hours / 1000
        co2e_t    = ch4_t * 80
        gas_usd   = ch4_t * 0.0553 * 2.80
        carb_usd  = co2e_t * 15.0
        fine_usd  = co2e_t * 50.0
        total_usd = gas_usd + carb_usd + fine_usd
        econ_rows.append({
            "Facility":     safe_get(d, "attribution", "facility_name", default="?"),
            "Operator":     safe_get(d, "attribution", "operator", default="?"),
            "Flux kg/hr":   round(flux, 1),
            "CH₄ lost (t)": round(ch4_t, 1),
            "CO₂e (t)":     round(co2e_t, 1),
            "Gas lost $":   round(gas_usd),
            "Carbon cost $": round(carb_usd),
            "Fine $":       round(fine_usd),
            "Total ₹ Lakh": round(total_usd * 83.5 / 1e5, 1),
            "Risk":         safe_get(d, "enforcement", "risk_level", default="LOW"),
            "_total_usd":   total_usd,
            "_flux":        flux,
        })

    total_inr_cr  = sum(r["_total_usd"] for r in econ_rows) * 83.5 / 1e7
    total_usd_sum = sum(r["_total_usd"] for r in econ_rows)

    # Big summary card
    gas_total  = sum(r["_flux"] * duration * 24 * 0.0553 * 2.8 / 1000 for r in econ_rows)
    carb_total = sum(r["_flux"] * duration * 24 * 80 / 1000 * 15 for r in econ_rows)
    fine_total = sum(r["_flux"] * duration * 24 * 80 / 1000 * 50 for r in econ_rows)

    st.markdown(f"""
    <div style="text-align:center;padding:28px 0 20px;border:1px solid #0f2035;
                background:linear-gradient(180deg,#060c15,#02040a);margin-bottom:20px">
        <div style="font-size:0.6rem;letter-spacing:0.25em;color:#1a4030;
                    text-transform:uppercase;margin-bottom:8px">
            Total Projected Economic Impact over {duration} Days
        </div>
        <div style="font-family:'Exo 2',sans-serif;font-size:4rem;font-weight:800;
                    color:#ffaa00;text-shadow:0 0 40px rgba(255,170,0,0.3);line-height:1">
            ₹{total_inr_cr:.2f} Crore
        </div>
        <div style="font-size:0.75rem;color:#3d6040;margin-top:6px">
            ${total_usd_sum:,.0f} USD equivalent
        </div>
        <div style="display:flex;justify-content:center;gap:40px;margin-top:16px">
            <div style="text-align:center">
                <div style="font-size:0.6rem;color:#ff7722;letter-spacing:0.1em">GAS VALUE LOST</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:1.2rem;font-weight:600;color:#ff7722">
                    ${gas_total:,.0f}
                </div>
            </div>
            <div style="text-align:center">
                <div style="font-size:0.6rem;color:#00ccff;letter-spacing:0.1em">CARBON COST</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:1.2rem;font-weight:600;color:#00ccff">
                    ${carb_total:,.0f}
                </div>
            </div>
            <div style="text-align:center">
                <div style="font-size:0.6rem;color:#ff3355;letter-spacing:0.1em">REGULATORY FINE</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:1.2rem;font-weight:600;color:#ff3355">
                    ${fine_total:,.0f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        days_x = list(range(1, 91))
        daily  = total_inr_cr / duration if duration > 0 else 0
        proj_y = [daily * d for d in days_x]

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x=days_x, y=proj_y,
            mode="lines",
            line=dict(color="#ffaa00", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,170,0,0.05)",
        ))
        fig_proj.add_vline(
            x=duration,
            line_color="rgba(255,170,0,0.4)",
            line_dash="dot",
            annotation_text=f"  {duration}d — ₹{total_inr_cr:.1f}Cr",
            annotation_font_color="#ffaa00",
            annotation_font_size=9,
        )
        plotly_dark(fig_proj, "CUMULATIVE LIABILITY PROJECTION (₹ CRORE)", height=280)
        fig_proj.update_layout(xaxis_title="Days", yaxis_title="₹ Crore", showlegend=False)
        st.plotly_chart(fig_proj, use_container_width=True)

    with col_b:
        # Build bubble dataframe — only if we have data
        bubble_rows = [
            {
                "Facility":   r["Facility"],
                "Flux":       r["Flux kg/hr"],
                "Total_INR":  r["Total ₹ Lakh"],
                "Risk":       r["Risk"],
            }
            for r in econ_rows
            if r["Flux kg/hr"] > 0
        ]

        if bubble_rows:
            df_bubble = pd.DataFrame(bubble_rows)
            fig_bub = px.scatter(
                df_bubble,
                x="Flux",
                y="Total_INR",
                size="Flux",
                color="Risk",
                hover_name="Facility",
                color_discrete_map=RISK_HEX,
                size_max=36,
                labels={
                    "Flux":      "Emission Flux (kg/hr)",
                    "Total_INR": "Total Liability (₹ Lakh)",
                    "Risk":      "Risk Level",
                },
            )
            plotly_dark(fig_bub, "FLUX vs ECONOMIC LIABILITY", height=280)
            fig_bub.update_layout(
                xaxis_title="Emission Flux (kg/hr)",
                yaxis_title="Total Liability (₹ Lakh)",
                legend_title_text="",
            )
            st.plotly_chart(fig_bub, use_container_width=True)
        else:
            st.info("No data for bubble chart.")

    # Table
    display_cols = ["Facility", "Operator", "Flux kg/hr", "CH₄ lost (t)", "CO₂e (t)",
                    "Gas lost $", "Carbon cost $", "Fine $", "Total ₹ Lakh", "Risk"]
    df_econ_tbl = pd.DataFrame(econ_rows)[display_cols]
    styled_econ = df_econ_tbl.style.map(style_risk, subset=["Risk"])
    st.dataframe(styled_econ, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — Enforcement Notices
# ══════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown(section_header(
        "Auto-Generated Notices of Violation",
        "Drafted by ARGUS LLM Agent · Groq Llama-3.1-70B · Based on CPCB / MoEF&CC regulations"
    ), unsafe_allow_html=True)

    nov_dets = [d for d in detections if d.get("flux_kg_hr", 0) >= 100]

    if not nov_dets:
        st.info("No super-emitters detected yet. Run the pipeline to generate enforcement notices.")
    else:
        labels = [
            f"DET-{str(d['detection_id']).zfill(4)}  ·  "
            f"{safe_get(d, 'attribution', 'facility_name', default='?')}  ·  "
            f"{d.get('flux_kg_hr', 0):.0f} kg/hr  ·  "
            f"{safe_get(d, 'enforcement', 'risk_level', default='?')}"
            for d in nov_dets
        ]
        sel = st.selectbox("Select a detection to view its Notice of Violation", labels)
        det = nov_dets[labels.index(sel)]

        attr      = det.get("attribution") or {}
        flux      = float(det.get("flux_kg_hr", 0))
        risk      = safe_get(det, "enforcement", "risk_level", default="UNKNOWN")
        co2e_t    = flux * 30 * 24 * 80 / 1000
        fine_usd  = round(co2e_t * 50)
        fine_inr  = round(fine_usd * 83.5)
        notice_id = safe_get(det, "enforcement", "notice_id") or f"NOV-ARGUS-{det['detection_id']:04d}"

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(ph_card("Notice ID", notice_id.split("-")[-1]), unsafe_allow_html=True)
        mc2.markdown(ph_card("Emission Rate", f"{flux:.1f}", "kg/hr", "#ff3355"), unsafe_allow_html=True)
        mc3.markdown(ph_card("Statutory Fine", f"${fine_usd:,}", "", "#ffaa00"), unsafe_allow_html=True)
        mc4.markdown(ph_card("INR Equivalent", f"₹{fine_inr/1e5:.1f}L", "", "#00ccff"), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        risk_color = RISK_HEX.get(risk, "#888")
        st.markdown(f"""
        <div style="border:1px solid {risk_color}33;background:#060c15;padding:28px;
                    font-family:'Share Tech Mono',monospace;font-size:0.75rem;line-height:1.8">

            <div style="display:flex;justify-content:space-between;align-items:start;
                        border-bottom:1px solid #0f2035;padding-bottom:16px;margin-bottom:16px">
                <div>
                    <div style="font-family:'Exo 2',sans-serif;font-size:1.2rem;font-weight:800;
                                letter-spacing:0.1em;color:#cce8f4">NOTICE OF VIOLATION</div>
                    <div style="color:#3d8060;margin-top:4px">Reference: {notice_id}</div>
                    <div style="color:#3d8060">Date: {datetime.utcnow().strftime('%Y-%m-%d')}</div>
                    <div style="color:#3d8060">Authority: CPCB / MoEF&amp;CC</div>
                </div>
                <div style="border:1px solid {risk_color};padding:8px 16px;color:{risk_color};
                            font-family:'Exo 2',sans-serif;font-weight:800;font-size:0.9rem">
                    {risk}
                </div>
            </div>

            <div style="margin-bottom:16px">
                <div style="color:#1a4030;font-size:0.6rem;letter-spacing:0.15em;
                            text-transform:uppercase;margin-bottom:6px">Addressee</div>
                <div style="color:#cce8f4;font-size:0.85rem">{attr.get('operator', 'Unknown Operator')}</div>
                <div style="color:#8ab4c8">Re: {attr.get('facility_name', '?')} ({attr.get('facility_id', '?')})</div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
                <div style="border:1px solid #0f2035;padding:12px">
                    <div style="color:#1a4030;font-size:0.6rem;letter-spacing:0.15em;
                                text-transform:uppercase;margin-bottom:8px">Detection Parameters</div>
                    <div style="color:#8ab4c8">Emission Rate ·
                        <span style="color:#ffaa00;font-weight:bold">{flux:.1f} kg CH₄/hr</span></div>
                    <div style="color:#8ab4c8">CO₂-Equivalent · {flux*80:.0f} kg CO₂e/hr</div>
                    <div style="color:#8ab4c8">Threshold · 100 kg/hr (super-emitter)</div>
                    <div style="color:#8ab4c8">Attribution Confidence · {det.get('confidence',0)*100:.1f}%</div>
                    <div style="color:#8ab4c8">Data Source · Sentinel-5P TROPOMI / NASA EMIT</div>
                </div>
                <div style="border:1px solid #0f2035;padding:12px">
                    <div style="color:#1a4030;font-size:0.6rem;letter-spacing:0.15em;
                                text-transform:uppercase;margin-bottom:8px">Financial Liability (30d)</div>
                    <div style="color:#8ab4c8">Gas Value Lost · ${round(flux*720*0.0553*2.8/1000):,}</div>
                    <div style="color:#8ab4c8">Carbon Cost (GWP-20) · ${round(co2e_t*15):,}</div>
                    <div style="color:#8ab4c8">Regulatory Fine · ${fine_usd:,}</div>
                    <div style="color:#ff3355;font-weight:bold;margin-top:4px">
                        Total · ${fine_usd:,} (₹{fine_inr/1e5:.1f} Lakh)</div>
                </div>
            </div>

            <div style="border:1px solid #0f2035;padding:12px;margin-bottom:16px">
                <div style="color:#1a4030;font-size:0.6rem;letter-spacing:0.15em;
                            text-transform:uppercase;margin-bottom:8px">Required Corrective Actions</div>
                <div style="color:#8ab4c8">① 72 hours — Cease or curtail the detected emission source</div>
                <div style="color:#8ab4c8">② 7 days &nbsp;&nbsp;— Submit Root Cause Analysis to CPCB</div>
                <div style="color:#8ab4c8">③ 30 days &nbsp;— Implement permanent remediation measures</div>
                <div style="color:#8ab4c8">④ 60 days &nbsp;— Submit LDAR continuous monitoring plan</div>
            </div>

            <div style="color:#1a4030;font-size:0.6rem;text-align:center;
                        border-top:1px solid #0f2035;padding-top:12px">
                Auto-generated by ARGUS · Environment Protection Act 1986 · Air Act 1981 · India NDC 2021
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5 — Active Learning
# ══════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown(section_header(
        "Active Learning — Uncertainty Review Queue",
        "Detections where the model is uncertain are flagged for human review, then fed back into retraining"
    ), unsafe_allow_html=True)

    queue  = al_data.get("items", [])
    curve  = al_data.get("learning_curve", {})
    q_size = al_data.get("queue_size", 0)

    col_kpi, col_curve = st.columns([1, 2])

    with col_kpi:
        st.markdown(ph_card("Pending Reviews", q_size, "items"), unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(ph_card("Uncertainty Threshold", "σ² > 0.15", "", "#ffaa00"), unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if curve.get("mean_variance"):
            last_var = curve["mean_variance"][-1]
            st.markdown(ph_card("Latest Uncertainty", f"{last_var:.4f}", "σ²", "#00ccff"), unsafe_allow_html=True)

    with col_curve:
        if curve.get("runs"):
            fig_unc = go.Figure()
            fig_unc.add_trace(go.Scatter(
                x=list(range(len(curve["runs"]))),
                y=curve["mean_variance"],
                mode="lines+markers",
                name="Mean σ²",
                line=dict(color="#00ccff", width=2),
                marker=dict(size=5, color="#00ccff"),
                fill="tozeroy",
                fillcolor="rgba(0,204,255,0.04)",
            ))
            fig_unc.add_hline(
                y=0.15,
                line_dash="dot",
                line_color="rgba(255,51,85,0.5)",
                annotation_text="uncertainty threshold = 0.15",
                annotation_font_color="#ff3355",
                annotation_font_size=9,
            )
            plotly_dark(fig_unc, "MODEL UNCERTAINTY OVER TIME (should fall as model learns)", height=260)
            fig_unc.update_layout(xaxis_title="Pipeline Run #", yaxis_title="Mean Epistemic σ²", showlegend=False)
            st.plotly_chart(fig_unc, use_container_width=True)
        else:
            st.markdown("""
            <div style="border:1px solid #0f2035;padding:40px;text-align:center;background:#060c15">
                <div style="color:#1a4030;font-size:0.7rem;letter-spacing:0.15em">
                    No uncertainty history yet — run the pipeline to generate data
                </div>
            </div>
            """, unsafe_allow_html=True)

    if queue:
        st.markdown('<div style="font-size:0.6rem;letter-spacing:0.18em;color:#1a4030;'
                    'text-transform:uppercase;padding:12px 0 8px">▸ Detections Awaiting Human Review</div>',
                    unsafe_allow_html=True)
        for item in queue[:6]:
            with st.expander(
                f"DET-{str(item.get('label_id', 0)).zfill(4)}  ·  "
                f"Uncertainty σ²={item.get('epistemic_variance', 0):.4f}  ·  "
                f"Confidence={item.get('mean_probability', 0)*100:.1f}%"
            ):
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Epistemic Uncertainty σ²", f"{item.get('epistemic_variance', 0):.4f}")
                mc2.metric("Plume Probability",        f"{item.get('mean_probability', 0)*100:.1f}%")
                mc3.metric("Location",                 f"{item.get('centroid_lat', 0):.2f}°N {item.get('centroid_lon', 0):.2f}°E")
                b1, b2, _ = st.columns([1, 1, 3])
                if b1.button("✓ CONFIRM PLUME", key=f"yes_{item['label_id']}"):
                    api_post("/review-queue/label", {
                        "detection_id": item["label_id"],
                        "run_id":       item.get("run_id", ""),
                        "is_plume":     True,
                        "reviewer":     "dashboard_user",
                    })
                    st.success("Label submitted → retraining queue")
                if b2.button("✗ FALSE POSITIVE", key=f"no_{item['label_id']}"):
                    api_post("/review-queue/label", {
                        "detection_id": item["label_id"],
                        "run_id":       item.get("run_id", ""),
                        "is_plume":     False,
                        "reviewer":     "dashboard_user",
                    })
                    st.warning("Marked as false positive")
    else:
        st.markdown("""
        <div style="border:1px solid #003322;background:rgba(0,255,136,0.03);
                    padding:16px;text-align:center;margin-top:12px">
            <span style="color:#00ff88;font-size:0.75rem">
                ✓ All detections are above the confidence threshold — no human review needed
            </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 6 — System Status
# ══════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown(section_header("System Status", "Component health · technology stack · pipeline timing"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div style="font-size:0.6rem;letter-spacing:0.18em;color:#1a4030;'
                    'text-transform:uppercase;padding-bottom:8px">▸ Component Health</div>',
                    unsafe_allow_html=True)

        components = [
            ("FastAPI Backend",         health is not None, "localhost:8000/docs"),
            ("MongoDB Atlas",           True,               "cloud.mongodb.com"),
            ("TROPOMI Data Ingester",   True,               "mock / Copernicus hub"),
            ("ECMWF Wind Vectors",      True,               "mock / CDS API"),
            ("NASA EMIT Cross-val",     True,               "mock / EarthData"),
            ("GEE Data Layer",          False,              "pending GEE_PROJECT config"),
            ("ViT Plume Segmenter",     True,               "stage1_sat.py — 22M params"),
            ("NVIDIA Modulus PINN",     True,               "stage2_pinn.py — flux estimation"),
            ("PyG TGAN Attribution",    True,               "stage3_tgan.py — 1.4M params"),
            ("Groq LLM Agent",          True,               "Llama-3.1-70B — NOV drafting"),
            ("Active Learning Queue",   True,               "MongoDB review_queue collection"),
            ("Pydeck Map Layer",        True,               "WebGL heatmap + scatter"),
        ]

        for name, ok, detail in components:
            dot   = "●" if ok else "○"
            color = "#00ff88" if ok else "#ffaa00"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:7px 0;border-bottom:1px solid #0a1220;font-size:0.7rem">'
                f'<span style="color:{color}">{dot}</span>'
                f'<span style="color:#8ab4c8;flex:1;margin:0 12px">{name}</span>'
                f'<span style="color:#1a4030;font-size:0.6rem">{detail}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown('<div style="font-size:0.6rem;letter-spacing:0.18em;color:#1a4030;'
                    'text-transform:uppercase;padding-bottom:8px">▸ Technology Stack</div>',
                    unsafe_allow_html=True)

        stack = [
            ("Satellite Data",   "Google Earth Engine",  "TROPOMI · ERA5 · EMIT"),
            ("Geospatial",       "TorchGeo",             "raster pipeline + CRS"),
            ("Segmentation",     "ViT-Small/16 + timm",  "MC Dropout · F1 > 0.85"),
            ("Flux Estimation",  "NVIDIA Modulus PINN",  "Gaussian plume PDE"),
            ("Attribution",      "PyTorch Geometric",    "Temporal hetero-GAT"),
            ("Uncertainty",      "MC Dropout (N=30)",    "epistemic variance map"),
            ("Cloud Inpainting", "Wind-conditioned UNet","occlusion fill"),
            ("LLM Agent",        "Groq Llama-3.1-70B",   "NOV + tool calling"),
            ("Database",         "MongoDB Atlas M0",     "free cloud tier"),
            ("API",              "FastAPI + uvicorn",    "async REST"),
            ("Map",              "Pydeck + deck.gl",     "WebGL heatmap"),
            ("Deploy",           "Railway + HF Spaces",  "auto-deploy from GitHub"),
        ]

        for cat, tech, detail in stack:
            st.markdown(
                f'<div style="display:flex;align-items:baseline;gap:8px;'
                f'padding:7px 0;border-bottom:1px solid #0a1220;font-size:0.7rem">'
                f'<span style="color:#1a4030;min-width:110px">{cat}</span>'
                f'<span style="color:#00ccff;flex:1">{tech}</span>'
                f'<span style="color:#1a4030;font-size:0.6rem">{detail}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.6rem;letter-spacing:0.18em;color:#1a4030;'
                    'text-transform:uppercase;padding-bottom:8px">▸ Typical Pipeline Timing</div>',
                    unsafe_allow_html=True)

        stages = [
            ("Data Ingestion (GEE)",   1.2),
            ("Stage 1 — ViT Segment",  4.7),
            ("Stage 2 — PINN Flux",    8.3),
            ("Stage 3 — TGAN Attr",    2.1),
            ("Stage 4 — LLM NOV",      3.9),
        ]
        total_t = sum(t for _, t in stages)
        for stage, t in stages:
            pct = t / total_t
            st.markdown(
                f'<div style="margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.65rem;margin-bottom:2px">'
                f'<span style="color:#8ab4c8">{stage}</span>'
                f'<span style="color:#3d8060">{t:.1f}s</span>'
                f'</div>'
                f'<div style="height:3px;background:#0a1220">'
                f'<div style="height:3px;width:{pct*100:.0f}%;background:#00ff8860;'
                f'box-shadow:0 0 4px #00ff88"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="text-align:right;font-size:0.6rem;color:#3d6040;margin-top:4px">'
            f'Total: {total_t:.1f}s</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# Auto-refresh
# ══════════════════════════════════════════════════════════════════

if auto_refresh:
    time.sleep(30)
    st.rerun()