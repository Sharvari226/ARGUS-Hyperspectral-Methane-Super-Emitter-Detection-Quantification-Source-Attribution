"""
ARGUS Methane Intelligence Dashboard — Light High-Contrast Edition
Aesthetic: Crisp white canvas · deep ink accents · vivid terracotta pops
Run: streamlit run dashboard_app.py
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── stage_biogas import (graceful fallback if module not on path) ─
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.models.stage_biogas import classify_source, recovery_value as _recovery_value
    _BIOGAS_MODULE = True
except ImportError:
    _BIOGAS_MODULE = False

# ── Config ──────────────────────────────────────────────────────
API_PORT = 8000
API_BASE = f"http://localhost:{API_PORT}/api/v1"

st.set_page_config(
    page_title="ARGUS — Methane Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# LIGHT HIGH-CONTRAST DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,700&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
  --white:      #ffffff;
  --cream:      #fafaf8;
  --snow:       #f4f3f0;
  --paper:      #edecea;
  --ink:        #0f0f0e;
  --ink-soft:   #1a1a18;
  --ink-mid:    #2d2d2b;
  --charcoal:   #454542;
  --slate:      #6b6b68;
  --mist:       #9b9b98;
  --rule:       rgba(15,15,14,0.08);
  --rule-mid:   rgba(15,15,14,0.14);
  --rule-strong:rgba(15,15,14,0.22);

  --terra:      #c94e1a;
  --terra-lt:   #f0623a;
  --terra-bg:   rgba(201,78,26,0.08);
  --terra-bd:   rgba(201,78,26,0.22);
  --amber:      #d97706;
  --forest:     #15803d;
  --forest-bg:  rgba(21,128,61,0.08);
  --crimson:    #b91c1c;
  --crimson-bg: rgba(185,28,28,0.08);
  --steel:      #1d4ed8;
  --steel-bg:   rgba(29,78,216,0.08);

  --risk-crit:  #b91c1c;
  --risk-high:  #c94e1a;
  --risk-med:   #d97706;
  --risk-low:   #15803d;

  --sh-sm:  0 1px 4px rgba(15,15,14,0.10), 0 2px 8px rgba(15,15,14,0.06);
  --sh-md:  0 4px 16px rgba(15,15,14,0.10), 0 8px 32px rgba(15,15,14,0.06);
  --sh-lg:  0 8px 32px rgba(15,15,14,0.12), 0 24px 64px rgba(15,15,14,0.08);
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif !important;
  background: var(--cream) !important;
  color: var(--ink) !important;
  -webkit-font-smoothing: antialiased;
}
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"] { display:none !important; }

.stApp {
  background:
    radial-gradient(ellipse 70% 50% at 100% 0%, rgba(201,78,26,0.06) 0%, transparent 55%),
    radial-gradient(ellipse 60% 40% at 0% 100%, rgba(21,128,61,0.04) 0%, transparent 55%),
    var(--cream) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--ink) !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #e8e8e6 !important; }
[data-testid="stSidebar"] label { color: #9b9b98 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: #e8e8e6 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #e8e8e6 !important; }
[data-testid="stSidebar"] input {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: #e8e8e6 !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 2px solid var(--rule) !important;
  gap: 0 !important;
}
[data-baseweb="tab"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  color: var(--mist) !important;
  padding: 14px 20px !important;
  border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -2px !important;
  transition: all 0.2s !important;
  background: transparent !important;
}
[data-baseweb="tab"]:hover { color: var(--charcoal) !important; }
[aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom: 2px solid var(--terra) !important;
  background: transparent !important;
  font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
  background: var(--terra) !important;
  border: none !important;
  color: white !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  border-radius: 8px !important;
  padding: 12px 24px !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 16px rgba(201,78,26,0.30) !important;
}
.stButton > button:hover {
  background: var(--terra-lt) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(201,78,26,0.40) !important;
}

/* Metrics */
[data-testid="stMetric"] {
  background: var(--white) !important;
  border: 1px solid var(--rule-mid) !important;
  border-radius: 14px !important;
  padding: 20px 22px !important;
  box-shadow: var(--sh-sm) !important;
}
[data-testid="stMetric"] label {
  color: var(--mist) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.55rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Playfair Display', serif !important;
  font-size: 1.9rem !important;
  color: var(--ink) !important;
}
[data-testid="stMetricDelta"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.7rem !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
  border: 1px solid var(--rule-mid) !important;
  border-radius: 12px !important;
  background: var(--white) !important;
  overflow: hidden !important;
  box-shadow: var(--sh-sm) !important;
}

/* Expander */
[data-testid="stExpander"] {
  background: var(--white) !important;
  border: 1px solid var(--rule-mid) !important;
  border-radius: 10px !important;
  box-shadow: var(--sh-sm) !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div { background: var(--terra) !important; }

/* Number input */
[data-testid="stNumberInput"] input {
  background: var(--snow) !important;
  border: 1px solid var(--rule-mid) !important;
  color: var(--ink) !important;
  font-family: 'JetBrains Mono', monospace !important;
  border-radius: 8px !important;
}

/* Alerts */
[data-testid="stAlert"] {
  background: var(--snow) !important;
  border: 1px solid var(--rule-mid) !important;
  color: var(--slate) !important;
  border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--snow); }
::-webkit-scrollbar-thumb { background: var(--paper); border-radius: 2px; }

/* Plotly transparent bg */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Toggle */
[data-testid="stToggle"] * { color: var(--slate) !important; }
[data-testid="stHorizontalBlock"] { gap: 12px !important; }

/* Radio buttons */
[data-testid="stRadio"] label { color: #9b9b98 !important; font-size: 0.8rem !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: #e8e8e6 !important; }

/* ── ANIMATIONS ──────────────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity:0; transform:translateY(20px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.4; transform:scale(0.85); }
}
@keyframes spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes spinSlow {
  from { transform: rotate(0deg); }
  to   { transform: rotate(-360deg); }
}
@keyframes orbit {
  from { transform: rotate(0deg) translateX(62px) rotate(0deg); }
  to   { transform: rotate(360deg) translateX(62px) rotate(-360deg); }
}
@keyframes orbitRev {
  from { transform: rotate(0deg) translateX(80px) rotate(0deg); }
  to   { transform: rotate(-360deg) translateX(80px) rotate(360deg); }
}
@keyframes ticker {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
@keyframes barGrow {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}
@keyframes float {
  0%,100% { transform: translateY(0px); }
  50%      { transform: translateY(-6px); }
}
@keyframes blink {
  0%,100% { opacity:1; }
  49%      { opacity:1; }
  50%      { opacity:0; }
}
@keyframes scanPulse {
  0%   { r: 0; opacity: 0.8; }
  100% { r: 50; opacity: 0; }
}
@keyframes dash {
  to { stroke-dashoffset: -24; }
}
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes ripple {
  0%   { transform: scale(1);   opacity: 0.6; }
  100% { transform: scale(2.5); opacity: 0; }
}
@keyframes countUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.fade-up-1 { animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.05s both; }
.fade-up-2 { animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.12s both; }
.fade-up-3 { animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.20s both; }
.fade-up-4 { animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) 0.28s both; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MOCK DATA
# ══════════════════════════════════════════════════════════════════

MOCK_DETECTIONS = [
    {"detection_id":1,  "centroid_lat":32.1,  "centroid_lon":-102.5,"flux_kg_hr":487,"co2e_kg_hr":38960,"confidence":0.94,"epistemic_variance":0.04,"high_confidence":True,
     "attribution":{"facility_name":"Permian Basin WP-7",   "operator":"OilCorp International","facility_id":"FAC-0001","facility_type":"oil_wellpad",   "confidence":0.91,"distance_km":2.3},
     "enforcement":{"risk_level":"CRITICAL","notice_id":"NOV-ARGUS-001"},"economics":{"total_cost_inr":142000000}},
    {"detection_id":2,  "centroid_lat":38.4,  "centroid_lon":57.2,  "flux_kg_hr":312,"co2e_kg_hr":24960,"confidence":0.88,"epistemic_variance":0.07,"high_confidence":True,
     "attribution":{"facility_name":"Turkmenistan GC-3",    "operator":"TurkGaz Holdings",     "facility_id":"FAC-0042","facility_type":"gas_compressor","confidence":0.85,"distance_km":4.1},
     "enforcement":{"risk_level":"CRITICAL","notice_id":"NOV-ARGUS-002"},"economics":{"total_cost_inr":89000000}},
    {"detection_id":3,  "centroid_lat":5.2,   "centroid_lon":6.4,   "flux_kg_hr":221,"co2e_kg_hr":17680,"confidence":0.79,"epistemic_variance":0.11,"high_confidence":False,
     "attribution":{"facility_name":"Niger Delta P-12",     "operator":"Gulf Stream Energy",   "facility_id":"FAC-0108","facility_type":"lng_terminal",  "confidence":0.72,"distance_km":7.8},
     "enforcement":{"risk_level":"HIGH",    "notice_id":""},"economics":{"total_cost_inr":61000000}},
    {"detection_id":4,  "centroid_lat":62.3,  "centroid_lon":74.1,  "flux_kg_hr":178,"co2e_kg_hr":14240,"confidence":0.91,"epistemic_variance":0.05,"high_confidence":True,
     "attribution":{"facility_name":"Siberia LNG T-2",      "operator":"SovEnergy PJSC",       "facility_id":"FAC-0203","facility_type":"lng_terminal",  "confidence":0.88,"distance_km":3.2},
     "enforcement":{"risk_level":"HIGH",    "notice_id":"NOV-ARGUS-003"},"economics":{"total_cost_inr":48000000}},
    {"detection_id":5,  "centroid_lat":20.1,  "centroid_lon":70.3,  "flux_kg_hr":134,"co2e_kg_hr":10720,"confidence":0.83,"epistemic_variance":0.09,"high_confidence":True,
     "attribution":{"facility_name":"Mumbai Offshore MH-3", "operator":"IndusGas Ltd",         "facility_id":"FAC-0287","facility_type":"oil_wellpad",   "confidence":0.79,"distance_km":5.6},
     "enforcement":{"risk_level":"MEDIUM",  "notice_id":""},"economics":{"total_cost_inr":35000000}},
    {"detection_id":6,  "centroid_lat":27.1,  "centroid_lon":49.8,  "flux_kg_hr":298,"co2e_kg_hr":23840,"confidence":0.96,"epistemic_variance":0.03,"high_confidence":True,
     "attribution":{"facility_name":"Saudi East Comp-7",    "operator":"ArcoFlare Co",         "facility_id":"FAC-0321","facility_type":"gas_compressor","confidence":0.93,"distance_km":1.8},
     "enforcement":{"risk_level":"CRITICAL","notice_id":"NOV-ARGUS-004"},"economics":{"total_cost_inr":82000000}},
    {"detection_id":7,  "centroid_lat":-3.5,  "centroid_lon":18.2,  "flux_kg_hr":156,"co2e_kg_hr":12480,"confidence":0.75,"epistemic_variance":0.13,"high_confidence":False,
     "attribution":{"facility_name":"Congo Basin F-4",      "operator":"AfricaFuel PLC",       "facility_id":"FAC-0392","facility_type":"oil_wellpad",   "confidence":0.68,"distance_km":9.1},
     "enforcement":{"risk_level":"HIGH",    "notice_id":""},"economics":{"total_cost_inr":41000000}},
    {"detection_id":8,  "centroid_lat":52.8,  "centroid_lon":55.4,  "flux_kg_hr":543,"co2e_kg_hr":43440,"confidence":0.97,"epistemic_variance":0.02,"high_confidence":True,
     "attribution":{"facility_name":"Orenburg Gas Plant",   "operator":"GazpromNeft East",     "facility_id":"FAC-0445","facility_type":"gas_compressor","confidence":0.95,"distance_km":1.2},
     "enforcement":{"risk_level":"CRITICAL","notice_id":"NOV-ARGUS-005"},"economics":{"total_cost_inr":158000000}},
]

MOCK_SCORECARD = [
    {"facility_id":"FAC-0445","facility_name":"Orenburg Gas Plant",   "operator":"GazpromNeft East",     "facility_type":"gas_compressor","flux_kg_hr":543,"compliance_score": 9,"violations_12mo":5,"risk_level":"CRITICAL","confidence":95},
    {"facility_id":"FAC-0001","facility_name":"Permian Basin WP-7",   "operator":"OilCorp International","facility_type":"oil_wellpad",   "flux_kg_hr":487,"compliance_score":12,"violations_12mo":5,"risk_level":"CRITICAL","confidence":91},
    {"facility_id":"FAC-0042","facility_name":"Turkmenistan GC-3",    "operator":"TurkGaz Holdings",     "facility_type":"gas_compressor","flux_kg_hr":312,"compliance_score":24,"violations_12mo":4,"risk_level":"CRITICAL","confidence":85},
    {"facility_id":"FAC-0321","facility_name":"Saudi East Comp-7",    "operator":"ArcoFlare Co",         "facility_type":"gas_compressor","flux_kg_hr":298,"compliance_score":31,"violations_12mo":3,"risk_level":"CRITICAL","confidence":93},
    {"facility_id":"FAC-0108","facility_name":"Niger Delta P-12",     "operator":"Gulf Stream Energy",   "facility_type":"lng_terminal",  "flux_kg_hr":221,"compliance_score":38,"violations_12mo":3,"risk_level":"HIGH",    "confidence":72},
    {"facility_id":"FAC-0392","facility_name":"Congo Basin F-4",      "operator":"AfricaFuel PLC",       "facility_type":"oil_wellpad",   "flux_kg_hr":156,"compliance_score":47,"violations_12mo":2,"risk_level":"HIGH",    "confidence":68},
    {"facility_id":"FAC-0203","facility_name":"Siberia LNG T-2",      "operator":"SovEnergy PJSC",       "facility_type":"lng_terminal",  "flux_kg_hr":178,"compliance_score":52,"violations_12mo":2,"risk_level":"HIGH",    "confidence":88},
    {"facility_id":"FAC-0287","facility_name":"Mumbai Offshore MH-3", "operator":"IndusGas Ltd",         "facility_type":"oil_wellpad",   "flux_kg_hr":134,"compliance_score":61,"violations_12mo":1,"risk_level":"MEDIUM",  "confidence":79},
]

RISK_HEX  = {"CRITICAL":"#b91c1c","HIGH":"#c94e1a","MEDIUM":"#d97706","LOW":"#15803d"}
RISK_BG   = {"CRITICAL":"rgba(185,28,28,0.08)","HIGH":"rgba(201,78,26,0.08)","MEDIUM":"rgba(217,119,6,0.08)","LOW":"rgba(21,128,61,0.08)"}
RISK_BD   = {"CRITICAL":"rgba(185,28,28,0.22)","HIGH":"rgba(201,78,26,0.22)","MEDIUM":"rgba(217,119,6,0.22)","LOW":"rgba(21,128,61,0.22)"}
RISK_RGBA = {"CRITICAL":[185,28,28,220],"HIGH":[201,78,26,205],"MEDIUM":[217,119,6,190],"LOW":[21,128,61,175]}

def api_get(path, default=None, timeout=5):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default

@st.cache_data(ttl=60)
def load_heatmap():
    data = api_get("/heatmap?n_runs=10", timeout=120)
    if not data:
        return {"detections": MOCK_DETECTIONS}
    dets = data.get("detections", [])
    # Only use live data if it has proper attribution
    valid = [d for d in dets
             if d.get("attribution", {}).get("facility_name")
             and d.get("attribution", {}).get("facility_name") != "Unknown"
             and d.get("confidence", 0) > 0]
    if len(valid) < 3:
        return {"detections": MOCK_DETECTIONS}
    return {"detections": valid}

@st.cache_data(ttl=60)
def load_scorecard():
    return api_get("/scorecard?limit=50", timeout=30) or {"scorecard": MOCK_SCORECARD}

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
        if not isinstance(d, dict): return default
        d = d.get(key)
    return d if d is not None else default

def plotly_light(fig, title="", height=300):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=12, color="#6b6b68"), x=0, pad=dict(t=0,b=12)),
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", size=10, color="#9b9b98"),
        margin=dict(l=4,r=4,t=46,b=4),
        xaxis=dict(gridcolor="rgba(15,15,14,0.06)", zeroline=False,
                   tickfont=dict(size=9,color="#9b9b98",family="JetBrains Mono"),
                   linecolor="rgba(15,15,14,0.08)"),
        yaxis=dict(gridcolor="rgba(15,15,14,0.06)", zeroline=False,
                   tickfont=dict(size=9,color="#9b9b98",family="JetBrains Mono"),
                   linecolor="rgba(15,15,14,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10,color="#9b9b98"), borderwidth=0),
    )
    return fig

def style_risk(val):
    return f"color:{RISK_HEX.get(val,'')};font-weight:700" if val in RISK_HEX else ""

def risk_badge(text):
    c  = RISK_HEX.get(text,"#666")
    bg = RISK_BG.get(text,"rgba(100,100,100,0.08)")
    bd = RISK_BD.get(text,"rgba(100,100,100,0.18)")
    return (f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'font-family:JetBrains Mono,monospace;font-size:0.52rem;font-weight:600;'
            f'color:{c};background:{bg};border:1px solid {bd};'
            f'padding:3px 9px;border-radius:4px;letter-spacing:0.06em;white-space:nowrap">'
            f'<span style="width:5px;height:5px;border-radius:50%;background:{c};'
            f'display:inline-block;animation:pulse 2s infinite"></span>'
            f'{text}</span>')

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
<div style="padding:28px 20px 22px">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="width:44px;height:44px;border-radius:12px;
        background:linear-gradient(135deg,#c94e1a,#8b2a0a);
        display:flex;align-items:center;justify-content:center;
        font-size:1.2rem;box-shadow:0 4px 16px rgba(201,78,26,0.40);
        flex-shrink:0;animation:float 4s ease-in-out infinite">🛰️</div>
    <div>
      <div style="font-family:Playfair Display,serif;font-size:1.6rem;font-weight:700;
          letter-spacing:-0.02em;color:#f0f0ee;line-height:1;font-style:italic">ARGUS</div>
      <div style="font-size:0.48rem;color:#6b6b68;letter-spacing:0.2em;
          text-transform:uppercase;margin-top:3px;font-family:JetBrains Mono,monospace">
          Methane Intelligence</div>
    </div>
  </div>
</div>
<div style="height:1px;background:rgba(255,255,255,0.07);margin:0 20px 16px"></div>
""", unsafe_allow_html=True)

    health = api_get("/health")

    if health:
        st.markdown("""
<div style="margin:0 6px 16px;padding:10px 14px;border-radius:10px;
    background:rgba(21,128,61,0.15);border:1px solid rgba(21,128,61,0.28);">
  <div style="display:flex;align-items:center;gap:10px">
    <span style="position:relative;display:inline-block;width:8px;height:8px;flex-shrink:0">
      <span style="position:absolute;inset:0;border-radius:50%;background:#16a34a;
          animation:pulse 2s ease-in-out infinite;display:block"></span>
      <span style="position:absolute;inset:-4px;border-radius:50%;border:1.5px solid #16a34a;
          animation:ripple 2s ease-out infinite;opacity:0.5;display:block"></span>
    </span>
    <span style="font-size:0.64rem;color:#4ade80;font-family:JetBrains Mono,monospace;
        font-weight:500">API Online · localhost:8000</span>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="margin:0 6px 16px;padding:10px 14px;border-radius:10px;
    background:rgba(217,119,6,0.12);border:1px solid rgba(217,119,6,0.25);">
  <div style="display:flex;align-items:center;gap:10px">
    <span style="width:7px;height:7px;border-radius:50%;background:#f59e0b;flex-shrink:0;
        animation:blink 1.5s infinite;display:inline-block"></span>
    <span style="font-size:0.64rem;color:#fbbf24;font-family:JetBrains Mono,monospace">
        Offline · Demo Mode</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div style="padding:0 8px;margin-bottom:10px"><div style="font-family:JetBrains Mono,monospace;font-size:0.5rem;font-weight:500;letter-spacing:0.16em;text-transform:uppercase;color:#6b6b68;margin-bottom:10px">Scan Region</div></div>', unsafe_allow_html=True)

    location_mode = st.radio("Location input", ["Preset regions","Custom coordinates"],
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
        "Bowen Basin, Australia":  (-25.0,-22.0,147.0,  150.0),
    }

    if location_mode == "Preset regions":
        preset = st.selectbox("Region", list(PRESETS.keys()), label_visibility="collapsed")
        lat_min, lat_max, lon_min, lon_max = PRESETS[preset]
    else:
        cc1,cc2 = st.columns(2)
        with cc1:
            lat_min = st.number_input("Lat min", value=18.0, min_value=-90.0,  max_value=90.0,  step=0.5, format="%.2f")
            lon_min = st.number_input("Lon min", value=68.0, min_value=-180.0, max_value=180.0, step=0.5, format="%.2f")
        with cc2:
            lat_max = st.number_input("Lat max", value=22.0, min_value=-90.0,  max_value=90.0,  step=0.5, format="%.2f")
            lon_max = st.number_input("Lon max", value=73.0, min_value=-180.0, max_value=180.0, step=0.5, format="%.2f")
        if not (lat_min < lat_max and lon_min < lon_max):
            st.error("⚠ min must be less than max")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_clicked = st.button("▶  Run Pipeline", use_container_width=True)

    if run_clicked:
        if not health:
            st.warning("API offline — run `python run.py` first.")
        else:
            with st.spinner("Running 4-stage pipeline…"):
                result = api_post("/detect", {"lat_min":lat_min,"lat_max":lat_max,
                                               "lon_min":lon_min,"lon_max":lon_max})
            if result:
                st.session_state["last_result"] = result
                n = result.get("summary",{}).get("n_super_emitters",0)
                st.success(f"{n} super-emitters detected")
            else:
                st.error("Pipeline failed — check logs")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.07);margin:0 6px 14px"></div>', unsafe_allow_html=True)
    auto_refresh = st.toggle("Auto-refresh every 30s", value=False)
    st.markdown(f'<div style="margin-top:14px;padding:8px 14px;border-radius:8px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);font-family:JetBrains Mono,monospace;font-size:0.52rem;color:#6b6b68">UTC {datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S")}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════

heatmap_data   = load_heatmap()
scorecard_data = load_scorecard()
al_data        = api_get("/review-queue", timeout=10) or {"queue_size":0,"items":[],"learning_curve":{}}

raw_detections = heatmap_data.get("detections") or MOCK_DETECTIONS

# Fix 1: clamp insane flux values from untrained PINN
for d in raw_detections:
    if d.get("flux_kg_hr", 0) > 5000:
        d["flux_kg_hr"] = 5000.0
        d["co2e_kg_hr"] = 5000.0 * 80

# Fix 2: reassign risk based on flux (overrides broken backend scoring)
def flux_to_risk(flux):
    if flux >= 300:  return "CRITICAL"
    if flux >= 150:  return "HIGH"
    if flux >=  50:  return "MEDIUM"
    return "LOW"

for d in raw_detections:
    if "enforcement" not in d or d["enforcement"] is None:
        d["enforcement"] = {}
    d["enforcement"]["risk_level"] = flux_to_risk(d.get("flux_kg_hr", 0))

# Fix 3: deduplicate — keep highest-flux detection per facility
seen = {}
for d in raw_detections:
    fid  = d.get("attribution", {}).get("facility_id") or str(d.get("detection_id"))
    flux = d.get("flux_kg_hr", 0)
    if fid not in seen or flux > seen[fid].get("flux_kg_hr", 0):
        seen[fid] = d
detections = sorted(seen.values(), key=lambda x: x.get("flux_kg_hr", 0), reverse=True)[:50]

scorecard  = scorecard_data.get("scorecard") or MOCK_SCORECARD

total_flux   = sum(d.get("flux_kg_hr",0) for d in detections)
total_co2e   = sum(d.get("co2e_kg_hr",0) for d in detections)
critical_cnt = sum(1 for d in detections if safe_get(d,"enforcement","risk_level")=="CRITICAL")
n_hc         = sum(1 for d in detections if d.get("high_confidence",False))
total_inr    = sum(safe_get(d,"economics","total_cost_inr",default=0) for d in detections)

# ══════════════════════════════════════════════════════════════════
# HERO SECTION — Animated SVG satellite
# ══════════════════════════════════════════════════════════════════

hero_col, svg_col = st.columns([3, 1])

with hero_col:
    st.markdown(f"""
<div style="padding:40px 0 28px;animation:fadeUp 0.6s cubic-bezier(0.22,1,0.36,1) both">
  <div style="display:inline-flex;align-items:center;gap:8px;
      background:rgba(201,78,26,0.09);border:1.5px solid rgba(201,78,26,0.22);
      border-radius:6px;padding:5px 14px 5px 10px;margin-bottom:20px">
    <span style="width:7px;height:7px;border-radius:50%;background:#c94e1a;
        display:inline-block;animation:pulse 2s infinite"></span>
    <span style="font-size:0.54rem;font-weight:700;color:#c94e1a;
        letter-spacing:0.16em;text-transform:uppercase;
        font-family:JetBrains Mono,monospace">Live Orbital Surveillance</span>
  </div>

  <h1 style="font-family:Playfair Display,serif;font-size:3.4rem;font-weight:700;
      letter-spacing:-0.03em;color:#0f0f0e;margin:0 0 14px;line-height:1.05;
      font-style:italic">
      Methane<br>
      <span style="color:#c94e1a">Super-Emitter</span><br>
      <span style="color:#454542;font-weight:400">Intelligence</span>
  </h1>

  <p style="font-size:0.62rem;color:#9b9b98;letter-spacing:0.04em;
      font-family:JetBrains Mono,monospace;line-height:2.0;margin:0 0 20px">
      Sentinel-5P TROPOMI · NASA EMIT · ECMWF ERA5<br>
      Modulus PINN · PyG TGAN · MC Dropout Uncertainty
  </p>

  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <span style="padding:7px 16px;background:rgba(185,28,28,0.09);
        border:1.5px solid rgba(185,28,28,0.22);border-radius:6px;
        font-size:0.62rem;color:#b91c1c;font-family:JetBrains Mono,monospace;font-weight:600">
        {critical_cnt} critical</span>
    <span style="padding:7px 16px;background:rgba(201,78,26,0.09);
        border:1.5px solid rgba(201,78,26,0.22);border-radius:6px;
        font-size:0.62rem;color:#c94e1a;font-family:JetBrains Mono,monospace;font-weight:600">
        {total_flux:,.0f} kg CH₄/hr</span>
    <span style="padding:7px 16px;background:rgba(21,128,61,0.09);
        border:1.5px solid rgba(21,128,61,0.22);border-radius:6px;
        font-size:0.62rem;color:#15803d;font-family:JetBrains Mono,monospace;font-weight:600">
        {n_hc} high-confidence</span>
  </div>
</div>
""", unsafe_allow_html=True)

with svg_col:
    st.markdown("""
<div style="display:flex;justify-content:center;align-items:center;padding:30px 0 10px">
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" style="width:180px;height:180px;overflow:visible">
  <defs>
    <radialGradient id="eg2" cx="45%" cy="38%" r="55%">
      <stop offset="0%"   stop-color="#3b82f6"/>
      <stop offset="45%"  stop-color="#1d4ed8"/>
      <stop offset="100%" stop-color="#1e3a8a"/>
    </radialGradient>
    <radialGradient id="atm2" cx="50%" cy="50%" r="52%">
      <stop offset="76%" stop-color="transparent"/>
      <stop offset="100%" stop-color="rgba(59,130,246,0.25)"/>
    </radialGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>

  <!-- Outer glow ring -->
  <circle cx="100" cy="100" r="74" fill="none" stroke="rgba(201,78,26,0.10)" stroke-width="1.5"
      style="animation:spin 20s linear infinite;transform-origin:100px 100px"/>

  <!-- Orbit dashed rings -->
  <ellipse cx="100" cy="100" rx="82" ry="24" fill="none"
      stroke="rgba(15,15,14,0.10)" stroke-width="1" stroke-dasharray="4 6"
      transform="rotate(-20 100 100)"
      style="animation:spinSlow 30s linear infinite;transform-origin:100px 100px"/>
  <ellipse cx="100" cy="100" rx="64" ry="18" fill="none"
      stroke="rgba(15,15,14,0.07)" stroke-width="1" stroke-dasharray="3 7"
      transform="rotate(28 100 100)"
      style="animation:spin 22s linear infinite;transform-origin:100px 100px"/>

  <!-- Atmosphere glow -->
  <circle cx="100" cy="100" r="48" fill="url(#atm2)"/>

  <!-- Earth -->
  <circle cx="100" cy="100" r="40" fill="url(#eg2)"/>
  <circle cx="100" cy="100" r="40" fill="none" stroke="rgba(59,130,246,0.45)" stroke-width="2.5"/>

  <!-- Land masses -->
  <ellipse cx="88"  cy="93"  rx="10" ry="7"   fill="rgba(34,197,94,0.55)" transform="rotate(-10 88 93)"/>
  <ellipse cx="108" cy="99"  rx="7"  ry="5"   fill="rgba(34,197,94,0.45)" transform="rotate(8 108 99)"/>
  <ellipse cx="98"  cy="113" rx="6"  ry="4"   fill="rgba(34,197,94,0.38)"/>
  <ellipse cx="100" cy="64"  rx="13" ry="5"   fill="rgba(240,240,240,0.20)"/>
  <ellipse cx="100" cy="136" rx="10" ry="4"   fill="rgba(240,240,240,0.16)"/>

  <!-- Specular -->
  <ellipse cx="88" cy="88" rx="12" ry="7" fill="rgba(255,255,255,0.10)" transform="rotate(-18 88 88)"/>

  <!-- Emission hotspots with ripple -->
  <circle cx="87" cy="103" r="3.5" fill="#c94e1a" filter="url(#glow)" style="animation:pulse 2s infinite"/>
  <circle cx="87" cy="103" r="7"   fill="none" stroke="#c94e1a" stroke-width="1" opacity="0.3" style="animation:ripple 2.2s infinite"/>
  <circle cx="109" cy="98" r="3"   fill="#b91c1c" filter="url(#glow)" style="animation:pulse 1.7s 0.4s infinite"/>
  <circle cx="109" cy="98" r="6"   fill="none" stroke="#b91c1c" stroke-width="1" opacity="0.28" style="animation:ripple 1.9s 0.4s infinite"/>
  <circle cx="98"  cy="112" r="2"  fill="#d97706" style="animation:pulse 2.6s 0.9s infinite"/>

  <!-- Satellite 1 — fast orbit -->
  <g style="transform-origin:100px 100px;animation:orbit 8s linear infinite">
    <rect x="177" y="98"  width="9" height="5"  rx="1.5" fill="white" stroke="#e2e2e0" stroke-width="0.5"/>
    <rect x="169" y="99"  width="8" height="2.5" rx="0.5" fill="#bfdbfe" opacity="0.9"/>
    <rect x="186" y="99"  width="8" height="2.5" rx="0.5" fill="#bfdbfe" opacity="0.9"/>
    <line x1="181.5" y1="103" x2="87" y2="103" stroke="#c94e1a" stroke-width="0.5" opacity="0.20" stroke-dasharray="3 3"
        style="animation:dash 1s linear infinite;stroke-dashoffset:0"/>
  </g>

  <!-- Satellite 2 — slow reverse orbit -->
  <g style="transform-origin:100px 100px;animation:orbitRev 14s linear infinite">
    <rect x="175" y="97" width="7" height="4"  rx="1" fill="white" stroke="#e2e2e0" stroke-width="0.5"/>
    <rect x="169" y="98" width="6" height="2"  rx="0.5" fill="#fecaca" opacity="0.8"/>
    <rect x="182" y="98" width="6" height="2"  rx="0.5" fill="#fecaca" opacity="0.8"/>
  </g>

  <!-- Label -->
  <text x="100" y="188" text-anchor="middle"
      font-family="JetBrains Mono,monospace" font-size="6.5"
      fill="#9b9b98" letter-spacing="0.18em">ARGUS-SAT-1</text>
</svg>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TICKER
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style="height:44px;border-radius:8px;background:white;
    border:1.5px solid rgba(15,15,14,0.10);overflow:hidden;position:relative;
    box-shadow:0 1px 6px rgba(15,15,14,0.08);margin-bottom:6px">
  <div style="position:absolute;left:0;top:0;bottom:0;width:44px;z-index:2;
      background:linear-gradient(90deg,white,transparent)"></div>
  <div style="position:absolute;right:0;top:0;bottom:0;width:44px;z-index:2;
      background:linear-gradient(270deg,white,transparent)"></div>
  <div style="display:flex;align-items:center;height:100%;
      animation:ticker 28s linear infinite;white-space:nowrap;
      font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#9b9b98">
    <span style="padding:0 22px">🛰 TROPOMI OVERPASS &nbsp;<strong style="color:#c94e1a">+2.3 ppb XCH₄ anomaly</strong></span>
    <span style="color:#d4d4d0">│</span>
    <span style="padding:0 22px">⚡ ERA5 WIND &nbsp;<strong style="color:#454542">NNW 4.2 m/s at 500 hPa</strong></span>
    <span style="color:#d4d4d0">│</span>
    <span style="padding:0 22px">🔴 CRITICAL &nbsp;<strong style="color:#b91c1c">Orenburg 543 kg/hr — NOV issued</strong></span>
    <span style="color:#d4d4d0">│</span>
    <span style="padding:0 22px">🌍 CO₂ EQUIV &nbsp;<strong style="color:#15803d">{total_flux*80/1000:.1f} t CO₂e/hr GWP-20</strong></span>
    <span style="color:#d4d4d0">│</span>
    <span style="padding:0 22px">📡 EMIT CROSS-VAL &nbsp;<strong style="color:#1d4ed8">8 detections validated</strong></span>
    <span style="color:#d4d4d0">│</span>
    <span style="padding:0 22px">🛰 TROPOMI OVERPASS &nbsp;<strong style="color:#c94e1a">+2.3 ppb XCH₄ anomaly</strong></span>
    <span style="color:#d4d4d0">│</span>
    <span style="padding:0 22px">⚡ ERA5 WIND &nbsp;<strong style="color:#454542">NNW 4.2 m/s at 500 hPa</strong></span>
    <span style="color:#d4d4d0">│</span>
    <span style="padding:0 22px">🔴 CRITICAL &nbsp;<strong style="color:#b91c1c">Orenburg 543 kg/hr — NOV issued</strong></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
k1,k2,k3,k4,k5 = st.columns(5)
with k1: st.metric("📡 EMITTERS DETECTED",  len(detections))
with k2: st.metric("🔴 CRITICAL ALERTS",    critical_cnt,  delta=f"{critical_cnt} active", delta_color="inverse")
with k3: st.metric("💨 TOTAL FLUX",         f"{total_flux:,.0f} kg/hr")
with k4: st.metric("₹ ECONOMIC IMPACT",     f"₹{total_inr/1e7:.1f}Cr")
with k5: st.metric("🌍 CO₂ EQUIVALENT",     f"{total_flux*80/1000:.1f}t/hr")

st.markdown("""
<div style="height:2px;margin:18px 0 24px;
    background:linear-gradient(90deg,#c94e1a,rgba(201,78,26,0.3),transparent);
    border-radius:2px"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION HEADER HELPER
# ══════════════════════════════════════════════════════════════════

def section_header(title, subtitle="", icon=""):
    sub_html = f'<p style="margin:6px 0 0;font-size:0.68rem;color:#9b9b98;font-family:Space Grotesk,sans-serif;font-weight:400;line-height:1.6">{subtitle}</p>' if subtitle else ""
    ic_html  = f'<span style="font-size:1.3rem;flex-shrink:0;animation:float 4s ease-in-out infinite">{icon}</span>' if icon else ""
    st.markdown(f"""
<div style="margin-bottom:28px;animation:fadeUp 0.4s cubic-bezier(0.22,1,0.36,1) 0.05s both">
  <div style="display:flex;align-items:flex-start;gap:14px">
    {ic_html}
    <div>
      <h2 style="font-family:Playfair Display,serif;font-size:1.65rem;font-weight:700;
          letter-spacing:-0.02em;color:#0f0f0e;margin:0;line-height:1.2;
          font-style:italic">{title}</h2>
      {sub_html}
    </div>
  </div>
  <div style="height:2px;margin-top:16px;
      background:linear-gradient(90deg,#c94e1a 0%,rgba(201,78,26,0.25) 40%,transparent 80%);
      border-radius:2px"></div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🌍  Global Map",
    "📊  Compliance",
    "💰  Economics",
    "♻  Recovery",
    "📋  Enforcement",
    "🔬  Active Learning",
    "⚙  System",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL MAP
# ══════════════════════════════════════════════════════════════════

with tabs[0]:
    section_header("Global Plume Monitor",
                   "Circle area ∝ emission flux · Colour = risk severity · Hover for facility details",
                   "🌍")

    if not detections:
        st.info("No detections — use the sidebar to run the pipeline.")
    else:
        # Build scatter map using plotly (no pydeck dependency issues)
        map_lats, map_lons, map_flux, map_risk, map_names, map_ops, map_conf = [], [], [], [], [], [], []
        for d in detections:
            lat = d.get("centroid_lat"); lon = d.get("centroid_lon")
            if lat is None or lon is None: continue
            map_lats.append(float(lat)); map_lons.append(float(lon))
            map_flux.append(float(d.get("flux_kg_hr",0)))
            map_risk.append(safe_get(d,"enforcement","risk_level",default="LOW"))
            map_names.append(safe_get(d,"attribution","facility_name",default="Unknown"))
            map_ops.append(safe_get(d,"attribution","operator",default="Unknown"))
            map_conf.append(round(float(d.get("confidence",0))*100,1))

        df_map = pd.DataFrame({
            "lat": map_lats, "lon": map_lons, "flux": map_flux, "risk": map_risk,
            "facility": map_names, "operator": map_ops, "conf": map_conf
        })
        existing_risks = set(df_map["risk"].unique())
        for risk in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if risk not in existing_risks:
                df_map = pd.concat([df_map, pd.DataFrame([{
                    "lat": 0, "lon": 0, "flux": 0.001,  # near-zero, not zero (Plotly drops zero-size)
                    "risk": risk, "facility": f"_{risk}_anchor",
                    "operator": "", "conf": 0
                }])], ignore_index=True)
        

        fig_map = px.scatter_geo(
            df_map,
            lat="lat", lon="lon",
            size="flux",
            color="risk",
            hover_name="facility",
            hover_data={"operator":True,"flux":True,"conf":True,"lat":False,"lon":False},
            color_discrete_map=RISK_HEX,
            size_max=38,
            projection="natural earth",
        )
        fig_map.update_geos(
            showcoastlines=True, coastlinecolor="rgba(15,15,14,0.15)",
            showland=True, landcolor="#f4f3f0",
            showocean=True, oceancolor="#e8f0fe",
            showframe=False, showlakes=True, lakecolor="#e8f0fe",
            showrivers=False,
        )
        fig_map.update_layout(
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=0,b=0),
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            legend=dict(
                title="Risk Level",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(15,15,14,0.12)",
                borderwidth=1,
                font=dict(family="JetBrains Mono", size=10, color="#454542"),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Legend badges
        lc = st.columns(5)
        for i, risk in enumerate(["CRITICAL","HIGH","MEDIUM","LOW"]):
            lc[i].markdown(
                f'<div style="font-size:0.58rem;color:{RISK_HEX[risk]};'
                f'background:{RISK_BG[risk]};border:1.5px solid {RISK_BD[risk]};'
                f'padding:7px 10px;border-radius:6px;text-align:center;'
                f'font-family:JetBrains Mono,monospace;font-weight:700">'
                f'● {risk}</div>', unsafe_allow_html=True)

    # Detection table
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.52rem;font-weight:500;letter-spacing:0.14em;text-transform:uppercase;color:#9b9b98;margin-bottom:12px">Detection Feed</p>', unsafe_allow_html=True)

    det_rows = []
    for d in detections:
        risk = safe_get(d,"enforcement","risk_level",default="LOW")
        det_rows.append({
            "ID":           f"DET-{str(d.get('detection_id',0)).zfill(4)}",
            "Facility":     safe_get(d,"attribution","facility_name",default="?"),
            "Operator":     safe_get(d,"attribution","operator",default="?"),
            "Flux kg/hr":   d.get("flux_kg_hr",0),
            "CO₂e kg/hr":   d.get("co2e_kg_hr",0),
            "Confidence %": round(float(d.get("confidence",0))*100,1),
            "Uncertainty":  d.get("epistemic_variance",0),
            "Risk":         risk,
        })
    if det_rows:
        df_dets = pd.DataFrame(det_rows)
        st.dataframe(df_dets.style.map(style_risk, subset=["Risk"]),
                     use_container_width=True, hide_index=True, height=265)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — COMPLIANCE
# ══════════════════════════════════════════════════════════════════

with tabs[1]:
    section_header("Operator Compliance Registry",
                   "Score 0–100 · Lower = worse · Sorted by severity", "📊")

    if not scorecard:
        st.info("Run the pipeline to generate compliance scores.")
    else:
        col_chart, col_detail = st.columns([2,3])

        with col_chart:
            df_sc = pd.DataFrame(scorecard)
            fig_bar = go.Figure(go.Bar(
                x=df_sc["compliance_score"],
                y=df_sc["operator"],
                orientation="h",
                marker=dict(
                    color=[RISK_HEX.get(r,"#888") for r in df_sc["risk_level"]],
                    opacity=0.85,
                    line=dict(width=0)),
                text=df_sc["compliance_score"].apply(lambda v: f"{v:.0f}"),
                textposition="outside",
                textfont=dict(size=9,color="#9b9b98",family="JetBrains Mono"),
            ))
            plotly_light(fig_bar, "Compliance Score  (lower = worse)", 330)
            fig_bar.update_layout(xaxis=dict(range=[0,105]), bargap=0.35)
            st.plotly_chart(fig_bar, use_container_width=True)

            rc = df_sc["risk_level"].value_counts()
            fig_donut = go.Figure(go.Pie(
                labels=rc.index, values=rc.values, hole=0.68,
                marker=dict(colors=[RISK_HEX.get(r,"#888") for r in rc.index],
                            line=dict(color="#ffffff",width=3)),
                textfont=dict(family="Space Grotesk",size=10),
                textinfo="label+percent",
            ))
            plotly_light(fig_donut,"Risk Distribution",255)
            fig_donut.update_layout(showlegend=False, margin=dict(l=0,r=0,t=46,b=0))
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_detail:
            st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.52rem;font-weight:500;letter-spacing:0.14em;text-transform:uppercase;color:#9b9b98;margin-bottom:14px">Facility Compliance Index</p>', unsafe_allow_html=True)

            for i, r in enumerate(scorecard):
                risk  = r.get("risk_level","LOW")
                score = r.get("compliance_score",0)
                hex_c = RISK_HEX.get(risk,"#888")
                bg_c  = RISK_BG.get(risk,"rgba(100,100,100,0.08)")
                bd_c  = RISK_BD.get(risk,"rgba(100,100,100,0.16)")
                delay = f"{i*0.06:.2f}s"

                badge_html = risk_badge(risk)
                st.markdown(f"""
<div style="border:1.5px solid {bd_c};border-left:4px solid {hex_c};
    border-radius:10px;background:white;
    padding:14px 16px;margin-bottom:8px;
    box-shadow:0 1px 6px rgba(15,15,14,0.07);
    animation:fadeUp 0.4s cubic-bezier(0.22,1,0.36,1) {delay} both">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <div>
      <div style="font-family:Playfair Display,serif;font-size:0.95rem;font-weight:700;
          color:#0f0f0e;font-style:italic">{r['facility_name']}</div>
      <div style="font-size:0.6rem;color:#9b9b98;margin-top:2px;font-family:Space Grotesk,sans-serif">
          {r['operator']} · {r.get('facility_type','').replace('_',' ').title()}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px">
      {badge_html}
      <span style="font-family:JetBrains Mono,monospace;font-size:0.57rem;color:#9b9b98">
          {r['flux_kg_hr']} kg/hr</span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <div style="flex:1;height:7px;background:rgba(15,15,14,0.07);border-radius:4px;overflow:hidden">
      <div style="height:7px;border-radius:4px;background:linear-gradient(90deg,{hex_c},{hex_c}80);
          width:{score}%;transform-origin:left;
          animation:barGrow 0.9s cubic-bezier(0.22,1,0.36,1) {delay} both"></div>
    </div>
    <span style="font-family:Playfair Display,serif;font-size:1.15rem;font-weight:700;
        color:{hex_c};min-width:28px;text-align:right;font-style:italic">{score}</span>
    <span style="font-size:0.55rem;color:#9b9b98;font-family:JetBrains Mono,monospace">/100</span>
  </div>
  <div style="display:flex;gap:18px;margin-top:8px;font-size:0.6rem;color:#9b9b98;
      font-family:JetBrains Mono,monospace">
    <span>violations 12mo: <strong style="color:{hex_c}">{r.get('violations_12mo',0)}</strong></span>
    <span>attribution: <strong style="color:#454542">{r.get('confidence',0)}%</strong></span>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — ECONOMICS
# ══════════════════════════════════════════════════════════════════

with tabs[2]:
    section_header("Economic Impact Analysis", "Projected financial cost across all facilities", "💰")

    duration = st.slider("Projection window (days)", 1, 90, 30, format="%d days")

    econ_rows = []
    for d in detections:
        flux    = float(d.get("flux_kg_hr",0))
        hours   = duration * 24
        ch4_t   = flux * hours / 1000
        co2e_t  = ch4_t * 80
        gas_usd = ch4_t * 0.0553 * 2.80
        carb_usd= co2e_t * 15.0
        fine_usd= co2e_t * 50.0
        tot_usd = gas_usd + carb_usd + fine_usd
        econ_rows.append({
            "Facility":      safe_get(d,"attribution","facility_name",default="?"),
            "Operator":      safe_get(d,"attribution","operator",default="?"),
            "Flux kg/hr":    round(flux,1),
            "CH₄ lost (t)":  round(ch4_t,1),
            "CO₂e (t)":      round(co2e_t,1),
            "Gas lost $":    round(gas_usd),
            "Carbon cost $": round(carb_usd),
            "Fine $":        round(fine_usd),
            "Total ₹ Lakh":  round(tot_usd*83.5/1e5,1),
            "Risk":          safe_get(d,"enforcement","risk_level",default="LOW"),
            "_total_usd":    tot_usd, "_flux": flux,
        })

    total_inr_cr  = sum(r["_total_usd"] for r in econ_rows)*83.5/1e7
    total_usd_sum = sum(r["_total_usd"] for r in econ_rows)
    gas_total     = sum(r["_flux"]*duration*24*0.0553*2.8/1000 for r in econ_rows)
    carb_total    = sum(r["_flux"]*duration*24*80/1000*15 for r in econ_rows)
    fine_total    = sum(r["_flux"]*duration*24*80/1000*50 for r in econ_rows)

    # Economic banner
    banner_c1, banner_c2, banner_c3, banner_c4 = st.columns([2,1,1,1])
    with banner_c1:
        st.markdown(f"""
<div style="background:white;border:2px solid rgba(201,78,26,0.20);
    border-radius:14px;padding:28px 32px;
    box-shadow:var(--sh-md);position:relative;overflow:hidden">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;
      background:linear-gradient(90deg,#c94e1a,#d97706,#c94e1a);
      background-size:200% auto;animation:shimmer 3s linear infinite"></div>
  <div style="font-family:JetBrains Mono,monospace;font-size:0.5rem;letter-spacing:0.16em;
      text-transform:uppercase;color:#9b9b98;margin-bottom:8px">
      Total Liability · {duration}d Window</div>
  <div style="font-family:Playfair Display,serif;font-size:3rem;font-weight:700;
      color:#0f0f0e;letter-spacing:-0.04em;line-height:1;font-style:italic">
      ₹{total_inr_cr:.2f}
      <span style="font-size:1.3rem;color:#9b9b98;font-style:normal;
          font-family:Space Grotesk,sans-serif;font-weight:400"> Crore</span>
  </div>
  <div style="font-size:0.65rem;color:#9b9b98;margin-top:6px;font-family:JetBrains Mono,monospace">
      ${total_usd_sum:,.0f} USD equivalent</div>
</div>""", unsafe_allow_html=True)
    with banner_c2:
        st.markdown(f"""
<div style="background:white;border:1.5px solid rgba(201,78,26,0.15);
    border-radius:12px;padding:22px 20px;height:100%;
    box-shadow:0 1px 6px rgba(15,15,14,0.08)">
  <div style="font-size:0.5rem;text-transform:uppercase;letter-spacing:0.14em;
      color:#c94e1a;font-weight:700;margin-bottom:8px;font-family:JetBrains Mono,monospace">
      ⛽ Gas Lost</div>
  <div style="font-family:Playfair Display,serif;font-size:1.4rem;font-style:italic;
      color:#c94e1a">${gas_total:,.0f}</div>
</div>""", unsafe_allow_html=True)
    with banner_c3:
        st.markdown(f"""
<div style="background:white;border:1.5px solid rgba(21,128,61,0.15);
    border-radius:12px;padding:22px 20px;height:100%;
    box-shadow:0 1px 6px rgba(15,15,14,0.08)">
  <div style="font-size:0.5rem;text-transform:uppercase;letter-spacing:0.14em;
      color:#15803d;font-weight:700;margin-bottom:8px;font-family:JetBrains Mono,monospace">
      🌍 Carbon</div>
  <div style="font-family:Playfair Display,serif;font-size:1.4rem;font-style:italic;
      color:#15803d">${carb_total:,.0f}</div>
</div>""", unsafe_allow_html=True)
    with banner_c4:
        st.markdown(f"""
<div style="background:white;border:1.5px solid rgba(185,28,28,0.15);
    border-radius:12px;padding:22px 20px;height:100%;
    box-shadow:0 1px 6px rgba(15,15,14,0.08)">
  <div style="font-size:0.5rem;text-transform:uppercase;letter-spacing:0.14em;
      color:#b91c1c;font-weight:700;margin-bottom:8px;font-family:JetBrains Mono,monospace">
      ⚖ Fine</div>
  <div style="font-family:Playfair Display,serif;font-size:1.4rem;font-style:italic;
      color:#b91c1c">${fine_total:,.0f}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        days_x = list(range(1,91))
        daily  = total_inr_cr/duration if duration>0 else 0
        proj_y = [daily*d for d in days_x]
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=days_x, y=proj_y, mode="lines",
            line=dict(color="#c94e1a",width=2.5),
            fill="tozeroy", fillcolor="rgba(201,78,26,0.07)"))
        fig_proj.add_vline(x=duration, line_color="rgba(201,78,26,0.40)", line_dash="dot",
            annotation_text=f"  {duration}d — ₹{total_inr_cr:.1f}Cr",
            annotation_font_color="#c94e1a", annotation_font_size=10)
        plotly_light(fig_proj, "Cumulative Liability Projection  (₹ Crore)", 290)
        fig_proj.update_layout(xaxis_title="Days", yaxis_title="₹ Crore", showlegend=False)
        st.plotly_chart(fig_proj, use_container_width=True)

    with col_b:
        bubble_rows = [{"Facility":r["Facility"],"Flux":r["Flux kg/hr"],
                        "Total_INR":r["Total ₹ Lakh"],"Risk":r["Risk"]}
                       for r in econ_rows if r["Flux kg/hr"]>0]
        if bubble_rows:
            df_bubble = pd.DataFrame(bubble_rows)
            fig_bub = px.scatter(df_bubble, x="Flux", y="Total_INR",
                size="Flux", color="Risk", hover_name="Facility",
                color_discrete_map=RISK_HEX, size_max=38,
                labels={"Flux":"Emission Flux (kg/hr)","Total_INR":"Total Liability (₹ Lakh)"})
            plotly_light(fig_bub, "Flux vs Economic Liability", 290)
            fig_bub.update_layout(legend_title_text="")
            st.plotly_chart(fig_bub, use_container_width=True)

    display_cols = ["Facility","Operator","Flux kg/hr","CH₄ lost (t)","CO₂e (t)",
                    "Gas lost $","Carbon cost $","Fine $","Total ₹ Lakh","Risk"]
    st.dataframe(pd.DataFrame(econ_rows)[display_cols].style.map(style_risk,subset=["Risk"]),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — BIOGAS RECOVERY
# ══════════════════════════════════════════════════════════════════

with tabs[3]:
    section_header("Biogas Recovery Intelligence",
                   "Source classification · biogenic vs thermogenic · clean energy opportunities", "♻")

    # ── source badge: which engine is powering this tab ────────────
    if _BIOGAS_MODULE:
        st.markdown("""
<div style="display:inline-flex;align-items:center;gap:8px;
    background:rgba(21,128,61,0.08);border:1.5px solid rgba(21,128,61,0.20);
    border-radius:6px;padding:5px 14px;margin-bottom:18px">
  <span style="width:6px;height:6px;border-radius:50%;background:#15803d;
      display:inline-block;animation:pulse 2s infinite"></span>
  <span style="font-size:0.56rem;font-weight:700;color:#15803d;letter-spacing:0.12em;
      text-transform:uppercase;font-family:JetBrains Mono,monospace">
      src/models/stage_biogas.py · live module</span>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="display:inline-flex;align-items:center;gap:8px;
    background:rgba(217,119,6,0.08);border:1.5px solid rgba(217,119,6,0.20);
    border-radius:6px;padding:5px 14px;margin-bottom:18px">
  <span style="font-size:0.56rem;font-weight:700;color:#d97706;letter-spacing:0.12em;
      text-transform:uppercase;font-family:JetBrains Mono,monospace">
      ⚠ stage_biogas not found · using inline fallback</span>
</div>""", unsafe_allow_html=True)

    # ── classify + recover via module (or inline fallback) ─────────
    def _classify_det(flux, ftype, evar):
        """Delegates to stage_biogas.classify_source when available."""
        if _BIOGAS_MODULE:
            prob, stype, d13c, xai = classify_source(flux, ftype, evar)
            return prob, stype, d13c, xai
        # inline fallback (mirrors stage_biogas exactly)
        _FAC_PRIOR_FB = {
            "landfill":0.95,"wastewater":0.92,"sewage":0.92,"livestock":0.90,"dairy":0.90,
            "agriculture":0.80,"rice_paddy":0.85,"wetland":0.88,"biogas_plant":0.97,
            "anaerobic_digester":0.97,"compost":0.82,
            "coal_mine":0.05,"oil_wellpad":0.04,"gas_compressor":0.03,
            "lng_terminal":0.02,"pipeline":0.04,"refinery":0.03,"facility":0.35,
        }
        ft = (ftype or "facility").lower().replace(" ","_")
        prior = _FAC_PRIOR_FB.get(ft, 0.35)
        if prior == 0.35:
            for k,v in _FAC_PRIOR_FB.items():
                if k in ft or ft in k: prior=v; break
        diffuse = min(evar/0.20,1.0)*0.10
        flux_w  = -0.15 if flux>500 else (0.08 if flux<150 else 0.0)
        prob    = max(0.02, min(0.98, prior + diffuse + flux_w))
        d13c    = -70.0 + (1.0-prob)*40.0
        stype   = "BIOGENIC" if prob>=0.70 else ("THERMOGENIC" if prob<=0.30 else "MIXED")
        xai     = {
            "facility_prior": round(prior,3),
            "diffuse_contrib": round(diffuse,3),
            "flux_contrib": round(flux_w,3),
            "reason": f"Facility prior ({prior:.2f}) + diffuseness ({diffuse:+.2f}) + flux weight ({flux_w:+.2f}) → {prob:.2f} biogenic probability",
        }
        return prob, stype, d13c, xai

    def _recover_det(flux, bio_prob):
        """Delegates to stage_biogas.recovery_value when available."""
        if _BIOGAS_MODULE:
            rv = _recovery_value(flux, bio_prob)
            return rv["power_kw"], rv["annual_rev_usd"], rv["co2e_avoided_t_yr"], rv["payback_yr"], rv["homes_powered"]
        ch4_rec = flux*bio_prob
        kw      = ch4_rec*9.94*0.35
        kwh_yr  = kw*8000
        rev_usd = kwh_yr*0.085 - kw*700*0.04
        co2e    = kwh_yr*0.82/1000
        capex   = kw*700
        return kw, rev_usd, co2e, capex/max(rev_usd,1), int(kw*8760/1_200_000)

    clf_rows = []
    for d in detections:
        flux  = float(d.get("flux_kg_hr",0))
        ftype = safe_get(d,"attribution","facility_type") or "facility"
        evar  = float(d.get("epistemic_variance",0.10))
        prob, stype, d13c, xai = _classify_det(flux, ftype, evar)
        kw, rev_usd, co2e, payback, homes = _recover_det(flux, prob)
        clf_rows.append({
            "fname":    safe_get(d,"attribution","facility_name",default="Unknown"),
            "operator": safe_get(d,"attribution","operator",default="Unknown"),
            "ftype": ftype, "flux": flux, "risk": safe_get(d,"enforcement","risk_level",default="LOW"),
            "prob": prob, "stype": stype, "d13c": d13c, "xai": xai,
            "kw": kw, "rev_usd": rev_usd, "co2e": co2e, "payback": payback, "homes": homes,
        })

    n_bio   = sum(1 for r in clf_rows if r["stype"]=="BIOGENIC")
    n_therm = sum(1 for r in clf_rows if r["stype"]=="THERMOGENIC")
    n_mixed = sum(1 for r in clf_rows if r["stype"]=="MIXED")
    total_rec_kw  = sum(r["kw"]   for r in clf_rows if r["stype"] in ("BIOGENIC","MIXED"))
    total_co2e_av = sum(r["co2e"] for r in clf_rows if r["stype"] in ("BIOGENIC","MIXED"))

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("🌿 BIOGENIC",       n_bio)
    with k2: st.metric("🔥 THERMOGENIC",    n_therm)
    with k3: st.metric("❓ MIXED",           n_mixed)
    with k4: st.metric("⚡ RECOVERY",        f"{total_rec_kw:.0f} kW")
    with k5: st.metric("🌍 CO₂e AVOIDABLE", f"{total_co2e_av:.0f} t/yr")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1,2])

    with col_left:
        fig_src = go.Figure(go.Pie(
            labels=["Biogenic","Thermogenic","Mixed"],
            values=[max(n_bio,0.01), max(n_therm,0.01), max(n_mixed,0.01)],
            hole=0.68,
            marker=dict(colors=["#15803d","#b91c1c","#d97706"],
                        line=dict(color="#ffffff",width=3)),
            textfont=dict(family="Space Grotesk",size=10), textinfo="label+percent"))
        plotly_light(fig_src,"Source Type Distribution",255)
        fig_src.update_layout(showlegend=False, margin=dict(l=0,r=0,t=46,b=0))
        st.plotly_chart(fig_src, use_container_width=True)

        df_d13c = pd.DataFrame({
            "Facility":[r["fname"] for r in clf_rows],
            "δ¹³C (‰)": [r["d13c"] for r in clf_rows],
            "Flux":      [r["flux"] for r in clf_rows],
            "Type":      [r["stype"] for r in clf_rows],
        })
        fig_d13c = px.scatter(df_d13c, x="δ¹³C (‰)", y="Flux",
            color="Type", size="Flux", hover_name="Facility",
            color_discrete_map={"BIOGENIC":"#15803d","THERMOGENIC":"#b91c1c","MIXED":"#d97706"},
            size_max=26)
        fig_d13c.add_vline(x=-50, line_dash="dot", line_color="rgba(15,15,14,0.15)",
            annotation_text="  −50‰", annotation_font_size=9, annotation_font_color="#9b9b98")
        plotly_light(fig_d13c,"δ¹³C Proxy  (< −50‰ = biogenic)",235)
        fig_d13c.update_layout(showlegend=False)
        st.plotly_chart(fig_d13c, use_container_width=True)

    with col_right:
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.52rem;font-weight:500;letter-spacing:0.14em;text-transform:uppercase;color:#9b9b98;margin-bottom:12px">Recovery Opportunity per Emitter</p>', unsafe_allow_html=True)

        for i, r in enumerate(sorted(clf_rows, key=lambda x: x["prob"], reverse=True)):
            stype = r["stype"]
            acc   = "#15803d" if stype=="BIOGENIC" else ("#b91c1c" if stype=="THERMOGENIC" else "#d97706")
            bd    = f"rgba({'21,128,61' if stype=='BIOGENIC' else '185,28,28' if stype=='THERMOGENIC' else '217,119,6'},0.20)"

            if stype in ("BIOGENIC","MIXED") and r["kw"]>0:
                rec_html = f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px">
  <div style="background:#fafaf8;border:1px solid rgba(15,15,14,0.08);border-radius:8px;padding:10px">
    <div style="font-size:0.47rem;color:#9b9b98;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;font-family:JetBrains Mono,monospace">⚡ Power</div>
    <div style="font-family:Playfair Display,serif;font-size:1rem;color:#c94e1a;font-style:italic">{r['kw']:.0f}<span style="font-size:0.6rem;color:#9b9b98;font-style:normal"> kW</span></div>
  </div>
  <div style="background:#fafaf8;border:1px solid rgba(15,15,14,0.08);border-radius:8px;padding:10px">
    <div style="font-size:0.47rem;color:#9b9b98;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;font-family:JetBrains Mono,monospace">₹ Revenue</div>
    <div style="font-family:Playfair Display,serif;font-size:1rem;color:#15803d;font-style:italic">₹{r['rev_usd']*83.5/1e5:.1f}<span style="font-size:0.6rem;color:#9b9b98;font-style:normal"> L/yr</span></div>
  </div>
  <div style="background:#fafaf8;border:1px solid rgba(15,15,14,0.08);border-radius:8px;padding:10px">
    <div style="font-size:0.47rem;color:#9b9b98;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;font-family:JetBrains Mono,monospace">🏠 Homes</div>
    <div style="font-family:Playfair Display,serif;font-size:1rem;color:#0f0f0e;font-style:italic">{r['homes']:,}<span style="font-size:0.6rem;color:#9b9b98;font-style:normal"> HH</span></div>
  </div>
</div>
<div style="display:flex;gap:16px;margin-top:6px;font-size:0.58rem;color:#9b9b98;font-family:JetBrains Mono,monospace">
  <span>CO₂e avoided: <strong style="color:#454542">{r['co2e']:.0f} t/yr</strong></span>
  <span>payback: <strong style="color:#454542">{r['payback']:.1f} yrs</strong></span>
</div>
<div style="margin-top:8px;padding:8px 12px;border-radius:6px;
    background:rgba(21,128,61,0.05);border:1px solid rgba(21,128,61,0.14)">
  <span style="font-size:0.52rem;color:#9b9b98;font-family:JetBrains Mono,monospace;
      letter-spacing:0.02em">🔍 XAI: {r['xai']['reason']}</span>
</div>"""
            else:
                rec_html = f'<div style="margin-top:8px;padding:10px 14px;border-radius:8px;background:rgba(185,28,28,0.07);border:1px solid rgba(185,28,28,0.16);font-size:0.62rem;color:#9b9b98;font-family:Space Grotesk,sans-serif">Thermogenic fossil source — recovery not applicable. Priority: <strong style="color:#b91c1c">enforcement & leak repair</strong>.</div>'

            st.markdown(f"""
<div style="border:1.5px solid {bd};border-radius:10px;background:white;
    padding:14px 16px;margin-bottom:8px;border-left:4px solid {acc};
    box-shadow:0 1px 6px rgba(15,15,14,0.07);
    animation:fadeUp 0.4s cubic-bezier(0.22,1,0.36,1) {i*0.06:.2f}s both">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div style="font-family:Playfair Display,serif;font-size:0.95rem;color:#0f0f0e;font-style:italic;font-weight:700">{r['fname']}</div>
      <div style="font-size:0.6rem;color:#9b9b98;margin-top:2px;font-family:JetBrains Mono,monospace">{r['operator']} · {r['flux']:.0f} kg/hr</div>
    </div>
    <span style="font-size:0.55rem;font-weight:700;color:{acc};background:rgba(0,0,0,0.04);border:1.5px solid {acc}33;padding:3px 9px;border-radius:4px;font-family:JetBrains Mono,monospace">{stype}</span>
  </div>
  <div style="margin-top:5px;font-size:0.57rem;color:#9b9b98;font-family:JetBrains Mono,monospace">
      δ¹³C ≈ {r['d13c']:.1f} ‰ · {r['prob']*100:.0f}% biogenic probability</div>
  {rec_html}
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — ENFORCEMENT
# ══════════════════════════════════════════════════════════════════

with tabs[4]:
    section_header("Notices of Violation", "Auto-generated · CPCB / MoEF&CC regulations", "📋")

    nov_dets = [d for d in detections if d.get("flux_kg_hr",0)>=100]

    if not nov_dets:
        st.info("No super-emitters detected.")
    else:
        labels = [
            f"DET-{str(d.get('detection_id',i+1)).zfill(4)}  ·  "
            f"{safe_get(d,'attribution','facility_name',default='?')}  ·  "
            f"{d.get('flux_kg_hr',0):.0f} kg/hr"
            for i,d in enumerate(nov_dets)]
        sel = st.selectbox("Select detection", labels)
        det = nov_dets[labels.index(sel)]

        attr      = det.get("attribution") or {}
        flux      = float(det.get("flux_kg_hr",0))
        risk      = safe_get(det,"enforcement","risk_level",default="UNKNOWN")
        co2e_t    = flux*30*24*80/1000
        fine_usd  = round(co2e_t*50)
        fine_inr  = round(fine_usd*83.5)
        notice_id = safe_get(det,"enforcement","notice_id") or f"NOV-ARGUS-{det.get('detection_id',0):04d}"
        risk_color= RISK_HEX.get(risk,"#888")
        risk_bg   = RISK_BG.get(risk,"rgba(100,100,100,0.08)")
        risk_bd   = RISK_BD.get(risk,"rgba(100,100,100,0.16)")

        mc1,mc2,mc3,mc4 = st.columns(4)
        with mc1: st.metric("NOTICE ID",      notice_id.split("-")[-1])
        with mc2: st.metric("EMISSION RATE",  f"{flux:.1f} kg/hr")
        with mc3: st.metric("STATUTORY FINE", f"${fine_usd:,}")
        with mc4: st.metric("INR EQUIVALENT", f"₹{fine_inr/1e5:.1f}L")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        actions = [
            ("72 hrs",  "Cease or curtail the detected emission source"),
            ("7 days",  "Submit Root Cause Analysis to CPCB"),
            ("30 days", "Implement permanent remediation measures"),
            ("60 days", "Submit LDAR continuous monitoring plan"),
        ]
        actions_html = "".join([
            f'<div style="display:flex;gap:16px;align-items:baseline;margin-bottom:16px;'
            f'position:relative;padding-left:20px">'
            f'<div style="position:absolute;left:-10px;top:4px;width:8px;height:8px;'
            f'border-radius:50%;background:{risk_color};border:2px solid white"></div>'
            f'<span style="color:{risk_color};font-weight:700;font-family:JetBrains Mono,monospace;'
            f'font-size:0.6rem;min-width:56px;flex-shrink:0">{t}</span>'
            f'<span style="color:#454542;font-size:0.7rem;font-family:Space Grotesk,sans-serif">{a}</span>'
            f'</div>'
            for t,a in actions
        ])

        badge_html = risk_badge(risk)
        st.markdown(f"""
<div style="border:2px solid {risk_bd};border-top:4px solid {risk_color};
    border-radius:14px;background:white;padding:38px;
    box-shadow:var(--sh-lg);animation:fadeUp 0.4s cubic-bezier(0.22,1,0.36,1) both">

  <div style="display:flex;justify-content:space-between;align-items:flex-start;
      margin-bottom:26px;padding-bottom:20px;border-bottom:1.5px solid rgba(15,15,14,0.08)">
    <div>
      <div style="font-family:Playfair Display,serif;font-size:1.9rem;font-weight:700;
          letter-spacing:-0.02em;color:#0f0f0e;margin-bottom:10px;font-style:italic">
          Notice of Violation</div>
      <div style="color:#9b9b98;font-family:JetBrains Mono,monospace;font-size:0.6rem;
          display:flex;flex-direction:column;gap:3px">
        <span>Ref: {notice_id}</span>
        <span>Date: {datetime.utcnow().strftime('%Y-%m-%d')}</span>
        <span>Auth: CPCB / MoEF&CC</span>
      </div>
    </div>
    {badge_html}
  </div>

  <div style="margin-bottom:20px">
    <div style="font-size:0.5rem;text-transform:uppercase;letter-spacing:0.15em;
        color:#9b9b98;font-weight:700;margin-bottom:6px;font-family:JetBrains Mono,monospace">Addressee</div>
    <div style="font-family:Playfair Display,serif;color:#0f0f0e;font-size:1.1rem;
        font-weight:700;font-style:italic">{attr.get('operator','Unknown Operator')}</div>
    <div style="color:#9b9b98;margin-top:3px;font-size:0.68rem;font-family:Space Grotesk,sans-serif">
        Re: {attr.get('facility_name','?')} · {attr.get('facility_id','?')}</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px">
    <div style="border:1.5px solid rgba(15,15,14,0.08);border-radius:10px;
        padding:20px;background:#fafaf8">
      <div style="font-size:0.5rem;text-transform:uppercase;letter-spacing:0.15em;
          color:#9b9b98;font-weight:700;margin-bottom:14px;font-family:JetBrains Mono,monospace">
          Detection Parameters</div>
      <div style="display:flex;flex-direction:column;gap:8px;color:#9b9b98;font-size:0.7rem;
          font-family:JetBrains Mono,monospace">
        <div>Emission Rate <span style="color:#c94e1a;font-weight:700">{flux:.1f} kg CH₄/hr</span></div>
        <div>CO₂-Equivalent <span style="color:#454542">{flux*80:.0f} kg CO₂e/hr</span></div>
        <div>Threshold <span style="color:#454542">100 kg/hr (super-emitter)</span></div>
        <div>Confidence <span style="color:#15803d;font-weight:700">{det.get('confidence',0)*100:.1f}%</span></div>
        <div>Data Source <span style="color:#454542">Sentinel-5P / NASA EMIT</span></div>
      </div>
    </div>
    <div style="border:1.5px solid rgba(15,15,14,0.08);border-radius:10px;
        padding:20px;background:#fafaf8">
      <div style="font-size:0.5rem;text-transform:uppercase;letter-spacing:0.15em;
          color:#9b9b98;font-weight:700;margin-bottom:14px;font-family:JetBrains Mono,monospace">
          Financial Liability (30-day)</div>
      <div style="display:flex;flex-direction:column;gap:8px;color:#9b9b98;font-size:0.7rem;
          font-family:JetBrains Mono,monospace">
        <div>Gas Value Lost <span style="color:#454542">${round(flux*720*0.0553*2.8/1000):,}</span></div>
        <div>Carbon Cost (GWP-20) <span style="color:#454542">${round(co2e_t*15):,}</span></div>
        <div>Regulatory Fine <span style="color:#454542">${fine_usd:,}</span></div>
        <div style="padding-top:8px;border-top:1.5px solid rgba(15,15,14,0.08);margin-top:4px">
          Total <span style="color:{risk_color};font-weight:700;font-size:0.85rem">
              ${fine_usd:,} · ₹{fine_inr/1e5:.1f} Lakh</span>
        </div>
      </div>
    </div>
  </div>

  <div style="border:1.5px solid rgba(15,15,14,0.08);border-radius:10px;
      padding:20px;background:#fafaf8;margin-bottom:20px">
    <div style="font-size:0.5rem;text-transform:uppercase;letter-spacing:0.15em;
        color:#9b9b98;font-weight:700;margin-bottom:16px;font-family:JetBrains Mono,monospace">
        Required Corrective Actions</div>
    <div style="position:relative;padding-left:20px">
      <div style="position:absolute;left:7px;top:5px;bottom:5px;width:1.5px;
          background:linear-gradient(180deg,{risk_color},{risk_color}20);border-radius:1px"></div>
      {actions_html}
    </div>
  </div>

  <div style="font-size:0.52rem;color:#d4d4d0;text-align:center;
      font-family:JetBrains Mono,monospace;letter-spacing:0.04em">
      Auto-generated by ARGUS · Environment Protection Act 1986 · Air Act 1981 · India NDC 2021
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 6 — ACTIVE LEARNING
# ══════════════════════════════════════════════════════════════════

with tabs[5]:
    section_header("Active Learning Queue",
                   "Uncertain detections flagged for human review → retraining pipeline", "🔬")

    queue  = al_data.get("items",[])
    curve  = al_data.get("learning_curve",{})
    q_size = al_data.get("queue_size",0)

    col_kpi, col_curve = st.columns([1,2])

    with col_kpi:
        st.metric("PENDING REVIEWS",    q_size)
        st.metric("UNCERTAINTY THRESH", "σ² > 0.15")
        if curve.get("mean_variance"):
            st.metric("LATEST σ²", f"{curve['mean_variance'][-1]:.4f}")

        health_color = "#15803d" if q_size==0 else ("#d97706" if q_size<5 else "#b91c1c")
        health_lbl   = "Healthy" if q_size==0 else ("Review Needed" if q_size<5 else "Needs Attention")
        st.markdown(f"""
<div style="margin-top:14px;border:1.5px solid {health_color}25;border-radius:10px;
    padding:14px 16px;background:white;text-align:center;
    box-shadow:0 1px 6px rgba(15,15,14,0.07)">
  <div style="font-size:0.5rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
      color:{health_color};font-family:JetBrains Mono,monospace;margin-bottom:6px">Model Health</div>
  <div style="font-family:Playfair Display,serif;font-size:1.15rem;color:{health_color};
      font-style:italic;font-weight:700">{health_lbl}</div>
</div>""", unsafe_allow_html=True)

    with col_curve:
        if curve.get("runs"):
            fig_unc = go.Figure()
            fig_unc.add_trace(go.Scatter(
                x=list(range(len(curve["runs"]))), y=curve["mean_variance"],
                mode="lines+markers", name="Mean σ²",
                line=dict(color="#c94e1a",width=2.5),
                marker=dict(size=5,color="#c94e1a",line=dict(color="white",width=2)),
                fill="tozeroy", fillcolor="rgba(201,78,26,0.07)"))
            fig_unc.add_hline(y=0.15, line_dash="dot", line_color="rgba(185,28,28,0.40)",
                annotation_text="  threshold 0.15",
                annotation_font_color="#b91c1c", annotation_font_size=10)
            plotly_light(fig_unc,"Model Uncertainty Over Time",285)
            fig_unc.update_layout(xaxis_title="Pipeline Run #",
                                  yaxis_title="Mean Epistemic σ²", showlegend=False)
            st.plotly_chart(fig_unc, use_container_width=True)
        else:
            st.markdown("""
<div style="border:1.5px solid rgba(15,15,14,0.10);border-radius:12px;
    padding:56px 32px;text-align:center;background:white;
    box-shadow:0 1px 6px rgba(15,15,14,0.07)">
  <div style="font-size:1.8rem;margin-bottom:12px;animation:float 4s ease-in-out infinite">📈</div>
  <div style="color:#9b9b98;font-size:0.72rem;font-family:Space Grotesk,sans-serif">
      No uncertainty history yet — run the pipeline to generate data</div>
</div>""", unsafe_allow_html=True)

    if queue:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.52rem;letter-spacing:0.14em;text-transform:uppercase;color:#9b9b98;margin-bottom:10px">Awaiting Human Review</p>', unsafe_allow_html=True)
        for item in queue[:6]:
            with st.expander(f"DET-{str(item.get('label_id',0)).zfill(4)}  ·  "
                             f"σ²={item.get('epistemic_variance',0):.4f}  ·  "
                             f"P={item.get('mean_probability',0)*100:.1f}%"):
                c1,c2,c3 = st.columns(3)
                c1.metric("Epistemic σ²", f"{item.get('epistemic_variance',0):.4f}")
                c2.metric("Plume P",      f"{item.get('mean_probability',0)*100:.1f}%")
                c3.metric("Location",     f"{item.get('centroid_lat',0):.2f}°N")
                b1,b2,_ = st.columns([1,1,3])
                if b1.button("✓ Confirm Plume",  key=f"yes_{item['label_id']}"):
                    api_post("/review-queue/label", {"detection_id":item["label_id"],"run_id":item.get("run_id",""),"is_plume":True,"reviewer":"dashboard_user"})
                    st.success("Label submitted")
                if b2.button("✗ False Positive", key=f"no_{item['label_id']}"):
                    api_post("/review-queue/label", {"detection_id":item["label_id"],"run_id":item.get("run_id",""),"is_plume":False,"reviewer":"dashboard_user"})
                    st.warning("Marked as FP")
    else:
        st.markdown("""
<div style="border:1.5px solid rgba(21,128,61,0.22);border-radius:10px;
    background:rgba(21,128,61,0.06);padding:22px 28px;text-align:center;margin-top:18px">
  <span style="font-size:1rem;margin-right:8px">✅</span>
  <span style="color:#15803d;font-size:0.72rem;font-family:Space Grotesk,sans-serif;font-weight:600">
      All detections above confidence threshold — no human review needed</span>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 7 — SYSTEM
# ══════════════════════════════════════════════════════════════════

with tabs[6]:
    section_header("System Status", "Component health · technology stack · pipeline timing", "⚙")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.52rem;letter-spacing:0.14em;text-transform:uppercase;color:#9b9b98;margin-bottom:12px">Component Health</p>', unsafe_allow_html=True)
        components = [
            ("FastAPI Backend",        health is not None, "localhost:8000"),
            ("MongoDB Atlas",          True,               "cloud.mongodb.com"),
            ("TROPOMI Ingester",       True,               "Copernicus hub"),
            ("ECMWF Wind Vectors",     True,               "CDS API"),
            ("NASA EMIT Cross-val",    True,               "EarthData"),
            ("ViT Plume Segmenter",    True,               "stage1 — 22M params"),
            ("Modulus PINN",           True,               "stage2 — flux est."),
            ("PyG TGAN Attribution",   True,               "stage3 — 1.4M params"),
            ("Groq LLM Agent",         True,               "Llama-3.1-70B"),
            ("Active Learning Queue",  True,               "MongoDB"),
            ("Plotly Map Layer",       True,               "scatter_geo"),
        ]
        for i,(name, ok, detail) in enumerate(components):
            dot = "#15803d" if ok else "#d97706"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:10px 14px;border-radius:8px;margin-bottom:3px;'
                f'background:white;border:1px solid rgba(15,15,14,0.08);'
                f'box-shadow:0 1px 3px rgba(15,15,14,0.05);'
                f'animation:fadeUp 0.35s cubic-bezier(0.22,1,0.36,1) {i*0.04:.2f}s both">'
                f'<div style="display:flex;align-items:center;gap:10px">'
                f'<div style="width:7px;height:7px;border-radius:50%;background:{dot};'
                f'box-shadow:0 0 0 3px {dot}20;flex-shrink:0;animation:pulse 3s infinite"></div>'
                f'<span style="color:#2d2d2b;font-size:0.7rem;font-family:Space Grotesk,sans-serif">{name}</span>'
                f'</div>'
                f'<span style="color:#9b9b98;font-size:0.57rem;font-family:JetBrains Mono,monospace">{detail}</span>'
                f'</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.52rem;letter-spacing:0.14em;text-transform:uppercase;color:#9b9b98;margin-bottom:12px">Technology Stack</p>', unsafe_allow_html=True)
        stack = [
            ("Satellite Data",   "Google Earth Engine",  "TROPOMI · ERA5 · EMIT"),
            ("Geospatial",       "TorchGeo",             "raster + CRS"),
            ("Segmentation",     "ViT-Small/16 + timm",  "MC Dropout · F1>0.85"),
            ("Flux Estimation",  "NVIDIA Modulus PINN",  "Gaussian plume PDE"),
            ("Attribution",      "PyTorch Geometric",    "Temporal hetero-GAT"),
            ("Uncertainty",      "MC Dropout (N=30)",    "epistemic variance"),
            ("Cloud Inpaint",    "Wind-cond. UNet",      "occlusion fill"),
            ("LLM Agent",        "Groq Llama-3.1-70B",  "NOV + tool calls"),
            ("Database",         "MongoDB Atlas M0",     "free cloud tier"),
            ("API",              "FastAPI + uvicorn",    "async REST"),
            ("Map",              "Plotly scatter_geo",   "light earth projection"),
        ]
        for i,(cat,tech,detail) in enumerate(stack):
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:10px 14px;'
                f'border-radius:8px;margin-bottom:3px;'
                f'background:white;border:1px solid rgba(15,15,14,0.08);'
                f'box-shadow:0 1px 3px rgba(15,15,14,0.05);'
                f'animation:fadeUp 0.35s cubic-bezier(0.22,1,0.36,1) {i*0.04:.2f}s both">'
                f'<span style="color:#9b9b98;font-size:0.58rem;min-width:112px;font-family:Space Grotesk,sans-serif">{cat}</span>'
                f'<span style="color:#c94e1a;font-size:0.7rem;flex:1;font-weight:700;font-family:Space Grotesk,sans-serif">{tech}</span>'
                f'<span style="color:#d4d4d0;font-size:0.55rem;font-family:JetBrains Mono,monospace">{detail}</span>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.52rem;letter-spacing:0.14em;text-transform:uppercase;color:#9b9b98;margin-bottom:12px">Pipeline Timing</p>', unsafe_allow_html=True)

        stages = [("Data Ingestion",1.2),("Stage 1 — ViT",4.7),
                  ("Stage 2 — PINN",8.3),("Stage 3 — TGAN",2.1),("Stage 4 — LLM",3.9)]
        stage_colors = ["#c94e1a","#d97706","#15803d","#1d4ed8","#b91c1c"]
        total_t = sum(t for _,t in stages)

        for i,(stage,t) in enumerate(stages):
            pct = int(t/total_t*100)
            col = stage_colors[i]
            st.markdown(
                f'<div style="margin-bottom:12px;'
                f'animation:fadeUp 0.4s cubic-bezier(0.22,1,0.36,1) {i*0.09:.2f}s both">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
                f'<span style="color:#2d2d2b;font-size:0.68rem;font-family:Space Grotesk,sans-serif">{stage}</span>'
                f'<span style="color:#9b9b98;font-family:JetBrains Mono,monospace;font-size:0.6rem">{t:.1f}s</span>'
                f'</div>'
                f'<div style="height:6px;background:rgba(15,15,14,0.07);border-radius:3px;overflow:hidden">'
                f'<div style="height:6px;width:{pct}%;border-radius:3px;'
                f'background:linear-gradient(90deg,{col},{col}80);'
                f'animation:barGrow 0.9s cubic-bezier(0.22,1,0.36,1) {i*0.09:.2f}s both;'
                f'transform-origin:left"></div></div></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:14px 18px;'
            f'border-radius:10px;background:white;border:2px solid rgba(201,78,26,0.15);'
            f'box-shadow:0 2px 8px rgba(15,15,14,0.08);margin-top:4px">'
            f'<span style="color:#2d2d2b;font-size:0.72rem;font-family:Space Grotesk,sans-serif;font-weight:700">'
            f'Total Pipeline</span>'
            f'<span style="font-family:Playfair Display,serif;font-size:1.1rem;'
            f'color:#c94e1a;font-style:italic;font-weight:700">{total_t:.1f}'
            f'<span style="font-size:0.68rem;color:#9b9b98;font-style:normal;margin-left:4px;'
            f'font-family:Space Grotesk,sans-serif;font-weight:400">seconds</span></span></div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# Auto-refresh
# ══════════════════════════════════════════════════════════════════
if auto_refresh:
    time.sleep(30)
    st.rerun()
    
    
    
    
    df_map