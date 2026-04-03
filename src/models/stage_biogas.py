""" src/models/stage_biogas.py
Biogas source classifier + recovery opportunity engine + full XAI transparency.
Used by Recovery tab + Global Map for gas differentiation.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple

_FAC_PRIOR = {
    "landfill": 0.95, "wastewater": 0.92, "sewage": 0.92, "livestock": 0.90,
    "dairy": 0.90, "agriculture": 0.80, "rice_paddy": 0.85, "wetland": 0.88,
    "biogas_plant": 0.97, "anaerobic_digester": 0.97, "compost": 0.82,
    "coal_mine": 0.05, "oil_wellpad": 0.04, "gas_compressor": 0.03,
    "lng_terminal": 0.02, "pipeline": 0.04, "refinery": 0.03, "facility": 0.35,
}

def classify_source(
    flux_kg_hr: float,
    facility_type: str = "facility",
    epistemic_var: float = 0.10
) -> Tuple[float, str, float, Dict[str, Any]]:
    """Returns (biogenic_probability, source_type, d13c_proxy, xai_explanation)"""
    ftype = (facility_type or "facility").lower().replace(" ", "_")
    prior = _FAC_PRIOR.get(ftype, 0.35)
    if prior == 0.35:
        for k, v in _FAC_PRIOR.items():
            if k in ftype or ftype in k:
                prior = v
                break

    diffuse_contrib = min(epistemic_var / 0.20, 1.0) * 0.10
    flux_contrib = -0.15 if flux_kg_hr > 500 else (0.08 if flux_kg_hr < 150 else 0.0)

    prob = max(0.02, min(0.98, prior + diffuse_contrib + flux_contrib))
    d13c = -70.0 + (1.0 - prob) * 40.0

    if prob >= 0.70:
        stype = "BIOGENIC"
    elif prob <= 0.30:
        stype = "THERMOGENIC"
    else:
        stype = "MIXED"

    xai = {
        "facility_prior": round(prior, 3),
        "diffuse_contrib": round(diffuse_contrib, 3),
        "flux_contrib": round(flux_contrib, 3),
        "reason": f"Facility prior ({prior:.2f}) + diffuseness ({diffuse_contrib:+.2f}) + flux weight ({flux_contrib:+.2f}) → {prob:.2f} biogenic probability"
    }

    return prob, stype, d13c, xai


def recovery_value(flux_kg_hr: float, bio_prob: float) -> Dict[str, float]:
    ch4_rec = flux_kg_hr * bio_prob
    kw = ch4_rec * 9.94 * 0.35
    kwh_yr = kw * 8000
    rev_usd = kwh_yr * 0.085 - kw * 700 * 0.04
    co2e_avoided = kwh_yr * 0.82 / 1000
    capex = kw * 700
    payback_yr = capex / max(rev_usd, 1)
    homes_powered = int(kw * 8760 / 1_200_000)

    return {
        "power_kw": round(kw, 1),
        "annual_rev_usd": round(rev_usd),
        "co2e_avoided_t_yr": round(co2e_avoided, 1),
        "payback_yr": round(payback_yr, 2),
        "homes_powered": homes_powered,
    }