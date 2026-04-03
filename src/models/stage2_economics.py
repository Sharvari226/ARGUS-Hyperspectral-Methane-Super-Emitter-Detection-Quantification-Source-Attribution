from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.utils.config import cfg


@dataclass
class EconomicImpact:
    flux_kg_hr:           float
    duration_hours:       float

    # Gas value lost
    gas_lost_mmbtu:       float
    gas_value_usd:        float
    gas_value_inr:        float

    # Climate cost
    co2e_tonnes:          float
    carbon_cost_usd:      float

    # Regulatory fine (India NDC baseline)
    regulatory_fine_usd:  float
    regulatory_fine_inr:  float

    # Combined
    total_cost_usd:       float
    total_cost_inr:       float

    # Formatted strings for dashboard
    summary_line:         str
    detail_lines:         list[str]


USD_TO_INR = 83.5    # approximate — in production, fetch live rate


def calculate_economic_impact(
    flux_kg_hr:     float,
    duration_hours: float = 720.0,   # default: 30 days
) -> EconomicImpact:
    """
    Converts a flux estimate (kg/hr) and emission duration into
    full economic impact across three dimensions:
        1. Wasted gas value (market price)
        2. Carbon cost (CO₂-equivalent × carbon price)
        3. Regulatory fine (NDC-aligned penalty formula)
    """
    gcfg = cfg["intelligence"]
    gas_price    = cfg["env"]["gas_price_usd"]      # USD/MMBtu
    carbon_price = cfg["env"]["carbon_price_usd"]   # USD/tonne CO₂e
    fine_rate    = gcfg["regulatory_fine_per_ton_co2e"]  # USD/tonne

    total_kg_ch4 = flux_kg_hr * duration_hours

    # ── Gas market value ──────────────────────────────────────────
    # CH₄ energy content: 1 kg CH₄ ≈ 0.0553 MMBtu
    mmbtu_per_kg    = gcfg["co2_mmbtu_per_kg_ch4"]
    gas_lost_mmbtu  = total_kg_ch4 * mmbtu_per_kg
    gas_value_usd   = gas_lost_mmbtu * gas_price

    # ── Carbon cost ───────────────────────────────────────────────
    # GWP-20: 1 kg CH₄ = 80 kg CO₂e  → tonnes: ÷ 1000
    gwp            = gcfg["methane_gwp_20yr"]
    co2e_tonnes    = (total_kg_ch4 * gwp) / 1000.0
    carbon_cost_usd = co2e_tonnes * carbon_price

    # ── Regulatory fine ───────────────────────────────────────────
    regulatory_fine_usd = co2e_tonnes * fine_rate

    # ── Totals ────────────────────────────────────────────────────
    total_cost_usd = gas_value_usd + carbon_cost_usd + regulatory_fine_usd
    total_cost_inr = total_cost_usd * USD_TO_INR

    # ── Human-readable summary ────────────────────────────────────
    days = duration_hours / 24
    summary_line = (
        f"₹{total_cost_inr/1e7:.1f} Cr total impact "
        f"over {days:.0f} days at {flux_kg_hr:.0f} kg/hr"
    )

    detail_lines = [
        f"CH₄ emitted:        {total_kg_ch4/1000:.1f} tonnes",
        f"CO₂-equivalent:     {co2e_tonnes:.1f} tonnes (GWP-20)",
        f"Gas value lost:     ${gas_value_usd:,.0f}  (₹{gas_value_usd*USD_TO_INR/1e5:.1f}L)",
        f"Carbon cost:        ${carbon_cost_usd:,.0f}  (₹{carbon_cost_usd*USD_TO_INR/1e5:.1f}L)",
        f"Regulatory fine:    ${regulatory_fine_usd:,.0f}  (₹{regulatory_fine_usd*USD_TO_INR/1e5:.1f}L)",
        f"──────────────────────────────────────────────",
        f"TOTAL IMPACT:       ${total_cost_usd:,.0f}  (₹{total_cost_inr/1e7:.2f} Cr)",
    ]

    return EconomicImpact(
        flux_kg_hr=flux_kg_hr,
        duration_hours=duration_hours,
        gas_lost_mmbtu=gas_lost_mmbtu,
        gas_value_usd=gas_value_usd,
        gas_value_inr=gas_value_usd * USD_TO_INR,
        co2e_tonnes=co2e_tonnes,
        carbon_cost_usd=carbon_cost_usd,
        regulatory_fine_usd=regulatory_fine_usd,
        regulatory_fine_inr=regulatory_fine_usd * USD_TO_INR,
        total_cost_usd=total_cost_usd,
        total_cost_inr=total_cost_inr,
        summary_line=summary_line,
        detail_lines=detail_lines,
    )