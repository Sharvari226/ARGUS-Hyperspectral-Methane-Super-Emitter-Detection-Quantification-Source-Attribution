"""
src/agents/stage4_llm.py  — Groq edition
─────────────────────────────────────────
Uses Groq API (free tier) with Llama-3.3-70B.
OpenAI-compatible interface — same tool calling pattern.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

from openai import OpenAI
from loguru import logger

from src.utils.config import cfg
from src.data.facility_db import get_facility_by_id
from src.models.stage2_economics import calculate_economic_impact, EconomicImpact
from src.models.stage3_tgan import AttributionResult


# ── Model — updated from decommissioned llama-3.1-70b-versatile ──
_DEFAULT_MODEL = "llama-3.3-70b-versatile"


# ── Groq client ───────────────────────────────────────────────────

def _get_client() -> OpenAI | None:
    api_key = cfg["env"].get("groq_api_key", "")
    if not api_key:
        logger.warning("Stage4: No GROQ_API_KEY — LLM enforcement will use mock")
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


# ── Tool definitions (OpenAI function-calling format) ─────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_operator",
            "description": "Retrieve full operator and facility profile from the ARGUS registry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "facility_id": {"type": "string", "description": "Facility identifier e.g. FAC-0042"},
                },
                "required": ["facility_id"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_penalty",
            "description": "Calculate full regulatory penalty for a super-emitter event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flux_kg_hr":     {"type": "number"},
                    "duration_days":  {"type": "number"},
                },
                "required": ["flux_kg_hr", "duration_days"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_historical_violations",
            "description": "Query prior emission events for an operator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operator_name": {"type": "string"},
                    "lookback_days": {"type": "integer", "default": 365},
                },
                "required": ["operator_name"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_notice",
            "description": "Generate a formal Notice of Violation (NOV).",
            "parameters": {
                "type": "object",
                "properties": {
                    "operator":        {"type": "string"},
                    "facility_id":     {"type": "string"},
                    "facility_name":   {"type": "string"},
                    "flux_kg_hr":      {"type": "number"},
                    "duration_days":   {"type": "number"},
                    "total_fine_usd":  {"type": "number"},
                    "facility_type":   {"type": "string"},
                    "co2e_tonnes":     {"type": "number"},
                    "total_fine_inr":  {"type": "number"},
                    "detection_date":  {"type": "string"},
                    "confidence_pct":  {"type": "number"},
                    "violations_12mo": {"type": "integer"},
                },
                "required": ["operator", "facility_id", "facility_name", "flux_kg_hr", "duration_days", "total_fine_usd"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assess_climate_risk",
            "description": "Generate climate risk assessment for a super-emitter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flux_kg_hr":    {"type": "number"},
                    "facility_type": {"type": "string"},
                    "country":       {"type": "string"},
                    "duration_days": {"type": "number"},
                },
                "required": ["flux_kg_hr", "facility_type"],
            },
        }
    },
]

SYSTEM_PROMPT = """
You are ARGUS, an autonomous regulatory enforcement AI for methane super-emitter monitoring.
You have tools to look up facilities, calculate penalties, query violations, assess climate risk,
and draft formal Notices of Violation.

When given a detection event, you MUST call all 5 tools in this order:
1. lookup_operator
2. calculate_penalty
3. query_historical_violations
4. assess_climate_risk
5. draft_notice

Be precise with numbers. Reference Indian environmental regulations where applicable.
Always complete all 5 steps.
""".strip()


# ── Tool executor ─────────────────────────────────────────────────

class ToolExecutor:

    @staticmethod
    def lookup_operator(facility_id: str) -> dict:
        record = get_facility_by_id(facility_id)
        if record is None:
            return {"error": f"Facility {facility_id} not found"}
        record["regulatory_contact"] = {
            "authority": "Central Pollution Control Board (CPCB)",
            "email":     "enforcement@cpcb.nic.in",
            "phone":     "+91-11-43102030",
        }
        record["outstanding_actions"] = (
            "Prior NOV issued — compliance deadline missed"
            if record.get("violations_12mo", 0) >= 3 else "None"
        )
        return record

    @staticmethod
    def calculate_penalty(flux_kg_hr: float, duration_days: float) -> dict:
        impact: EconomicImpact = calculate_economic_impact(
            flux_kg_hr=flux_kg_hr, duration_hours=duration_days * 24
        )
        return {
            "flux_kg_hr":          flux_kg_hr,
            "duration_days":       duration_days,
            "ch4_emitted_tonnes":  round(flux_kg_hr * duration_days * 24 / 1000, 2),
            "co2e_tonnes":         round(impact.co2e_tonnes, 2),
            "gas_value_lost_usd":  round(impact.gas_value_usd, 2),
            "regulatory_fine_usd": round(impact.regulatory_fine_usd, 2),
            "regulatory_fine_inr": round(impact.regulatory_fine_inr, 2),
            "total_liability_usd": round(impact.total_cost_usd, 2),
            "total_liability_inr": round(impact.total_cost_inr, 2),
            "summary":             impact.summary_line,
        }

    @staticmethod
    def query_historical_violations(operator_name: str, lookback_days: int = 365) -> dict:
        import numpy as np
        rng   = np.random.default_rng(abs(hash(operator_name)) % 2**32)
        n_evt = int(rng.integers(1, 7))
        now   = datetime.utcnow()
        events = [{
            "detection_date": (now - timedelta(days=int(rng.integers(1, lookback_days)))).strftime("%Y-%m-%d"),
            "facility_id":    f"FAC-{rng.integers(0, 9999):04d}",
            "flux_kg_hr":     round(float(rng.uniform(80, 600)), 1),
            "status":         rng.choice(["NOV Issued", "Fine Paid", "Under Investigation"]),
            "fine_usd":       round(float(rng.uniform(5000, 250000))),
        } for _ in range(n_evt)]
        events.sort(key=lambda x: x["detection_date"], reverse=True)
        return {
            "operator": operator_name, "total_events": n_evt,
            "repeat_offender": n_evt >= 3, "events": events,
            "total_fines_usd": sum(e["fine_usd"] for e in events),
        }

    @staticmethod
    def draft_notice(
        operator: str, facility_id: str, facility_name: str,
        flux_kg_hr: float, duration_days: float, total_fine_usd: float,
        facility_type: str = "facility", co2e_tonnes: float = 0.0,
        total_fine_inr: float = 0.0, detection_date: str = "",
        confidence_pct: float = 95.0, violations_12mo: int = 0,
    ) -> dict:
        detection_date = detection_date or datetime.utcnow().strftime("%Y-%m-%d")
        notice_id  = f"NOV-ARGUS-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        deadline   = (datetime.utcnow() + timedelta(days=30)).strftime("%d %B %Y")
        notice_md  = f"""
# NOTICE OF VIOLATION — {notice_id}
**Date:** {detection_date} | **Authority:** CPCB / MoEF&CC

**To:** {operator} | **Facility:** {facility_name} ({facility_id})

## Violation
- Emission Rate: **{flux_kg_hr:.1f} kg CH₄/hr**
- Duration: {duration_days:.0f} days
- CO₂-Equivalent: {co2e_tonnes:.1f} tonnes CO₂e (GWP-20)
- Attribution Confidence: {confidence_pct:.1f}%

## Liability
Total Fine: **${total_fine_usd:,.0f} (₹{total_fine_inr:,.0f})**
Payment Deadline: {deadline}

## Required Actions
1. 72 hours — Cease emission source
2. 7 days — Submit RCA to CPCB
3. 30 days — Implement remediation
4. 60 days — Submit LDAR plan

*ARGUS Autonomous Regulatory Intelligence System*
        """.strip()
        logger.info(f"NOV drafted: {notice_id} for {operator}")
        return {
            "notice_id": notice_id, "notice_md": notice_md,
            "deadline": deadline, "fine_usd": total_fine_usd,
            "fine_inr": total_fine_inr, "facility_id": facility_id,
        }

    @staticmethod
    def assess_climate_risk(
        flux_kg_hr: float, facility_type: str,
        country: str = "Unknown", duration_days: float = 30.0,
    ) -> dict:
        total_ch4_t = flux_kg_hr * duration_days * 24 / 1000
        co2e_t      = total_ch4_t * 80
        priority    = (
            "P1 — CRITICAL" if flux_kg_hr >= 500 else
            "P2 — HIGH"     if flux_kg_hr >= 200 else
            "P3 — MEDIUM"   if flux_kg_hr >= 100 else
            "P4 — LOW"
        )
        return {
            "flux_kg_hr": flux_kg_hr, "co2e_tonnes": round(co2e_t, 2),
            "priority_level": priority, "facility_type": facility_type,
            "global_cars_equiv": round(co2e_t / 4.6),
        }


# ── Agentic loop ──────────────────────────────────────────────────

class ARGUSAgent:

    def __init__(self):
        self.client   = _get_client()
        self.executor = ToolExecutor()
        # Use config model but fall back to known-good model if config still has old value
        _cfg_model = cfg["stage4"].get("model", _DEFAULT_MODEL)
        self.model = _DEFAULT_MODEL if "3.1" in _cfg_model else _cfg_model

    def process_detection(
        self,
        detection:   dict,
        attribution: AttributionResult,
        flux_kg_hr:  float,
        co2e_kg_hr:  float,
    ) -> dict:
        if not self.client:
            return self._mock_result(detection, attribution, flux_kg_hr)

        user_message = (
            f"ARGUS super-emitter detection:\n"
            f"- Facility: {attribution.facility_name} ({attribution.facility_id})\n"
            f"- Operator: {attribution.operator}\n"
            f"- Flux: {flux_kg_hr:.1f} kg CH4/hr\n"
            f"- CO2e: {co2e_kg_hr:.1f} kg/hr\n"
            f"- Confidence: {attribution.confidence*100:.1f}%\n"
            f"- Location: {detection['centroid_lat']:.4f}N, {detection['centroid_lon']:.4f}E\n\n"
            f"Run the full enforcement pipeline."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        results = {
            "notice": None, "penalty": None, "history": None,
            "climate_risk": None, "operator_record": None, "final_summary": "",
        }

        for turn in range(10):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=cfg["stage4"]["max_tokens"],
                temperature=cfg["stage4"]["temperature"],
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            msg        = response.choices[0].message
            tool_calls = msg.tool_calls or []

            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]})

            if not tool_calls:
                results["final_summary"] = msg.content or ""
                break

            for tc in tool_calls:
                name   = tc.function.name
                inputs = json.loads(tc.function.arguments)
                logger.info(f"ARGUSAgent: calling {name}")

                fn = getattr(ToolExecutor, name, None)
                result_data = fn(**inputs) if fn else {"error": f"Unknown tool {name}"}

                if name == "lookup_operator":              results["operator_record"] = result_data
                elif name == "calculate_penalty":          results["penalty"]         = result_data
                elif name == "query_historical_violations": results["history"]        = result_data
                elif name == "draft_notice":               results["notice"]          = result_data
                elif name == "assess_climate_risk":        results["climate_risk"]    = result_data

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result_data),
                })

        return results

    def _mock_result(self, detection, attribution, flux_kg_hr) -> dict:
        penalty = ToolExecutor.calculate_penalty(flux_kg_hr, 30.0)
        notice  = ToolExecutor.draft_notice(
            operator=attribution.operator,
            facility_id=attribution.facility_id,
            facility_name=attribution.facility_name,
            facility_type=attribution.facility_type,
            flux_kg_hr=flux_kg_hr, duration_days=30.0,
            co2e_tonnes=penalty["co2e_tonnes"],
            total_fine_usd=penalty["total_liability_usd"],
            total_fine_inr=penalty["total_liability_inr"],
            confidence_pct=attribution.confidence * 100,
        )
        risk = ToolExecutor.assess_climate_risk(flux_kg_hr, attribution.facility_type)
        return {
            "notice": notice, "penalty": penalty, "climate_risk": risk,
            "history": None, "operator_record": None,
            "final_summary": f"NOV generated for {attribution.operator} — {flux_kg_hr:.0f} kg/hr",
        }


# ── Batch processor ───────────────────────────────────────────────

class BatchEnforcementProcessor:

    def __init__(self):
        self.agent = ARGUSAgent()

    def process_all(
        self,
        detections:   list[dict],
        attributions: list[AttributionResult],
        flux_outputs: list,
    ) -> list[dict]:
        results   = []
        threshold = cfg["pipeline"]["flux_threshold_kg_hr"]

        for det, attr, flux in zip(detections, attributions, flux_outputs):
            if flux.flux_kg_hr < threshold:
                continue
            try:
                result = self.agent.process_detection(
                    detection=det, attribution=attr,
                    flux_kg_hr=flux.flux_kg_hr, co2e_kg_hr=flux.co2e_kg_hr,
                )
                result["detection_id"] = det.get("label_id", det.get("detection_id", 0))
                result["facility_id"]  = attr.facility_id
                result["flux_kg_hr"]   = flux.flux_kg_hr
                results.append(result)
                logger.info(f"NOV: {attr.facility_name} — {flux.flux_kg_hr:.0f} kg/hr")
            except Exception as e:
                logger.error(f"BatchProcessor error {attr.facility_id}: {e}")

        return results