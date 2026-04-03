"""
src/pipeline/orchestrator.py  — GEE + Modulus edition
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger
from torch.linalg import det

from src.utils.config import cfg

from src.data.gee import (
    GEETROPOMIIngester,
    GEEWindIngester,
    GEEEMITIngester,
    gee_status,
)
from src.data.torchgeo_pipeline import preprocess_tropomi
from src.data.facility_db import load_facilities

from src.models.stage1_sat import (
    load_model as load_stage1,
    mc_predict,
    extract_plume_detections,
)
from src.models.stage2_pinn import (
    PINNFluxEstimator,
    apply_inpainting,
    FluxOutput,
)
from src.models.stage2_economics import calculate_economic_impact
from src.models.stage3_tgan import (
    SourceAttributor,
    build_compliance_scorecard,
    AttributionResult,
)
from src.agents.stage4_llm import BatchEnforcementProcessor
from src.agents.active_learning import ActiveLearningQueue


# ═════════════════════════════════════════════════════════════════
# Pipeline result container
# ═════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    run_id:           str
    bbox:             dict
    timestamp:        str
    duration_seconds: float

    detections:       list[dict]              = field(default_factory=list)
    flux_outputs:     list[FluxOutput]        = field(default_factory=list)
    attributions:     list[AttributionResult] = field(default_factory=list)
    enforcement:      list[dict]              = field(default_factory=list)
    scorecard:        list[dict]              = field(default_factory=list)
    review_queue:     list[dict]              = field(default_factory=list)

    emit_validated:   bool  = False
    cloud_inpainted:  bool  = False
    gee_live:         bool  = False

    n_super_emitters: int   = 0
    total_flux_kg_hr: float = 0.0
    total_co2e_kg_hr: float = 0.0
    total_impact_usd: float = 0.0
    total_impact_inr: float = 0.0

    timing:   dict      = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_api_dict(self) -> dict:
        return {
            "run_id":           self.run_id,
            "bbox":             self.bbox,
            "timestamp":        self.timestamp,
            "duration_seconds": round(self.duration_seconds, 2),
            "gee_live":         self.gee_live,
            "summary": {
                "n_detections":     len(self.detections),
                "n_super_emitters": self.n_super_emitters,
                "total_flux_kg_hr": round(self.total_flux_kg_hr, 2),
                "total_co2e_kg_hr": round(self.total_co2e_kg_hr, 2),
                "total_impact_usd": round(self.total_impact_usd, 2),
                "total_impact_inr": round(self.total_impact_inr, 2),
                "emit_validated":   self.emit_validated,
                "cloud_inpainted":  self.cloud_inpainted,
            },
            "detections": [
                {
                    "detection_id":       d["label_id"],
                    "centroid_lat":       d["centroid_lat"],
                    "centroid_lon":       d["centroid_lon"],
                    "flux_kg_hr":         round(d.get("flux_kg_hr", 0), 2),
                    "co2e_kg_hr":         round(d.get("co2e_kg_hr", 0), 2),
                    "confidence":         round(d["mean_probability"], 4),
                    "epistemic_variance": round(d["epistemic_variance"], 4),
                    "high_confidence":    d["high_confidence"],
                    "flux_uncertainty":   round(d.get("flux_uncertainty", 0), 2),
                    "attribution": {
                        "facility_id":   d.get("facility_id", ""),
                        "facility_name": d.get("facility_name", ""),
                        "operator":      d.get("operator", ""),
                        "confidence":    round(d.get("attribution_confidence", 0), 4),
                        "distance_km":   round(d.get("distance_km", 0), 2),
                    },
                    "economics":   d.get("economics", {}),
                    "enforcement": {
                        "notice_id":  d.get("notice_id", ""),
                        "risk_level": d.get("risk_level", ""),
                    },
                }
                for d in self.detections
            ],
            "scorecard":         self.scorecard,
            "review_queue_size": len(self.review_queue),
            "timing":            self.timing,
            "warnings":          self.warnings,
        }


# ═════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════

class ARGUSPipeline:

    def __init__(self, device: str = "cpu"):
        self.device = device
        logger.info("ARGUSPipeline: initialising...")

        self.tropomi = GEETROPOMIIngester()
        self.wind    = GEEWindIngester()
        self.emit    = GEEEMITIngester()

        self.stage1  = load_stage1(device=device)
        self.stage2  = PINNFluxEstimator(device=device)
        self.stage3  = SourceAttributor(device=device)
        self.stage4  = BatchEnforcementProcessor()
        self.al      = ActiveLearningQueue()

        self.gee_info = gee_status()
        logger.info(
            f"ARGUSPipeline: ready | "
            f"GEE={'live' if self.gee_info['available'] else 'mock'}"
        )

    def run(
        self,
        lat_min:  float,
        lat_max:  float,
        lon_min:  float,
        lon_max:  float,
        date:     datetime | None = None,
        history:  list[dict] | None = None,
    ) -> PipelineResult:

        run_id   = f"RUN-{uuid.uuid4().hex[:8].upper()}"
        t_start  = datetime.utcnow()
        timing   = {}
        warnings = []

        bbox = {
            "lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max,
        }

        logger.info(
            f"ARGUSPipeline [{run_id}]: "
            f"bbox=({lat_min:.2f},{lat_max:.2f},{lon_min:.2f},{lon_max:.2f})"
        )

        result = PipelineResult(
            run_id=run_id, bbox=bbox,
            timestamp=t_start.isoformat(),
            duration_seconds=0.0,
            gee_live=self.gee_info["available"],
        )

        # ── Stage 0: GEE data ingestion ───────────────────────────
        t0 = datetime.utcnow()
        try:
            ds_tropomi = self.tropomi.fetch(lat_min, lat_max, lon_min, lon_max, date)
            ds_wind    = self.wind.fetch(lat_min, lat_max, lon_min, lon_max, date)
            u_ms = float(ds_wind["u10"].values.mean())
            v_ms = float(ds_wind["v10"].values.mean())
            logger.info(f"Wind: u={u_ms:.2f} v={v_ms:.2f} m/s")
        except Exception as e:
            warnings.append(f"Data ingestion: {e}")
            logger.error(f"Data error: {e}")
            ds_tropomi = GEETROPOMIIngester._mock(lat_min, lat_max, lon_min, lon_max)
            u_ms, v_ms = 5.0, -1.5

        timing["stage0_gee"] = (datetime.utcnow() - t0).total_seconds()

        # ── Cloud inpainting ──────────────────────────────────────
        t0 = datetime.utcnow()
        try:
            cloud_frac = float(ds_tropomi["cloud_fraction"].values.mean())
            if cloud_frac > cfg["pipeline"]["cloud_mask_threshold"]:
                logger.info(f"Cloud cover {cloud_frac:.1%} — running inpainting")
                inpainted = apply_inpainting(ds_tropomi, u_ms, v_ms)
                ds_tropomi["methane_mixing_ratio_bias_corrected"].values[...] = inpainted
                result.cloud_inpainted = True
        except Exception as e:
            warnings.append(f"Inpainting: {e}")
        timing["inpainting"] = (datetime.utcnow() - t0).total_seconds()

        # ── Stage 1: TorchGeo + ViT segmentation ─────────────────
        t0 = datetime.utcnow()
        detections = []
        try:
            tensor     = preprocess_tropomi(ds_tropomi).to(self.device)
            seg_out    = mc_predict(self.stage1, tensor, device=self.device)
            detections = extract_plume_detections(
                mask_mean=seg_out.mask_mean.squeeze(0),
                mask_var =seg_out.mask_variance.squeeze(0),
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
            )
            emb = seg_out.embeddings.squeeze(0).cpu().numpy()
            for det in detections:
                det["cls_embedding"] = emb.tolist()
        except Exception as e:
            warnings.append(f"Stage 1: {e}")
            logger.error(f"Stage 1 error: {e}")

        timing["stage1_vit"] = (datetime.utcnow() - t0).total_seconds()
        logger.info(f"Stage 1: {len(detections)} plumes detected")

        if not detections:
            result.duration_seconds = (datetime.utcnow() - t_start).total_seconds()
            result.timing   = timing
            result.warnings = warnings
            return result

        # ── Active learning triage ────────────────────────────────
        # evaluate_and_queue returns an int (count queued), not a tuple
        # We keep all detections for processing and let the queue run in background
        queued_count = self.al.evaluate_and_queue(detections, run_id=run_id)
        process_dets = detections   # process all detections
        uncertain    = []           # uncertain ones are already saved to MongoDB by evaluate_and_queue
        result.review_queue = []

        if queued_count > 0:
            logger.info(f"Active learning: {queued_count} detections queued for review")

        # ── Stage 2: Modulus PINN flux ────────────────────────────
        t0 = datetime.utcnow()
        flux_outputs: list[FluxOutput] = []

        for det in process_dets:
            try:
                flux = self.stage2.estimate(
                    detection=det, ds_tropomi=ds_tropomi,
                    u_ms=u_ms, v_ms=v_ms,
                    lat_min=lat_min, lat_max=lat_max,
                    lon_min=lon_min, lon_max=lon_max,
                    n_epochs=300,
                )
                det["flux_kg_hr"]       = flux.flux_kg_hr
                det["flux_uncertainty"] = flux.flux_uncertainty
                det["co2e_kg_hr"]       = flux.co2e_kg_hr
                det["transport_age_hr"] = flux.transport_age_hr
                econ = calculate_economic_impact(flux.flux_kg_hr)
                det["economics"] = {
                    "total_cost_usd": round(econ.total_cost_usd, 2),
                    "total_cost_inr": round(econ.total_cost_inr, 2),
                    "summary":        econ.summary_line,
                }
                flux_outputs.append(flux)
            except Exception as e:
                warnings.append(f"Stage 2 det-{det['label_id']}: {e}")
                fallback  = float(det.get("pixel_area", 1.0) * 2.5)
                mock_flux = FluxOutput(
                    flux_kg_hr=fallback,
                    flux_uncertainty=fallback * 0.3,
                    plume_length_km=1.0,
                    effective_wind_ms=abs(u_ms),
                    transport_age_hr=1.0,
                    co2e_kg_hr=fallback * cfg["intelligence"]["methane_gwp_20yr"],
                )
                det["flux_kg_hr"] = fallback
                det["co2e_kg_hr"] = mock_flux.co2e_kg_hr
                det["economics"]  = {}
                flux_outputs.append(mock_flux)

        timing["stage2_modulus_pinn"] = (datetime.utcnow() - t0).total_seconds()

        # ── EMIT cross-validation ─────────────────────────────────
        t0 = datetime.utcnow()
        try:
            ds_emit   = self.emit.fetch(lat_min, lat_max, lon_min, lon_max)
            emit_peak = float(ds_emit["ch4_enhancement"].values.max())
            if emit_peak > 0:
                result.emit_validated = True
                logger.info(f"EMIT cross-validation: peak={emit_peak:.1f} ppb ✓")
        except Exception as e:
            warnings.append(f"EMIT: {e}")
        timing["emit_crossval"] = (datetime.utcnow() - t0).total_seconds()

        # ── Stage 3: TGAN attribution ─────────────────────────────
        t0 = datetime.utcnow()
        attributions: list[AttributionResult] = []
        try:
            attributions = self.stage3.attribute(
                detections=process_dets, u_ms=u_ms, v_ms=v_ms, history=history,
            )
            for det, attr in zip(process_dets, attributions):
                det["facility_id"]            = attr.facility_id
                det["facility_name"]          = attr.facility_name
                det["operator"]               = attr.operator
                det["attribution_confidence"] = attr.confidence
                det["distance_km"]            = attr.distance_km
        except Exception as e:
            warnings.append(f"Stage 3: {e}")
            logger.error(f"Stage 3 error: {e}")
            all_fac = load_facilities()
            for det in process_dets:
                row = all_fac.iloc[0]
                attributions.append(AttributionResult(
                    facility_id=str(row["facility_id"]),
                    facility_name=str(row.get("facility_name", "?")),
                    operator=str(row.get("operator", "?")),
                    facility_type=str(row.get("type", "unknown")),
                    confidence=0.5, distance_km=0.0,
                    back_traj_lat=det["centroid_lat"],
                    back_traj_lon=det["centroid_lon"],
                    all_candidates=[],
                ))

        timing["stage3_tgan"] = (datetime.utcnow() - t0).total_seconds()

        # ── Compliance scorecard ──────────────────────────────────
        try:
            scorecard = build_compliance_scorecard(attributions, flux_outputs)
            result.scorecard = scorecard
            for det in process_dets:
                fid = det.get("facility_id", "")
                for row in scorecard:
                    if row["facility_id"] == fid:
                        det["risk_level"] = row["risk_level"]
                        break
        except Exception as e:
            warnings.append(f"Scorecard: {e}")
            logger.error(f"Scorecard error: {e}")

        # ── Stage 4: Groq LLM enforcement ─────────────────────────
        t0 = datetime.utcnow()
        enforcement_results = []
        try:
            threshold    = cfg["pipeline"]["flux_threshold_kg_hr"]
            super_dets   = [d for d, f in zip(process_dets, flux_outputs) if f.flux_kg_hr >= threshold]
            super_fluxes = [f for f in flux_outputs if f.flux_kg_hr >= threshold]
            super_attrs  = [
                a for d, a, f in zip(process_dets, attributions, flux_outputs)
                if f.flux_kg_hr >= threshold
            ]

            if super_dets:
                enforcement_results = self.stage4.process_all(
                    detections=super_dets,
                    attributions=super_attrs,
                    flux_outputs=super_fluxes,
                )
                for enf in enforcement_results:
                    for det in process_dets:
                        if det["label_id"] == enf.get("detection_id"):
                            notice = enf.get("notice") or {}
                            det["notice_id"] = notice.get("notice_id", "")
                            det["fine_usd"]  = notice.get("fine_usd", 0.0)
                            det["fine_inr"]  = notice.get("fine_inr", 0.0)
        except Exception as e:
            warnings.append(f"Stage 4: {e}")
            logger.error(f"Stage 4 error: {e}")

        timing["stage4_groq"] = (datetime.utcnow() - t0).total_seconds()

        # ── Aggregate results ─────────────────────────────────────
        result.detections   = process_dets
        result.flux_outputs = flux_outputs
        result.attributions = attributions
        result.enforcement  = enforcement_results
        result.warnings     = warnings
        result.timing       = timing

        result.n_super_emitters = sum(
            1 for f in flux_outputs
            if f.flux_kg_hr >= cfg["pipeline"]["flux_threshold_kg_hr"]
        )
        result.total_flux_kg_hr = float(sum(f.flux_kg_hr for f in flux_outputs))
        result.total_co2e_kg_hr = float(sum(f.co2e_kg_hr for f in flux_outputs))
        result.total_impact_usd = float(sum(
            d.get("economics", {}).get("total_cost_usd", 0) for d in process_dets
        ))
        result.total_impact_inr = result.total_impact_usd * 83.5
        result.duration_seconds = (datetime.utcnow() - t_start).total_seconds()

        logger.info(
            f"ARGUSPipeline [{run_id}] COMPLETE | "
            f"{result.n_super_emitters} super-emitters | "
            f"{result.total_flux_kg_hr:.1f} kg/hr | "
            f"${result.total_impact_usd:,.0f} | "
            f"{result.duration_seconds:.1f}s"
        )
        return result


# ═════════════════════════════════════════════════════════════════
# Run store — MongoDB backed
# ═════════════════════════════════════════════════════════════════

class RunStore:
    def __init__(self):
        from src.db.mongo import get_sync_db
        self.db = get_sync_db()

    def save(self, result: PipelineResult) -> None:
        doc      = result.to_api_dict()
        doc["_id"] = result.run_id
        self.db.runs.replace_one({"_id": result.run_id}, doc, upsert=True)

        for det in result.detections:
            det_doc = {
                **det,
                "run_id":       result.run_id,
                "timestamp":    result.timestamp,
                "bbox":         result.bbox,
                "detection_id": det.get("label_id"),  # normalize here
            }

            lat = det.get("centroid_lat")
            lon = det.get("centroid_lon")
            if lat and lon:
                det_doc["location"] = {"type": "Point", "coordinates": [lon, lat]}
           

            det_id = det.get("detection_id") or det.get("label_id") or det.get("id")
            self.db.detections.replace_one(
                {"run_id": result.run_id, "detection_id": det_id},
                {**det_doc, "detection_id": det_id},
                upsert=True,
            )

        for row in result.scorecard:
            self.db.scorecard.replace_one(
                {"facility_id": row["facility_id"]},
                {**row, "updated_at": result.timestamp},
                upsert=True,
            )
        logger.debug(f"RunStore: saved {result.run_id}")

    def load_recent(self, n: int = 20) -> list[dict]:
        return list(
            self.db.runs.find({}, {"_id": 0}).sort("timestamp", -1).limit(n)
        )

    def load_all_detections(self, limit: int = 1000) -> list[dict]:
        return list(
            self.db.detections.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        )