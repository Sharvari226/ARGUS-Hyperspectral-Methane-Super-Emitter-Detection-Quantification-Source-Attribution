from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import NamedTuple
from datetime import datetime

import torch_geometric as pyg
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax as pyg_softmax
from loguru import logger

from src.utils.config import cfg
from src.utils.geo import haversine_km, back_propagate_wind
from src.data.facility_db import find_nearest_facilities, load_facilities

CKPT_PATH = Path(cfg["stage3"]["checkpoint"])


# ── Output container ──────────────────────────────────────────────────────────

class AttributionResult(NamedTuple):
    facility_id:        str
    facility_name:      str
    operator:           str
    facility_type:      str
    confidence:         float          # 0–1 attribution confidence
    distance_km:        float          # plume centroid → facility centroid
    back_traj_lat:      float          # wind back-trajectory source estimate
    back_traj_lon:      float
    all_candidates:     list[dict]     # ranked list of all considered facilities


# ── Node & edge feature dimensions ───────────────────────────────────────────

PLUME_DIM    = 8    # [lat, lon, flux, uncertainty, area, prob, u_wind, v_wind]
FACILITY_DIM = 6    # [lat, lon, compliance_score, violations, type_enc, distance]
EDGE_DIM     = 4    # [distance_km, bearing_deg, time_delta_hr, wind_alignment]


# ── Temporal edge encoder ─────────────────────────────────────────────────────

class TemporalEdgeEncoder(nn.Module):
    """
    Encodes edge features including the time-delta between
    historical detections (temporal edges) and spatial edges.
    Time decay: older edges get exponentially down-weighted.
    """

    def __init__(self, edge_dim: int = EDGE_DIM, out_dim: int = 32):
        super().__init__()
        self.decay = cfg["stage3"]["edge_time_decay"]   # 0.85
        self.net   = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
        )

    def forward(
        self,
        edge_attr:       torch.Tensor,   # (E, EDGE_DIM)
        time_delta_hr:   torch.Tensor,   # (E,) hours since detection
    ) -> torch.Tensor:
        # Temporal decay weight — older edges contribute less
        decay_w = self.decay ** time_delta_hr.unsqueeze(1)   # (E, 1)
        encoded = self.net(edge_attr)                         # (E, out_dim)
        return encoded * decay_w


# ── Heterogeneous GAT layer ───────────────────────────────────────────────────

class HeteroGATLayer(nn.Module):
    """
    One layer of heterogeneous graph attention over two node types:
        - 'plume'    nodes: current detections
        - 'facility' nodes: infrastructure polygons

    Three edge types:
        - ('plume',    'near',     'facility')   spatial proximity
        - ('facility', 'history',  'plume')      historical co-occurrence
        - ('plume',    'temporal', 'plume')       same-plume across time steps
    """

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.conv = HeteroConv(
            {
                ("plume",    "near",     "facility"): GATConv(
                    (hidden, hidden), hidden, heads=heads,
                    add_self_loops=False, edge_dim=32,
                ),
                ("facility", "history",  "plume"):    GATConv(
                    (hidden, hidden), hidden, heads=heads,
                    add_self_loops=False, edge_dim=32,
                ),
                ("plume",    "temporal", "plume"):    SAGEConv(
                    hidden, hidden,
                ),
            },
            aggr="sum",
        )
        self.norms = nn.ModuleDict({
            "plume":    nn.LayerNorm(hidden * heads),
            "facility": nn.LayerNorm(hidden * heads),
        })
        self.proj = nn.ModuleDict({
            "plume":    nn.Linear(hidden * heads, hidden),
            "facility": nn.Linear(hidden * heads, hidden),
        })

    def forward(
        self,
        x_dict:        dict[str, torch.Tensor],
        edge_index_dict: dict,
        edge_attr_dict:  dict,
    ) -> dict[str, torch.Tensor]:

        out = self.conv(x_dict, edge_index_dict, edge_attr_dict)

        result = {}
        for ntype, tensor in out.items():
            normed  = self.norms[ntype](tensor)
            proj    = self.proj[ntype](normed)
            # Residual: only if dimensions match
            if x_dict[ntype].shape == proj.shape:
                result[ntype] = F.gelu(proj + x_dict[ntype])
            else:
                result[ntype] = F.gelu(proj)

        return result


# ── Full TGAN model ───────────────────────────────────────────────────────────

class TemporalGraphAttributor(nn.Module):
    """
    Multi-layer Temporal Graph Attention Network for source attribution.

    Given a heterogeneous graph of:
        - Current plume detections (nodes)
        - Nearby facility polygons (nodes)
        - Spatial + temporal edges between them

    Outputs a probability distribution over facilities for each plume.

    The key insight: by aggregating historical co-occurrence patterns
    (which facilities have been near plumes before, and when),
    the GNN learns to rule out facilities that are upwind or historically clean.
    """

    def __init__(self):
        super().__init__()
        hidden = cfg["stage3"]["gat_hidden"]     # 128
        heads  = cfg["stage3"]["gat_heads"]      # 4
        layers = cfg["stage3"]["gat_layers"]     # 3

        # Input projections
        self.plume_proj    = nn.Linear(PLUME_DIM,    hidden)
        self.facility_proj = nn.Linear(FACILITY_DIM, hidden)

        # Temporal edge encoder
        self.edge_encoder = TemporalEdgeEncoder(EDGE_DIM, out_dim=32)

        # Stack of hetero-GAT layers
        self.gat_layers = nn.ModuleList([
            HeteroGATLayer(hidden, heads) for _ in range(layers)
        ])

        # Attribution scoring head
        # Takes (plume_embed ‖ facility_embed) → scalar score
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        logger.info(
            f"TemporalGraphAttributor: "
            f"{sum(p.numel() for p in self.parameters()):,} params | "
            f"{layers} GAT layers × {heads} heads"
        )

    def forward(
        self,
        data: HeteroData,
    ) -> torch.Tensor:
        """
        Returns (P, F) score matrix:
            P = number of plume nodes
            F = number of facility nodes
        Higher score = stronger attribution.
        """
        # ── Input projections ─────────────────────────────────────
        x_dict = {
            "plume":    F.gelu(self.plume_proj(data["plume"].x)),
            "facility": F.gelu(self.facility_proj(data["facility"].x)),
        }

        # ── Encode temporal edge features ─────────────────────────
        edge_attr_dict = {}
        for rel, attr in data.edge_attr_dict.items():
            t_delta = attr[:, 2]   # time_delta_hr is the 3rd edge feature
            edge_attr_dict[rel] = self.edge_encoder(attr, t_delta)

        # ── Message passing ───────────────────────────────────────
        for layer in self.gat_layers:
            x_dict = layer(x_dict, data.edge_index_dict, edge_attr_dict)

        # ── Score every (plume, facility) pair ────────────────────
        P = x_dict["plume"].shape[0]
        F_ = x_dict["facility"].shape[0]

        p_exp = x_dict["plume"].unsqueeze(1).expand(-1, F_, -1)      # (P, F, H)
        f_exp = x_dict["facility"].unsqueeze(0).expand(P, -1, -1)    # (P, F, H)
        pairs = torch.cat([p_exp, f_exp], dim=-1)                     # (P, F, 2H)

        scores = self.scorer(pairs).squeeze(-1)   # (P, F)
        return scores


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_attribution_graph(
    detections:     list[dict],
    u_ms:           float,
    v_ms:           float,
    history:        list[dict] | None = None,
    radius_km:      float | None = None,
) -> tuple[HeteroData, list, list]:
    """
    Constructs the HeteroData graph from:
        - Current plume detections (from Stage 1)
        - Nearby facility polygons (from facility_db)
        - Historical detection records (from database)

    Returns:
        data          : HeteroData graph
        plume_ids     : list of detection label_ids (maps graph index → detection)
        facility_ids  : list of facility_ids (maps graph index → facility)
    """
    radius_km = radius_km or cfg["pipeline"]["attribution_radius_km"]

    # ── Collect all unique nearby facilities ──────────────────────
    facility_rows = []
    facility_id_to_idx = {}

    for det in detections:
        nearby = find_nearest_facilities(
            det["centroid_lat"], det["centroid_lon"],
            radius_km=radius_km, top_k=5,
        )
        for _, row in nearby.iterrows():
            fid = row["facility_id"]
            if fid not in facility_id_to_idx:
                facility_id_to_idx[fid] = len(facility_rows)
                facility_rows.append(row)

    if not facility_rows:
        logger.warning("TGAN: no facilities found within radius — using global fallback")
        all_fac = load_facilities()
        for _, row in all_fac.iterrows():
            fid = row["facility_id"]
            facility_id_to_idx[fid] = len(facility_rows)
            facility_rows.append(row)

    facility_ids = list(facility_id_to_idx.keys())

    # ── Encode facility types ─────────────────────────────────────
    TYPE_MAP = {
        "oil_wellpad": 0, "gas_compressor": 1, "lng_terminal": 2,
        "pipeline_station": 3, "refinery": 4, "gas_storage": 5,
    }

    def encode_facility(row) -> list[float]:
        ftype = TYPE_MAP.get(str(row.get("type", "")), 0)
        return [
            float(row.geometry.centroid.y),           # lat
            float(row.geometry.centroid.x),           # lon
            float(row.get("compliance_score", 70.0)) / 100.0,
            float(row.get("violations_12mo", 0)) / 6.0,
            float(ftype) / 6.0,
            0.0,                                      # distance — filled below
        ]

    # ── Encode plume features ─────────────────────────────────────
    def encode_plume(det: dict) -> list[float]:
        return [
            float(det["centroid_lat"]),
            float(det["centroid_lon"]),
            float(det.get("flux_kg_hr", 0.0)) / 1000.0,    # normalised
            float(det.get("epistemic_variance", 0.05)),
            float(det["pixel_area"]) / 500.0,
            float(det["mean_probability"]),
            float(u_ms) / 20.0,
            float(v_ms) / 20.0,
        ]

    plume_feats    = [encode_plume(d) for d in detections]
    facility_feats = [encode_facility(r) for r in facility_rows]
    plume_ids      = [d["label_id"] for d in detections]

    # ── Build spatial edges (plume → facility) ────────────────────
    near_src, near_dst, near_attr = [], [], []

    for p_idx, det in enumerate(detections):
        p_lat, p_lon = det["centroid_lat"], det["centroid_lon"]

        for f_idx, row in enumerate(facility_rows):
            f_lat = float(row.geometry.centroid.y)
            f_lon = float(row.geometry.centroid.x)
            dist  = haversine_km(p_lat, p_lon, f_lat, f_lon)

            if dist > radius_km:
                continue

            bearing = _bearing(p_lat, p_lon, f_lat, f_lon)

            # Wind alignment: how much does wind vector point FROM facility TO plume?
            wind_dir  = np.degrees(np.arctan2(v_ms, u_ms))
            alignment = np.cos(np.radians(bearing - wind_dir))   # −1 to 1

            near_src.append(p_idx)
            near_dst.append(f_idx)
            near_attr.append([dist / radius_km, bearing / 360.0, 0.0, float(alignment)])

            # Update distance feature in facility node
            facility_feats[f_idx][5] = dist / radius_km

    # ── Build reverse edges (facility → plume history) ────────────
    hist_src, hist_dst, hist_attr = [], [], []
    if history:
        for h_det in history:
            h_lat = h_det.get("centroid_lat", 0.0)
            h_lon = h_det.get("centroid_lon", 0.0)
            h_hrs = h_det.get("age_hours", 0.0)

            for f_idx, row in enumerate(facility_rows):
                f_lat = float(row.geometry.centroid.y)
                f_lon = float(row.geometry.centroid.x)
                dist  = haversine_km(h_lat, h_lon, f_lat, f_lon)
                if dist < radius_km:
                    bearing   = _bearing(f_lat, f_lon, h_lat, h_lon)
                    wind_dir  = np.degrees(np.arctan2(v_ms, u_ms))
                    alignment = float(np.cos(np.radians(bearing - wind_dir)))
                    for p_idx in range(len(detections)):
                        hist_src.append(f_idx)
                        hist_dst.append(p_idx)
                        hist_attr.append([
                            dist / radius_km,
                            bearing / 360.0,
                            float(h_hrs),
                            alignment,
                        ])

    # ── Temporal plume edges (same plume at different times) ──────
    temp_src, temp_dst = [], []
    if len(detections) > 1:
        for i in range(len(detections)):
            for j in range(len(detections)):
                if i != j:
                    temp_src.append(i)
                    temp_dst.append(j)

    # ── Assemble HeteroData ───────────────────────────────────────
    data = HeteroData()

    data["plume"].x    = torch.tensor(plume_feats,    dtype=torch.float)
    data["facility"].x = torch.tensor(facility_feats, dtype=torch.float)

    if near_src:
        data["plume",    "near",     "facility"].edge_index = torch.tensor(
            [near_src, near_dst], dtype=torch.long
        )
        data["plume",    "near",     "facility"].edge_attr  = torch.tensor(
            near_attr, dtype=torch.float
        )
    else:
        # Empty edges — model handles this gracefully
        data["plume",    "near",     "facility"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["plume",    "near",     "facility"].edge_attr  = torch.zeros((0, EDGE_DIM))

    if hist_src:
        data["facility", "history",  "plume"].edge_index = torch.tensor(
            [hist_src, hist_dst], dtype=torch.long
        )
        data["facility", "history",  "plume"].edge_attr  = torch.tensor(
            hist_attr, dtype=torch.float
        )
    else:
        data["facility", "history",  "plume"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["facility", "history",  "plume"].edge_attr  = torch.zeros((0, EDGE_DIM))

    if temp_src:
        data["plume", "temporal", "plume"].edge_index = torch.tensor(
            [temp_src, temp_dst], dtype=torch.long
        )
        # Temporal edges use zero edge_attr (SAGEConv doesn't use edge_attr)
    else:
        data["plume", "temporal", "plume"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    return data, plume_ids, facility_ids


# ── Full attribution pipeline ─────────────────────────────────────────────────

class SourceAttributor:
    """
    End-to-end wrapper:
        detections + wind → AttributionResult list

    Falls back to distance-only heuristic if model weights absent.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model  = TemporalGraphAttributor().to(device)

        if CKPT_PATH.exists():
            state = torch.load(CKPT_PATH, map_location=device)
            self.model.load_state_dict(state)
            logger.info(f"SourceAttributor: loaded checkpoint from {CKPT_PATH}")
        else:
            logger.warning(
                "SourceAttributor: no checkpoint — using wind back-trajectory heuristic"
            )

    @torch.no_grad()
    def attribute(
        self,
        detections:  list[dict],
        u_ms:        float,
        v_ms:        float,
        history:     list[dict] | None = None,
    ) -> list[AttributionResult]:
        """
        Attribute each plume detection to its most likely source facility.
        Returns one AttributionResult per detection, sorted by confidence desc.
        """
        if not detections:
            return []

        # Build graph
        data, plume_ids, facility_ids = build_attribution_graph(
            detections, u_ms, v_ms, history
        )
        data = data.to(self.device)

        # Model inference
        if CKPT_PATH.exists():
            scores = self.model(data)                        # (P, F)
            probs  = torch.softmax(scores, dim=-1).cpu().numpy()
        else:
            probs  = self._heuristic_scores(detections, facility_ids, u_ms, v_ms)

        results = []
        all_fac = load_facilities()

        for p_idx, det in enumerate(detections):
            best_f_idx = int(np.argmax(probs[p_idx]))
            confidence = float(probs[p_idx][best_f_idx])
            fid        = facility_ids[best_f_idx]

            fac_row = all_fac[all_fac["facility_id"] == fid].iloc[0]
            f_lat   = float(fac_row.geometry.centroid.y)
            f_lon   = float(fac_row.geometry.centroid.x)
            dist    = haversine_km(
                det["centroid_lat"], det["centroid_lon"], f_lat, f_lon
            )

            # Lagrangian back-trajectory
            wind_speed = max(float(np.sqrt(u_ms**2 + v_ms**2)), 0.5)
            bt_lat, bt_lon = back_propagate_wind(
                det["centroid_lat"], det["centroid_lon"],
                u_ms, v_ms,
                duration_hours=det.get("transport_age_hr", 1.0),
            )

            # Build ranked candidate list
            candidates = []
            for f_idx_c, fid_c in enumerate(facility_ids):
                candidates.append({
                    "facility_id": fid_c,
                    "score":       float(probs[p_idx][f_idx_c]),
                    "rank":        0,
                })
            candidates.sort(key=lambda x: x["score"], reverse=True)
            for rank, c in enumerate(candidates):
                c["rank"] = rank + 1

            results.append(AttributionResult(
                facility_id=fid,
                facility_name=str(fac_row.get("facility_name", fid)),
                operator=str(fac_row.get("operator", "Unknown")),
                facility_type=str(fac_row.get("type", "unknown")),
                confidence=confidence,
                distance_km=dist,
                back_traj_lat=bt_lat,
                back_traj_lon=bt_lon,
                all_candidates=candidates[:5],
            ))

        return results

    # ------------------------------------------------------------------
    def _heuristic_scores(
        self,
        detections:   list[dict],
        facility_ids: list[str],
        u_ms:         float,
        v_ms:         float,
    ) -> np.ndarray:
        """
        Distance + wind-alignment heuristic used when model weights absent.
        Upwind facilities score higher than downwind ones.
        """
        all_fac = load_facilities()
        P, F    = len(detections), len(facility_ids)
        scores  = np.zeros((P, F), dtype=np.float32)
        wind_dir = np.degrees(np.arctan2(v_ms, u_ms))

        for p_idx, det in enumerate(detections):
            p_lat, p_lon = det["centroid_lat"], det["centroid_lon"]

            for f_idx, fid in enumerate(facility_ids):
                row   = all_fac[all_fac["facility_id"] == fid].iloc[0]
                f_lat = float(row.geometry.centroid.y)
                f_lon = float(row.geometry.centroid.x)

                dist = haversine_km(p_lat, p_lon, f_lat, f_lon) + 1e-3
                bearing   = _bearing(p_lat, p_lon, f_lat, f_lon)
                alignment = float(np.cos(np.radians(bearing - wind_dir + 180)))

                # Score: closer + more upwind = higher
                dist_score = 1.0 / dist
                wind_score = (alignment + 1) / 2.0   # normalise to [0,1]
                scores[p_idx, f_idx] = 0.6 * dist_score + 0.4 * wind_score

        # Normalise rows to sum to 1
        row_sums = scores.sum(axis=1, keepdims=True) + 1e-8
        return scores / row_sums


# ── Compliance scorecard ──────────────────────────────────────────────────────

def build_compliance_scorecard(
    attribution_results: list[AttributionResult],
    flux_outputs:        list,    # list[FluxOutput] from Stage 2
) -> list[dict]:
    """
    Aggregates attribution + flux results into a per-operator
    compliance scorecard that feeds the dashboard leaderboard.

    Score formula:
        risk_score = (violations_12mo × 20)
                   + (1 - confidence) × 30
                   + clamp(flux_kg_hr / 500, 0, 50)
    """
    all_fac = load_facilities()
    records = []

    for attr, flux in zip(attribution_results, flux_outputs):
        row = all_fac[all_fac["facility_id"] == attr.facility_id]
        if row.empty:
            continue
        row = row.iloc[0]

        violations = int(row.get("violations_12mo", 0))
        base_score = float(row.get("compliance_score", 70.0))
        risk_penalty = (
            violations * 20
            + (1 - attr.confidence) * 30
            + min(flux.flux_kg_hr / 500.0 * 50, 50)
        )
        final_score = max(0.0, base_score - risk_penalty)

        records.append({
            "facility_id":     attr.facility_id,
            "facility_name":   attr.facility_name,
            "operator":        attr.operator,
            "facility_type":   attr.facility_type,
            "flux_kg_hr":      round(flux.flux_kg_hr, 1),
            "co2e_kg_hr":      round(flux.co2e_kg_hr, 1),
            "confidence":      round(attr.confidence * 100, 1),
            "violations_12mo": violations,
            "compliance_score": round(final_score, 1),
            "risk_level":      _risk_label(final_score),
            "distance_km":     round(attr.distance_km, 2),
        })

    records.sort(key=lambda x: x["compliance_score"])
    return records


def _risk_label(score: float) -> str:
    if score < 20:  return "CRITICAL"   # was 30
    if score < 40:  return "HIGH"       # was 50
    if score < 58:  return "MEDIUM"     # was 70
    return "LOW"                        # now reachable for low-flux, clean facilities             # only reachable if score >= 70 after penalties


# ── Geometry helper ───────────────────────────────────────────────────────────

def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 to point 2 in degrees [0, 360)."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_tgan(model: TemporalGraphAttributor) -> None:
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    logger.info(f"TGAN: saved → {CKPT_PATH}")


def load_tgan(device: str = "cpu") -> TemporalGraphAttributor:
    model = TemporalGraphAttributor().to(device)
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        logger.info(f"TGAN: loaded checkpoint from {CKPT_PATH}")
    else:
        logger.warning("TGAN: no checkpoint — heuristic fallback active")
    return model