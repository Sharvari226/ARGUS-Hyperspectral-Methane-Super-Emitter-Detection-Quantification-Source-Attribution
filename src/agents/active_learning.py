from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.config import cfg

REVIEW_QUEUE_PATH = Path("data/active_learning/review_queue.jsonl")
REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)


class ActiveLearningQueue:

    def __init__(self):
        from src.db.mongo import get_sync_db
        self.db        = get_sync_db()
        self.threshold = cfg["pipeline"]["uncertainty_max"]

    def _write_to_queue(self, det: dict) -> None:
        record = {k: v for k, v in det.items()
                  if k not in ("pixel_ys", "pixel_xs")}
        self.db.review_queue.insert_one(record)

    def load_queue(self) -> list[dict]:
        cursor = (
            self.db.review_queue
            .find({"labeled": {"$exists": False}}, {"_id": 0})
            .sort("queued_at", -1)
            .limit(200)
        )
        return list(cursor)

    def queue_size(self) -> int:
        return self.db.review_queue.count_documents(
            {"labeled": {"$exists": False}}
        )

    def submit_label(
        self,
        detection_id: int,
        run_id: str,
        is_plume: bool,
        reviewer: str = "human_expert",
        notes: str = "",
    ) -> None:
        self.db.review_queue.update_one(
            {"label_id": detection_id, "run_id": run_id},
            {"$set": {
                "labeled":    True,
                "is_plume":   is_plume,
                "reviewer":   reviewer,
                "notes":      notes,
                "labeled_at": datetime.utcnow().isoformat(),
            }},
        )

    def learning_curve(self) -> dict:
        pipeline = [
            {"$group": {
                "_id":           "$run_id",
                "count":         {"$sum": 1},
                "mean_variance": {"$avg": "$epistemic_variance"},
            }},
            {"$sort": {"_id": 1}},
        ]
        docs = list(self.db.review_queue.aggregate(pipeline))
        return {
            "runs":          [d["_id"]           for d in docs],
            "queue_sizes":   [d["count"]          for d in docs],
            "mean_variance": [d["mean_variance"]  for d in docs],
        }

    def evaluate_and_queue(self, detections: list, run_id: str) -> int:
        """
        Evaluates detections for uncertainty and queues high-uncertainty ones.
        Returns number of items added to queue.
        """
        queued = 0
        for det in detections:
            variance = det.get("epistemic_variance", 0)
            if variance > self.threshold:
                record = {
                    "label_id":           det.get("detection_id", 0),
                    "run_id":             run_id,
                    "centroid_lat":       det.get("centroid_lat"),
                    "centroid_lon":       det.get("centroid_lon"),
                    "flux_kg_hr":         det.get("flux_kg_hr", 0),
                    "epistemic_variance": variance,
                    "confidence":         det.get("confidence", 0),
                    "queued_at":          datetime.utcnow().isoformat(),
                }
                try:
                    self.db.review_queue.insert_one(record)
                    queued += 1
                except Exception:
                    pass
        return queued