from __future__ import annotations

import time
import threading
from datetime import datetime

from loguru import logger

from src.pipeline.orchestrator import ARGUSPipeline, RunStore
from src.utils.config import cfg


MONITORING_BBOXES = [
    {"name": "Permian Basin, USA",      "lat_min": 31.0, "lat_max": 33.0, "lon_min": -104.0, "lon_max": -101.0},
    {"name": "Turkmenistan Gas Fields", "lat_min": 37.0, "lat_max": 40.0, "lon_min":   55.0, "lon_max":   60.0},
    {"name": "Niger Delta, Nigeria",    "lat_min":  4.0, "lat_max":  6.0, "lon_min":    5.0, "lon_max":    8.0},
    {"name": "Siberia Gas Fields",      "lat_min": 60.0, "lat_max": 65.0, "lon_min":   70.0, "lon_max":   80.0},
    {"name": "Mumbai Offshore, India",  "lat_min": 18.0, "lat_max": 22.0, "lon_min":   68.0, "lon_max":   73.0},
    {"name": "Saudi Aramco East",       "lat_min": 26.0, "lat_max": 28.0, "lon_min":   49.0, "lon_max":   51.0},
]


class PipelineScheduler:
    """
    Runs the ARGUS pipeline on all monitoring bboxes at a set interval.
    Runs in a background daemon thread so the API stays responsive.
    """

    def __init__(
        self,
        interval_seconds: int = 300,
        bboxes: list[dict] | None = None,
    ):
        self.interval  = interval_seconds
        self.bboxes    = bboxes or MONITORING_BBOXES
        self.pipeline  = ARGUSPipeline()
        self.store     = RunStore()
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(
            target=self._loop, daemon=True, name="argus-scheduler"
        )

    def start(self) -> None:
        logger.info(
            f"PipelineScheduler: starting — "
            f"{len(self.bboxes)} regions every {self.interval}s"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        logger.info("PipelineScheduler: stopped")

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            for bbox in self.bboxes:
                if self._stop_evt.is_set():
                    break
                try:
                    logger.info(f"Scheduler: running pipeline for {bbox['name']}")
                    result = self.pipeline.run(
                        lat_min=bbox["lat_min"],
                        lat_max=bbox["lat_max"],
                        lon_min=bbox["lon_min"],
                        lon_max=bbox["lon_max"],
                    )
                    self.store.save(result)
                    logger.info(
                        f"Scheduler: {bbox['name']} done — "
                        f"{result.n_super_emitters} super-emitters | "
                        f"{result.duration_seconds:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"Scheduler error ({bbox['name']}): {e}")

            self._stop_evt.wait(timeout=self.interval)