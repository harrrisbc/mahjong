from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mahjong_projector.types import Detection, VisionResult
from mahjong_projector.vision.seats import pick_winner, seat_for_point, seat_regions

log = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(self, model: Any | None = None, confidence_threshold: float = 0.50) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_weights(cls, weights: Path, confidence_threshold: float = 0.50) -> VisionPipeline:
        if not weights.is_file():
            log.warning("YOLO weights missing at %s", weights)
            return cls(model=None, confidence_threshold=confidence_threshold)
        from ultralytics import YOLO

        return cls(model=YOLO(str(weights)), confidence_threshold=confidence_threshold)

    def analyze(self, frame: np.ndarray) -> VisionResult:
        if self.model is None or frame is None or frame.size == 0:
            return VisionResult(winner_seat=None, tiles=[], confidence=0.0, raw_detections=[])
        h, w = frame.shape[:2]
        regions = seat_regions(w, h)
        results = self.model(frame, verbose=False)
        names = self.model.names
        detections: list[Detection] = []
        confs: list[float] = []
        boxes = results[0].boxes
        if boxes is None:
            return VisionResult(None, [], 0.0, [])
        for xyxy, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            c = float(conf)
            if c < self.confidence_threshold:
                continue
            tile = names[int(cls_id)]
            x1, y1, x2, y2 = map(float, xyxy)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            seat = seat_for_point(cx, cy, regions)
            detections.append(Detection(tile, (x1, y1, x2, y2), c, seat))
            confs.append(c)
        winner = pick_winner(detections)
        tiles = [d.tile for d in detections if d.seat == winner and d.tile != "face_down"]
        confidence = sum(confs) / len(confs) if confs else 0.0
        return VisionResult(winner_seat=winner, tiles=tiles, confidence=confidence, raw_detections=detections)
