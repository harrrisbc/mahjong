from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    content_dir: Path = Path("content")
    manifest_path: Path = Path("content/manifest.json")
    yolo_weights: Path = Path("models/tiles.pt")
    vosk_model_dir: Path = Path("models/vosk-model-small-cn-0.22")
    camera_index: int = 0
    display_index: int = 1
    vision_confidence_threshold: float = 0.50
    vision_detection_threshold: float = 0.25
    sample_rate: int = 16000

    @classmethod
    def load(cls) -> "AppConfig":
        return cls()
