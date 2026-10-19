from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Seat = Literal["east", "south", "west", "north"]


@dataclass(frozen=True)
class VoiceEvent:
    keyword: str
    ts: float


@dataclass(frozen=True)
class Detection:
    tile: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    seat: Seat | None


@dataclass(frozen=True)
class VisionResult:
    winner_seat: Seat | None
    tiles: list[str]
    confidence: float
    raw_detections: list[Detection] = field(default_factory=list)


@dataclass(frozen=True)
class HandState:
    tiles: list[str]
    winner_seat: Seat | None


@dataclass(frozen=True)
class FanHit:
    id: str
    fan: int


@dataclass(frozen=True)
class FanResult:
    matched: list[FanHit]
    selected: FanHit | None


@dataclass(frozen=True)
class PlayRequest:
    video_path: str | None
    reason: str
