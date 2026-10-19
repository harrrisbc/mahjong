from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from mahjong_projector.content_map.mapper import ContentMapper
from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.player.playback import Player
from mahjong_projector.types import HandState, PlayRequest, VoiceEvent
from mahjong_projector.voice.listener import VoiceListener

log = logging.getLogger(__name__)

State = Literal["idle", "listening", "capturing", "classifying", "playing"]


class Orchestrator:
    def __init__(
        self,
        voice: VoiceListener,
        vision: Any,
        fan_engine: FanEngine,
        content: ContentMapper,
        player: Player,
        capture_frame: Callable[[], np.ndarray | None],
        confidence_threshold: float = 0.50,
    ) -> None:
        self.voice = voice
        self.vision = vision
        self.fan_engine = fan_engine
        self.content = content
        self.player = player
        self.capture_frame = capture_frame
        self.confidence_threshold = confidence_threshold
        self.state: State = "listening"

    def on_voice(self, event: VoiceEvent) -> PlayRequest | None:
        if self.state == "playing":
            log.info("ignore 食 while playing")
            return None
        if event.keyword != "食":
            return None
        return self.handle_sik()

    def handle_sik(self) -> PlayRequest:
        self.state = "capturing"
        frame = self.capture_frame()
        if frame is None:
            log.error("camera frame failed")
            self.state = "listening"
            req = self.content.resolve(self.fan_engine.score(HandState([], None)))
            # force default/skip path via empty fan — ContentMapper no_fan → default/skip
            self._play(req)
            return req

        self.state = "classifying"
        vision = self.vision.analyze(frame)
        log.info(
            "vision seat=%s tiles=%s confidence=%.2f",
            vision.winner_seat,
            vision.tiles,
            vision.confidence,
        )
        if (
            vision.winner_seat is None
            or not vision.tiles
            or vision.confidence < self.confidence_threshold
        ):
            fan_result = self.fan_engine.score(HandState([], vision.winner_seat))
        else:
            fan_result = self.fan_engine.score(HandState(vision.tiles, vision.winner_seat))
        log.info("fans matched=%s selected=%s", fan_result.matched, fan_result.selected)
        req = self.content.resolve(fan_result)
        log.info("play path=%s reason=%s", req.video_path, req.reason)
        self._play(req)
        return req

    def _play(self, req: PlayRequest) -> None:
        self.state = "playing"
        try:
            self.player.play(req)
        finally:
            self.state = "listening"

    def run(
        self,
        read_chunk: Callable[[], bytes | None],
    ) -> None:
        self.state = "listening"
        self.voice.listen_forever(
            on_event=lambda event: self.on_voice(event),
            read_chunk=read_chunk,
        )
