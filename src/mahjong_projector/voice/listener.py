from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from mahjong_projector.types import VoiceEvent

log = logging.getLogger(__name__)

KEYWORD = "食"


class VoiceListener:
    def __init__(
        self,
        model_dir: Path | None,
        sample_rate: int = 16000,
        recognizer: Any | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._recognizer = recognizer
        if recognizer is None and model_dir is not None and Path(model_dir).is_dir():
            from vosk import KaldiRecognizer, Model

            self._recognizer = KaldiRecognizer(Model(str(model_dir)), sample_rate)
        # Vosk repeats the growing partial on every chunk; latch so one utterance fires once.
        self._latched = False

    def event_from_text(self, text: str) -> VoiceEvent | None:
        if KEYWORD in text:
            return VoiceEvent(keyword=KEYWORD, ts=time.time())
        return None

    def reset(self) -> None:
        self._latched = False
        reset = getattr(self._recognizer, "Reset", None)
        if callable(reset):
            reset()

    def _fire(self, text: str) -> VoiceEvent | None:
        event = self.event_from_text(text)
        if event is not None:
            self.reset()
            self._latched = True
        return event

    def accept_audio(self, chunk: bytes) -> VoiceEvent | None:
        if self._recognizer is None:
            return None
        if self._recognizer.AcceptWaveform(chunk):
            text = json.loads(self._recognizer.Result()).get("text", "")
            if self._latched:
                self._latched = False
                return None
            event = self.event_from_text(text)
            if event is not None:
                self.reset()
            return event
        if self._latched:
            return None
        partial = json.loads(self._recognizer.PartialResult()).get("partial", "")
        return self._fire(partial)

    def listen_forever(
        self,
        on_event: Callable[[VoiceEvent], None],
        read_chunk: Callable[[], bytes | None],
    ) -> None:
        while True:
            chunk = read_chunk()
            if chunk is None:
                break
            event = self.accept_audio(chunk)
            if event is not None:
                on_event(event)
