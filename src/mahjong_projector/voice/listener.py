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

    def event_from_text(self, text: str) -> VoiceEvent | None:
        if KEYWORD in text:
            return VoiceEvent(keyword=KEYWORD, ts=time.time())
        return None

    def accept_audio(self, chunk: bytes) -> VoiceEvent | None:
        if self._recognizer is None:
            return None
        if self._recognizer.AcceptWaveform(chunk):
            payload = json.loads(self._recognizer.Result())
            return self.event_from_text(payload.get("text", ""))
        partial = json.loads(self._recognizer.PartialResult()).get("partial", "")
        return self.event_from_text(partial)

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
