from __future__ import annotations

import logging
import sys
from collections.abc import Callable

import numpy as np

from mahjong_projector.config import AppConfig
from mahjong_projector.content_map.mapper import ContentMapper
from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.orchestrator import Orchestrator
from mahjong_projector.player.playback import Player
from mahjong_projector.vision.pipeline import VisionPipeline
from mahjong_projector.voice.listener import VoiceListener

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mahjong_projector")


def build_orchestrator(cfg: AppConfig, open_camera: bool = True) -> Orchestrator:
    cap = None
    if open_camera:
        import cv2

        cap = cv2.VideoCapture(cfg.camera_index)

    def capture_frame() -> np.ndarray | None:
        if cap is None:
            return None
        ok, frame = cap.read()
        return frame if ok else None

    return Orchestrator(
        voice=VoiceListener(cfg.vosk_model_dir if cfg.vosk_model_dir.is_dir() else None),
        vision=VisionPipeline.from_weights(cfg.yolo_weights, cfg.vision_confidence_threshold),
        fan_engine=FanEngine(),
        content=ContentMapper(cfg.manifest_path, cfg.content_dir),
        player=Player(display_index=cfg.display_index),
        capture_frame=capture_frame,
        confidence_threshold=cfg.vision_confidence_threshold,
    )


def _mic_chunk_reader(sample_rate: int):
    import sounddevice as sd

    q: list[bytes] = []

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        q.append(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
    )
    stream.start()

    def read() -> bytes | None:
        if not q:
            sd.sleep(50)
            return b""
        return q.pop(0)

    def drain() -> None:
        q.clear()

    read.stream = stream  # type: ignore[attr-defined]
    read.drain = drain  # type: ignore[attr-defined]
    return read


def voice_loop(
    orch: Orchestrator,
    read: Callable[[], bytes | None],
    drain: Callable[[], None],
) -> None:
    while True:
        chunk = read()
        if chunk is None:
            return
        if not chunk:
            continue
        event = orch.voice.accept_audio(chunk)
        if event is None:
            continue
        orch.on_voice(event)
        # audio captured while classifying/playing must not re-trigger the next cycle
        drain()
        orch.voice.reset()


def main(argv: list[str] | None = None) -> int:
    cfg = AppConfig.load()
    orch = build_orchestrator(cfg, open_camera=True)
    log.info("listening for 食 — ctrl+c to exit")
    read = _mic_chunk_reader(cfg.sample_rate)
    try:
        voice_loop(orch, read, read.drain)  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        log.info("bye")
        return 0
    finally:
        stream = getattr(read, "stream", None)
        if stream is not None:
            stream.stop()
            stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
