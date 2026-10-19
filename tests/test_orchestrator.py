import numpy as np

from mahjong_projector.content_map.mapper import ContentMapper
from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.orchestrator import Orchestrator
from mahjong_projector.player.playback import Player
from mahjong_projector.types import VisionResult, VoiceEvent
from mahjong_projector.voice.listener import VoiceListener


class FakeVision:
    def analyze(self, frame):
        return VisionResult(
            winner_seat="east",
            tiles=["1m","1m","1m","2m","3m","4m","5m","6m","7m","8m","9m","9m","9m","5m"],
            confidence=0.9,
            raw_detections=[],
        )


def test_full_cycle_plays_highest_fan(tmp_path):
    import json
    from pathlib import Path

    fans = tmp_path / "fans"
    fans.mkdir()
    video = fans / "qing_yi_se.mp4"
    video.write_bytes(b"x")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"default": "default.mp4", "fans": {"qing_yi_se": {"fan": 7, "file": "fans/qing_yi_se.mp4"}}})
    )
    (tmp_path / "default.mp4").write_bytes(b"d")
    played: list[str] = []

    orch = Orchestrator(
        voice=VoiceListener(model_dir=None),
        vision=FakeVision(),
        fan_engine=FanEngine(),
        content=ContentMapper(tmp_path / "manifest.json", tmp_path),
        player=Player(runner=lambda cmd: played.append(cmd[-1]) or 0, which=lambda n: "/bin/mpv" if n == "mpv" else None),
        capture_frame=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        confidence_threshold=0.5,
    )
    req = orch.handle_sik()
    assert req.reason == "highest_fan"
    assert played and played[0].endswith("qing_yi_se.mp4")
    assert orch.state == "listening"


def test_low_confidence_defaults(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps({"default": "default.mp4", "fans": {}}))
    default = tmp_path / "default.mp4"
    default.write_bytes(b"d")

    class LowVision:
        def analyze(self, frame):
            return VisionResult("east", ["1m"], 0.1, [])

    played: list[str] = []
    orch = Orchestrator(
        voice=VoiceListener(model_dir=None),
        vision=LowVision(),
        fan_engine=FanEngine(),
        content=ContentMapper(tmp_path / "manifest.json", tmp_path),
        player=Player(runner=lambda cmd: played.append(cmd[-1]) or 0, which=lambda n: "/bin/mpv" if n == "mpv" else None),
        capture_frame=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        confidence_threshold=0.5,
    )
    req = orch.handle_sik()
    assert req.reason == "default"
    assert played[0].endswith("default.mp4")


def test_on_voice_ignores_sik_while_playing(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps({"default": "default.mp4", "fans": {}}))
    (tmp_path / "default.mp4").write_bytes(b"d")
    play_count = 0

    def runner(_cmd):
        nonlocal play_count
        play_count += 1
        return 0

    orch = Orchestrator(
        voice=VoiceListener(model_dir=None),
        vision=FakeVision(),
        fan_engine=FanEngine(),
        content=ContentMapper(tmp_path / "manifest.json", tmp_path),
        player=Player(runner=runner, which=lambda n: "/bin/mpv" if n == "mpv" else None),
        capture_frame=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
    )
    orch.state = "playing"
    assert orch.on_voice(VoiceEvent("食", 0.0)) is None
    assert play_count == 0
