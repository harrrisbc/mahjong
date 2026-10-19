import numpy as np

from mahjong_projector.types import Detection
from mahjong_projector.vision.pipeline import VisionPipeline
from mahjong_projector.vision.seats import pick_winner, seat_for_point, seat_regions


def test_seat_regions_cover_four_sides():
    regions = seat_regions(1000, 1000)
    assert set(regions) == {"east", "south", "west", "north"}
    assert seat_for_point(500, 50, regions) == "north"


def test_pick_winner_by_count():
    dets = [
        Detection("1m", (100, 800, 140, 860), 0.9, "south"),
        Detection("2m", (150, 800, 190, 860), 0.9, "south"),
        Detection("3m", (800, 400, 840, 460), 0.9, "east"),
    ]
    assert pick_winner(dets) == "south"


def _fake_model(xyxy, conf, cls, names=None):
    class FakeModel:
        def __call__(self, frame, verbose=False):
            class Boxes:
                pass

            Boxes.xyxy = np.array(xyxy)
            Boxes.conf = np.array(conf)
            Boxes.cls = np.array(cls)

            class R:
                boxes = Boxes()

            return [R()]

    FakeModel.names = names or {0: "1m", 1: "2m"}
    return FakeModel()


def test_pipeline_with_fake_model():
    model = _fake_model(
        [[460.0, 800.0, 540.0, 860.0], [560.0, 800.0, 640.0, 860.0]], [0.9, 0.8], [0.0, 1.0]
    )
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = VisionPipeline(model=model).analyze(frame)
    assert result.winner_seat == "south"
    assert result.tiles == ["1m", "2m"]
    assert result.confidence == 0.8


def test_confidence_is_min_of_winner_tiles_only():
    model = _fake_model(
        [
            [460.0, 800.0, 540.0, 860.0],  # south
            [560.0, 800.0, 640.0, 860.0],  # south, weak
            [800.0, 400.0, 840.0, 460.0],  # east, strong but not winner
        ],
        [0.95, 0.3, 0.99],
        [0.0, 1.0, 0.0],
    )
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = VisionPipeline(model=model, detection_threshold=0.25).analyze(frame)
    assert result.winner_seat == "south"
    assert result.tiles == ["1m", "2m"]
    assert result.confidence == 0.3


def test_detections_below_floor_are_dropped():
    model = _fake_model(
        [[460.0, 800.0, 540.0, 860.0], [560.0, 800.0, 640.0, 860.0]], [0.9, 0.1], [0.0, 1.0]
    )
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = VisionPipeline(model=model, detection_threshold=0.25).analyze(frame)
    assert result.tiles == ["1m"]
    assert result.confidence == 0.9


def test_weak_winner_tile_gates_orchestrator_to_default(tmp_path):
    import json

    from mahjong_projector.content_map.mapper import ContentMapper
    from mahjong_projector.fan_engine.engine import FanEngine
    from mahjong_projector.orchestrator import Orchestrator
    from mahjong_projector.player.playback import Player
    from mahjong_projector.voice.listener import VoiceListener

    hand = ["1m","2m","3m","4m","5m","6m","7m","7m","7m","8m","8m","8m","9m","9m"]
    names = {i: t for i, t in enumerate(sorted(set(hand)))}
    ids = {t: i for i, t in names.items()}
    boxes = [[300.0 + 25 * i, 800.0, 320.0 + 25 * i, 860.0] for i in range(len(hand))]
    confs = [0.95] * (len(hand) - 1) + [0.3]
    model = _fake_model(boxes, confs, [float(ids[t]) for t in hand], names=names)

    (tmp_path / "fans").mkdir()
    (tmp_path / "fans" / "qing_yi_se.mp4").write_bytes(b"x")
    (tmp_path / "default.mp4").write_bytes(b"d")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"default": "default.mp4", "fans": {"qing_yi_se": {"fan": 7, "file": "fans/qing_yi_se.mp4"}}})
    )
    played: list[str] = []
    orch = Orchestrator(
        voice=VoiceListener(model_dir=None),
        vision=VisionPipeline(model=model, detection_threshold=0.25),
        fan_engine=FanEngine(),
        content=ContentMapper(tmp_path / "manifest.json", tmp_path),
        player=Player(runner=lambda cmd: played.append(cmd[-1]) or 0, which=lambda n: "/bin/mpv"),
        capture_frame=lambda: np.zeros((1000, 1000, 3), dtype=np.uint8),
        confidence_threshold=0.5,
    )
    req = orch.handle_sik()
    assert req.reason == "default"
    assert played[0].endswith("default.mp4")


def test_from_weights_missing_returns_empty():
    from pathlib import Path

    pipe = VisionPipeline.from_weights(Path("/nonexistent/tiles.pt"))
    assert pipe.model is None
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipe.analyze(frame)
    assert result.winner_seat is None
    assert result.tiles == []
    assert result.confidence == 0.0
