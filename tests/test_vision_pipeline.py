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


def test_pipeline_with_fake_model():
    class FakeModel:
        names = {0: "1m", 1: "2m"}

        def __call__(self, frame, verbose=False):
            class Boxes:
                xyxy = np.array([[460.0, 800.0, 540.0, 860.0], [560.0, 800.0, 640.0, 860.0]])
                conf = np.array([0.9, 0.8])
                cls = np.array([0.0, 1.0])

            class R:
                boxes = Boxes()

            return [R()]

    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = VisionPipeline(model=FakeModel(), confidence_threshold=0.5).analyze(frame)
    assert result.winner_seat == "south"
    assert result.tiles == ["1m", "2m"]
    assert result.confidence >= 0.5


def test_from_weights_missing_returns_empty():
    from pathlib import Path

    pipe = VisionPipeline.from_weights(Path("/nonexistent/tiles.pt"))
    assert pipe.model is None
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipe.analyze(frame)
    assert result.winner_seat is None
    assert result.tiles == []
    assert result.confidence == 0.0
