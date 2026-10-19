import json
from pathlib import Path

from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.types import HandState


def test_qing_yi_se_selected():
    tiles = ["1m","1m","1m","2m","3m","4m","5m","6m","7m","8m","9m","9m","9m","5m"]
    result = FanEngine().score(HandState(tiles=tiles, winner_seat="east"))
    assert result.selected is not None
    assert result.selected.id == "qing_yi_se"
    assert result.selected.fan == 7


def test_fixture_hands():
    data = json.loads(Path("tests/fixtures/hands.json").read_text())
    engine = FanEngine()
    for case in data:
        result = engine.score(HandState(tiles=case["tiles"], winner_seat=case.get("winner_seat")))
        assert result.selected is not None, case["name"]
        assert result.selected.id == case["expected_fan_id"], case["name"]
