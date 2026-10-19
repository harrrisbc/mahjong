import json
from pathlib import Path

import pytest

from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.types import HandState

QING_YI_SE = ["1m","2m","3m","4m","5m","6m","7m","7m","7m","8m","8m","8m","9m","9m"]
THIRTEEN_ORPHANS = ["1m","9m","1p","9p","1s","9s","we","ws","ww","wn","dr","dg","dw","dw"]


def _score(tiles):
    return FanEngine().score(HandState(tiles=tiles, winner_seat="east"))


def test_qing_yi_se_selected():
    result = _score(QING_YI_SE)
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


@pytest.mark.parametrize(
    "tiles",
    [
        ["1m"],
        ["we"],
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"],
        ["1m", "9m", "1p", "9p", "1s", "9s"],
        ["we", "we", "we", "ws", "ws", "ws", "dr", "dr"],
        ["f1", "f2"],
    ],
)
def test_incomplete_hand_scores_nothing(tiles):
    result = _score(tiles)
    assert result.selected is None
    assert result.matched == []


def test_flowers_do_not_break_zi_yi_se():
    tiles = ["we","we","we","ws","ws","ws","ww","ww","ww","dr","dr","dr","dg","dg","f1","f5"]
    result = _score(tiles)
    assert result.selected is not None
    assert result.selected.id == "zi_yi_se"


def test_flowers_do_not_break_shi_san_yao():
    result = _score(THIRTEEN_ORPHANS + ["f3"])
    assert result.selected is not None
    assert result.selected.id == "shi_san_yao"


def test_flowers_do_not_break_qing_yi_se():
    result = _score(QING_YI_SE + ["f2"])
    assert result.selected is not None
    assert result.selected.id == "qing_yi_se"
