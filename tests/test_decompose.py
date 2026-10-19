from mahjong_projector.fan_engine.decompose import analyze_hand
from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.types import HandState


def test_analyze_all_pungs():
    tiles = ["1m", "1m", "1m", "2p", "2p", "2p", "3s", "3s", "3s", "we", "we", "we", "dr", "dr"]
    analysis = analyze_hand(tiles)
    assert analysis is not None
    assert analysis.pair == ("dr", "dr")
    assert sum(1 for m in analysis.melds if m.kind == "pung") == 4


def test_analyze_rejects_garbage():
    assert analyze_hand(["1m"] * 14) is None


def test_four_of_a_kind_in_14_tiles_is_not_a_kong():
    tiles = ["1m"] * 4 + ["2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "dr", "dr"]
    analysis = analyze_hand(tiles)
    assert analysis is not None
    assert all(m.kind != "kong" for m in analysis.melds)


def test_one_kong_15_tiles():
    tiles = ["1m"] * 4 + ["2p", "3p", "4p", "5s", "6s", "7s", "we", "we", "we", "dr", "dr"]
    analysis = analyze_hand(tiles)
    assert analysis is not None
    assert [m.kind for m in analysis.melds].count("kong") == 1
    assert len(analysis.melds) == 4


def test_four_kongs_18_tiles():
    tiles = ["1m"] * 4 + ["2p"] * 4 + ["3s"] * 4 + ["we"] * 4 + ["dr", "dr"]
    analysis = analyze_hand(tiles)
    assert analysis is not None
    assert [m.kind for m in analysis.melds].count("kong") == 4


def test_kong_lengths_reject_non_win_shapes():
    assert analyze_hand(["1m"] * 4 + ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"]) is None
    assert analyze_hand(["1m"] * 4 + ["2p"] * 4 + ["3s"] * 4 + ["we"] * 4 + ["dr", "dg", "dw"]) is None


def test_kong_counts_toward_da_san_yuan():
    tiles = ["dr"] * 4 + ["dg"] * 3 + ["dw"] * 3 + ["1m", "2m", "3m", "9p", "9p"]
    result = FanEngine().score(HandState(tiles=tiles, winner_seat="east"))
    assert result.selected is not None
    assert result.selected.id == "da_san_yuan"
