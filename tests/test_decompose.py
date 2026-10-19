from mahjong_projector.fan_engine.decompose import analyze_hand


def test_analyze_all_pungs():
    tiles = ["1m", "1m", "1m", "2p", "2p", "2p", "3s", "3s", "3s", "we", "we", "we", "dr", "dr"]
    analysis = analyze_hand(tiles)
    assert analysis is not None
    assert analysis.pair == ("dr", "dr")
    assert sum(1 for m in analysis.melds if m.kind == "pung") == 4


def test_analyze_rejects_garbage():
    assert analyze_hand(["1m"] * 14) is None
