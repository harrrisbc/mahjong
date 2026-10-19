from mahjong_projector.types import FanResult, FanHit, HandState


def test_fan_result_selected_is_highest_shaped():
    hand = HandState(tiles=["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9m", "9m", "5m"], winner_seat="east")
    assert hand.winner_seat == "east"
    hit = FanHit(id="qing_yi_se", fan=7)
    result = FanResult(matched=[hit], selected=hit)
    assert result.selected.id == "qing_yi_se"
