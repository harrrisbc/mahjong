from mahjong_projector.tiles import count_tiles, is_honor, is_valid_tile, suit_of


def test_suit_and_honor():
    assert is_valid_tile("3p")
    assert suit_of("3p") == "p"
    assert is_honor("dr")
    assert not is_honor("3p")
    assert count_tiles(["1m", "1m", "dr"])["1m"] == 2
