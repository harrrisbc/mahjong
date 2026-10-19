from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mahjong_projector.fan_engine.decompose import HandAnalysis
from mahjong_projector.tiles import (
    DRAGONS,
    WINDS,
    count_tiles,
    is_honor,
    is_terminal,
    is_terminal_or_honor,
    suit_of,
)
from mahjong_projector.types import HandState


@dataclass(frozen=True)
class FanSpec:
    id: str
    fan: int
    check: Callable[[HandState, HandAnalysis | None], bool]


def _suits_used(tiles: list[str]) -> set[str]:
    return {s for t in tiles if (s := suit_of(t)) is not None}


def has_thirteen_orphans(tiles: list[str]) -> bool:
    need = {f"{n}{s}" for s in "mps" for n in (1, 9)} | set(WINDS) | set(DRAGONS)
    c = count_tiles(tiles)
    if set(c) != need:
        return False
    return sum(1 for v in c.values() if v == 2) == 1 and all(v >= 1 for v in c.values())


def has_nine_gates(tiles: list[str]) -> bool:
    suits = _suits_used(tiles)
    if len(suits) != 1 or any(is_honor(t) for t in tiles):
        return False
    s = next(iter(suits))
    c = count_tiles(tiles)
    need = {f"1{s}": 3, f"9{s}": 3}
    for n in range(2, 9):
        need[f"{n}{s}"] = 1
    # 1112345678999 covers 13 tiles; a 14th on any rank 1-9 completes the gates
    if len(tiles) != 14 or not set(c) <= set(need):
        return False
    return all(c.get(k, 0) >= v for k, v in need.items())


def _dragon_pung_count(analysis: HandAnalysis) -> tuple[int, bool]:
    pungs = sum(1 for m in analysis.melds if m.kind == "pung" and m.tiles[0] in DRAGONS)
    pair_dragon = analysis.pair[0] in DRAGONS
    return pungs, pair_dragon


def _wind_pung_count(analysis: HandAnalysis) -> tuple[int, bool]:
    pungs = sum(1 for m in analysis.melds if m.kind == "pung" and m.tiles[0] in WINDS)
    pair_wind = analysis.pair[0] in WINDS
    return pungs, pair_wind


def _check_peng_peng(hand: HandState, analysis: HandAnalysis | None) -> bool:
    return analysis is not None and analysis.is_all_pungs


def _check_hun_yi_se(hand: HandState, analysis: HandAnalysis | None) -> bool:
    tiles = hand.tiles
    suits = _suits_used(tiles)
    return len(suits) == 1 and any(is_honor(t) for t in tiles)


def _check_qing_yi_se(hand: HandState, analysis: HandAnalysis | None) -> bool:
    tiles = hand.tiles
    suits = _suits_used(tiles)
    return len(suits) == 1 and not any(is_honor(t) for t in tiles) and len(tiles) > 0


def _check_da_san_yuan(hand: HandState, analysis: HandAnalysis | None) -> bool:
    if analysis is None:
        return False
    pungs, _ = _dragon_pung_count(analysis)
    return pungs == 3


def _check_xiao_san_yuan(hand: HandState, analysis: HandAnalysis | None) -> bool:
    if analysis is None:
        return False
    pungs, pair_dragon = _dragon_pung_count(analysis)
    return pungs == 2 and pair_dragon


def _check_da_si_xi(hand: HandState, analysis: HandAnalysis | None) -> bool:
    if analysis is None:
        return False
    pungs, _ = _wind_pung_count(analysis)
    return pungs == 4


def _check_xiao_si_xi(hand: HandState, analysis: HandAnalysis | None) -> bool:
    if analysis is None:
        return False
    pungs, pair_wind = _wind_pung_count(analysis)
    return pungs == 3 and pair_wind


def _check_zi_yi_se(hand: HandState, analysis: HandAnalysis | None) -> bool:
    return bool(hand.tiles) and all(is_honor(t) for t in hand.tiles)


def _check_qing_yao_jiu(hand: HandState, analysis: HandAnalysis | None) -> bool:
    return bool(hand.tiles) and all(is_terminal(t) for t in hand.tiles)


def _check_hun_yao_jiu(hand: HandState, analysis: HandAnalysis | None) -> bool:
    tiles = hand.tiles
    if not tiles or not all(is_terminal_or_honor(t) for t in tiles):
        return False
    return any(is_terminal(t) for t in tiles) and any(is_honor(t) for t in tiles)


def _check_shi_san_yao(hand: HandState, analysis: HandAnalysis | None) -> bool:
    return has_thirteen_orphans(hand.tiles)


def _check_jiu_lian(hand: HandState, analysis: HandAnalysis | None) -> bool:
    return has_nine_gates(hand.tiles)


def _check_ji_hu(hand: HandState, analysis: HandAnalysis | None) -> bool:
    return analysis is not None


def _never(hand: HandState, analysis: HandAnalysis | None) -> bool:
    return False


# Ordered: higher fans first among equals uses this list order for tie-break
FAN_TABLE: list[FanSpec] = [
    FanSpec("da_si_xi", 13, _check_da_si_xi),
    FanSpec("shi_san_yao", 13, _check_shi_san_yao),
    FanSpec("jiu_lian_bao_deng", 13, _check_jiu_lian),
    FanSpec("qing_yao_jiu", 13, _check_qing_yao_jiu),
    FanSpec("zi_yi_se", 13, _check_zi_yi_se),
    FanSpec("da_san_yuan", 8, _check_da_san_yuan),
    FanSpec("qing_yi_se", 7, _check_qing_yi_se),
    FanSpec("xiao_si_xi", 6, _check_xiao_si_xi),
    FanSpec("xiao_san_yuan", 5, _check_xiao_san_yuan),
    FanSpec("hun_yao_jiu", 5, _check_hun_yao_jiu),
    FanSpec("peng_peng_hu", 3, _check_peng_peng),
    FanSpec("hun_yi_se", 3, _check_hun_yi_se),
    FanSpec("ji_hu", 1, _check_ji_hu),
    # context-only placeholders (always false in v1)
    FanSpec("zi_mo", 1, _never),
    FanSpec("men_qing", 1, _never),
    FanSpec("wu_hua", 1, _never),
]
