from __future__ import annotations

from mahjong_projector.fan_engine.decompose import analyze_hand, strip_flowers
from mahjong_projector.fan_engine.fans import FAN_TABLE, has_nine_gates, has_thirteen_orphans
from mahjong_projector.types import FanHit, FanResult, HandState


class FanEngine:
    def score(self, hand: HandState) -> FanResult:
        tiles = strip_flowers(hand.tiles)
        analysis = analyze_hand(tiles)
        complete = analysis is not None or (
            len(tiles) == 14 and (has_thirteen_orphans(tiles) or has_nine_gates(tiles))
        )
        if not complete:
            return FanResult(matched=[], selected=None)
        body = HandState(tiles=tiles, winner_seat=hand.winner_seat)
        matched: list[FanHit] = []
        for spec in FAN_TABLE:
            if spec.check(body, analysis):
                matched.append(FanHit(id=spec.id, fan=spec.fan))
        if not matched:
            return FanResult(matched=[], selected=None)
        best_fan = max(m.fan for m in matched)
        selected = next(m for m in matched if m.fan == best_fan)
        return FanResult(matched=matched, selected=selected)
