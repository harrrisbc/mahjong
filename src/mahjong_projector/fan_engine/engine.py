from __future__ import annotations

from mahjong_projector.fan_engine.decompose import analyze_hand
from mahjong_projector.fan_engine.fans import FAN_TABLE
from mahjong_projector.types import FanHit, FanResult, HandState


class FanEngine:
    def score(self, hand: HandState) -> FanResult:
        analysis = analyze_hand(hand.tiles)
        matched: list[FanHit] = []
        for spec in FAN_TABLE:
            if spec.check(hand, analysis):
                matched.append(FanHit(id=spec.id, fan=spec.fan))
        if not matched:
            return FanResult(matched=[], selected=None)
        best_fan = max(m.fan for m in matched)
        selected = next(m for m in matched if m.fan == best_fan)
        return FanResult(matched=matched, selected=selected)
