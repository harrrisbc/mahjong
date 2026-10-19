from __future__ import annotations

from mahjong_projector.types import Detection, Seat

# normalized x0,y0,x1,y1
_PRESETS: dict[Seat, tuple[float, float, float, float]] = {
    "north": (0.25, 0.00, 0.75, 0.28),
    "south": (0.25, 0.72, 0.75, 1.00),
    "west": (0.00, 0.25, 0.28, 0.75),
    "east": (0.72, 0.25, 1.00, 0.75),
}


def seat_regions(width: int, height: int) -> dict[Seat, tuple[int, int, int, int]]:
    out: dict[Seat, tuple[int, int, int, int]] = {}
    for seat, (x0, y0, x1, y1) in _PRESETS.items():
        out[seat] = (
            int(x0 * width),
            int(y0 * height),
            int(x1 * width),
            int(y1 * height),
        )
    return out


def seat_for_point(x: float, y: float, regions: dict[Seat, tuple[int, int, int, int]]) -> Seat | None:
    for seat, (x0, y0, x1, y1) in regions.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return seat
    return None


def pick_winner(detections: list[Detection]) -> Seat | None:
    counts: dict[Seat, int] = {"east": 0, "south": 0, "west": 0, "north": 0}
    for d in detections:
        if d.tile == "face_down" or d.seat is None:
            continue
        counts[d.seat] += 1
    best = max(counts.values())
    if best == 0:
        return None
    leaders = [s for s, n in counts.items() if n == best]
    if len(leaders) != 1:
        return None
    return leaders[0]
