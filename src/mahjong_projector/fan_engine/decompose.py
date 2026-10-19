from __future__ import annotations

from dataclasses import dataclass

from mahjong_projector.tiles import FLOWERS, count_tiles, suit_of


@dataclass(frozen=True)
class Meld:
    kind: str  # "pung" | "chow" | "kong"
    tiles: tuple[str, ...]


@dataclass(frozen=True)
class HandAnalysis:
    tiles: tuple[str, ...]
    pair: tuple[str, str]
    melds: tuple[Meld, ...]

    @property
    def is_all_pungs(self) -> bool:
        return all(m.kind in {"pung", "kong"} for m in self.melds)


def _strip_flowers(tiles: list[str]) -> list[str]:
    return [t for t in tiles if t not in FLOWERS]


def _chow_options(tile: str, counts: dict[str, int]) -> list[tuple[str, str, str]]:
    s = suit_of(tile)
    if s is None:
        return []
    n = int(tile[0])
    out: list[tuple[str, str, str]] = []
    for start in (n - 2, n - 1, n):
        if start < 1 or start > 7:
            continue
        group = (f"{start}{s}", f"{start+1}{s}", f"{start+2}{s}")
        trial = counts.copy()
        ok = True
        for g in group:
            if trial.get(g, 0) <= 0:
                ok = False
                break
            trial[g] -= 1
        if ok:
            out.append(group)
    return out


def _search(counts: dict[str, int], melds: list[Meld]) -> list[Meld] | None:
    if sum(counts.values()) == 0:
        return melds
    tile = next(t for t, c in counts.items() if c > 0)
    # pung
    if counts[tile] >= 3:
        counts[tile] -= 3
        melds.append(Meld("pung", (tile, tile, tile)))
        if _search(counts, melds) is not None:
            return melds
        melds.pop()
        counts[tile] += 3
    # chow
    for group in _chow_options(tile, counts):
        for g in group:
            counts[g] -= 1
        melds.append(Meld("chow", group))
        if _search(counts, melds) is not None:
            return melds
        melds.pop()
        for g in group:
            counts[g] += 1
    return None


def analyze_hand(tiles: list[str]) -> HandAnalysis | None:
    body = _strip_flowers(tiles)
    if len(body) not in {14, 11, 8}:  # allow partial revealed hands only at 14 for v1 win shape
        # v1: require 14-tile body for structured fans; callers may still use pattern fans on raw tiles
        pass
    if len(body) != 14:
        return None
    counts = dict(count_tiles(body))
    if any(n > 4 for n in counts.values()):
        return None
    for tile, n in list(counts.items()):
        if n < 2:
            continue
        counts[tile] -= 2
        found = _search(counts, [])
        if found is not None and len(found) == 4:
            return HandAnalysis(tuple(body), (tile, tile), tuple(found))
        counts[tile] += 2
    return None
