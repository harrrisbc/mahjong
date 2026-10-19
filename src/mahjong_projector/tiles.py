from __future__ import annotations

from collections import Counter

SUITS = ("m", "p", "s")
WINDS = ("we", "ws", "ww", "wn")
DRAGONS = ("dr", "dg", "dw")
FLOWERS = tuple(f"f{i}" for i in range(1, 9))

VALID_TILES: frozenset[str] = frozenset(
    [f"{n}{s}" for s in SUITS for n in range(1, 10)]
    + list(WINDS)
    + list(DRAGONS)
    + list(FLOWERS)
)


def is_valid_tile(code: str) -> bool:
    return code in VALID_TILES


def suit_of(code: str) -> str | None:
    if len(code) == 2 and code[0].isdigit() and code[1] in SUITS:
        return code[1]
    return None


def is_honor(code: str) -> bool:
    return code in WINDS or code in DRAGONS


def is_wind(code: str) -> bool:
    return code in WINDS


def is_dragon(code: str) -> bool:
    return code in DRAGONS


def is_terminal(code: str) -> bool:
    return len(code) == 2 and code[0] in {"1", "9"} and code[1] in SUITS


def is_terminal_or_honor(code: str) -> bool:
    return is_terminal(code) or is_honor(code)


def count_tiles(tiles: list[str]) -> Counter[str]:
    return Counter(tiles)
