# Mahjong Classify & Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python desktop app that hears 「食」, reads the winning hand via overhead YOLO vision, scores Hong Kong fans, and fullscreen-plays the highest-fan video on a projector.

**Architecture:** Single Python package with a linear orchestrator state machine (`idle → listening → capturing → classifying → playing → idle`). Modules communicate only through typed dataclasses (`VoiceEvent`, `VisionResult`, `HandState`, `FanResult`, `PlayRequest`). Content is external under `content/`; Phase 2 projector warp is not built.

**Tech Stack:** Python 3.11+, pytest, Vosk (voice), OpenCV + Ultralytics YOLO11 (vision), mpv/ffplay (playback), JSON manifest for fan→video mapping.

## Global Constraints

- Tile wire codes only: `1m`–`9m`, `1p`–`9p`, `1s`–`9s`, winds `we/ws/ww/wn`, dragons `dr/dg/dw`, optional flowers `f1`–`f8`
- Ruleset: Hong Kong mahjong; multi-fan → play **highest fan only**; tie-break = first in fan table order
- Voice trigger keyword: exactly 「食」
- Vision model: YOLO11 multi-class (Ultralytics)
- Playback: mpv preferred, ffplay fallback; fullscreen to projector/secondary display
- Errors (v1): camera/vision/fan/content failures → `default.mp4` else skip; ignore 「食」 while playing
- No manual override UI, no Phase 2 warp, no network sync
- Package import root: `mahjong_projector`
- Default vision confidence threshold: `0.50`

---

## File structure

```text
pyproject.toml
README.md
src/mahjong_projector/
  __init__.py
  __main__.py
  types.py                 # shared dataclasses + Seat literal
  config.py                # paths, thresholds, display index
  tiles.py                 # tile helpers / validation
  fan_engine/
    __init__.py
    decompose.py           # 14-tile meld search (pung/chow/kong/pair)
    fans.py                # HK fan registry + checkers
    engine.py              # score hand → FanResult
  content_map/
    __init__.py
    mapper.py              # manifest + filesystem fallbacks
  voice/
    __init__.py
    listener.py            # Vosk keyword listener
  vision/
    __init__.py
    seats.py               # 4 seat region presets + winner heuristic
    pipeline.py            # YOLO infer → VisionResult
  player/
    __init__.py
    playback.py            # mpv/ffplay fullscreen
  orchestrator.py          # state machine wiring
content/
  manifest.json
  fans/.gitkeep
  README.md                # how to drop videos
tests/
  test_tiles.py
  test_decompose.py
  test_fan_engine.py
  test_content_mapper.py
  test_player.py
  test_voice_listener.py
  test_vision_pipeline.py
  test_orchestrator.py
  fixtures/
    hands.json             # known hands → expected highest fan
    sample_frame.jpg       # optional; may be generated in tests
models/
  README.md                # where to put YOLO + Vosk model files
```

---

### Task 1: Project scaffold + shared types

**Files:**
- Create: `pyproject.toml`
- Create: `src/mahjong_projector/__init__.py`
- Create: `src/mahjong_projector/types.py`
- Create: `src/mahjong_projector/config.py`
- Create: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Seat`, `VoiceEvent`, `Detection`, `VisionResult`, `HandState`, `FanHit`, `FanResult`, `PlayRequest`, `AppConfig.load()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
from mahjong_projector.types import FanResult, FanHit, HandState


def test_fan_result_selected_is_highest_shaped():
    hand = HandState(tiles=["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9m", "9m", "5m"], winner_seat="east")
    assert hand.winner_seat == "east"
    hit = FanHit(id="qing_yi_se", fan=7)
    result = FanResult(matched=[hit], selected=hit)
    assert result.selected.id == "qing_yi_se"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace && pip install -e ".[dev]" && pytest tests/test_types.py -v`  
Expected: FAIL (package / types missing) or install fails until `pyproject.toml` exists — create scaffold in Step 3 then re-run; first run before implementation should show import error.

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mahjong-projector"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "opencv-python>=4.8",
  "numpy>=1.26",
  "ultralytics>=8.3",
  "vosk>=0.3.45",
  "sounddevice>=0.4.6",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
mahjong-projector = "mahjong_projector.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
# src/mahjong_projector/__init__.py
__version__ = "0.1.0"
```

```python
# src/mahjong_projector/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Seat = Literal["east", "south", "west", "north"]


@dataclass(frozen=True)
class VoiceEvent:
    keyword: str
    ts: float


@dataclass(frozen=True)
class Detection:
    tile: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    seat: Seat | None


@dataclass(frozen=True)
class VisionResult:
    winner_seat: Seat | None
    tiles: list[str]
    confidence: float
    raw_detections: list[Detection] = field(default_factory=list)


@dataclass(frozen=True)
class HandState:
    tiles: list[str]
    winner_seat: Seat | None


@dataclass(frozen=True)
class FanHit:
    id: str
    fan: int


@dataclass(frozen=True)
class FanResult:
    matched: list[FanHit]
    selected: FanHit | None


@dataclass(frozen=True)
class PlayRequest:
    video_path: str | None
    reason: str
```

```python
# src/mahjong_projector/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    content_dir: Path = Path("content")
    manifest_path: Path = Path("content/manifest.json")
    yolo_weights: Path = Path("models/tiles.pt")
    vosk_model_dir: Path = Path("models/vosk-model-small-cn-0.22")
    camera_index: int = 0
    display_index: int = 1
    vision_confidence_threshold: float = 0.50
    sample_rate: int = 16000

    @classmethod
    def load(cls) -> "AppConfig":
        return cls()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_types.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/mahjong_projector/__init__.py src/mahjong_projector/types.py src/mahjong_projector/config.py tests/test_types.py
git commit -m "feat: scaffold package and shared types"
```

---

### Task 2: Tile helpers

**Files:**
- Create: `src/mahjong_projector/tiles.py`
- Create: `tests/test_tiles.py`

**Interfaces:**
- Consumes: canonical tile code strings from Global Constraints
- Produces: `VALID_TILES`, `is_valid_tile(code)`, `suit_of(code)`, `is_honor(code)`, `is_terminal(code)`, `is_dragon(code)`, `is_wind(code)`, `count_tiles(tiles) -> Counter`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tiles.py
from mahjong_projector.tiles import count_tiles, is_honor, is_valid_tile, suit_of


def test_suit_and_honor():
    assert is_valid_tile("3p")
    assert suit_of("3p") == "p"
    assert is_honor("dr")
    assert not is_honor("3p")
    assert count_tiles(["1m", "1m", "dr"])["1m"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tiles.py -v`  
Expected: FAIL with `ModuleNotFoundError` or import error for `tiles`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mahjong_projector/tiles.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tiles.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mahjong_projector/tiles.py tests/test_tiles.py
git commit -m "feat: add tile code helpers"
```

---

### Task 3: Hand decomposer (melds)

**Files:**
- Create: `src/mahjong_projector/fan_engine/__init__.py`
- Create: `src/mahjong_projector/fan_engine/decompose.py`
- Create: `tests/test_decompose.py`

**Interfaces:**
- Consumes: `count_tiles`, tile helpers
- Produces: `Meld(kind, tiles)`, `HandAnalysis(pair, melds, tiles)`, `analyze_hand(tiles) -> HandAnalysis | None` (None if no standard 4-meld+pair win shape after stripping flowers)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decompose.py
from mahjong_projector.fan_engine.decompose import analyze_hand


def test_analyze_all_pungs():
    tiles = ["1m", "1m", "1m", "2p", "2p", "2p", "3s", "3s", "3s", "we", "we", "we", "dr", "dr"]
    analysis = analyze_hand(tiles)
    assert analysis is not None
    assert analysis.pair == ("dr", "dr")
    assert sum(1 for m in analysis.melds if m.kind == "pung") == 4


def test_analyze_rejects_garbage():
    assert analyze_hand(["1m"] * 14) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_decompose.py -v`  
Expected: FAIL import / missing `analyze_hand`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mahjong_projector/fan_engine/__init__.py
"""Fan engine package. FanEngine export is added in Task 4."""
```

```python
# src/mahjong_projector/fan_engine/decompose.py
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
    for tile, n in list(counts.items()):
        if n < 2:
            continue
        counts[tile] -= 2
        found = _search(counts, [])
        if found is not None and len(found) == 4:
            return HandAnalysis(tuple(body), (tile, tile), tuple(found))
        counts[tile] += 2
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_decompose.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mahjong_projector/fan_engine/__init__.py src/mahjong_projector/fan_engine/decompose.py tests/test_decompose.py
git commit -m "feat: add winning-hand meld decomposer"
```

---

### Task 4: Fan registry + FanEngine

**Files:**
- Create: `src/mahjong_projector/fan_engine/fans.py`
- Create: `src/mahjong_projector/fan_engine/engine.py`
- Create: `tests/test_fan_engine.py`
- Create: `tests/fixtures/hands.json`
- Modify: `src/mahjong_projector/fan_engine/__init__.py` (export already planned)

**Interfaces:**
- Consumes: `HandState`, `analyze_hand`, tile helpers
- Produces: `FAN_TABLE` ordered list; `FanEngine.score(hand: HandState) -> FanResult`

**HK fan table (v1 pattern-capable subset — full IDs reserved in manifest even if checker returns false without context):**

| id | fan | checker basis |
|----|-----|----------------|
| `ji_hu` | 1 | valid 14-tile win shape, no higher pattern fan |
| `peng_peng_hu` | 3 | `analysis.is_all_pungs` |
| `hun_yi_se` | 3 | honors + exactly one suit |
| `qing_yi_se` | 7 | exactly one suit, no honors |
| `xiao_san_yuan` | 5 | 2 dragon pungs + dragon pair |
| `da_san_yuan` | 8 | 3 dragon pungs |
| `xiao_si_xi` | 6 | 3 wind pungs + wind pair |
| `da_si_xi` | 13 | 4 wind pungs |
| `zi_yi_se` | 13 | all honors |
| `hun_yao_jiu` | 5 | all terminals/honors, mixed |
| `qing_yao_jiu` | 13 | all terminals (no honors) |
| `shi_san_yao` | 13 | 13 orphans pattern |
| `jiu_lian_bao_deng` | 13 | nine gates pattern in one suit |

Context-only fans (`zi_mo`, `men_qing`, `wu_hua`, seat winds, etc.) appear in `manifest.json` with video slots but **v1 checkers always false** (unknown context).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fan_engine.py
import json
from pathlib import Path

from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.types import HandState


def test_qing_yi_se_selected():
    tiles = ["1m","1m","1m","2m","3m","4m","5m","6m","7m","8m","9m","9m","9m","5m"]
    result = FanEngine().score(HandState(tiles=tiles, winner_seat="east"))
    assert result.selected is not None
    assert result.selected.id == "qing_yi_se"
    assert result.selected.fan == 7


def test_fixture_hands():
    data = json.loads(Path("tests/fixtures/hands.json").read_text())
    engine = FanEngine()
    for case in data:
        result = engine.score(HandState(tiles=case["tiles"], winner_seat=case.get("winner_seat")))
        assert result.selected is not None, case["name"]
        assert result.selected.id == case["expected_fan_id"], case["name"]
```

Use this exact `tests/fixtures/hands.json` content (JSON array):

```json
[
  {
    "name": "peng_peng_hu",
    "tiles": ["1m","1m","1m","2p","2p","2p","3s","3s","3s","we","we","we","dr","dr"],
    "winner_seat": "east",
    "expected_fan_id": "peng_peng_hu"
  },
  {
    "name": "qing_yi_se",
    "tiles": ["1m","1m","1m","2m","3m","4m","5m","6m","7m","8m","9m","9m","9m","5m"],
    "winner_seat": "south",
    "expected_fan_id": "qing_yi_se"
  },
  {
    "name": "zi_yi_se",
    "tiles": ["we","we","we","ws","ws","ws","ww","ww","ww","wn","wn","wn","dr","dr"],
    "winner_seat": "west",
    "expected_fan_id": "da_si_xi"
  }
]
```

For the `zi_yi_se` row above: four wind pungs + dragon pair should select `da_si_xi` (13) over `zi_yi_se` (13) by **table order** — put `da_si_xi` before `zi_yi_se` in `FAN_TABLE`, and keep this fixture expecting `da_si_xi`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fan_engine.py -v`  
Expected: FAIL missing `FanEngine`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mahjong_projector/fan_engine/fans.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mahjong_projector.fan_engine.decompose import HandAnalysis, analyze_hand
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


def _has_thirteen_orphans(tiles: list[str]) -> bool:
    need = {f"{n}{s}" for s in "mps" for n in (1, 9)} | set(WINDS) | set(DRAGONS)
    c = count_tiles(tiles)
    if set(c) != need:
        return False
    return sum(1 for v in c.values() if v == 2) == 1 and all(v >= 1 for v in c.values())


def _has_nine_gates(tiles: list[str]) -> bool:
    suits = _suits_used(tiles)
    if len(suits) != 1 or any(is_honor(t) for t in tiles):
        return False
    s = next(iter(suits))
    c = count_tiles(tiles)
    need = {f"1{s}": 3, f"9{s}": 3}
    for n in range(2, 9):
        need[f"{n}{s}"] = 1
    # nine gates: 1112345678999 + one extra in-suit
    base_ok = all(c.get(k, 0) >= v for k, v in need.items())
    return base_ok and len(tiles) == 14 and set(suit_of(t) for t in tiles) == {s}


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
    return _has_thirteen_orphans(hand.tiles)


def _check_jiu_lian(hand: HandState, analysis: HandAnalysis | None) -> bool:
    return _has_nine_gates(hand.tiles)


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
```

```python
# src/mahjong_projector/fan_engine/engine.py
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
```

Fix `__init__.py` if Step from Task 3 imported engine before it existed — ensure:

```python
from mahjong_projector.fan_engine.engine import FanEngine
__all__ = ["FanEngine"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fan_engine.py tests/test_decompose.py -v`  
Expected: PASS

If `peng_peng_hu` loses to nothing else, OK. If `da_si_xi` hand also matches `zi_yi_se` and `peng_peng_hu`, selected must be `da_si_xi`.

- [ ] **Step 5: Commit**

```bash
git add src/mahjong_projector/fan_engine/fans.py src/mahjong_projector/fan_engine/engine.py src/mahjong_projector/fan_engine/__init__.py tests/test_fan_engine.py tests/fixtures/hands.json
git commit -m "feat: add Hong Kong fan engine and fixtures"
```

---

### Task 5: ContentMapper + manifest

**Files:**
- Create: `src/mahjong_projector/content_map/__init__.py`
- Create: `src/mahjong_projector/content_map/mapper.py`
- Create: `content/manifest.json`
- Create: `content/fans/.gitkeep`
- Create: `content/README.md`
- Create: `tests/test_content_mapper.py`

**Interfaces:**
- Consumes: `FanResult`, `AppConfig.manifest_path` / `content_dir`
- Produces: `ContentMapper.resolve(fan_result: FanResult) -> PlayRequest`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_content_mapper.py
import json
from pathlib import Path

from mahjong_projector.content_map.mapper import ContentMapper
from mahjong_projector.types import FanHit, FanResult


def test_maps_highest_fan_file(tmp_path: Path):
    fans = tmp_path / "fans"
    fans.mkdir()
    video = fans / "qing_yi_se.mp4"
    video.write_bytes(b"fake")
    manifest = {
        "default": "default.mp4",
        "fans": {
            "qing_yi_se": {"fan": 7, "file": "fans/qing_yi_se.mp4"},
            "peng_peng_hu": {"fan": 3, "file": "fans/peng_peng_hu.mp4"},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "default.mp4").write_bytes(b"d")
    mapper = ContentMapper(tmp_path / "manifest.json", tmp_path)
    req = mapper.resolve(FanResult(matched=[FanHit("qing_yi_se", 7)], selected=FanHit("qing_yi_se", 7)))
    assert req.video_path == str(video.resolve())
    assert req.reason == "highest_fan"


def test_missing_fan_uses_default(tmp_path: Path):
    manifest = {"default": "default.mp4", "fans": {"qing_yi_se": {"fan": 7, "file": "fans/missing.mp4"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    default = tmp_path / "default.mp4"
    default.write_bytes(b"d")
    mapper = ContentMapper(tmp_path / "manifest.json", tmp_path)
    req = mapper.resolve(FanResult(matched=[FanHit("qing_yi_se", 7)], selected=FanHit("qing_yi_se", 7)))
    assert req.video_path == str(default.resolve())
    assert req.reason == "default"


def test_missing_default_skips(tmp_path: Path):
    manifest = {"default": "default.mp4", "fans": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    mapper = ContentMapper(tmp_path / "manifest.json", tmp_path)
    req = mapper.resolve(FanResult(matched=[], selected=None))
    assert req.video_path is None
    assert req.reason == "skip"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_content_mapper.py -v`  
Expected: FAIL missing mapper

- [ ] **Step 3: Write minimal implementation**

```python
# src/mahjong_projector/content_map/__init__.py
from mahjong_projector.content_map.mapper import ContentMapper
__all__ = ["ContentMapper"]
```

```python
# src/mahjong_projector/content_map/mapper.py
from __future__ import annotations

import json
from pathlib import Path

from mahjong_projector.types import FanResult, PlayRequest


class ContentMapper:
    def __init__(self, manifest_path: Path, content_dir: Path) -> None:
        self.content_dir = content_dir
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._default_rel = data.get("default", "default.mp4")
        self._fans: dict[str, dict] = data.get("fans", {})

    def resolve(self, fan_result: FanResult) -> PlayRequest:
        if fan_result.selected is None:
            return self._default_or_skip("no_fan")
        entry = self._fans.get(fan_result.selected.id)
        if entry and entry.get("file"):
            path = (self.content_dir / entry["file"]).resolve()
            if path.is_file():
                return PlayRequest(video_path=str(path), reason="highest_fan")
        return self._default_or_skip("missing_fan_video")

    def _default_or_skip(self, reason: str) -> PlayRequest:
        path = (self.content_dir / self._default_rel).resolve()
        if path.is_file():
            return PlayRequest(video_path=str(path), reason="default")
        return PlayRequest(video_path=None, reason="skip")
```

`content/manifest.json` must list every `FAN_TABLE` id plus context placeholders:

```json
{
  "default": "default.mp4",
  "fans": {
    "da_si_xi": {"fan": 13, "file": "fans/da_si_xi.mp4"},
    "shi_san_yao": {"fan": 13, "file": "fans/shi_san_yao.mp4"},
    "jiu_lian_bao_deng": {"fan": 13, "file": "fans/jiu_lian_bao_deng.mp4"},
    "qing_yao_jiu": {"fan": 13, "file": "fans/qing_yao_jiu.mp4"},
    "zi_yi_se": {"fan": 13, "file": "fans/zi_yi_se.mp4"},
    "da_san_yuan": {"fan": 8, "file": "fans/da_san_yuan.mp4"},
    "qing_yi_se": {"fan": 7, "file": "fans/qing_yi_se.mp4"},
    "xiao_si_xi": {"fan": 6, "file": "fans/xiao_si_xi.mp4"},
    "xiao_san_yuan": {"fan": 5, "file": "fans/xiao_san_yuan.mp4"},
    "hun_yao_jiu": {"fan": 5, "file": "fans/hun_yao_jiu.mp4"},
    "peng_peng_hu": {"fan": 3, "file": "fans/peng_peng_hu.mp4"},
    "hun_yi_se": {"fan": 3, "file": "fans/hun_yi_se.mp4"},
    "ji_hu": {"fan": 1, "file": "fans/ji_hu.mp4"},
    "zi_mo": {"fan": 1, "file": "fans/zi_mo.mp4"},
    "men_qing": {"fan": 1, "file": "fans/men_qing.mp4"},
    "wu_hua": {"fan": 1, "file": "fans/wu_hua.mp4"}
  }
}
```

```markdown
# content/README.md

Drop your fan videos as `fans/<fan_id>.mp4` matching keys in `manifest.json`.
If a fan file is missing, playback uses `default.mp4`, or skips if that is missing too.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_content_mapper.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mahjong_projector/content_map content/manifest.json content/fans/.gitkeep content/README.md tests/test_content_mapper.py
git commit -m "feat: add content mapper and fan video manifest"
```

---

### Task 6: Player (mpv / ffplay)

**Files:**
- Create: `src/mahjong_projector/player/__init__.py`
- Create: `src/mahjong_projector/player/playback.py`
- Create: `tests/test_player.py`

**Interfaces:**
- Consumes: `PlayRequest`, `display_index: int`
- Produces: `Player.play(request: PlayRequest) -> bool` (True if a player process ran successfully; False on skip/failure)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_player.py
from mahjong_projector.player.playback import Player
from mahjong_projector.types import PlayRequest


def test_skip_when_no_path():
    assert Player(display_index=0, runner=lambda cmd: 0).play(PlayRequest(None, "skip")) is False


def test_builds_mpv_command(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    ok = Player(display_index=1, runner=runner, which=lambda name: "/usr/bin/mpv" if name == "mpv" else None).play(
        PlayRequest(str(video), "highest_fan")
    )
    assert ok is True
    assert calls[0][0] == "/usr/bin/mpv"
    assert "--fs" in calls[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_player.py -v`  
Expected: FAIL missing Player

- [ ] **Step 3: Write minimal implementation**

```python
# src/mahjong_projector/player/__init__.py
from mahjong_projector.player.playback import Player
__all__ = ["Player"]
```

```python
# src/mahjong_projector/player/playback.py
from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from mahjong_projector.types import PlayRequest

log = logging.getLogger(__name__)


class Player:
    def __init__(
        self,
        display_index: int = 1,
        runner: Callable[[list[str]], int] | None = None,
        which: Callable[[str], str | None] | None = None,
    ) -> None:
        self.display_index = display_index
        self._runner = runner or self._default_runner
        self._which = which or shutil.which

    def play(self, request: PlayRequest) -> bool:
        if not request.video_path:
            log.info("playback skip reason=%s", request.reason)
            return False
        path = Path(request.video_path)
        if not path.is_file():
            log.error("playback missing file %s", path)
            return False
        mpv = self._which("mpv")
        if mpv:
            cmd = [mpv, "--fs", f"--fs-screen={self.display_index}", "--really-quiet", str(path)]
        else:
            ffplay = self._which("ffplay")
            if not ffplay:
                log.error("neither mpv nor ffplay found")
                return False
            cmd = [ffplay, "-fs", "-autoexit", "-loglevel", "quiet", str(path)]
        try:
            code = self._runner(cmd)
            return code == 0
        except Exception:
            log.exception("playback failed")
            return False

    @staticmethod
    def _default_runner(cmd: list[str]) -> int:
        return subprocess.call(cmd)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_player.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mahjong_projector/player tests/test_player.py
git commit -m "feat: add mpv/ffplay fullscreen player"
```

---

### Task 7: VoiceListener (Vosk keyword)

**Files:**
- Create: `src/mahjong_projector/voice/__init__.py`
- Create: `src/mahjong_projector/voice/listener.py`
- Create: `tests/test_voice_listener.py`
- Create: `models/README.md`

**Interfaces:**
- Consumes: `vosk_model_dir`, `sample_rate`, audio source callback
- Produces: `VoiceListener.poll() -> VoiceEvent | None`; `VoiceListener.listen_forever(on_event)` optional helper

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voice_listener.py
from mahjong_projector.voice.listener import VoiceListener


def test_detects_sik_in_partial_text():
    listener = VoiceListener(model_dir=None, recognizer=None)
    event = listener.event_from_text("嘅食啦")
    assert event is not None
    assert event.keyword == "食"


def test_ignores_other_text():
    listener = VoiceListener(model_dir=None, recognizer=None)
    assert listener.event_from_text("打牌") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_voice_listener.py -v`  
Expected: FAIL missing VoiceListener

- [ ] **Step 3: Write minimal implementation**

```python
# src/mahjong_projector/voice/__init__.py
from mahjong_projector.voice.listener import VoiceListener
__all__ = ["VoiceListener"]
```

```python
# src/mahjong_projector/voice/listener.py
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from mahjong_projector.types import VoiceEvent

log = logging.getLogger(__name__)

KEYWORD = "食"


class VoiceListener:
    def __init__(
        self,
        model_dir: Path | None,
        sample_rate: int = 16000,
        recognizer: Any | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._recognizer = recognizer
        if recognizer is None and model_dir is not None and Path(model_dir).is_dir():
            from vosk import KaldiRecognizer, Model

            self._recognizer = KaldiRecognizer(Model(str(model_dir)), sample_rate)

    def event_from_text(self, text: str) -> VoiceEvent | None:
        if KEYWORD in text:
            return VoiceEvent(keyword=KEYWORD, ts=time.time())
        return None

    def accept_audio(self, chunk: bytes) -> VoiceEvent | None:
        if self._recognizer is None:
            return None
        if self._recognizer.AcceptWaveform(chunk):
            payload = json.loads(self._recognizer.Result())
            return self.event_from_text(payload.get("text", ""))
        partial = json.loads(self._recognizer.PartialResult()).get("partial", "")
        return self.event_from_text(partial)

    def listen_forever(self, on_event: Callable[[VoiceEvent], None], read_chunk: Callable[[], bytes | None]) -> None:
        while True:
            chunk = read_chunk()
            if chunk is None:
                break
            event = self.accept_audio(chunk)
            if event is not None:
                on_event(event)
```

Write `models/README.md` explaining:

- put `tiles.pt` (YOLO11 mahjong weights) in `models/`
- put Vosk Chinese model at `models/vosk-model-small-cn-0.22/`
- download with: `curl -LO https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip` then unzip inside `models/`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_voice_listener.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mahjong_projector/voice tests/test_voice_listener.py models/README.md
git commit -m "feat: add Vosk voice listener for 食"
```

---

### Task 8: Vision seats + YOLO pipeline

**Files:**
- Create: `src/mahjong_projector/vision/__init__.py`
- Create: `src/mahjong_projector/vision/seats.py`
- Create: `src/mahjong_projector/vision/pipeline.py`
- Create: `tests/test_vision_pipeline.py`

**Interfaces:**
- Consumes: BGR `numpy.ndarray` frame, YOLO model wrapper, confidence threshold
- Produces: `seat_regions(width, height) -> dict[Seat, (x1,y1,x2,y2)]`, `pick_winner(detections) -> Seat | None`, `VisionPipeline.analyze(frame) -> VisionResult`

Seat preset (normalized fractions of frame):

| seat | x0 | y0 | x1 | y1 |
|------|----|----|----|----|
| north | 0.25 | 0.00 | 0.75 | 0.28 |
| south | 0.25 | 0.72 | 0.75 | 1.00 |
| west | 0.00 | 0.25 | 0.28 | 0.75 |
| east | 0.72 | 0.25 | 1.00 | 0.75 |

Winner heuristic: seat with the most face-up tile detections (excluding class `face_down` if present); ties → `None`.

YOLO class names must equal canonical tile codes (`1m`, `dr`, …). Missing weights → pipeline returns empty result with confidence `0.0` (orchestrator will default/skip).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vision_pipeline.py
import numpy as np

from mahjong_projector.types import Detection
from mahjong_projector.vision.pipeline import VisionPipeline
from mahjong_projector.vision.seats import pick_winner, seat_for_point, seat_regions


def test_seat_regions_cover_four_sides():
    regions = seat_regions(1000, 1000)
    assert set(regions) == {"east", "south", "west", "north"}
    assert seat_for_point(500, 50, regions) == "north"


def test_pick_winner_by_count():
    dets = [
        Detection("1m", (100, 800, 140, 860), 0.9, "south"),
        Detection("2m", (150, 800, 190, 860), 0.9, "south"),
        Detection("3m", (800, 400, 840, 460), 0.9, "east"),
    ]
    assert pick_winner(dets) == "south"


def test_pipeline_with_fake_model():
    class FakeModel:
        names = {0: "1m", 1: "2m"}

        def __call__(self, frame, verbose=False):
            class Boxes:
                xyxy = np.array([[100.0, 800.0, 140.0, 860.0], [150.0, 800.0, 190.0, 860.0]])
                conf = np.array([0.9, 0.8])
                cls = np.array([0.0, 1.0])

            class R:
                boxes = Boxes()

            return [R()]

    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = VisionPipeline(model=FakeModel(), confidence_threshold=0.5).analyze(frame)
    assert result.winner_seat == "south"
    assert result.tiles == ["1m", "2m"]
    assert result.confidence >= 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_pipeline.py -v`  
Expected: FAIL missing vision modules

- [ ] **Step 3: Write minimal implementation**

```python
# src/mahjong_projector/vision/__init__.py
from mahjong_projector.vision.pipeline import VisionPipeline
__all__ = ["VisionPipeline"]
```

```python
# src/mahjong_projector/vision/seats.py
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
```

```python
# src/mahjong_projector/vision/pipeline.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mahjong_projector.types import Detection, VisionResult
from mahjong_projector.vision.seats import pick_winner, seat_for_point, seat_regions

log = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(self, model: Any | None = None, confidence_threshold: float = 0.50) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_weights(cls, weights: Path, confidence_threshold: float = 0.50) -> "VisionPipeline":
        if not weights.is_file():
            log.warning("YOLO weights missing at %s", weights)
            return cls(model=None, confidence_threshold=confidence_threshold)
        from ultralytics import YOLO

        return cls(model=YOLO(str(weights)), confidence_threshold=confidence_threshold)

    def analyze(self, frame: np.ndarray) -> VisionResult:
        if self.model is None or frame is None or frame.size == 0:
            return VisionResult(winner_seat=None, tiles=[], confidence=0.0, raw_detections=[])
        h, w = frame.shape[:2]
        regions = seat_regions(w, h)
        results = self.model(frame, verbose=False)
        names = self.model.names
        detections: list[Detection] = []
        confs: list[float] = []
        boxes = results[0].boxes
        if boxes is None:
            return VisionResult(None, [], 0.0, [])
        for xyxy, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            c = float(conf)
            if c < self.confidence_threshold:
                continue
            tile = names[int(cls_id)]
            x1, y1, x2, y2 = map(float, xyxy)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            seat = seat_for_point(cx, cy, regions)
            detections.append(Detection(tile, (x1, y1, x2, y2), c, seat))
            confs.append(c)
        winner = pick_winner(detections)
        tiles = [d.tile for d in detections if d.seat == winner and d.tile != "face_down"]
        confidence = sum(confs) / len(confs) if confs else 0.0
        return VisionResult(winner_seat=winner, tiles=tiles, confidence=confidence, raw_detections=detections)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_pipeline.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mahjong_projector/vision tests/test_vision_pipeline.py
git commit -m "feat: add seat regions and YOLO vision pipeline"
```

---

### Task 9: Orchestrator state machine

**Files:**
- Create: `src/mahjong_projector/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `VoiceListener`, `VisionPipeline`, `FanEngine`, `ContentMapper`, `Player`, camera `capture_frame() -> np.ndarray | None`, `AppConfig`
- Produces: `Orchestrator.handle_sik() -> PlayRequest` (one-shot cycle); `Orchestrator.run()` loop; property `state`

States: `idle | listening | capturing | classifying | playing`

Rules from spec:

- low confidence / no tiles / no winner / no fan → mapper default/skip path
- while `playing`, ignore additional 「食」
- log seat, tiles, matched fans, selected, video

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
import numpy as np

from mahjong_projector.content_map.mapper import ContentMapper
from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.orchestrator import Orchestrator
from mahjong_projector.player.playback import Player
from mahjong_projector.types import PlayRequest, VisionResult
from mahjong_projector.voice.listener import VoiceListener


class FakeVision:
    def analyze(self, frame):
        return VisionResult(
            winner_seat="east",
            tiles=["1m","1m","1m","2m","3m","4m","5m","6m","7m","8m","9m","9m","9m","5m"],
            confidence=0.9,
            raw_detections=[],
        )


def test_full_cycle_plays_highest_fan(tmp_path):
    import json
    from pathlib import Path

    fans = tmp_path / "fans"
    fans.mkdir()
    video = fans / "qing_yi_se.mp4"
    video.write_bytes(b"x")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"default": "default.mp4", "fans": {"qing_yi_se": {"fan": 7, "file": "fans/qing_yi_se.mp4"}}})
    )
    (tmp_path / "default.mp4").write_bytes(b"d")
    played: list[str] = []

    orch = Orchestrator(
        voice=VoiceListener(model_dir=None),
        vision=FakeVision(),
        fan_engine=FanEngine(),
        content=ContentMapper(tmp_path / "manifest.json", tmp_path),
        player=Player(runner=lambda cmd: played.append(cmd[-1]) or 0, which=lambda n: "/bin/mpv" if n == "mpv" else None),
        capture_frame=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        confidence_threshold=0.5,
    )
    req = orch.handle_sik()
    assert req.reason == "highest_fan"
    assert played and played[0].endswith("qing_yi_se.mp4")
    assert orch.state == "listening"


def test_low_confidence_defaults(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps({"default": "default.mp4", "fans": {}}))
    default = tmp_path / "default.mp4"
    default.write_bytes(b"d")

    class LowVision:
        def analyze(self, frame):
            return VisionResult("east", ["1m"], 0.1, [])

    played: list[str] = []
    orch = Orchestrator(
        voice=VoiceListener(model_dir=None),
        vision=LowVision(),
        fan_engine=FanEngine(),
        content=ContentMapper(tmp_path / "manifest.json", tmp_path),
        player=Player(runner=lambda cmd: played.append(cmd[-1]) or 0, which=lambda n: "/bin/mpv" if n == "mpv" else None),
        capture_frame=lambda: np.zeros((10, 10, 3), dtype=np.uint8),
        confidence_threshold=0.5,
    )
    req = orch.handle_sik()
    assert req.reason == "default"
    assert played[0].endswith("default.mp4")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py -v`  
Expected: FAIL missing Orchestrator

- [ ] **Step 3: Write minimal implementation**

```python
# src/mahjong_projector/orchestrator.py
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from mahjong_projector.content_map.mapper import ContentMapper
from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.player.playback import Player
from mahjong_projector.types import HandState, PlayRequest, VoiceEvent
from mahjong_projector.voice.listener import VoiceListener

log = logging.getLogger(__name__)

State = Literal["idle", "listening", "capturing", "classifying", "playing"]


class Orchestrator:
    def __init__(
        self,
        voice: VoiceListener,
        vision: Any,
        fan_engine: FanEngine,
        content: ContentMapper,
        player: Player,
        capture_frame: Callable[[], np.ndarray | None],
        confidence_threshold: float = 0.50,
    ) -> None:
        self.voice = voice
        self.vision = vision
        self.fan_engine = fan_engine
        self.content = content
        self.player = player
        self.capture_frame = capture_frame
        self.confidence_threshold = confidence_threshold
        self.state: State = "listening"

    def on_voice(self, event: VoiceEvent) -> PlayRequest | None:
        if self.state == "playing":
            log.info("ignore 食 while playing")
            return None
        if event.keyword != "食":
            return None
        return self.handle_sik()

    def handle_sik(self) -> PlayRequest:
        self.state = "capturing"
        frame = self.capture_frame()
        if frame is None:
            log.error("camera frame failed")
            self.state = "listening"
            req = self.content.resolve(self.fan_engine.score(HandState([], None)))
            # force default/skip path via empty fan — ContentMapper no_fan → default/skip
            self._play(req)
            return req

        self.state = "classifying"
        vision = self.vision.analyze(frame)
        log.info(
            "vision seat=%s tiles=%s confidence=%.2f",
            vision.winner_seat,
            vision.tiles,
            vision.confidence,
        )
        if (
            vision.winner_seat is None
            or not vision.tiles
            or vision.confidence < self.confidence_threshold
        ):
            fan_result = self.fan_engine.score(HandState([], vision.winner_seat))
        else:
            fan_result = self.fan_engine.score(HandState(vision.tiles, vision.winner_seat))
        log.info("fans matched=%s selected=%s", fan_result.matched, fan_result.selected)
        req = self.content.resolve(fan_result)
        log.info("play path=%s reason=%s", req.video_path, req.reason)
        self._play(req)
        return req

    def _play(self, req: PlayRequest) -> None:
        self.state = "playing"
        try:
            self.player.play(req)
        finally:
            self.state = "listening"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mahjong_projector/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: wire classify-playback orchestrator"
```

---

### Task 10: CLI entrypoint + README smoke path

**Files:**
- Create: `src/mahjong_projector/__main__.py`
- Create: `README.md`
- Modify: `models/README.md` only if needed (already created)

**Interfaces:**
- Consumes: `AppConfig.load()`, real camera via OpenCV, sounddevice mic chunks
- Produces: `python -m mahjong_projector` / `mahjong-projector` runnable loop

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main_import.py
from mahjong_projector.__main__ import build_orchestrator
from mahjong_projector.config import AppConfig


def test_build_orchestrator_with_missing_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "content" / "fans").mkdir(parents=True)
    (tmp_path / "content" / "manifest.json").write_text(
        '{"default":"default.mp4","fans":{}}', encoding="utf-8"
    )
    cfg = AppConfig(
        content_dir=tmp_path / "content",
        manifest_path=tmp_path / "content" / "manifest.json",
        yolo_weights=tmp_path / "missing.pt",
        vosk_model_dir=tmp_path / "missing-vosk",
    )
    orch = build_orchestrator(cfg, open_camera=False)
    assert orch.state == "listening"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main_import.py -v`  
Expected: FAIL missing `__main__.build_orchestrator`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mahjong_projector/__main__.py
from __future__ import annotations

import logging
import sys

import numpy as np

from mahjong_projector.config import AppConfig
from mahjong_projector.content_map.mapper import ContentMapper
from mahjong_projector.fan_engine.engine import FanEngine
from mahjong_projector.orchestrator import Orchestrator
from mahjong_projector.player.playback import Player
from mahjong_projector.vision.pipeline import VisionPipeline
from mahjong_projector.voice.listener import VoiceListener

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mahjong_projector")


def build_orchestrator(cfg: AppConfig, open_camera: bool = True) -> Orchestrator:
    cap = None
    if open_camera:
        import cv2

        cap = cv2.VideoCapture(cfg.camera_index)

    def capture_frame() -> np.ndarray | None:
        if cap is None:
            return None
        ok, frame = cap.read()
        return frame if ok else None

    return Orchestrator(
        voice=VoiceListener(cfg.vosk_model_dir if cfg.vosk_model_dir.is_dir() else None),
        vision=VisionPipeline.from_weights(cfg.yolo_weights, cfg.vision_confidence_threshold),
        fan_engine=FanEngine(),
        content=ContentMapper(cfg.manifest_path, cfg.content_dir),
        player=Player(display_index=cfg.display_index),
        capture_frame=capture_frame,
        confidence_threshold=cfg.vision_confidence_threshold,
    )


def _mic_chunk_reader(sample_rate: int):
    import sounddevice as sd

    q: list[bytes] = []

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        q.append(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
    )
    stream.start()

    def read() -> bytes | None:
        if not q:
            sd.sleep(50)
            return b""
        return q.pop(0)

    read.stream = stream  # type: ignore[attr-defined]
    return read


def main(argv: list[str] | None = None) -> int:
    cfg = AppConfig.load()
    orch = build_orchestrator(cfg, open_camera=True)
    log.info("listening for 食 — ctrl+c to exit")
    read = _mic_chunk_reader(cfg.sample_rate)
    try:
        while True:
            chunk = read()
            if not chunk:
                continue
            event = orch.voice.accept_audio(chunk)
            if event is not None:
                orch.on_voice(event)
    except KeyboardInterrupt:
        log.info("bye")
        return 0
    finally:
        stream = getattr(read, "stream", None)
        if stream is not None:
            stream.stop()
            stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```markdown
# Mahjong Projector

Voice 「食」 → overhead YOLO tile read → Hong Kong fan classify → fullscreen fan video.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# install mpv: sudo apt install mpv
# download Vosk + YOLO weights into models/ (see models/README.md)
# add videos under content/fans/ per content/manifest.json
```

## Run

```bash
mahjong-projector
# or
python -m mahjong_projector
```

## Test

```bash
pytest -v
```

## Design

See `docs/superpowers/specs/2026-10-19-mahjong-projector-classify-playback-design.md`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -v`  
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/mahjong_projector/__main__.py README.md tests/test_main_import.py
git commit -m "feat: add CLI entrypoint and project README"
```

---

## Plan self-review

**1. Spec coverage**

| Spec requirement | Task |
|------------------|------|
| Voice 「食」 (Vosk) | Task 7, 10 |
| Overhead YOLO vision + seat heuristic | Task 8 |
| HK fan engine, highest fan only | Task 3–4 |
| Content manifest + default/skip | Task 5 |
| mpv/ffplay fullscreen projector | Task 6 |
| Orchestrator states + ignore while playing | Task 9 |
| Canonical tile codes | Task 2 + Global Constraints |
| Error → default/skip | Task 5, 9 |
| Phase 2 warp deferred | Non-goal; no task |
| Unit tests + fixtures | Tasks 1–10 |
| Manual smoke path documented | Task 10 README |

**2. Placeholder scan:** no TBD/TODO steps; concrete code in each implementation step.

**3. Type consistency:** `FanHit.id/fan`, `PlayRequest.video_path/reason`, `VisionResult.winner_seat/tiles/confidence`, `HandState.tiles/winner_seat`, seat literals `east|south|west|north` used consistently; tile codes `we/ws/ww/wn` + `dr/dg/dw`.

**Gap fixed during planning:** context-only fans listed in manifest for content slots; checkers return false until Phase 2+ context inputs exist.
