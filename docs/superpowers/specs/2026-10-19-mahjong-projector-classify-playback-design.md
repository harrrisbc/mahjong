# Mahjong Projector — Classify & Playback Design

**Date:** 2026-10-19  
**Repo:** https://github.com/harrrisbc/mahjong.git  
**Status:** Draft for implementation planning  

## Goal

Build a laptop/desktop **classify + playback** system for Hong Kong mahjong:

1. Hear the voice trigger **「食」**
2. Use an overhead webcam + computer vision to find the winning seat and read exposed tiles
3. Score the hand with Hong Kong mahjong fan rules
4. Play the video mapped to the **highest** matched fan on a projector (fullscreen)

The user supplies all video content. The program only classifies and plays back.

**Out of scope for Phase 1:** geometric table projection mapping, polished override UI, perfect fan coverage when context (自摸/門清/風位) is unknown.

**Phase 2 (later):** table projection mapping (homography warp) inserted before playback, without changing classify logic.

## Approach

**Python desktop monolith** with clear pipeline modules:

`VoiceListener` → `VisionPipeline` → `FanEngine` → `ContentMapper` → `Player`

Chosen over Electron+sidecar (heavier for v1) and classify-stub-first (weaker “食就自動” experience).

## Architecture

```
Microphone → VoiceListener(「食」)
                ↓
         CaptureFrame(overhead webcam)
                ↓
         VisionPipeline
           - 4 seat regions
           - pick winning seat (revealed hand heuristic)
           - read exposed tiles → HandState
                ↓
         FanEngine(HK rules)
           - list matched fans
           - select highest fan
                ↓
         ContentMapper(fan_id → video path)
                ↓
         Player(fullscreen → projector)
```

**Orchestrator state machine:**  
`idle → listening → capturing → classifying → playing → idle`

Phase 2 adds `ProjectorMapper` immediately before `Player`.

## Components

### VoiceListener

- Continuously listen to the default microphone
- Trigger on keyword **「食」**
- V1 stack: **Vosk** offline Chinese/Cantonese-capable small model (low latency)
- No trigger → stay in listen loop; no UI prompt

### VisionPipeline

- Input: single overhead webcam frame (or short burst after trigger)
- Model: **YOLO11 (Ultralytics) multi-class** for tile detect + classify
- Classes: standard HK tile set (~34 suits/honors; optional flowers `f1–f8`; optional helpers like `face_down`)
- Seats: four preset regions (east/south/west/north); calibration can come later
- Winner selection (v1): heuristic — which seat region shows a newly revealed face-up hand after 「食」
- Training: bootstrap from public mahjong tile datasets, fine-tune on real table overhead photos
- Output: `VisionResult` with seat, tile codes, confidence, raw detections

### FanEngine

- Ruleset: **Hong Kong mahjong**, as-complete fan table as practical
- Input: `HandState` (tiles + winner seat + optional unknown context)
- Output: all matched fans + **selected highest fan** (`fan_id`, fan value)
- V1 limitation: only fans decidable from exposed tile face patterns are reliable. Unknown 自摸/食糊、門清、圈風/位風 → those fans may be missed; acceptable for v1

### ContentMapper

- Reads `content/manifest.json`
- Maps `fan_id` → video file under `content/fans/`
- Missing file → `content/default.mp4`
- Missing default → skip playback

### Player

- Fullscreen playback on projector / secondary display
- V1: **mpv** (fallback ffplay)
- During playback, ignore new 「食」 events until return to idle

### App / Orchestrator

- Wires modules; logs `seat / tiles / fans / selected / video` for debugging
- No blocking manual UI in v1

## Data contracts

### VoiceEvent

```json
{ "keyword": "食", "ts": 0 }
```

### VisionResult

```json
{
  "winner_seat": "east",
  "tiles": ["1m", "1m", "1m", "2m"],
  "confidence": 0.82,
  "raw_detections": []
}
```

`winner_seat`: `"east" | "south" | "west" | "north" | null`

### HandState (FanEngine input)

- `tiles`: recognized exposed tiles
- `winner_seat`
- Optional context fields may exist later; v1 may leave them unknown

### Tile codes (canonical)

| Group | Codes |
|-------|--------|
| Characters (萬) | `1m`–`9m` |
| Dots (筒) | `1p`–`9p` |
| Bamboo (索) | `1s`–`9s` |
| Winds | `we`, `ws`, `ww`, `wn` |
| Dragons | `dr` (紅中), `dg` (發財), `dw` (白板) |
| Flowers (optional) | `f1`–`f8` |

These codes are the only wire format between VisionPipeline and FanEngine.

### FanResult

```json
{
  "matched": [{ "id": "qing_yi_se", "fan": 7 }],
  "selected": { "id": "qing_yi_se", "fan": 7 }
}
```

Multi-fan hands: **play only the highest-fan video** (`selected`). Tie-break: stable order in the fan table (first highest wins).

### PlayRequest

```json
{ "video_path": "content/fans/qing_yi_se.mp4", "reason": "highest_fan" }
```

or `{ "video_path": null, "reason": "skip" }`

## Content layout

```text
content/
  manifest.json
  default.mp4
  fans/
    qing_yi_se.mp4
    peng_peng_hu.mp4
    ...
```

`manifest.json` responsibilities:

- List known HK `fan_id`s and fan values (priority)
- Map each `fan_id` to a relative video path (optional until user adds content)
- Declare `default` video path

User owns all creative assets; empty mappings are normal in early use.

## Error handling (v1 — rough)

| Case | Behavior |
|------|----------|
| No 「食」 detected | Keep listening |
| Camera / frame failure | Log + skip |
| No winner or tiles | `default.mp4`, else skip |
| Low vision confidence | `default.mp4`, else skip |
| Zero matched fans | `default.mp4`, else skip |
| Fan matched but file missing | `default.mp4`, else skip |
| Playback failure | Log + return idle |
| 「食」 while playing | Ignore until idle |

Confidence threshold is a config value (default chosen during implementation; start conservative).

## Testing strategy

- **FanEngine:** unit tests from known hands → expected highest `fan_id`
- **ContentMapper:** missing fan file, missing default, happy path
- **Voice / Vision:** fixtures (sample audio clip, sample overhead images); mocks in orchestrator tests
- **Manual smoke:** say 「食」 → inject/sample frame → confirm fullscreen video on target display

## Non-goals (v1)

- Manual seat/fan override UI
- Perfect live scoring with full table context
- Multi-video sequences or combo-specific clips
- Geometric projector calibration / warp (Phase 2)
- Network / multi-device sync

## Success criteria (v1)

1. Saying 「食」 starts one classify→playback cycle on a laptop with webcam + projector
2. Overhead vision returns a seat + tile list often enough for demo use (failures degrade to default/skip)
3. FanEngine selects a highest HK fan from tile patterns it can evaluate
4. Matching video plays fullscreen when content exists; otherwise default/skip
5. Modules are independently mockable for tests

## Phase 2 preview

- Add `ProjectorMapper` (homography from calibration points)
- Optional seat-region calibration UI
- Optional manual override and richer context inputs (自摸、門清、風位) to unlock more fans
