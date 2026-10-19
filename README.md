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
