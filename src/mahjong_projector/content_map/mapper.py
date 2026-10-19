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
