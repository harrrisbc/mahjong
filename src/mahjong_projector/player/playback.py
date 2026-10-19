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
