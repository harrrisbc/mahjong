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
