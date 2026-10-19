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
