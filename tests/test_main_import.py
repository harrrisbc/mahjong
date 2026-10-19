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
