from mahjong_projector.voice.listener import VoiceListener


def test_detects_sik_in_partial_text():
    listener = VoiceListener(model_dir=None, recognizer=None)
    event = listener.event_from_text("嘅食啦")
    assert event is not None
    assert event.keyword == "食"


def test_ignores_other_text():
    listener = VoiceListener(model_dir=None, recognizer=None)
    assert listener.event_from_text("打牌") is None
