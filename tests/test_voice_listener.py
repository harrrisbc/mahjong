import json

from mahjong_projector.voice.listener import VoiceListener


def test_detects_sik_in_partial_text():
    listener = VoiceListener(model_dir=None, recognizer=None)
    event = listener.event_from_text("嘅食啦")
    assert event is not None
    assert event.keyword == "食"


def test_ignores_other_text():
    listener = VoiceListener(model_dir=None, recognizer=None)
    assert listener.event_from_text("打牌") is None


class FakeVosk:
    """Replays a scripted sequence of (is_final, text) like KaldiRecognizer."""

    def __init__(self, script: list[tuple[bool, str]]) -> None:
        self.script = list(script)
        self.current: tuple[bool, str] = (False, "")
        self.resets = 0

    def AcceptWaveform(self, chunk: bytes) -> bool:
        self.current = self.script.pop(0)
        return self.current[0]

    def Result(self) -> str:
        return json.dumps({"text": self.current[1]})

    def PartialResult(self) -> str:
        return json.dumps({"partial": self.current[1]})

    def Reset(self) -> None:
        self.resets += 1


def _feed(listener: VoiceListener, n: int) -> list:
    events = []
    for _ in range(n):
        ev = listener.accept_audio(b"\x00\x00")
        if ev is not None:
            events.append(ev)
    return events


def test_partial_partial_final_fires_once():
    rec = FakeVosk([(False, "食"), (False, "食 糊"), (True, "食 糊 啦")])
    listener = VoiceListener(model_dir=None, recognizer=rec)
    events = _feed(listener, 3)
    assert len(events) == 1
    assert rec.resets == 1


def test_latch_clears_after_final_for_next_utterance():
    rec = FakeVosk(
        [
            (False, "食"),
            (False, "食 糊"),
            (True, "食 糊"),
            (False, "打"),
            (False, "食"),
            (True, "食"),
        ]
    )
    listener = VoiceListener(model_dir=None, recognizer=rec)
    assert len(_feed(listener, 6)) == 2


def test_keyword_only_in_final_fires_once():
    rec = FakeVosk([(False, "嘅"), (True, "嘅 食")])
    listener = VoiceListener(model_dir=None, recognizer=rec)
    assert len(_feed(listener, 2)) == 1


def test_voice_loop_runs_one_cycle_per_utterance():
    from mahjong_projector.__main__ import voice_loop

    class CountingOrch:
        def __init__(self, voice):
            self.voice = voice
            self.cycles = 0

        def on_voice(self, event):
            self.cycles += 1

    rec = FakeVosk([(False, "食"), (False, "食 糊"), (True, "食 糊 啦"), (False, "食")])
    orch = CountingOrch(VoiceListener(model_dir=None, recognizer=rec))
    queue = [b"a", b"b", b"c", b"d"]
    drained = []

    def read():
        return queue.pop(0) if queue else None

    def drain():
        drained.append(len(queue))
        queue.clear()

    voice_loop(orch, read, drain)
    assert orch.cycles == 1
    assert drained == [3]


def test_reset_clears_latch():
    rec = FakeVosk([(False, "食"), (False, "食")])
    listener = VoiceListener(model_dir=None, recognizer=rec)
    assert len(_feed(listener, 1)) == 1
    listener.reset()
    assert len(_feed(listener, 1)) == 1
