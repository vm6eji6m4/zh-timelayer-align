import numpy as np
import pytest
import soundfile as sf

import zh_timelayer_align.timelayer as tl


class FakeAligner:
    """Returns char k at (0.5 + k*0.3, 0.7 + k*0.3) within the chunk."""
    def __init__(self, score=-3.0, fail=False):
        self.score, self.fail, self.last_score = score, fail, None

    def align(self, chunk, chars):
        if self.fail:
            raise ValueError("boom")
        self.last_score = self.score
        return [(0.5 + k * 0.3, 0.7 + k * 0.3) for k in range(len(chars))]


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros(16000 * 60, dtype="float32"), 16000)
    return p


def segs():
    return [{"start": 31.0, "end": 33.0, "text": "你好世界"}, {"start": 33.0, "end": 36.0, "text": "今天天氣"}]


def test_high_confidence_moves_times(wav):
    s = segs()
    st = tl.apply_timelayer(s, wav, FakeAligner(-3.0))
    assert s[0]["start"] == pytest.approx(30.5)            # window 1 starts at 30 s
    assert s[1]["start"] == pytest.approx(30.5 + 4 * 0.3)
    assert st["aligned_windows"] == 1 and st["segments_flagged"] == 0


def test_low_confidence_keeps_and_flags(wav):
    s = segs()
    st = tl.apply_timelayer(s, wav, FakeAligner(-8.0))
    assert [x["start"] for x in s] == [31.0, 33.0]
    assert all(x["time_review"] for x in s) and st["low_conf_windows"] == 1


def test_failure_is_counted(wav):
    s = segs()
    st = tl.apply_timelayer(s, wav, FakeAligner(fail=True))
    assert st["failed_windows"] == 1 and [x["start"] for x in s] == [31.0, 33.0]
    assert all(x["time_review"] for x in s)


def test_no_han_untouched(wav):
    s = [{"start": 5.0, "end": 6.0, "text": "OK ..."}]
    st = tl.apply_timelayer(s, wav, FakeAligner())
    assert s[0]["start"] == 5.0 and "time_review" not in s[0] and st["windows"] == 0


def test_no_overlap_after(wav):
    s = segs()
    tl.apply_timelayer(s, wav, FakeAligner())
    assert s[0]["end"] <= s[1]["start"]


def test_negative_control_threshold_is_what_keeps_times(wav, monkeypatch):
    monkeypatch.setattr(tl, "CUT", -999.0)
    s = segs()
    tl.apply_timelayer(s, wav, FakeAligner(-8.0))
    assert s[0]["start"] != 31.0   # without the gate, the low-confidence times would be taken


def test_missing_opencc_fails_loudly(monkeypatch):
    """No silent identity fallback: Traditional text would be mis-aligned without any error."""
    pytest.importorskip("torchaudio")
    pytest.importorskip("transformers")
    import sys
    from zh_timelayer_align import aligner
    monkeypatch.setitem(sys.modules, "opencc", None)
    with pytest.raises(ImportError):
        aligner._default_t2s()
