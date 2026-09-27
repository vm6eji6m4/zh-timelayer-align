"""Re-time subtitle segments with forced alignment, gated by alignment confidence.

Rule (measured, see README):
  * chunk confidence >= CUT  -> use the aligned start/end
  * chunk confidence <  CUT  -> keep the original times, mark the segment for review
  * aligner raises           -> same as low confidence, and counted in failed_windows
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import soundfile as sf

CUT = -5.712
WINDOW = 30.0
HAN = re.compile(r"[一-鿿]")


def apply_timelayer(segments: list[dict], audio: str | Path, aligner, cut: float | None = None,
                    window: float = WINDOW) -> dict:
    """Modify ``segments`` (dicts with start/end/text) in place. Returns statistics."""
    cut = CUT if cut is None else cut
    t0 = time.time()
    wav, sr = sf.read(str(audio), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        raise ValueError(f"audio must be 16 kHz, got {sr}")
    dur = len(wav) / sr
    segments.sort(key=lambda s: s["start"])
    old_start = [s["start"] for s in segments]

    windows: dict[int, list[tuple[int, list[str]]]] = {}
    for k, s in enumerate(segments):
        chars = HAN.findall(s["text"])
        if chars:
            windows.setdefault(int(((s["start"] + s["end"]) / 2) // window), []).append((k, chars))

    stats = dict(status="applied", cut=cut, windows=len(windows), aligned_windows=0,
                 low_conf_windows=0, failed_windows=0)
    for w, items in sorted(windows.items()):
        a = w * window
        chunk = wav[int(a * sr):int(min(a + window, dur) * sr)]
        flat = [c for _, cs in items for c in cs]
        try:
            times = aligner.align(chunk, flat)
            score = aligner.last_score
        except Exception:
            stats["failed_windows"] += 1
            times, score = None, None
        if times is None or score is None or score < cut:
            if times is not None:
                stats["low_conf_windows"] += 1
            for k, _ in items:
                segments[k]["time_review"] = True
            continue
        stats["aligned_windows"] += 1
        pos = 0
        for k, cs in items:
            first, last = times[pos], times[pos + len(cs) - 1]
            pos += len(cs)
            segments[k]["start"] = a + first[0]
            segments[k]["end"] = max(a + last[1], segments[k]["start"] + 0.2)

    for k in range(len(segments) - 1):
        if segments[k]["end"] > segments[k + 1]["start"]:
            segments[k]["end"] = max(segments[k]["start"] + 0.1, segments[k + 1]["start"])

    stats["segments_moved"] = sum(abs(s["start"] - o) > 0.05 for s, o in zip(segments, old_start))
    stats["segments_flagged"] = sum(1 for s in segments if s.get("time_review"))
    stats["elapsed"] = round(time.time() - t0, 2)
    return stats
