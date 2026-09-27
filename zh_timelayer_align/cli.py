"""Command line: re-time an SRT against its audio.

    python -m zh_timelayer_align.cli --audio talk.wav --srt talk.srt --model ./wav2vec2-zh-align --out talk.aligned.srt

Audio must be 16 kHz mono (convert with: ffmpeg -i in.mp3 -ac 1 -ar 16000 talk.wav).
Writes the new SRT and, next to it, <out>.review.json listing segments whose
times were kept because alignment confidence was low.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .aligner import CtcAligner
from .timelayer import CUT, apply_timelayer


def _ts(t: float) -> str:
    ms = int(round(max(t, 0) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def read_srt(path: Path) -> list[dict]:
    def sec(ts):
        h, m, s = ts.strip().replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    out = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.strip().splitlines()
        arrow = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if arrow is None:
            continue
        a, b = lines[arrow].split("-->")
        out.append({"start": sec(a), "end": sec(b), "text": "\n".join(lines[arrow + 1:])})
    return out


def write_srt(segments: list[dict], path: Path) -> None:
    path.write_text("\n".join(f"{i}\n{_ts(s['start'])} --> {_ts(s['end'])}\n{s['text']}\n"
                              for i, s in enumerate(segments, 1)), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--srt", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path, help="folder with model.safetensors + vocab.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cut", type=float, default=CUT, help=f"confidence threshold (default {CUT})")
    a = ap.parse_args(argv)
    segs = read_srt(a.srt)
    stats = apply_timelayer(segs, a.audio, CtcAligner(a.model, a.device), cut=a.cut)
    write_srt(segs, a.out)
    stats["review"] = [[i, round(s["start"], 3), round(s["end"], 3)] for i, s in enumerate(segs, 1) if s.get("time_review")]
    a.out.with_suffix(".review.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "review"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
