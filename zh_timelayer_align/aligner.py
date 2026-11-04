"""CTC forced aligner for Mandarin (wav2vec2 + torchaudio.functional.forced_align).

Given a 16 kHz mono audio chunk and the characters that were spoken in it,
return a (start, end) time for every character, plus a confidence score for
the whole chunk (mean per-character log-probability of the aligned frames).

The confidence matters: forced alignment never refuses. If the text does not
match the audio it still returns times, just wrong ones. The score is how you
tell the two cases apart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC

SAMPLE_RATE = 16000


def _default_t2s():
    """The model vocabulary is Simplified Chinese; map each char 1:1 before lookup."""
    try:
        import opencc
    except ImportError as e:  # an identity fallback would silently mis-align Traditional text
        raise ImportError(
            "zh_timelayer_align needs opencc (pip install opencc-python-reimplemented). "
            "If the text is already Simplified, pass to_model_char=lambda ch: ch explicitly.") from e
    cc = opencc.OpenCC("t2s")
    return lambda ch: (cc.convert(ch) or ch)[0]


class CtcAligner:
    def __init__(self, model_dir: str | os.PathLike, device: str = "cuda", to_model_char=None):
        model_dir = Path(model_dir)
        self.vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        self.blank = self.vocab["<pad>"]
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        self.device = device
        self.to_model_char = to_model_char or _default_t2s()
        self.last_score: float | None = None

    def align(self, chunk: np.ndarray, chars: list[str]) -> list[tuple[float, float]]:
        """Return per-character (start, end) in seconds relative to the chunk start.

        Characters missing from the model vocabulary get interpolated times.
        Raises ValueError when nothing can be aligned; callers must count that.
        """
        self.last_score = None
        dur = len(chunk) / SAMPLE_RATE
        x = (chunk - chunk.mean()) / (chunk.std() + 1e-7)  # do_normalize, as in preprocessor_config.json
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(x.astype("float32"))[None].to(self.device)).logits
            em = torch.log_softmax(logits, -1).float().cpu()
        keep = [(i, self.vocab[s]) for i, s in enumerate(self.to_model_char(c) for c in chars) if s in self.vocab]
        if not keep:
            raise ValueError("no characters in model vocabulary")
        frames = em.shape[1]
        if len(keep) > frames:
            raise ValueError("more characters than audio frames")
        targets = torch.tensor([[t for _, t in keep]], dtype=torch.int32)
        ali, scores = torchaudio.functional.forced_align(em, targets, blank=self.blank)
        spans = torchaudio.functional.merge_tokens(ali[0], scores[0].exp(), blank=self.blank)
        if len(spans) != len(keep):
            raise ValueError(f"aligned {len(spans)} tokens, expected {len(keep)}")
        self.last_score = float(np.mean([float(scores[0, sp.start:sp.end].mean()) for sp in spans]))
        sec_per_frame = dur / frames
        st = np.full(len(chars), np.nan)
        en = np.full(len(chars), np.nan)
        for (i, _), sp in zip(keep, spans):
            st[i] = sp.start * sec_per_frame
            en[i] = sp.end * sec_per_frame
        idx = np.arange(len(chars))
        ok = ~np.isnan(st)
        st = np.interp(idx, idx[ok], st[ok])
        en = np.where(np.isnan(en), st, en)
        return list(zip(st.tolist(), en.tolist()))
