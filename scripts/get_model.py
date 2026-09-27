"""Download the alignment model once and save it as safetensors.

    python scripts/get_model.py ./wav2vec2-zh-align

Why convert: recent transformers versions refuse to load .bin weights unless
torch >= 2.6 (CVE-2025-32434). safetensors loads everywhere. The script checks
that the converted model gives bit-identical logits, with a negative control.
"""
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import Wav2Vec2ForCTC

REPO = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"  # Apache-2.0
REVISION = "99ccb2737be22b8bb50dcfcc39ad4d567fb90cfd"

dst = Path(sys.argv[1] if len(sys.argv) > 1 else "wav2vec2-zh-align")
if dst.exists():
    sys.exit(f"{dst} already exists")
src = Path(snapshot_download(REPO, revision=REVISION,
                             allow_patterns=["*.json", "pytorch_model.bin"]))
m = Wav2Vec2ForCTC.from_pretrained(src).eval()
m.save_pretrained(dst, safe_serialization=True)
for f in ("vocab.json", "preprocessor_config.json", "special_tokens_map.json"):
    shutil.copy(src / f, dst / f)

m2 = Wav2Vec2ForCTC.from_pretrained(dst).eval()
x = torch.randn(1, 16000 * 3)
with torch.inference_mode():
    same = torch.equal(m(x).logits, m2(x).logits)
with torch.no_grad():
    m2.lm_head.weight[0, 0] += 1e-2  # negative control: a weight that inference actually uses
with torch.inference_mode():
    control = torch.equal(m(x).logits, m2(x).logits)
print(f"saved {dst}; identical={same}; negative control detected change={not control}")
if not same or control:
    sys.exit("verification failed")
