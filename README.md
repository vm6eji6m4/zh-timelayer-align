# zh-timelayer-align

把字幕的「時間」從「文字」裡拆出來，獨立成一層：用 wav2vec2 CTC 強制對齊替每一句重新定開頭與結尾，
**再用對齊信心決定要不要採用**。適用於國語（含台語夾雜）的長錄音：演講、課程、法會、訪談。

*Re-time Mandarin subtitles with CTC forced alignment, and use the alignment's own confidence to decide
whether to trust it. English summary at the bottom.*

## 為什麼需要

Whisper 系引擎自己給的時間碼，在我們的量測裡句子開頭中位誤差約 0.5 秒；
換成另一個引擎的文字（例如台語專用模型），時間可能整段漂掉十幾秒。
強制對齊只負責「這個字在第幾秒」，不管文字是哪個引擎寫的，所以可以接在任何 ASR 後面。

但強制對齊有個陷阱：**文字和聲音對不上時，它不會失敗，照樣給你時間**。
我們把文字故意錯位一整塊去對齊，67 塊裡 0 塊報錯。所以本專案的重點不只是對齊，而是**把關**。

## 做法

1. 以 30 秒為一塊，把落在這塊的字幕句的漢字串起來，送進
   [`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn)（Apache-2.0）
   ＋ `torchaudio.functional.forced_align`，得到每個字的時間。
2. 算這一塊的**信心**：每個字對到的音框，平均對數機率。
3. 信心 ≥ −5.712 → 採用對齊時間；信心 < −5.712 → **保留原時間，標記待人工校對**。
4. 對齊丟例外 → 同低信心處理，並計數（不靜默退回）。

不需要 WhisperX；在 torch 2.3／transformers 4.46 就能跑。

## 量測結果

| 量測 | 素材 | 結果 |
|---|---|---|
| 對齊器本身的準確度 | 90 分鐘台語劇，專業字幕 1,144 句 | 句開頭中位誤差 **0.15–0.16 秒**（引擎自帶時間 0.49 秒）；結尾 0.35–0.37 秒。不論文字來自 turbo、large-v3、SenseVoice 或台語模型，結果都一樣 |
| 信心能否抓出「文字對不上」 | 157 支長錄音（約 30 小時），正確文字 vs 錯位一塊的文字 | AUC **0.91** |
| 人工盲聽 | 兩輪共 80 段「新舊時間差 > 1 秒」的句子，A／B 隨機對調、不顯示秒數 | 信心 ≥ 門檻：**28 段新時間對、0 段舊時間對**。信心 < 門檻：新 18／舊 26 |
| 規則整體 | 同上 72 段（排除「都對／都不對／無法判斷」） | 「高信心用新、低信心保留」選對 **75%**；全部換新 64%；全部保留 36% |

門檻 −5.712 是在人工聽之前、由資料分布定下的，不是看聽判結果調的
（事後掃描最好的門檻 −6.5 可到 79%，差距小且會過度擬合，未採用）。

### 已知限制

- 信心低的地方，新舊時間**都**常常不對（信心 −9～−5.7 之間是一半一半）：規則只保證不讓它變差，這些句子仍需要人工。
- 腔調重、遠場、唱誦的錄音，低信心比例會高很多（某批舊錄音有 55–72% 的句子低於門檻）。門檻可能要依錄音品質調整。
- 同一句在短時間內重複很多次（例如儀式唱誦），人耳也常分不出是哪一次。
- 台語被 ASR 翻成國語文字時，文字與聲音不符，信心會低 → 保留原時間，這是設計如此。
- 模型字表是簡體；程式逐字轉簡體後查表（需要 `opencc`），時間配回原字。

## 安裝與使用

```bash
pip install -e .
python scripts/get_model.py ./wav2vec2-zh-align      # 下載並轉成 safetensors（約 1.28 GB），附位元相同驗證
ffmpeg -i talk.mp3 -ac 1 -ar 16000 talk.wav
python -m zh_timelayer_align.cli --audio talk.wav --srt talk.srt --model ./wav2vec2-zh-align --out talk.aligned.srt
```

輸出 `talk.aligned.srt`，以及 `talk.aligned.review.json`（統計＋被標記待校的句子）。

程式內使用：

```python
from zh_timelayer_align import CtcAligner, apply_timelayer
segs = [{"start": 3.4, "end": 10.0, "text": "今天我們來談時間碼"}, ...]
stats = apply_timelayer(segs, "talk.wav", CtcAligner("./wav2vec2-zh-align"))
# segs 就地更新；低信心的句子有 segs[i]["time_review"] = True
```

測試（不需 GPU、不需模型）：`pytest -q`。其中有一個負對照：把門檻拿掉時，低信心的時間會被改寫——證明測試真的在驗門檻。

## 量測時踩過的坑

- **假成功**：WhisperX 與 stable-ts 第一次跑「全部完成」，其實每一塊都失敗、退回平均分配；只有把退回次數記下來才看得到。
- **不報錯的失敗**：另一個對齊器在文字不符時回傳大量「長度 0」的字，看起來最快最乾淨。
- **參考資料本身**：一份人工三校字幕的時間是照畫面定的，不是照聲音，拿來當標準會量出假誤差。
- **自我一致**：用同一個對齊器對「正確文字」產生的參考時間去評估同一個對齊器，會得到中位誤差 0——那只證明它自己跟自己一致。
- 新版 transformers 在 torch < 2.6 時拒讀 `.bin` 權重 → `get_model.py` 轉 safetensors。

## English summary

Forced alignment gives accurate per-character timing (median onset error 0.15 s against professional
subtitles on a 90-min drama, vs 0.49 s for Whisper's own timestamps) regardless of which ASR produced
the text. But it never fails: misaligned text still gets times. We gate on the chunk's mean aligned
log-probability (AUC 0.91 for matched vs shifted text). In blind A/B listening (80 clips, two rounds),
chunks above the pre-registered threshold −5.712 were right 28/28 times; below it, the original times
were kept and flagged for human review. No WhisperX needed; runs on torch 2.3 / transformers 4.46.

## 授權

程式碼 Apache-2.0。模型權重屬原作者（Apache-2.0），不隨本專案散布，由 `get_model.py` 下載。
