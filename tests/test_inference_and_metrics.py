import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_utils import resample_audio, truncate_audio_to_last_n_seconds


def test_lazy_import_without_cwd_model(monkeypatch, tmp_path):
    """Importing inference must not fail when no ONNX model sits in the CWD."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMART_TURN_ONNX_PATH", raising=False)
    import importlib
    import inference

    reloaded = importlib.reload(inference)
    assert reloaded.TARGET_SAMPLE_RATE == 16000


def test_resample_8khz_to_16khz_preserves_duration_and_frequency():
    sr_in = 8000
    t = np.arange(sr_in, dtype=np.float32) / sr_in
    tone_8k = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    tone_16k = resample_audio(tone_8k, orig_sr=8000, target_sr=16000)
    assert tone_16k.shape == (16000,)
    assert tone_16k.dtype == np.float32

    spectrum = np.abs(np.fft.rfft(tone_16k))
    freqs = np.fft.rfftfreq(tone_16k.size, d=1.0 / 16000)
    peak_hz = freqs[int(np.argmax(spectrum))]
    assert abs(peak_hz - 440.0) <= 2.0


def test_resample_invalid_sample_rate_raises():
    with pytest.raises(ValueError, match="positive"):
        resample_audio(np.zeros(100, dtype=np.float32), orig_sr=0, target_sr=16000)


def test_truncate_or_pad_8_seconds():
    short = np.ones(4000, dtype=np.float32)
    padded = truncate_audio_to_last_n_seconds(short, n_seconds=8, sample_rate=16000)
    assert padded.shape == (128000,)
    assert np.all(padded[:-4000] == 0.0)
    assert np.all(padded[-4000:] == 1.0)


def test_predict_endpoint_8khz_matches_16khz():
    onnx_file = ROOT / "smart-turn-v3.2-cpu.onnx"
    if not onnx_file.exists():
        pytest.skip("smart-turn-v3.2-cpu.onnx not downloaded")

    import inference

    t16 = np.arange(16000 * 2, dtype=np.float32) / 16000.0
    sig16 = (
        0.25 * np.sin(2.0 * np.pi * 220.0 * t16)
        + 0.12 * np.sin(2.0 * np.pi * 660.0 * t16)
        + 0.06 * np.sin(2.0 * np.pi * 1320.0 * t16)
    ).astype(np.float32)
    sig8 = sig16[::2].copy()

    r16 = inference.predict_endpoint(sig16, sample_rate=16000, onnx_path=str(onnx_file))
    r8 = inference.predict_endpoint(sig8, sample_rate=8000, onnx_path=str(onnx_file))

    assert r16["prediction"] == r8["prediction"]
    assert abs(r16["probability"] - r8["probability"]) < 0.06


def test_confusion_fpr_fnr_denominators():
    """FPR must divide by actual negatives (FP + TN), and FNR by actual positives (FN + TP)."""
    # 50 negatives (10 FP, 40 TN) -> true FPR = 20.0% (not 10/100 = 10.0%)
    # 50 positives (5 FN, 45 TP)  -> true FNR = 10.0% (1 - recall = 1 - 0.90)
    labels = np.array([0] * 50 + [1] * 50)
    probs = np.array([0.9] * 10 + [0.1] * 40 + [0.1] * 5 + [0.9] * 45, dtype=np.float32)

    preds = (probs > 0.5).astype(int)
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    neg = np.sum(labels == 0)
    pos = np.sum(labels == 1)

    fpr = float(fp / neg) * 100.0
    fnr = float(fn / pos) * 100.0
    assert fpr == pytest.approx(20.0)
    assert fnr == pytest.approx(10.0)
