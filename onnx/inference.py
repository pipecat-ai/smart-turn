#!/usr/bin/env python3
"""ONNX inference helper for Smart-Turn.

Mirrors `inference.py` (PyTorch) but uses ONNX Runtime so that other
scripts can simply do:

    from onnx_inference import predict_endpoint

and get the same dictionary-style result.
"""
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
from transformers import Wav2Vec2Processor
import time

# Determine project root (parent of this 'onnx' directory)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_DIR = ROOT / "smart-turn-v2-onnx"
DEFAULT_MODEL_DIR = ROOT / "v2-model"

MODEL_HUB_PATH: str = os.getenv("SMART_TURN_MODEL", str(DEFAULT_MODEL_DIR))
MODEL_PATH_ONNX: str = os.getenv("SMART_TURN_ONNX", str(DEFAULT_ONNX_DIR / "model.onnx"))
RATE: int = 16000
MAX_DURATION_SECONDS: int = 16
PREDICTION_THRESHOLD: float = 0.5

def _load_session() -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    if not Path(MODEL_PATH_ONNX).exists():
        raise FileNotFoundError(
            f"ONNX model not found at '{MODEL_PATH_ONNX}'. Set SMART_TURN_ONNX env var or export the model first."
        )
    return ort.InferenceSession(MODEL_PATH_ONNX, providers=providers)

print("🔄 Loading ONNX model and feature-extractor …", flush=True)
_processor = Wav2Vec2Processor.from_pretrained(MODEL_HUB_PATH)
_session   = _load_session()
print(f"✅  Ready – provider: {_session.get_providers()[0]}")


def predict_endpoint(audio_array: np.ndarray) -> Dict[str, float]:
    """Return turn-completion prediction for a 16-kHz mono `audio_array`."""
    if not isinstance(audio_array, np.ndarray):
        raise TypeError("audio_array must be a NumPy array")

    inputs = _processor(
        audio_array,
        sampling_rate=RATE,
        padding="max_length",
        truncation=True,
        max_length=RATE * MAX_DURATION_SECONDS,
        return_attention_mask=True,
        return_tensors="pt",
    )

    ort_inputs = {
        "input_values": inputs["input_values"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy().astype(np.int64),
    }
    t0 = time.perf_counter()
    probs = _session.run(["probabilities"], ort_inputs)[0]
    inference_ms = (time.perf_counter() - t0) * 1000.0
    probability = float(probs[0][0])
    prediction  = 1 if probability > PREDICTION_THRESHOLD else 0
    return {
        "prediction": prediction,
        "probability": probability,
        "inference_ms": inference_ms,
    }


if __name__ == "__main__":
    import numpy.random as npr

    dummy = npr.randn(RATE * 2).astype(np.float32)
    out = predict_endpoint(dummy)
    print(out) 
