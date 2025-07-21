import time

import coremltools as ct
import numpy as np
from transformers import Wav2Vec2Processor

MODEL_PATH = "smart_turn_classifier.mlpackage"
TORCH_MODEL_PATH = "pipecat-ai/smart-turn-v2"

print("Loading Core ML model …")
start_time = time.perf_counter()
mlmodel = ct.models.MLModel(MODEL_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)
print(f"Model loaded in {time.perf_counter() - start_time:.2f}s")

# Create a random 8-second audio clip at 16 kHz
audio_random = np.random.randn(16000 * 8).astype(np.float32)

processor = Wav2Vec2Processor.from_pretrained(TORCH_MODEL_PATH)
inputs = processor(
    audio_random,
    sampling_rate=16000,
    padding="max_length",
    truncation=True,
    max_length=16000 * 16,
    return_attention_mask=True,
    return_tensors="np",
)

print("Testing inference time …")
times = []
for _ in range(10):
    start = time.perf_counter()
    output = mlmodel.predict(dict(inputs))
    times.append(time.perf_counter() - start)

print(f"Max: {max(times) * 1000:.2f} ms")
print(f"Min: {min(times) * 1000:.2f} ms")
print(f"Median: {np.median(times) * 1000:.2f} ms")
print("Sample output:", output)
