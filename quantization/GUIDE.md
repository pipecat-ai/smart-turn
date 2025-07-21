# Quantization Guide

## Overview

This tool applies **Dynamic Quantization** to the Smart Turn v2 model, achieving:

- **Model size reduction**: ~90% (361MB → 34MB)
- **Memory usage reduction**: ~75% (803MB → 200MB)
- **Speed improvement**: Platform-dependent (faster on x86, slower on Apple Silicon)

## Limitations

- CPU-only optimization
- Significantly slower on Apple Silicon

## Quick Start

```bash
# Basic quantization
python quantization/quantize.py

# With benchmarking
python quantization/quantize.py --benchmark
```

## Testing Guide

```bash
# Basic benchmark (CPU + quantized)
python quantization/benchmark.py

# GPU test (tests float32 on GPU, quantized on CPU)
python quantization/benchmark.py --device cuda

# MPS test for Apple Silicon (tests float32 on MPS, quantized on CPU)
python quantization/benchmark.py --device mps

# Quick test with 10% of test data
python quantization/benchmark.py --sample-percent 10

# Single-core CPU test (for consistent benchmarking)
python quantization/benchmark.py --single-core
```

### Low Memory Mode

For systems with limited RAM (e.g., 1GB AWS instances):

```bash
python quantization/benchmark.py --device cpu --low-memory
```

**Note**: `--low-memory` loads a smaller subset of data (1% from start + 1% from end) for balanced distribution.

## Using Quantized Model

```python
import torch
from model import Wav2Vec2ForEndpointing
from transformers import Wav2Vec2Processor

# Set backend (important!)
torch.backends.quantized.engine = 'qnnpack'  # or 'fbgemm' for x86
torch.set_num_threads(4)  # Adjust based on your CPU

# Load model
model = Wav2Vec2ForEndpointing.from_pretrained('quantized_model')
model.load_state_dict(torch.load('quantized_model/pytorch_model.bin'))
model.eval()

# Load processor
processor = Wav2Vec2Processor.from_pretrained('quantized_model')

# Inference example
audio = your_audio_array  # numpy array or list
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
with torch.no_grad():
    outputs = model(**inputs)
```

## Performance Summary

| Platform      | Expected Speed                     |
| ------------- | ---------------------------------- |
| Apple Silicon | 0.36~0.7x slower → Use MPS instead |
| Intel/AMD x86 | 1.16~1.2x faster                   |
| ARM servers   | Not tested                         |

### Memory & Model Size Comparison

| Metric                 | Float32 (Original) | INT8 (Quantized) | Reduction |
| ---------------------- | ------------------ | ---------------- | --------- |
| Model size (MB)        | 361.57             | 34.18            | 90.5%     |
| Peak memory usage (MB) | 803.36             | 200.25           | 75.1%     |

Model size reduced by 90.5%, and peak memory usage reduced by 75.1% with quantization.

## Benchmark Results on Apple Silicon

Below are benchmark results measured on Apple Silicon (Apple M2, 1 core, arm64) using 403 samples (10% test split).

| Metric                  | Float32 (Original) | INT8 (Quantized) |
| ----------------------- | ------------------ | ---------------- |
| Accuracy                | 0.9876             | 0.9826           |
| Precision               | 0.9889             | 0.9780           |
| Recall                  | 0.9834             | 0.9834           |
| F1 Score                | 0.9861             | 0.9807           |
| Avg Inference Time (ms) | 330.71 ± 126.00    | 913.45 ± 307.53  |
| Predictions (Pos/Neg)   | 180 / 223          | 182 / 221        |
| Speedup                 | 1.00x              | 0.36x~0.7x       |

- ✅ **Accuracy**: Minimal performance drop (0.9876 → 0.9826, only -0.5%)
- ❌ **Speed**: Significantly slower on Apple Silicon (0.36x~0.7x slower)

The poor performance is due to QNNPACK backend not being optimized for Apple Silicon architecture. INT8 quantization is primarily beneficial on x86 processors with FBGEMM backend.

## Benchmark Results on AWS EC2

Tested with a small number of samples due to low-spec Intel CPU. 32 samples are used for testing. Focus only on the inference speed.

| Model            | Avg Inference Time (ms) | Speedup     |
| ---------------- | ----------------------- | ----------- |
| Float32          | 1122.64 ± 467.32        | 1.00x       |
| INT8 (Quantized) | 939.57 ± 269.90         | 1.16x~1.19x |

Speedup is 1.16x~1.19x on x86 servers with FBGEMM backend.

### Why Dynamic over Static Quantization?

- No calibration dataset needed
- Better accuracy preservation
- Simpler implementation

## Notes

- Apple Silicon users: Use `device='mps'` instead of quantization for better performance
- x86 servers: Quantization works great with FBGEMM backend
- Quantized models only run on CPU (PyTorch limitation)
- Results vary by hardware
