#!/usr/bin/env python3
"""
Optimized quantization script with various performance improvements
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress urllib3 warnings
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from model import Wav2Vec2ForEndpointing
import platform
import warnings
import json
import time
import numpy as np


def optimize_for_cpu():
    """Optimize PyTorch settings for CPU inference"""
    # Determine optimal number of threads
    import psutil

    n_cores = psutil.cpu_count(logical=False)

    # For inference, fewer threads often work better (avoid hyperthreading)
    optimal_threads = min(4, n_cores)

    print(f"CPU optimization:")
    print(f"  Physical cores: {n_cores}")
    print(f"  Setting threads to: {optimal_threads}")

    torch.set_num_threads(optimal_threads)
    torch.set_num_interop_threads(optimal_threads)

    # Disable gradients globally
    torch.set_grad_enabled(False)

    # Set CPU optimizations
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def select_best_quantization_backend():
    """Select the best quantization backend for the current platform"""
    system = platform.system()
    machine = platform.machine()

    print(f"\nPlatform detection:")
    print(f"  System: {system}")
    print(f"  Architecture: {machine}")

    # Default backend
    backend = "qnnpack"

    # Platform-specific selection
    if system == "Linux":
        if "x86" in machine or "amd64" in machine.lower():
            # x86_64 Linux typically works better with fbgemm
            backend = "fbgemm"
        else:
            # ARM Linux (including M1 under Rosetta)
            backend = "qnnpack"
    elif system == "Darwin":  # macOS
        # Apple Silicon or Intel Mac
        backend = "qnnpack"
    elif system == "Windows":
        # Windows typically works better with fbgemm
        backend = "fbgemm"

    # Try to set the backend
    try:
        torch.backends.quantized.engine = backend
        print(f"  Selected backend: {backend}")
    except RuntimeError as e:
        print(f"  Failed to set {backend}, falling back to qnnpack")
        torch.backends.quantized.engine = "qnnpack"
        backend = "qnnpack"

    return backend


def create_optimized_quantized_model(model_path):
    """Create an optimized quantized model with various improvements

    Args:
        model_path: Path to the pretrained model
    """

    # Optimize CPU settings
    optimize_for_cpu()

    # Select best backend
    backend = select_best_quantization_backend()

    print(f"\nLoading model from {model_path}")

    # Load model (quantization only works on CPU)
    model = Wav2Vec2ForEndpointing.from_pretrained(model_path)
    model.eval()

    # Get model size before quantization
    param_size = 0
    buffer_size = 0
    for name, param in model.named_parameters():
        param_size += param.nelement() * param.element_size()
    for name, buffer in model.named_buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    original_size = (param_size + buffer_size) / 1024 / 1024  # MB
    print(f"Original model size: {original_size:.2f} MB")

    print("\nApplying optimized dynamic quantization...")

    # Configure quantization
    quantization_config = {
        "dtype": torch.qint8,
    }

    # List of modules to quantize
    # For Wav2Vec2, we mainly want to quantize Linear layers
    modules_to_quantize = {nn.Linear}

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, modules_to_quantize, **quantization_config
    )

    # Additional optimizations
    print("\nApplying additional optimizations...")

    # 1. Fuse modules where possible (this is model-specific)
    # For Wav2Vec2, there might not be many fusion opportunities

    # 2. Remove unnecessary attributes to save memory
    for name, module in quantized_model.named_modules():
        if hasattr(module, "qconfig"):
            delattr(module, "qconfig")

    # 3. Optimize the model for inference
    quantized_model.eval()

    # Calculate actual quantized model size
    quantized_param_size = 0
    quantized_buffer_size = 0
    for name, param in quantized_model.named_parameters():
        quantized_param_size += param.nelement() * param.element_size()
    for name, buffer in quantized_model.named_buffers():
        quantized_buffer_size += buffer.nelement() * buffer.element_size()

    quantized_size = (quantized_param_size + quantized_buffer_size) / 1024 / 1024
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {(1 - quantized_size / original_size) * 100:.1f}%")

    return quantized_model, backend, original_size


def benchmark_model(model, processor, num_runs=10):
    """Quick benchmark of the model"""
    print(f"\nRunning quick benchmark ({num_runs} runs)...")

    # Create dummy input
    dummy_audio = np.random.randn(16000 * 2).astype(np.float32)  # 2 seconds
    inputs = processor(
        dummy_audio,
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=16000 * 16,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            _ = model(**inputs)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"  Average inference time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min/Max: {np.min(times):.2f} / {np.max(times):.2f} ms")

    return mean_time


def create_torchscript_model(model, processor):
    """Create TorchScript version for additional optimization"""
    print("\nCreating TorchScript version...")

    # Ensure model is on CPU for TorchScript
    model = model.cpu()

    # Create example input
    dummy_audio = torch.randn(16000 * 2)
    inputs = processor(
        dummy_audio.numpy(),
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=16000 * 16,
        return_attention_mask=True,
        return_tensors="pt",
    )

    try:
        # Trace the model
        traced_model = torch.jit.trace(
            model, (inputs["input_values"], inputs["attention_mask"])
        )

        # Optimize for inference
        traced_model = torch.jit.optimize_for_inference(traced_model)

        print("  ✓ TorchScript model created successfully")
        return traced_model
    except Exception as e:
        print(f"  ✗ TorchScript creation failed: {e}")
        return None


def save_optimization_info(output_dir, backend, original_size, quantized_size):
    """Save optimization information to JSON file"""
    optimization_info = {
        "quantization_backend": backend,
        "quantized_modules": ["nn.Linear"],
        "optimization_settings": {
            "num_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },
        "model_sizes": {
            "original_mb": round(original_size, 2),
            "quantized_mb": round(quantized_size, 2),
            "compression_ratio": round((1 - quantized_size / original_size) * 100, 1),
        },
    }

    with open(output_dir / "optimization_info.json", "w") as f:
        json.dump(optimization_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Optimized quantization for Smart Turn model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic quantization
  python tools/quantization/quantize.py
  
  # Full optimization with benchmarks
  python tools/quantization/quantize.py --benchmark --create-torchscript
""",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="pipecat-ai/smart-turn-v2",
        help="Path or identifier of the pretrained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="quantized_model",
        help="Directory to save the quantized model",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparison",
    )
    parser.add_argument(
        "--create-torchscript",
        action="store_true",
        help="Also create TorchScript version",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export reminder",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Optimized Smart Turn Model Quantization")
    print("=" * 80)

    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)

    # Benchmark original model if requested
    original_time = None
    if args.benchmark:
        print("\nBenchmarking original Float32 model...")
        original_model = Wav2Vec2ForEndpointing.from_pretrained(args.model_path)
        original_model.eval()
        original_time = benchmark_model(original_model, processor)
        del original_model

    # Create optimized quantized model
    quantized_model, backend, original_size = create_optimized_quantized_model(
        args.model_path
    )

    # Benchmark quantized model if requested
    if args.benchmark:
        quantized_time = benchmark_model(quantized_model, processor)
        speedup = original_time / quantized_time
        print(f"\nSpeedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    # Save the model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving optimized model to {output_dir}")

    # Save model state dict
    torch.save(quantized_model.state_dict(), output_dir / "pytorch_model.bin")

    # Save config
    config = quantized_model.config
    config.save_pretrained(output_dir)

    # Save processor
    processor.save_pretrained(output_dir)

    # Calculate actual quantized model size
    quantized_param_size = 0
    quantized_buffer_size = 0
    for name, param in quantized_model.named_parameters():
        quantized_param_size += param.nelement() * param.element_size()
    for name, buffer in quantized_model.named_buffers():
        quantized_buffer_size += buffer.nelement() * buffer.element_size()

    actual_quantized_size = (quantized_param_size + quantized_buffer_size) / 1024 / 1024

    # Save optimization info
    save_optimization_info(output_dir, backend, original_size, actual_quantized_size)

    # Create TorchScript version if requested
    if args.create_torchscript:
        traced_model = create_torchscript_model(quantized_model, processor)
        if traced_model:
            torch.jit.save(traced_model, output_dir / "model_traced.pt")

            if args.benchmark:
                print("\nBenchmarking TorchScript model...")

                # Create a wrapper for benchmark
                class TracedWrapper:
                    def __init__(self, model):
                        self.model = model

                    def __call__(self, input_values, attention_mask):
                        return self.model(input_values, attention_mask)

                wrapped = TracedWrapper(traced_model)
                traced_time = benchmark_model(wrapped, processor)
                if original_time:
                    traced_speedup = original_time / traced_time
                    print(f"TorchScript speedup: {traced_speedup:.2f}x")

    print("\n" + "=" * 80)
    print("Optimization Summary")
    print("=" * 80)
    print(f"✓ Quantization backend: {backend}")
    print(f"✓ CPU threads: {torch.get_num_threads()}")
    print(f"✓ Model saved to: {output_dir}")
    print(f"✓ Original size: {original_size:.1f} MB")
    print(
        f"✓ Quantized size: {actual_quantized_size:.1f} MB ({(1 - actual_quantized_size / original_size) * 100:.0f}% reduction)"
    )

    if args.create_torchscript and traced_model:
        print(f"✓ TorchScript model saved")

    if not args.skip_onnx:
        print("\n💡 For ONNX export, use:")
        print(f"   python tools/onnx/export_onnx.py --model-path {args.model_path}")

    print("\n📌 Loading the quantized model:")
    print("```python")
    print(f"torch.backends.quantized.engine = '{backend}'")
    print(f"torch.set_num_threads({torch.get_num_threads()})")
    print("model = Wav2Vec2ForEndpointing.from_pretrained('" + str(output_dir) + "')")
    print(
        "model.load_state_dict(torch.load('"
        + str(output_dir / "pytorch_model.bin")
        + "'))"
    )
    print("```")

    print("\nDone!")


if __name__ == "__main__":
    main()
