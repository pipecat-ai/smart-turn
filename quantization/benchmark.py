#!/usr/bin/env python3
"""
Smart Turn v2 Quantization Benchmark Script
Simple version that always uses test data only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings
import warnings
import logging

# Suppress urllib3 warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

# Suppress PyTorch quantization warnings
warnings.filterwarnings(
    "ignore", message="Currently, qnnpack incorrectly ignores reduce_range"
)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress C++ warnings from PyTorch
import os

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_PRINT_REPRO_ON_FAILURE"] = "0"
os.environ["TORCH_LOGS"] = "-all"

# Reduce logging level for PyTorch
logging.getLogger("torch").setLevel(logging.ERROR)

import torch
import numpy as np
import time
import json
import argparse
from datetime import datetime
from transformers import Wav2Vec2Processor
from model import Wav2Vec2ForEndpointing
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc
import platform
import subprocess
import psutil


def get_cpu_info():
    """Get CPU information across different platforms."""
    cpu_info = {
        "model": "Unknown",
        "cores": torch.get_num_threads(),
        "architecture": platform.machine(),
    }

    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info["model"] = result.stdout.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info["model"] = line.split(":")[1].strip()
                        break
        elif system == "Windows":
            cmd = ["wmic", "cpu", "get", "name", "/value"]
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Name=" in line:
                        cpu_info["model"] = line.split("=")[1].strip()
                        break
    except Exception:
        pass

    return cpu_info


def get_os_info():
    """Get OS information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def benchmark_model(
    model, processor, test_dataset, n_samples, test_labels, device="cpu"
):
    """Run benchmark on a model configuration."""
    model = model.to(device)
    model.eval()

    # Get model size
    model_size_mb = get_model_size_mb(model)

    # Get initial memory usage
    if device == "cpu":
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

    predictions = []
    inference_times = []

    print(f"\nRunning inference on {n_samples} samples...")
    print(f"Model size: {model_size_mb:.2f} MB")

    with torch.no_grad():
        for i in range(n_samples):
            # Load sample on-demand to save memory
            sample = test_dataset[i]
            audio = sample["audio"]["array"]

            # Clear any lingering tensors
            if i > 0 and i % 10 == 0:
                gc.collect()

            # Prepare input
            inputs = processor(
                audio,
                sampling_rate=16000,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(device)

            # Time inference
            start_time = time.perf_counter()
            outputs = model(**inputs)
            end_time = time.perf_counter()

            # Get prediction
            # Model returns dict with 'logits' key containing probabilities
            probs = outputs["logits"]
            pred = (probs > 0.5).long().item()  # Binary classification
            predictions.append(pred)
            inference_times.append((end_time - start_time) * 1000)  # ms

            # Clear intermediate tensors
            del inputs
            del outputs
            del probs

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_samples} samples...")

    # Calculate metrics with zero_division parameter to handle edge cases
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)

    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)

    # Get peak memory usage
    peak_memory_mb = None
    if device == "cpu":
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        peak_memory_mb = current_memory_mb - initial_memory_mb

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_inference_time_ms": avg_time,
        "std_inference_time_ms": std_time,
        "total_samples": n_samples,
        "predictions": predictions,
        "model_size_mb": model_size_mb,
    }

    if peak_memory_mb is not None:
        result["peak_memory_usage_mb"] = peak_memory_mb

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Smart Turn v2 Quantization Benchmark (Simplified)"
    )
    parser.add_argument(
        "--sample-percent",
        type=float,
        default=100.0,
        help="Percentage of test set to use (default: 100%%)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to test on (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results (default: benchmark_results)",
    )
    parser.add_argument(
        "--single-core",
        action="store_true",
        help="Use only single CPU core",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Low memory mode: load only last 10% of data directly (may not match train.py splits exactly)",
    )

    args = parser.parse_args()

    # Set single core if requested
    if args.single_core:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print("Using single CPU core")

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # Get system info
    cpu_info = get_cpu_info()
    os_info = get_os_info()

    print(f"\nSystem Information:")
    print(f"  OS: {os_info['system']} {os_info['release']}")
    print(f"  CPU: {cpu_info['model']}")
    print(f"  CPU Cores: {cpu_info['cores']}")
    print(f"  Architecture: {cpu_info['architecture']}")

    if args.device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if args.low_memory:
        # In low memory mode, load a smaller portion but with better distribution
        print(
            "\nLow memory mode: Loading smaller dataset with balanced distribution..."
        )
        print("Note: This may not match exact train.py test split")
        # Load 20% to get a mix of positive and negative samples
        # First 10% has all positive, last 10% has all negative
        # So we take a bit from each to get balanced data
        dataset_first = load_dataset("pipecat-ai/rime_2", split="train[:1%]")
        dataset_last = load_dataset("pipecat-ai/rime_2", split="train[-1%:]")

        # Combine both parts
        from datasets import concatenate_datasets

        test_dataset = concatenate_datasets([dataset_first, dataset_last])
        print(f"Loaded {len(test_dataset)} samples (1% from start + 1% from end)")
    else:
        # Load dataset and create test split using same methodology as train.py
        print("\nLoading dataset and creating test split (matching train.py)...")
        dataset = load_dataset("pipecat-ai/rime_2", split="train")

        # Reproduce train.py splits with same seed
        # First split: 80% train, 20% (eval+test)
        dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)
        eval_test_combined = dataset_dict["test"]

        # Second split: Split the 20% into 10% eval, 10% test
        eval_test_dict = eval_test_combined.train_test_split(test_size=0.5, seed=42)
        test_dataset = eval_test_dict["test"]

        print(f"Created test split with {len(test_dataset)} samples (10% of total)")

        # Clean up to save memory
        del dataset
        del dataset_dict
        del eval_test_combined
        del eval_test_dict
        gc.collect()

    # Don't load all samples into memory at once
    total_samples = len(test_dataset)

    # Apply sample percentage if needed
    if args.sample_percent < 100.0:
        n_samples = int(total_samples * args.sample_percent / 100.0)
        print(f"Using {n_samples} samples ({args.sample_percent}% of test set)")
    else:
        n_samples = total_samples

    # Just collect labels first (lightweight)
    test_labels = []
    for i in range(n_samples):
        sample = test_dataset[i]
        label = 1 if sample["endpoint_bool"] else 0
        test_labels.append(label)

    # Print distribution
    positive_count = sum(test_labels)
    negative_count = len(test_labels) - positive_count
    print(f"\nTest set distribution:")
    print(
        f"  Positive (complete): {positive_count} ({positive_count / len(test_labels) * 100:.1f}%)"
    )
    print(
        f"  Negative (incomplete): {negative_count} ({negative_count / len(test_labels) * 100:.1f}%)"
    )

    # Warning for small sample sizes
    if len(test_labels) < 20:
        print(
            f"\n⚠️  WARNING: Using only {len(test_labels)} samples. Results may not be representative."
        )
        print("  Consider using a larger sample percentage for more reliable metrics.")

    # Load models
    model_path = "pipecat-ai/smart-turn-v2"
    processor = Wav2Vec2Processor.from_pretrained(model_path)

    results = {}

    # Test original model
    print(f"\n{'=' * 60}")
    print("Testing original Float32 model...")
    print("=" * 60)

    if args.device == "cuda":
        model = Wav2Vec2ForEndpointing.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
    else:
        model = Wav2Vec2ForEndpointing.from_pretrained(model_path)

    results["float32"] = benchmark_model(
        model, processor, test_dataset, n_samples, test_labels, args.device
    )
    print(f"\nFloat32 Results:")
    print(f"  Model size: {results['float32']['model_size_mb']:.2f} MB")
    if "peak_memory_usage_mb" in results["float32"]:
        print(
            f"  Peak memory usage: {results['float32']['peak_memory_usage_mb']:.2f} MB"
        )
    print(f"  Accuracy: {results['float32']['accuracy']:.4f}")
    print(f"  Precision: {results['float32']['precision']:.4f}")
    print(f"  Recall: {results['float32']['recall']:.4f}")
    print(f"  F1 Score: {results['float32']['f1']:.4f}")
    print(
        f"  Avg inference: {results['float32']['avg_inference_time_ms']:.2f} ± {results['float32']['std_inference_time_ms']:.2f} ms"
    )

    # Show prediction distribution
    pred_positive = sum(results["float32"]["predictions"])
    pred_negative = len(results["float32"]["predictions"]) - pred_positive
    print(f"  Predictions: {pred_positive} positive, {pred_negative} negative")

    del model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Test quantized model (always runs on CPU for comparison)
    print(f"\n{'=' * 60}")
    print("Testing quantized INT8 model (CPU only)...")
    print("=" * 60)

    # Set quantization backend
    backend = "fbgemm" if platform.system() in ["Linux", "Windows"] else "qnnpack"

    # Try to use fbgemm on macOS if available (to avoid qnnpack warnings)
    if platform.system() == "Darwin":
        try:
            torch.backends.quantized.engine = "fbgemm"
            backend = "fbgemm"
        except RuntimeError:
            torch.backends.quantized.engine = "qnnpack"
            backend = "qnnpack"
    else:
        torch.backends.quantized.engine = backend

    print(f"Using quantization backend: {backend}")

    # Load and quantize model
    model = Wav2Vec2ForEndpointing.from_pretrained(model_path)
    model.eval()

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    results["quantized"] = benchmark_model(
        quantized_model, processor, test_dataset, n_samples, test_labels, "cpu"
    )
    print(f"\nQuantized Results:")
    print(f"  Model size: {results['quantized']['model_size_mb']:.2f} MB")
    if "peak_memory_usage_mb" in results["quantized"]:
        print(
            f"  Peak memory usage: {results['quantized']['peak_memory_usage_mb']:.2f} MB"
        )
    print(f"  Accuracy: {results['quantized']['accuracy']:.4f}")
    print(f"  Precision: {results['quantized']['precision']:.4f}")
    print(f"  Recall: {results['quantized']['recall']:.4f}")
    print(f"  F1 Score: {results['quantized']['f1']:.4f}")
    print(
        f"  Avg inference: {results['quantized']['avg_inference_time_ms']:.2f} ± {results['quantized']['std_inference_time_ms']:.2f} ms"
    )

    # Show prediction distribution
    q_pred_positive = sum(results["quantized"]["predictions"])
    q_pred_negative = len(results["quantized"]["predictions"]) - q_pred_positive
    print(f"  Predictions: {q_pred_positive} positive, {q_pred_negative} negative")

    # Calculate speedup
    speedup = (
        results["float32"]["avg_inference_time_ms"]
        / results["quantized"]["avg_inference_time_ms"]
    )
    print(f"\nComparison:")
    print(f"  Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(
        f"  Model size reduction: {(1 - results['quantized']['model_size_mb'] / results['float32']['model_size_mb']) * 100:.1f}%"
    )
    if (
        "peak_memory_usage_mb" in results["float32"]
        and "peak_memory_usage_mb" in results["quantized"]
    ):
        memory_reduction = (
            1
            - results["quantized"]["peak_memory_usage_mb"]
            / results["float32"]["peak_memory_usage_mb"]
        ) * 100
        print(f"  Memory usage reduction: {memory_reduction:.1f}%")

    del quantized_model
    gc.collect()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "sample_percent": args.sample_percent,
        "total_test_samples": n_samples,
        "single_core": args.single_core,
        "system_info": {
            "os": os_info,
            "cpu": cpu_info,
        },
    }

    if args.device == "cuda":
        results["metadata"]["system_info"]["gpu"] = torch.cuda.get_device_name(0)

    # Save JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create simple markdown report
    report = f"""# Benchmark Results

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Device**: {args.device}  
**System**: {os_info["system"]} {os_info["release"]}  
**CPU**: {cpu_info["model"]}  
"""

    if args.device == "cuda":
        report += f"**GPU**: {torch.cuda.get_device_name(0)}  \n"

    report += f"""
## Test Dataset
- Total samples: {n_samples}
- Positive: {positive_count} ({positive_count / len(test_labels) * 100:.1f}%)
- Negative: {negative_count} ({negative_count / len(test_labels) * 100:.1f}%)

## Results

### Float32 Model
- Model Size: {results["float32"]["model_size_mb"]:.2f} MB
- Accuracy: {results["float32"]["accuracy"]:.4f}
- Precision: {results["float32"]["precision"]:.4f}
- Recall: {results["float32"]["recall"]:.4f}
- F1 Score: {results["float32"]["f1"]:.4f}
- Avg Inference: {results["float32"]["avg_inference_time_ms"]:.2f} ± {results["float32"]["std_inference_time_ms"]:.2f} ms
"""

    if "quantized" in results:
        report += f"""
### Quantized Model
- Model Size: {results["quantized"]["model_size_mb"]:.2f} MB
- Accuracy: {results["quantized"]["accuracy"]:.4f}
- Precision: {results["quantized"]["precision"]:.4f}
- Recall: {results["quantized"]["recall"]:.4f}
- F1 Score: {results["quantized"]["f1"]:.4f}
- Avg Inference: {results["quantized"]["avg_inference_time_ms"]:.2f} ± {results["quantized"]["std_inference_time_ms"]:.2f} ms

### Comparison
- Speedup: {speedup:.2f}x {"faster" if speedup > 1 else "slower"}
- Model size reduction: {(1 - results["quantized"]["model_size_mb"] / results["float32"]["model_size_mb"]) * 100:.1f}%
- Accuracy drop: {(results["float32"]["accuracy"] - results["quantized"]["accuracy"]) * 100:.2f}%
"""

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
