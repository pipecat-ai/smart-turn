import time
import os
import subprocess
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
from collections import defaultdict

import modal
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor

from datasets import load_dataset, load_from_disk

app = modal.App("endpointing-benchmark")
volume = modal.Volume.from_name("endpointing", create_if_missing=False)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy",
        "torch",
        "datasets==3.2.0",
        "transformers[torch]==4.48.2",
        "scikit-learn==1.6.1",
        "onnxruntime-gpu",
        "librosa",
        "soundfile"
    )
)

SAMPLING_RATE = 16000
N_MELS = 80
N_FRAMES = 800  # 8s → 800 frames
FEATURE_SHAPE = (1, N_MELS, N_FRAMES)
AUDIO_SECONDS = 8

# Language code to full name mapping with flag emojis
LANGUAGE_MAPPING = {
    "eng": "🇬🇧 🇺🇸 English",
    "rus": "🇷🇺 Russian", 
    "por": "🇵🇹 Portuguese",
    "nld": "🇳🇱 Dutch",
    "deu": "🇩🇪 German",
    "hin": "🇮🇳 Hindi",
    "spa": "🇪🇸 Spanish",
    "fra": "🇫🇷 French",
    "vie": "🇻🇳 Vietnamese",
    "ind": "🇮🇩 Indonesian",
    "nor": "🇳🇴 Norwegian",
    "fin": "🇫🇮 Finnish",
    "ben": "🇧🇩 Bengali",
    "pol": "🇵🇱 Polish",
    "ara": "🇸🇦 Arabic",
    "tur": "🇹🇷 Turkish",
    "zho": "🇨🇳 Chinese",
    "ukr": "🇺🇦 Ukrainian",
    "kor": "🇰🇷 Korean",
    "jpn": "🇯🇵 Japanese",
    "dan": "🇩🇰 Danish",
    "ita": "🇮🇹 Italian",
    "mar": "🇮🇳 Marathi"
}


def log_progress(message: str, level: str = "INFO"):
    """Log progress with timestamp."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"[{timestamp}] [{level}] {message}")


def get_gpu_model_name() -> str:
    """Get the GPU model name using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]  # Get first GPU
            # Extract just the model name (e.g., "NVIDIA L4" -> "L4")
            if ' ' in gpu_name:
                model_name = gpu_name.split()[-1]  # Get last part
                return model_name
            return gpu_name
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback: try to get from CUDA device properties
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if ' ' in device_name:
                model_name = device_name.split()[-1]
                return model_name
            return device_name
    except Exception:
        pass
    
    # Final fallback
    return "GPU"


def generate_markdown_output_path(onnx_path: str) -> str:
    """Generate dynamic markdown output path based on ONNX path and current datetime."""
    # Try to extract model name from /data/output/{model_name}/... format
    model_name = None
    if onnx_path.startswith("/data/output/"):
        # Split the path and get the part after /data/output/
        path_parts = onnx_path.split("/")
        if len(path_parts) >= 4:  # /data/output/{model_name}/...
            model_name = path_parts[3]
    
    # Fallback to filename if path doesn't match expected format
    if model_name is None:
        onnx_filename = os.path.basename(onnx_path)
        model_name = os.path.splitext(onnx_filename)[0]
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output path
    output_path = f"/data/benchmark/{model_name}/report_{timestamp}.md"
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_path


def load_dataset_at(path: str):
    log_progress(f"Loading dataset from: {path}")
    if path.startswith('/'):
        dataset = load_from_disk(path)["train"]
    else:
        dataset = load_dataset(path)["train"]
    log_progress(f"Dataset loaded successfully. Size: {len(dataset):,} samples")
    return dataset


def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=SAMPLING_RATE):
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    return audio_array


class OnDemandWhisperDataset(Dataset):
    def __init__(self, hf_dataset, feature_extractor):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_array = sample["audio"]["array"]
        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)
        label = 1 if sample["endpoint_bool"] else 0

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=AUDIO_SECONDS * SAMPLING_RATE,
            truncation=True,
            do_normalize=True,
        )
        return {
            "input_features": inputs["input_features"].squeeze(0),  # (80,800) tensor
            "labels": torch.tensor(label, dtype=torch.long),
            "language": sample.get("language", "unknown"),  # Include language
            "dataset": sample.get("dataset", "unknown"),  # Include dataset
        }


@dataclass
class WhisperDataCollator:
    def __call__(self, features: List[Dict[str, Union[torch.Tensor, int, str]]]) -> Dict[
        str, Union[torch.Tensor, List[str]]]:
        input_features = torch.stack([f["input_features"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        languages = [f["language"] for f in features]
        datasets = [f["dataset"] for f in features]
        return {
            "input_features": input_features,
            "labels": labels,
            "languages": languages,
            "datasets": datasets
        }


def process_predictions(logits: np.ndarray):
    probs = logits.squeeze()
    preds = (probs > 0.5).astype(int)
    return probs, preds


def compute_metrics_with_confusion(probs: np.ndarray, labels: np.ndarray):
    """Compute metrics including false positive and false negative rates."""
    preds = (probs > 0.5).astype(int)

    # Calculate confusion matrix components
    false_positives = np.sum((preds == 1) & (labels == 0))
    false_negatives = np.sum((preds == 0) & (labels == 1))

    total = len(labels)

    return {
        "sample_count": int(total),
        "accuracy": float(accuracy_score(labels, preds)) * 100,
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "false_positive_rate": float(false_positives / total) * 100 if total > 0 else 0.0,
        "false_negative_rate": float(false_negatives / total) * 100 if total > 0 else 0.0,
    }


def compute_per_category_metrics(probs: np.ndarray, labels: np.ndarray, categories: List[str], category_name: str) -> \
        Dict[str, Dict[str, float]]:
    """Compute metrics for each category (language or dataset) separately."""
    log_progress(f"Computing per-{category_name} metrics...")
    category_metrics = {}

    # Group by category
    cat_data = defaultdict(lambda: {"probs": [], "labels": []})
    for prob, label, cat in zip(probs, labels, categories):
        cat_data[cat]["probs"].append(prob)
        cat_data[cat]["labels"].append(label)

    # Compute metrics for each category
    unique_categories = list(cat_data.keys())
    log_progress(f"Found {len(unique_categories)} unique {category_name}s: {sorted(unique_categories)}")

    for i, (cat, data) in enumerate(cat_data.items()):
        cat_probs = np.array(data["probs"])
        cat_labels = np.array(data["labels"])

        if len(cat_labels) > 0:  # Only compute if we have samples
            category_metrics[cat] = compute_metrics_with_confusion(cat_probs, cat_labels)
            log_progress(f"  [{i + 1}/{len(unique_categories)}] {cat}: {len(cat_labels)} samples, "
                         f"accuracy: {category_metrics[cat]['accuracy']:.2f}%")
        else:
            category_metrics[cat] = {
                "sample_count": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
            }
            log_progress(f"  [{i + 1}/{len(unique_categories)}] {cat}: 0 samples")

    return category_metrics


def format_language_name(lang_code: str) -> str:
    """Convert language code to full name with flag emoji."""
    return LANGUAGE_MAPPING.get(lang_code, lang_code)


def format_markdown_report(results: Dict, gpu_model_name: str = "GPU") -> str:
    """Format the results into a comprehensive Markdown report."""
    md_lines = []

    # Header
    md_lines.append("# Endpointing Model Benchmark Report")
    md_lines.append(f"\n**Model:** `{results['onnx_path']}`")
    md_lines.append(f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    # Accuracy Results
    if "accuracy" in results and "note" not in results["accuracy"]:
        acc_data = results["accuracy"]

        md_lines.append("\n## Accuracy Results")
        md_lines.append(f"\n**Total Samples:** {acc_data['total_samples']:,}")
        formatted_languages = [format_language_name(lang) for lang in acc_data['unique_languages']]
        md_lines.append(f"\n**Unique Languages:** {', '.join(formatted_languages)}")
        if 'unique_datasets' in acc_data:
            md_lines.append(f"\n**Unique Datasets:** {', '.join(acc_data['unique_datasets'])}")

        # Overall Accuracy Table
        md_lines.append("\n### Overall Performance")
        md_lines.append("| Metric | Sample Count | Accuracy (%) | False Positives (%) | False Negatives (%) |")
        md_lines.append("|--------|--------------|--------------|---------------------|---------------------|")

        overall = acc_data["overall"]
        md_lines.append(
            f"| Overall | {overall['sample_count']:,} | {overall['accuracy']:.2f} | {overall['false_positive_rate']:.2f} | {overall['false_negative_rate']:.2f} |")

        # Per-Language Accuracy Table
        if "per_language" in acc_data and acc_data["per_language"]:
            md_lines.append("\n### Performance by Language")
            md_lines.append("| Language | Sample Count | Accuracy (%) | False Positives (%) | False Negatives (%) |")
            md_lines.append("|----------|--------------|--------------|---------------------|---------------------|")

            # Sort languages by accuracy in descending order
            sorted_languages = sorted(acc_data["per_language"].keys(), 
                                    key=lambda lang: acc_data["per_language"][lang]['accuracy'], 
                                    reverse=True)
            
            for lang in sorted_languages:
                metrics = acc_data["per_language"][lang]
                formatted_lang = format_language_name(lang)
                md_lines.append(
                    f"| {formatted_lang} | {metrics['sample_count']:,} | {metrics['accuracy']:.2f} | {metrics['false_positive_rate']:.2f} | {metrics['false_negative_rate']:.2f} |")

        # Per-Dataset Accuracy Table
        if "per_dataset" in acc_data and acc_data["per_dataset"]:
            md_lines.append("\n### Performance by Dataset")
            md_lines.append("| Dataset | Sample Count | Accuracy (%) | False Positives (%) | False Negatives (%) |")
            md_lines.append("|---------|--------------|--------------|---------------------|---------------------|")

            # Sort datasets by accuracy in descending order
            sorted_datasets = sorted(acc_data["per_dataset"].keys(), 
                                   key=lambda dataset: acc_data["per_dataset"][dataset]['accuracy'], 
                                   reverse=True)
            
            for dataset in sorted_datasets:
                metrics = acc_data["per_dataset"][dataset]
                md_lines.append(
                    f"| {dataset} | {metrics['sample_count']:,} | {metrics['accuracy']:.2f} | {metrics['false_positive_rate']:.2f} | {metrics['false_negative_rate']:.2f} |")

    # Performance Results
    md_lines.append("\n## Inference Performance")

    # Direct Inference (pre-computed features)
    md_lines.append("\n### Direct Inference Performance")
    md_lines.append("*Using pre-computed zero features (inference only)*")
    md_lines.append(
        "\n| Provider | P50 Latency (ms) | P90 Latency (ms) | Mean Latency (ms) | Throughput (samples/sec) |")
    md_lines.append("|----------|------------------|------------------|-------------------|--------------------------|")

    if "perf_cpu" in results:
        cpu_perf = results["perf_cpu"]
        md_lines.append(
            f"| CPU | {cpu_perf['latency_ms_p50']:.2f} | {cpu_perf['latency_ms_p90']:.2f} | {cpu_perf['latency_ms_mean']:.2f} | {cpu_perf['throughput_sps']:.1f} |")

    if "perf_gpu" in results and "note" not in results["perf_gpu"]:
        gpu_perf = results["perf_gpu"]
        md_lines.append(
            f"| {gpu_model_name} | {gpu_perf['latency_ms_p50']:.2f} | {gpu_perf['latency_ms_p90']:.2f} | {gpu_perf['latency_ms_mean']:.2f} | {gpu_perf['throughput_sps']:.1f} |")

    # Feature Extraction Performance
    if "perf_feature_extractor" in results:
        md_lines.append("\n### Feature Extraction Performance")
        md_lines.append("*Whisper feature extraction from 8-second audio*")
        md_lines.append(
            "\n| Component | P50 Latency (ms) | P90 Latency (ms) | Mean Latency (ms) | Throughput (samples/sec) |")
        md_lines.append(
            "|-----------|------------------|------------------|-------------------|--------------------------|")

        fe_perf = results["perf_feature_extractor"]
        md_lines.append(
            f"| Feature Extractor | {fe_perf['latency_ms_p50']:.2f} | {fe_perf['latency_ms_p90']:.2f} | {fe_perf['latency_ms_mean']:.2f} | {fe_perf['throughput_sps']:.1f} |")

    # End-to-End Performance
    md_lines.append("\n### End-to-End Performance")
    md_lines.append("*Feature extraction + inference from raw audio*")
    md_lines.append(
        "\n| Provider | P50 Latency (ms) | P90 Latency (ms) | Mean Latency (ms) | Throughput (samples/sec) |")
    md_lines.append("|----------|------------------|------------------|-------------------|--------------------------|")

    if "perf_e2e_cpu" in results:
        e2e_cpu = results["perf_e2e_cpu"]
        md_lines.append(
            f"| CPU | {e2e_cpu['latency_ms_p50']:.2f} | {e2e_cpu['latency_ms_p90']:.2f} | {e2e_cpu['latency_ms_mean']:.2f} | {e2e_cpu['throughput_sps']:.1f} |")

    if "perf_e2e_gpu" in results and "note" not in results["perf_e2e_gpu"]:
        e2e_gpu = results["perf_e2e_gpu"]
        md_lines.append(
            f"| {gpu_model_name} | {e2e_gpu['latency_ms_p50']:.2f} | {e2e_gpu['latency_ms_p90']:.2f} | {e2e_gpu['latency_ms_mean']:.2f} | {e2e_gpu['throughput_sps']:.1f} |")

    # Add notes about any skipped measurements
    notes = []
    if "perf_gpu" in results and "note" in results["perf_gpu"]:
        notes.append("- GPU inference: " + results["perf_gpu"]["note"])
    if "perf_e2e_gpu" in results and "note" in results["perf_e2e_gpu"]:
        notes.append("- GPU end-to-end: " + results["perf_e2e_gpu"]["note"])
    if "accuracy" in results and "note" in results["accuracy"]:
        notes.append("- Accuracy evaluation: " + results["accuracy"]["note"])

    if notes:
        md_lines.append("\n## Notes")
        md_lines.extend(notes)

    return "\n".join(md_lines)


def _zero_audio(n_seconds: int = AUDIO_SECONDS, sample_rate: int = SAMPLING_RATE) -> np.ndarray:
    return np.zeros(n_seconds * sample_rate, dtype=np.float32)


def _extract_features_np(
        fe: WhisperFeatureExtractor,
        audio: np.ndarray,
) -> np.ndarray:
    """Return (1, 80, 800) np.float32 features from 8s audio."""
    out = fe(
        audio,
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",  # returns numpy arrays
        padding="max_length",
        max_length=AUDIO_SECONDS * SAMPLING_RATE,
        truncation=True,
        do_normalize=True,
    )["input_features"].astype(np.float32)
    # Ensure (1,80,800)
    if out.shape != FEATURE_SHAPE:
        out = out.reshape(FEATURE_SHAPE)
    return out


def _latency_stats(times: List[float]) -> Dict[str, float]:
    p50 = np.percentile(times, 50) * 1000
    p90 = np.percentile(times, 90) * 1000
    mean = np.mean(times) * 1000
    return {
        "latency_ms_p50": float(p50),
        "latency_ms_p90": float(p90),
        "latency_ms_mean": float(mean),
        "throughput_sps": float(1.0 / np.mean(times)),
    }


def run_fe_perf(
        fe: WhisperFeatureExtractor,
        audio: np.ndarray,
        runs: int = 1000,
        warmup: int = 100,
) -> Dict[str, float]:
    log_progress(f"Running feature extraction performance test ({warmup} warmup + {runs} runs)")

    # warmup
    log_progress("  Warming up feature extractor...")
    for i in range(warmup):
        _ = _extract_features_np(fe, audio)
        if (i + 1) % max(1, warmup // 4) == 0:
            log_progress(f"    Warmup progress: {i + 1}/{warmup}")

    log_progress("  Running timed feature extraction...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        _ = _extract_features_np(fe, audio)
        times.append(time.perf_counter() - t0)

        if (i + 1) % max(1, runs // 10) == 0:
            current_mean = np.mean(times) * 1000
            log_progress(f"    Progress: {i + 1}/{runs} | Current mean latency: {current_mean:.2f}ms")

    stats = _latency_stats(times)
    log_progress(f"  Feature extraction complete - Mean: {stats['latency_ms_mean']:.2f}ms, "
                 f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms")
    return stats


def run_e2e_perf(
        session: ort.InferenceSession,
        fe: WhisperFeatureExtractor,
        audio: np.ndarray,
        runs: int = 1000,
        warmup: int = 100,
) -> Dict[str, float]:
    provider_name = session.get_providers()[0]
    log_progress(f"Running end-to-end performance test on {provider_name} ({warmup} warmup + {runs} runs)")

    inp_name = session.get_inputs()[0].name

    # warmup
    log_progress("  Warming up end-to-end pipeline...")
    for i in range(warmup):
        feats = _extract_features_np(fe, audio)  # (1,80,800)
        _ = session.run(None, {inp_name: feats})
        if (i + 1) % max(1, warmup // 4) == 0:
            log_progress(f"    Warmup progress: {i + 1}/{warmup}")

    log_progress("  Running timed end-to-end inference...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        feats = _extract_features_np(fe, audio)
        _ = session.run(None, {inp_name: feats})
        times.append(time.perf_counter() - t0)

        if (i + 1) % max(1, runs // 10) == 0:
            current_mean = np.mean(times) * 1000
            log_progress(f"    Progress: {i + 1}/{runs} | Current mean latency: {current_mean:.2f}ms")

    stats = _latency_stats(times)
    log_progress(f"  End-to-end {provider_name} complete - Mean: {stats['latency_ms_mean']:.2f}ms, "
                 f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms")
    return stats


def run_perf(session: ort.InferenceSession, runs: int = 100, warmup: int = 10):
    provider_name = session.get_providers()[0]
    log_progress(f"Running inference performance test on {provider_name} ({warmup} warmup + {runs} runs)")

    inp = np.zeros(FEATURE_SHAPE, dtype=np.float32)
    feed = {session.get_inputs()[0].name: inp}

    # warmup
    log_progress("  Warming up inference session...")
    for i in range(warmup):
        session.run(None, feed)
        if (i + 1) % max(1, warmup // 4) == 0:
            log_progress(f"    Warmup progress: {i + 1}/{warmup}")

    log_progress("  Running timed inference...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append(time.perf_counter() - t0)

        if (i + 1) % max(1, runs // 10) == 0:
            current_mean = np.mean(times) * 1000
            log_progress(f"    Progress: {i + 1}/{runs} | Current mean latency: {current_mean:.2f}ms")

    stats = _latency_stats(times)
    log_progress(f"  Inference {provider_name} complete - Mean: {stats['latency_ms_mean']:.2f}ms, "
                 f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms")
    return stats


def build_session(onnx_path: str, providers: List[str]):
    log_progress(f"Building ONNX session with providers: {providers}")
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    log_progress(f"Session created successfully. Active provider: {session.get_providers()[0]}")
    return session


def run_accuracy(onnx_path: str, dataset_path: str, limit: Optional[int], batch_size: int = 64):
    log_progress("=" * 50)
    log_progress("ACCURACY EVALUATION")
    log_progress("=" * 50)

    log_progress("Loading dataset and preparing data loader...")
    base = load_dataset_at(dataset_path)
    fe = WhisperFeatureExtractor(chunk_length=AUDIO_SECONDS)  # 8 seconds
    wrapped = OnDemandWhisperDataset(base, fe)

    if limit is not None:
        n = min(limit, len(wrapped))
        indices = list(range(n))
        wrapped = torch.utils.data.Subset(wrapped, indices)
        log_progress(f"Limited dataset to {n:,} samples (limit: {limit:,})")
    else:
        log_progress(f"Using full dataset: {len(wrapped):,} samples")

    loader = DataLoader(wrapped, batch_size=batch_size, shuffle=False, collate_fn=WhisperDataCollator())
    total_batches = len(loader)
    log_progress(f"Created data loader: {total_batches} batches of size {batch_size}")

    log_progress("Building inference session...")
    # Prefer CUDA if available, fallback to CPU
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        session_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        log_progress("Using CUDA provider for accuracy evaluation")
    else:
        session_providers = ["CPUExecutionProvider"]
        log_progress("CUDA not available, using CPU provider for accuracy evaluation")
    
    sess = build_session(onnx_path, providers=session_providers)
    inp_name = sess.get_inputs()[0].name

    log_progress("Starting inference on dataset...")
    probs_all = []
    labels_all = []
    languages_all = []
    datasets_all = []

    samples_processed = 0
    batch_start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        batch_inference_start = time.time()

        x = batch["input_features"].numpy()  # (B,80,800)
        y = batch["labels"].numpy()
        langs = batch["languages"]
        dsets = batch["datasets"]

        # Run inference
        out = sess.run(None, {inp_name: x})
        probs, _ = process_predictions(out[0])

        probs_all.append(probs)
        labels_all.append(y)
        languages_all.extend(langs)
        datasets_all.extend(dsets)

        samples_processed += len(y)
        batch_inference_time = time.time() - batch_inference_start

        # Progress logging
        if (batch_idx + 1) % max(1, total_batches // 20) == 0 or batch_idx + 1 == total_batches:
            elapsed = time.time() - batch_start_time
            samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0
            eta_seconds = (len(wrapped) - samples_processed) / samples_per_sec if samples_per_sec > 0 else 0
            eta_str = f"{int(eta_seconds // 60)}:{int(eta_seconds % 60):02d}" if eta_seconds < float('inf') else "N/A"

            log_progress(f"  Batch {batch_idx + 1:,}/{total_batches:,} | "
                         f"Samples: {samples_processed:,}/{len(wrapped):,} | "
                         f"Rate: {samples_per_sec:.1f} samples/sec | "
                         f"Batch time: {batch_inference_time:.3f}s | "
                         f"ETA: {eta_str}")

    log_progress("Concatenating results...")
    probs_all = np.concatenate(
        [p if p.ndim == 1 else p.squeeze() for p in probs_all], axis=0
    ).astype(np.float32)
    labels_all = np.concatenate(labels_all, axis=0).astype(np.int32)

    log_progress(f"Computing metrics for {len(labels_all):,} samples...")

    # Compute overall metrics
    log_progress("Computing overall metrics...")
    overall_metrics = compute_metrics_with_confusion(probs_all, labels_all)
    log_progress(f"  Overall accuracy: {overall_metrics['accuracy']:.2f}%")

    # Compute per-language metrics
    per_language_metrics = compute_per_category_metrics(probs_all, labels_all, languages_all, "language")

    # Compute per-dataset metrics
    per_dataset_metrics = compute_per_category_metrics(probs_all, labels_all, datasets_all, "dataset")

    log_progress("Accuracy evaluation complete!")
    return {
        "overall": overall_metrics,
        "per_language": per_language_metrics,
        "per_dataset": per_dataset_metrics,
        "total_samples": len(labels_all),
        "unique_languages": sorted(list(set(languages_all))),
        "unique_datasets": sorted(list(set(datasets_all)))
    }


@app.function(
    image=image,
    gpu="T4",
    memory=8192,
    cpu=6.0,
    volumes={"/data": volume},
    timeout=60 * 60,
)
def benchmark(onnx_path: str,
              dataset_path: Optional[str] = None,
              limit: Optional[int] = None,
              perf_runs: int = 100,
              markdown_output: Optional[str] = None):
    # Generate markdown output path if not provided
    if markdown_output is None:
        markdown_output = generate_markdown_output_path(onnx_path)
    
    log_progress("=" * 80)
    log_progress("Starting benchmark")
    log_progress("=" * 80)
    log_progress(f"Model: {onnx_path}")
    log_progress(f"Dataset: {dataset_path if dataset_path else 'None (performance only)'}")
    log_progress(f"Sample limit: {limit if limit else 'None'}")
    log_progress(f"Performance runs: {perf_runs}")
    log_progress(f"Output file: {markdown_output}")
    log_progress("")

    results = {"onnx_path": onnx_path}

    # Detect GPU model name
    gpu_model_name = get_gpu_model_name()
    log_progress(f"Detected GPU model: {gpu_model_name}")

    # Providers
    log_progress("Checking available ONNX providers...")
    providers = ort.get_available_providers()
    log_progress(f"Available providers: {providers}")

    cpu_prov = ["CPUExecutionProvider"]
    gpu_prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else None

    if gpu_prov:
        log_progress("GPU (CUDA) provider available")
    else:
        log_progress("GPU (CUDA) provider not available - will skip GPU benchmarks")

    # Sessions
    log_progress("Building inference sessions...")
    cpu_sess = build_session(onnx_path, providers=cpu_prov)
    gpu_sess = build_session(onnx_path, providers=gpu_prov) if gpu_prov else None

    # ---------- Performance (zeros → direct) ----------
    log_progress("=" * 50)
    log_progress("Direct inference performance")
    log_progress("=" * 50)

    results["perf_cpu"] = run_perf(cpu_sess, runs=perf_runs)
    if gpu_sess:
        results["perf_gpu"] = run_perf(gpu_sess, runs=perf_runs)
    else:
        results["perf_gpu"] = {"note": "CUDAExecutionProvider not available; skipped."}
        log_progress("Skipping GPU inference performance test (CUDA not available)")

    # ---------- Feature extraction on 8s zero audio ----------
    log_progress("=" * 50)
    log_progress("Feature extraction performance")
    log_progress("=" * 50)

    fe = WhisperFeatureExtractor(chunk_length=AUDIO_SECONDS)
    zero_audio = _zero_audio(AUDIO_SECONDS, SAMPLING_RATE)
    results["perf_feature_extractor"] = run_fe_perf(fe, zero_audio, runs=perf_runs)

    # ---------- End-to-end (feature extraction + inference) ----------
    log_progress("=" * 50)
    log_progress("End-to-end performance")
    log_progress("=" * 50)

    results["perf_e2e_cpu"] = run_e2e_perf(cpu_sess, fe, zero_audio, runs=perf_runs)
    if gpu_sess:
        results["perf_e2e_gpu"] = run_e2e_perf(gpu_sess, fe, zero_audio, runs=perf_runs)
    else:
        results["perf_e2e_gpu"] = {"note": "CUDAExecutionProvider not available; skipped."}

    # ---------- Accuracy (dataset) ----------
    if dataset_path:
        results["accuracy"] = run_accuracy(onnx_path, dataset_path, limit=limit)
    else:
        results["accuracy"] = {"note": "No dataset_path provided; skipped."}

    # Generate markdown report
    markdown_report = format_markdown_report(results, gpu_model_name)

    # Write markdown report to file
    with open(markdown_output, 'w', encoding='utf-8') as f:
        f.write(markdown_report)

    # Print both the raw results (for debugging) and confirmation of file write
    print("=" * 80)
    print("Raw results:")
    print("=" * 80)
    print(results)
    print("\n" + "=" * 80)
    print(f"Markdown report written to: {markdown_output}")

    return results


@app.local_entrypoint()
def main(onnx_path: str,
         dataset_path: str = "",
         limit: Optional[int] = None,
         perf_runs: int = 100,
         markdown_output: Optional[str] = None):
    res = benchmark.remote(onnx_path, dataset_path if dataset_path else None, limit, perf_runs, markdown_output)
    print(res)