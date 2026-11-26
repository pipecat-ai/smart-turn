import os
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from audio_utils import resample_audio, truncate_audio_to_last_n_seconds

TARGET_SAMPLE_RATE = 16000
DEFAULT_ONNX_CANDIDATES = (
    "smart-turn-v3.2-cpu.onnx",
    "smart-turn-v3.1-cpu.onnx",
    "smart-turn-v3.1.onnx",
)
ONNX_MODEL_PATH = os.environ.get("SMART_TURN_ONNX_PATH", DEFAULT_ONNX_CANDIDATES[0])

_feature_extractor = None
_sessions = {}


def _resolve_onnx_path(onnx_path=None):
    if onnx_path:
        return onnx_path
    env_path = os.environ.get("SMART_TURN_ONNX_PATH")
    if env_path:
        return env_path
    for candidate in DEFAULT_ONNX_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return ONNX_MODEL_PATH


def build_session(onnx_path, intra_op_num_threads=None):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    if intra_op_num_threads is None:
        intra_op_num_threads = int(os.environ.get("SMART_TURN_INTRA_OP_THREADS", "2"))
    if intra_op_num_threads > 0:
        so.intra_op_num_threads = intra_op_num_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def _get_runtime(onnx_path=None):
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = WhisperFeatureExtractor(chunk_length=8)
    resolved = _resolve_onnx_path(onnx_path)
    if resolved not in _sessions:
        _sessions[resolved] = build_session(resolved)
    return _feature_extractor, _sessions[resolved]


def predict_endpoint(audio_array, sample_rate=TARGET_SAMPLE_RATE, onnx_path=None):
    """
    Predict whether an audio segment is complete (turn ended) or incomplete.

    Args:
        audio_array: 1-D Numpy array containing mono PCM audio samples in [-1, 1]
        sample_rate: Input audio sample rate in Hz (default: 16000; resamples e.g. 8000 Hz telephony)
        onnx_path: Optional explicit path to the Smart Turn ONNX checkpoint

    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion (sigmoid output)
    """
    audio_array = np.asarray(audio_array, dtype=np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_array = resample_audio(audio_array, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

    # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
    audio_array = truncate_audio_to_last_n_seconds(
        audio_array, n_seconds=8, sample_rate=TARGET_SAMPLE_RATE
    )

    fe, sess = _get_runtime(onnx_path=onnx_path)

    # Process audio using Whisper's feature extractor
    inputs = fe(
        audio_array,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=8 * TARGET_SAMPLE_RATE,
        truncation=True,
        do_normalize=True,
    )

    # Extract features and ensure correct shape for ONNX
    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

    # Run ONNX inference
    outputs = sess.run(None, {"input_features": input_features})

    # Extract probability (ONNX model returns sigmoid probabilities)
    probability = outputs[0][0].item()

    # Make prediction (1 for Complete, 0 for Incomplete)
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }


# Example usage
if __name__ == "__main__":
    # Create a dummy audio array for testing (1 second of random audio)
    dummy_audio = np.random.randn(16000).astype(np.float32)

    result = predict_endpoint(dummy_audio)
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
