import os
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from .audio_utils import truncate_audio_to_last_n_seconds

ONNX_MODEL_PATH = os.environ.get("SMART_TURN_ONNX_MODEL_PATH", "smart-turn-v3.1.onnx")
DEFAULT_ONNX_MODEL_PATH = ONNX_MODEL_PATH


def build_session(onnx_path):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=so)


@lru_cache(maxsize=4)
def get_session(onnx_path: str = DEFAULT_ONNX_MODEL_PATH):
    return build_session(onnx_path)


@lru_cache(maxsize=1)
def get_feature_extractor():
    return WhisperFeatureExtractor(chunk_length=8)


def predict_endpoint(
        audio_array,
        onnx_path=DEFAULT_ONNX_MODEL_PATH,
        session=None,
        feature_extractor=None,
):
    """
    Predict whether an audio segment is complete (turn ended) or incomplete.

    Args:
        audio_array: Numpy array containing audio samples at 16kHz
        onnx_path: Path to the Smart Turn ONNX model. Defaults to
            SMART_TURN_ONNX_MODEL_PATH or smart-turn-v3.1.onnx.
        session: Optional pre-built ONNX Runtime session.
        feature_extractor: Optional Whisper feature extractor.

    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion (sigmoid output)
    """
    session = session or get_session(str(onnx_path))
    feature_extractor = feature_extractor or get_feature_extractor()

    # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

    # Process audio using Whisper's feature extractor
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    )

    # Extract features and ensure correct shape for ONNX
    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

    # Run ONNX inference
    outputs = session.run(None, {"input_features": input_features})

    # Extract probability (ONNX model returns sigmoid probabilities)
    probability = np.asarray(outputs[0]).reshape(-1)[0].item()

    # Make prediction (1 for Complete, 0 for Incomplete)
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }


def _example_main():
    # Create a dummy audio array for testing (1 second of random audio)
    dummy_audio = np.random.randn(16000).astype(np.float32)

    result = predict_endpoint(dummy_audio)
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")


if __name__ == "__main__":
    _example_main()
