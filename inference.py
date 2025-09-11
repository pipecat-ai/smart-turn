import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

ONNX_MODEL_PATH = "smart-turn-v3.0.onnx"

def build_session(onnx_path):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so)

feature_extractor = WhisperFeatureExtractor(chunk_length=8)
session = build_session(ONNX_MODEL_PATH)

def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Pad with zeros at the beginning
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
    return audio_array


def predict_endpoint(audio_array):
    """
    Predict whether an audio segment is complete (turn ended) or incomplete.

    Args:
        audio_array: Numpy array containing audio samples at 16kHz

    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion (sigmoid output)
    """

    # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

    # Process audio using Whisper's feature extractor
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    )

    # Convert to numpy and ensure correct shape for ONNX
    input_features = inputs.input_features.squeeze(0).numpy().astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

    # Run ONNX inference
    outputs = session.run(None, {"input_features": input_features})

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
