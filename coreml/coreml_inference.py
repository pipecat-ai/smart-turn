import torch
import coremltools as ct
from transformers import Wav2Vec2BertForSequenceClassification, AutoFeatureExtractor
import numpy as np
import hashlib

TORCH_MODEL_PATH = "pipecat-ai/smart-turn"
MODEL_PATH = "smart_turn_classifier.mlpackage"

# Load model and processor
# todo: we shouldn't need to load the torch_model for coreml-only operation. how do we get
# processor without the model loaded?
torch_model = Wav2Vec2BertForSequenceClassification.from_pretrained(TORCH_MODEL_PATH)
processor = AutoFeatureExtractor.from_pretrained(TORCH_MODEL_PATH)
model = ct.models.MLModel(MODEL_PATH)


def predict_endpoint(audio_array):
    """
    Predict whether an audio segment is complete (turn ended) or incomplete.

    Args:
        audio_array: Numpy array containing audio samples at 16kHz

    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion class
    """

    # Calculate basic statistics
    audio_min = np.min(audio_array)
    audio_max = np.max(audio_array)
    audio_mean = np.mean(audio_array)
    audio_std = np.std(audio_array)

    # Calculate a hash of the input data
    audio_hash = hashlib.md5(audio_array.tobytes()).hexdigest()[:10]

    print(
        f"Input Audio Stats - Min: {audio_min:.4f}, Max: {audio_max:.4f}, Mean: {audio_mean:.4f}, Std: {audio_std:.4f}"
    )
    print(f"Input Audio Hash: {audio_hash}")
    print(f"Input Audio Shape: {audio_array.shape}")

    inputs = processor(
        audio_array,
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=800,  # Maximum length as specified in training
        return_attention_mask=True,
        return_tensors="pt",
    )

    print(dict(inputs))
    output = model.predict(dict(inputs))
    logits = output["turn_probabilities"]  # Core ML returns numpy array
    print("logits", logits)
    # Convert numpy array to PyTorch tensor before applying softmax
    logits_tensor = torch.tensor(logits)
    probabilities = torch.nn.functional.softmax(logits_tensor, dim=1)
    completion_prob = probabilities[0, 1].item()  # Probability of class 1 (Complete)

    # Make prediction (1 for Complete, 0 for Incomplete)
    prediction = 1 if completion_prob > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": completion_prob,
    }
