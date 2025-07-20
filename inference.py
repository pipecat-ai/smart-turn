import torch
from transformers import Wav2Vec2Processor
from model import Wav2Vec2ForEndpointing

# MODEL_PATH = "path/to/your/trained/model"
MODEL_PATH = "pipecat-ai/smart-turn-v2"

# Load model and processor
model = Wav2Vec2ForEndpointing.from_pretrained(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

# Set model to evaluation mode and move to platform-optimized backend if available.
# MPS for Apple silicon, CUDA for NVIDIA.
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
model = model.to(device)
model.eval()


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

    # Process audio
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=16000 * 16,  # 16 seconds at 16kHz as specified in training
        return_attention_mask=True,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

        # The model returns sigmoid probabilities directly in the logits field
        probability = outputs["logits"][0].item()

        # Make prediction (1 for Complete, 0 for Incomplete)
        prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Create a dummy audio array for testing (1 second of random audio)
    dummy_audio = np.random.randn(16000).astype(np.float32)

    result = predict_endpoint(dummy_audio)
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")