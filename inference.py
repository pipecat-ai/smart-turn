import torch
from transformers import Wav2Vec2BertForSequenceClassification, AutoFeatureExtractor

# MODEL_PATH = "model-v1"
MODEL_PATH = "pipecat-ai/smart-turn"

# Load model and processor
model = Wav2Vec2BertForSequenceClassification.from_pretrained(MODEL_PATH)
processor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

# Set model to evaluation mode and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        - probability: Probability of completion class
    """

    # Process audio
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=800,  # Maximum length as specified in training
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        completion_prob = probabilities[0, 1].item()  # Probability of class 1 (Complete)

        # Make prediction (1 for Complete, 0 for Incomplete)
        prediction = 1 if completion_prob > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": completion_prob,
    }
