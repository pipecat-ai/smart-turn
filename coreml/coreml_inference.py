import logging
import os

import coremltools as ct
from transformers import Wav2Vec2Processor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TORCH_MODEL_PATH = "pipecat-ai/smart-turn-v2"
# Get the model path from environment variable, with a fallback default value
MODEL_PATH = os.environ.get("COREML_MODEL_PATH", "smart_turn_classifier.mlpackage")
logger.info(f"Loading Core ML model from: {MODEL_PATH}")

# Load processor (no need to load the large PyTorch model)
processor = Wav2Vec2Processor.from_pretrained(TORCH_MODEL_PATH)
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

    inputs = processor(
        audio_array,
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=16000 * 16,  # 16-second maximum length used in training
        return_attention_mask=True,
        return_tensors="np",
    )

    # Core ML expects numpy arrays, but processor returns int64 masks – cast to int32
    inputs["attention_mask"] = inputs["attention_mask"].astype("int32")
    output = model.predict(dict(inputs))
    probabilities = output["logits"]  # Already sigmoid output in our Core ML model
    completion_prob = float(probabilities[0][0])
    prediction = 1 if completion_prob > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": completion_prob,
    }
