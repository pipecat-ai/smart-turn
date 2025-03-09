import coremltools as ct
from transformers import Wav2Vec2BertForSequenceClassification, AutoFeatureExtractor

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

    inputs = processor(
        audio_array,
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=800,  # Maximum length as specified in training
        return_attention_mask=True,
        return_tensors="pt",
    )

    return model.predict(dict(inputs))
