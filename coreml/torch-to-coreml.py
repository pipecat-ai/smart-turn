import torch
from transformers import Wav2Vec2BertForSequenceClassification, AutoFeatureExtractor
import coremltools as ct

MODEL_PATH = "pipecat-ai/smart-turn"

# Load model and processor
model = Wav2Vec2BertForSequenceClassification.from_pretrained(MODEL_PATH)
processor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model.eval()

print(f"Model config: {model.config}")

# Sample input for tracing (8 seconds of audio at 16kHz)
audio_random = torch.randn(1, 16000 * 8)
sample_input = processor(
    audio_random,
    sampling_rate=16000,
    padding="max_length",
    truncation=True,
    max_length=800,
    return_attention_mask=True,
    return_tensors="pt",
)
print(sample_input)

print("input_features shape: ", sample_input["input_features"].shape)
print("attention_mask shape: ", sample_input["attention_mask"].shape)


# Create a class to better handle the forward pass for CoreML conversion
class TurnClassifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, attention_mask):
        # Run the model
        print("x", input_features, attention_mask)
        outputs = self.model(input_features, attention_mask)

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs


# Create and trace the model
turn_classifier = TurnClassifier(model)
turn_classifier.eval()

traced_model = torch.jit.trace(
    turn_classifier, (sample_input["input_features"], sample_input["attention_mask"])
)
model_for_conversion = traced_model
print("Successfully traced the model")

print("Exporting to CoreML...")

# Define the proper input shape for audio
input_shape = (1, 16000 * 8)  # (batch_size, audio_samples)

# Convert to CoreML
coreml_model = ct.convert(
    model_for_conversion,
    inputs=[
        ct.TensorType(name="input_features", shape=sample_input["input_features"].shape),
        ct.TensorType(name="attention_mask", shape=sample_input["attention_mask"].shape),
    ],
    outputs=[ct.TensorType(name="turn_probabilities")],
    minimum_deployment_target=ct.target.iOS15,
)

# Set model metadata
coreml_model.author = "Smart Turn Model"
coreml_model.license = "BSD"
coreml_model.short_description = "Turn detection classifier for audio"
coreml_model.version = "1.0"

# Save the CoreML model
output_path = "smart_turn_classifier.mlpackage"
coreml_model.save(output_path)

print(f"CoreML model saved successfully as '{output_path}'")
