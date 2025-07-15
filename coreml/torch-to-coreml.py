import torch
from transformers import Wav2Vec2Processor
from model import Wav2Vec2ForEndpointing

import coremltools as ct
import numpy as np

MODEL_PATH = "pipecat-ai/smart-turn-v2"

# Load model and processor
model = Wav2Vec2ForEndpointing.from_pretrained(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model.eval()

print(f"Model config: {model.config}")

# Sample input for tracing (16 seconds of audio at 16kHz)
audio_random = torch.randn(16000 * 16)  # 16-second dummy audio
sample_input = processor(
    audio_random,
    sampling_rate=16000,
    padding="max_length",
    truncation=True,
    max_length=16000 * 16,
    return_attention_mask=True,
    return_tensors="pt",
)
print(sample_input)
print("input_values dtype: ", sample_input["input_values"].dtype)
print("attention_mask dtype: ", sample_input["attention_mask"].dtype)

print("input_values shape: ", sample_input["input_values"].shape)
print("attention_mask shape: ", sample_input["attention_mask"].shape)


# Create a class to better handle the forward pass for CoreML conversion
class TurnClassifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values, attention_mask):
        # Run the model
        print("inputs", input_values, attention_mask)
        print("----")
        outputs = self.model(input_values, attention_mask)
        print("outputs", outputs)
        logits = outputs["logits"]
        print("outputs logits shape", logits.shape)
        print("----")

        # Apply softmax to get probabilities
        # probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        # return probs

        return logits


# Create and trace the model
turn_classifier = TurnClassifier(model)
# turn_classifier = model
turn_classifier.eval()

input_values = sample_input["input_values"]
attention_mask = sample_input["attention_mask"]

print("input and attention mask shapes:", input_values.shape, attention_mask.shape)

traced_model = torch.jit.trace(turn_classifier, (input_values, attention_mask))
model_for_conversion = traced_model
print("Successfully traced the model")

print("Exporting to CoreML...")

# Convert to CoreML
# First convert in FP32
coreml_model = ct.convert(
    model_for_conversion,
    inputs=[
        ct.TensorType(name="input_values", shape=(1, 16000 * 16), dtype=np.float32),
        # note: if we specify the dtype here and don't disable quantization (using the
        # compute_precision argument below), export works but the model crashes during
        # inference.
        ct.TensorType(name="attention_mask", shape=(1, 16000 * 16), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    minimum_deployment_target=ct.target.iOS15,
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT32,
    # Export to FP16 hangs forever?
    # compute_precision=ct.precision.FLOAT16,
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
