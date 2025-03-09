import coremltools as ct
import numpy as np
import time

MODEL_PATH = "smart_turn_classifier.mlpackage"

print("Loading model")
start_time = time.perf_counter()
model = ct.models.MLModel(MODEL_PATH)
print(f"Model loaded in {time.perf_counter() - start_time} seconds")

input_features_shape = [1, 400, 160]
attention_mask_shape = [1, 400]

input_features = np.random.rand(*input_features_shape)
attention_mask = np.ones(attention_mask_shape)

print("Testing inference time")
times = []
for _ in range(10):
    start_time = time.perf_counter()
    output_dict = model.predict(
        {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
    )
    end_time = time.perf_counter()
    times.append(end_time - start_time)

print(f"Max: {max(times) * 1000:.2f}ms")
print(f"Min: {min(times) * 1000:.2f}ms")
print(f"Median: {np.median(times) * 1000:.2f}ms")
