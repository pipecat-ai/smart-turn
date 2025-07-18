# Smart-Turn – ONNX Toolkit

This folder contains everything you need to

1. export the fine-tuned **Smart-Turn** endpointing model to ONNX,
2. run batch/off-line inference, and
3. demo real-time endpointing with a microphone.

---
## 1.  Requirements

```bash
pip install -r onnx/requirements.txt   # optimum, onnxruntime, transformers …
```
All scripts work on CPU by default.  If you have a CUDA-enabled GPU install
`onnxruntime-gpu` instead of the CPU build.

---
## 2.  Export the model

```bash
python onnx/onnx_export.py
```
This will:
* load the fine-tuned PyTorch model from `v2-model/`
* export it to ONNX opset 14
* write `smart-turn-v2-onnx/model.onnx`
* validate that ONNX and PyTorch outputs match.

### Custom paths / options

| Method             | Flag / Variable             | Example                                 |
| ------------------ | --------------------------- | --------------------------------------- |
| CLI flags          | `--model` / `--out`         | `--model /path/to/model`                |
| Environment var    | `SMART_TURN_MODEL`          | `export SMART_TURN_MODEL=/my/model`      |
| Environment var    | `SMART_TURN_ONNX`           | `export SMART_TURN_ONNX=/tmp/onnx_out`   |

The script prints the resolved paths so you always know what it’s using.

---
## 3.  Batch inference helper

```python
from onnx.onnx_inference import predict_endpoint
import numpy as np

audio = np.random.randn(16000*2).astype(np.float32)  # 2-s fake clip
print(predict_endpoint(audio))
```
Output:
```json
{"prediction": 1, "probability": 0.87, "inference_ms": 12.4}
```
`predict_endpoint` automatically picks up the ONNX model created in step 2
(or the path in `SMART_TURN_ONNX`).

---
## 4.  Real-time microphone demo

```bash
python onnx/onnx_record_and_predict.py
```
The script will:
1. load Silero VAD (Torch Hub)
2. listen to your default microphone (16 kHz)
3. detect speech, trim it, feed it to ONNX
4. print prediction, probability and inference time.

Stop with <kbd>Ctrl+C</kbd>.

---
## 6.  Troubleshooting

* `FileNotFoundError: ONNX model not found …`  → run the exporter or set `SMART_TURN_ONNX` correctly.
* `onnxruntime not installed` → `pip install onnxruntime` (or `onnxruntime-gpu`).
---
