# CoreML model export

The scripts below work, but at the moment the CoreML export is not quantized, and runs ~8x more slowly than running the PyTorch model with the torch Apple Silicon `mps` backend.

So for now, it's better to use the PyTorch model on Apple Silicon. It would be great to figure out how to produce an optimized CoreML model. At the moment, the blockers are:
  - ANECompilerService runs forever at 100% CPU when trying to export to fp16 with compute_units=ct.ComputeUnit.ALL, or when trying to load an exported model with any setting *other* than compute_units=ct.ComputeUnit.CPU_ONLY
  - When running in CPU_ONLY mode, both the fp32 and fp16 models run reasonably fast, but still 50% slower than the torch mps backend.
  - When running in CPU_ONLY mode, there is an interaction with Python asyncio that causes segfaults. We spent quite a bit of time trying to narrow down the cause of this and formulate a workaround, but have not been completely successful here.

## ... old ...

Work in progress towards producing a performant coreml modal automatically from PyTorch model checkpoints.

The coreml model package is in the [coreml directory of the Hugging Face](https://huggingface.co/pipecat-ai/smart-turn/tree/main/coreml) repo.

## Current status

Model export and inference work.

The exported model is not quantized. coremltools quantizes to float16 by default, but
quantized model produces meaningless output. This means the model occupies >2GB
of memory.

On an M4 MacBook, inference runs in ~125ms.

## Files

- smart_turn_classifier.mlpackage -- the exported model; available at https://huggingface.co/pipecat-ai/smart-turn
- torch-to-coreml.py -- export script
- test-coreml-load.py -- sanity check that the exported model loads properly
- test-coreml-predict.py -- very basic prediction test with random data
- coreml_record_and_predict.py -- port of ../record_and_predicty.py. Run this for an interactive test with the system mic
- coreml_inference.py -- implementation of ../inference.py

## Run locally

1. Install git-lfs (Large File Storage)

   ```bash
   # Git LFS (Large File Storage)
   brew install git-lfs
   # Hugging Face uses LFS to store large model files, including .mlpackage
   git lfs install
   ```

2. Clone the repo with the smart_turn_classifier.mlpackage

   ```bash
   git clone https://huggingface.co/pipecat-ai/smart-turn
   ```

3. Set up the environment variable for where your model is installed

   ```bash
   export COREML_MODEL_PATH=/path/to/smart_turn_classifier.mlpackage
   ```

   For example, if you clone the smart-turn model repo into smart-turn/coreml, you would use:

   ```bash
   export COREML_MODEL_PATH=smart-turn/coreml/smart_turn_classifier.mlpackage
   ```

4. Run the model

   ```bash
   python coreml_record_and_predict.py
   ```

   Note: This takes around ~30 seconds to start.

## Todo

- Fix issues with quantization.
- Write a wrapper that loads platform-specific versions of the model from HuggingFace.
