# CoreML model export

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
