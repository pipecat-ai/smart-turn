# CoreML model export

Work in progress towards producing a performant coreml modal automatically from PyTorch model checkpoints.

The coreml model package is in the [coreml directory of the Hugging Face](https://huggingface.co/pipecat-ai/smart-turn/tree/main/coreml) repo.

## Current status

Model export and inference work.

The exported model is not quantized. coremltools quantizes to float16 by default, but
quantized model produces meaningless output. This means the model occupies >2GB
of memory.

On an m4 macbook, inference runs in ~125ms.

## Files

  - smart_turn_classifier.mlpackage -- the exported model. Not checked into the repo.
  - torch-to-coreml.py -- export script
  - test-coreml-load.py -- sanity check that the exported model loads properly
  - test-coreml-predict.py -- very basic prediction test with random data
  - coreml_record_and_predict.py -- port of ../record_and_predicty.py. Run this for an interactive test with the system mic
  - coreml_inference.py -- implementation of ../inference.py

## Todo

  - Fix issues with quantization.
  - Write a wrapper that loads platform-specific versions of the model from HuggingFace.


