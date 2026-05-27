# Smart Turn v3.2

An open source, community-driven, native audio turn detection model.

* [HuggingFace page with model weights](https://huggingface.co/pipecat-ai/smart-turn-v3)

Turn detection is one of the most important functions of a conversational voice AI technology stack. Turn detection means deciding when a voice agent should respond to human speech.

Most voice agents today use *voice activity detection (VAD)* as the basis for turn detection. VAD segments audio into "speech" and "non-speech" segments. VAD can't take into account the actual linguistic or acoustic content of the speech. Humans do turn detection based on grammar, tone and pace of speech, and various other complex audio and semantic cues. We want to build a model that matches human expectations more closely than the VAD-based approach can.

This is a truly open model (BSD 2-clause license). Anyone can use, fork, and contribute to this project. This model started its life as a work in progress component of the [Pipecat](https://pipecat.ai) ecosystem. Pipecat is an open source, vendor neutral framework for building voice and multimodal realtime AI agents.

 ## Features

* **Support for 23 languages**
  * 🇸🇦 Arabic, 🇧🇩 Bengali, 🇨🇳 Chinese, 🇩🇰 Danish, 🇳🇱 Dutch, 🇩🇪 German, 🇬🇧 🇺🇸 English, 🇫🇮 Finnish, 🇫🇷 French, 🇮🇳 Hindi, 🇮🇩 Indonesian, 🇮🇹 Italian, 🇯🇵 Japanese, 🇰🇷 Korean, 🇮🇳 Marathi, 🇳🇴 Norwegian, 🇵🇱 Polish, 🇵🇹 Portuguese, 🇷🇺 Russian, 🇪🇸 Spanish, 🇹🇷 Turkish, 🇺🇦 Ukrainian, and 🇻🇳 Vietnamese.
* **Fast inference time**
  * Runs in as little as 10ms on some CPUs, and under 100ms on most cloud instances
  * Works in conjunction with a lightweight VAD model like Silero, meaning Smart Turn only needs to run during periods of silence
* **Available in CPU (8MB quantized) and GPU (32MB unquantized) versions**
  * The GPU version uses `fp32` weights, meaning it runs slightly faster on GPUs, and has slightly improved accuracy by around 1%
  * The CPU version is quantized to `int8`, making it significantly smaller and faster for CPU inference, at a slight accuracy cost
* **Audio native**
  * The model works directly with PCM audio samples, rather than text transcriptions, allowing it to take into account subtle prosody cues in the user's speech
* **Fully open source**
  * The datasets, training script, and model weights are all open source.

 ## Run the model locally

**Set up the environment:**

```
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[inference,examples]"
```

For training dependencies, install `pip install -e ".[training]"`. For Modal entrypoints, also install
`pip install -e ".[modal]"`.

You may need to install PortAudio development libraries if not already installed as those are required for PyAudio:

**Ubuntu/Debian**

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
```

**macOS (using Homebrew)**

```bash
brew install portaudio
```

**Run the utility**

Run a command-line utility that streams audio from the system microphone, detects segment start/stop using VAD, and sends each segment to the model for a phrase endpoint prediction.

```
# 
# It will take about 30 seconds to start up the first time.
#

# Try:
#
#   - "I can't seem to, um ..."
#   - "I can't seem to, um, find the return label."

smart-turn-record-and-predict --model /path/to/smart-turn-v3.1.onnx
```

## Model usage

### With Pipecat (and Pipecat Cloud)

Pipecat supports local inference using `LocalSmartTurnAnalyzerV3` (available in v0.0.85).

For more information, see the Pipecat documentation:

https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview

Smart Turn v3 has been extensively tested on [Pipecat Cloud](https://www.daily.co/products/pipecat-cloud/), with inference completing in around 65ms on a standard 1x instance when using `LocalSmartTurnAnalyzerV3`.

### With local inference

Install Smart Turn as a package and invoke `predict_endpoint()` with 16kHz mono audio:

```bash
pip install "smart-turn[inference] @ git+https://github.com/pipecat-ai/smart-turn.git"
```

```python
from smart_turn.inference import predict_endpoint

result = predict_endpoint(audio_array, onnx_path="/path/to/smart-turn-v3.1.onnx")
```

For a command-line example, run:

```bash
smart-turn-predict /path/to/audio.wav --model /path/to/smart-turn-v3.1.onnx
```

### Notes on input format

Smart Turn takes 16kHz mono PCM audio as input. Up to 8 seconds of audio is supported, and we recommend providing the full audio of the user's current turn.

The model is designed to be used in conjunction with a lightweight VAD model such as Silero. Once the VAD model detects silence, run Smart Turn on the entire recording of the user's turn, truncating from the beginning to shorten the audio to around 8 seconds if necessary.

If the input data is shorter than 8 seconds, insert padding at the beginning to make up the remaining length, such that the audio data is at the end of the input vector, and the padding zeroes are at the beginning.

If additional speech is detected from the user before Smart Turn has finished executing, re-run Smart Turn on the entire turn recording, including the new audio, rather than just the new segment. Smart Turn works best when given sufficient context, and is not designed to run on very short audio segments.

Note that audio from previous turns does not need to be included. 


## Project goals

The current version of this model is based on the Whisper Tiny backbone. More on model architecture below.

The high-level goal of this project is to build a state-of-the-art turn detection model that:
  - Anyone can use,
  - Is easy to deploy in production,
  - Is easy to fine-tune for specific application needs.

Medium-term goals:
  - Support for additional languages
  - Experiment with further optimizations and architecture improvements
  - Gather more human data for training and evaluation
  - Text conditioning of the model, to support "modes" like credit card, telephone number, and address entry.

## Model architecture

Smart Turn v3 uses Whisper Tiny as a base, with a linear classifier layer. The model is transformer-based and has approximately 8M parameters, and is available in both int8 quantized and full fp32 versions.

We have experimented with multiple architectures and base models, including wav2vec2-BERT, wav2vec2, LSTM, and additional transformer classifier layers.


## Inference

Sample code for inference is included in `smart_turn/inference.py`. See the `smart-turn-predict` and
`smart-turn-record-and-predict` commands for usage examples.

## Training

Training code is available from the `smart_turn.train` package module.

The training code will download datasets from the [pipecat-ai](https://huggingface.co/pipecat-ai) HuggingFace repository. (But of course you can modify it to use your own datasets.)

You can run training locally or using [Modal](https://modal.com) (using `train_modal.py`). Training runs are logged to [Weights & Biases](https://www.wandb.ai) unless you disable logging.

```bash
# Install training dependencies from a source checkout.
pip install -e ".[training]"

# Or depend on this repo from another project.
pip install "smart-turn[training] @ git+https://github.com/pipecat-ai/smart-turn.git"
```

```python
from smart_turn.train import CONFIG, do_training_run

CONFIG["datasets_training"] = ["/path/to/local/train_dataset"]
CONFIG["datasets_test"] = ["/path/to/local/test_dataset"]

model_path = do_training_run(run_name="my-smart-turn-run", output_dir="./output")
```

```
# To run a training job on Modal, run:
modal run --detach train_modal.py
```

### Collecting and contributing data

Currently, the following datasets are used for training and evaluation:

* pipecat-ai/smart-turn-data-v3.2-train
* pipecat-ai/smart-turn-data-v3.2-test

## Things to do

### Categorize training data

We're looking for people to help manually classify the training data and remove any invalid samples. If you'd like to help with this, please visit the following page:

https://smart-turn-dataset.pipecat.ai/

### Human training data

It's possible to contribute data to the project by playing the [turn training games](https://turn-training.pipecat.ai/). Alternatively, please feel free to [contribute samples directly](https://github.com/pipecat-ai/smart-turn/blob/main/docs/data_generation_contribution_guide.md) by following the linked README.

### Architecture experiments

The current model architecture is relatively simple. It would be interesting to experiment with other approaches to improve performance, have the model output additional information about the audio, or receive additional context as input.

For example, it would be great to provide the model with additional context to condition the inference. A use case for this would be for the model to "know" that the user is currently reciting a credit card number, or a phone number, or an email address.

### Supporting training on more platforms

We trained early versions of this model on Google Colab. We should support Colab as a training platform, again! It would be great to have quickstarts for training on a wide variety of platforms.

## Contributors

- [Marcus](https://github.com/marcus-daily)
- [Eli](https://github.com/ebb351)
- [Mark](https://github.com/markbackman)
- [Kwindla](https://github.com/kwindla)

Thank you to the following organisations for contributing audio datasets:

- [Liva AI](https://www.theliva.ai/)
- [Midcentury](https://www.midcentury.xyz/)
- [MundoAI](https://mundoai.world/)
