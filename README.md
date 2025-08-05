# Smart turn detection

This is an open source, community-driven, native audio turn detection model.

> [!NOTE]
> Smart Turn V2 has now been released, with faster inference and support for 14 languages.
> 
> https://www.daily.co/blog/smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/
> 
> Contribute to model development. Generate turn detection data by playing the [turn training games](https://turn-training.pipecat.ai/), or help improve the quality of existing training data using the [classification tool](https://smart-turn-dataset.pipecat.ai/).

HuggingFace page: [pipecat-ai/smart-turn-v2](https://huggingface.co/pipecat-ai/smart-turn-v2)

Turn detection is one of the most important functions of a conversational voice AI technology stack. Turn detection means deciding when a voice agent should respond to human speech.

 Most voice agents today use *voice activity detection (VAD)* as the basis for turn detection. VAD segments audio into "speech" and "non-speech" segments. VAD can't take into account the actual linguistic or acoustic content of the speech. Humans do turn detection based on grammar, tone and pace of speech, and various other complex audio and semantic cues. We want to build a model that matches human expectations more closely than the VAD-based approach can.

This is a truly open model (BSD 2-clause license). Anyone can use, fork, and contribute to this project. This model started its life as a work in progress component of the [Pipecat](https://pipecat.ai) ecosystem. Pipecat is an open source, vendor neutral framework for building voice and multimodal realtime AI agents.

 ## Current state of the model

Smart Turn v2 supports 14 different languages, and was trained on a range of synthetic and human data. Performance has also increased 3x since the first version, and we're hopeful that this can be optimized even further.

Currently supported languages: 🇬🇧 🇺🇸 English, 🇫🇷 French, 🇩🇪 German, 🇪🇸 Spanish, 🇵🇹 Portuguese, 🇨🇳 Chinese, 🇯🇵 Japanese, 🇮🇳 Hindi, 🇮🇹 Italian, 🇰🇷 Korean, 🇳🇱 Dutch, 🇵🇱 Polish, 🇷🇺 Russian, and 🇹🇷 Turkish.

 ## Run the model locally

**Set up the environment:**

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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

python record_and_predict.py
```

## Model usage

### With Pipecat

Pipecat supports local inference using `LocalSmartTurnAnalyzerV2` (available in v0.0.77), and also supports using the instance hosted on [Fal](https://fal.ai/) using `FalSmartTurnAnalyzer`.

For more information, see the Pipecat documentation:

https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview

### With Pipecat Cloud

Pipecat Cloud users can make use of Fal's hosted Smart Turn v2 inference using `FalSmartTurnAnalyzer`. This service is provided at no extra cost.

See the following page for details:

https://pipecat-cloud.mintlify.app/pipecat-in-production/smart-turn

### With local inference

From the Smart Turn source repository, obtain the files `model.py` and `inference.py`. Import these files into your project and invoke the `predict_endpoint()` function with your audio. For an example, please see `predict.py`:

https://github.com/pipecat-ai/smart-turn/blob/main/predict.py

### With Fal hosted inference

[Fal](https://fal.ai/) provides a hosted Smart Turn endpoint which has been updated with the latest v2 model.

https://fal.ai/models/fal-ai/smart-turn/api

Please see the link above for documentation, or try the sample `curl` command below.

```bash
curl -X POST --url https://fal.run/fal-ai/smart-turn \
    --header "Authorization: Key $FAL_KEY" \
    --header "Content-Type: application/json" \
    --data '{ "audio_url": "https://fal.media/files/panda/5-QaAOC32rB_hqWaVdqEH.mpga" }'
```

### Notes on input format

Smart Turn takes 16kHz PCM audio as input. Up to 16 seconds of audio is supported, and we recommend providing approximately the last 8 seconds of speech.

The model is designed to be used in conjunction with a lightweight VAD model such as Silero. Once the VAD model detects silence, run Smart Turn on the entire recording of the user's turn, truncating from the beginning to shorten the audio to around 8 seconds if necessary.

If additional speech is detected from the user before Smart Turn has finished executing, re-run Smart Turn on the entire turn recording, including the new audio, rather than just the new segment. Smart Turn works best when given sufficient context, and is not designed to run on very short audio segments.

Note that audio from previous turns does not need to be included. 


## Project goals

The current version of this model is based on Meta AI's Wav2Vec2 backbone. More on model architecture below.

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

Smart Turn v2 uses Wav2Vec2 as a base, with a linear classifier layer. The model is transformer-based and has approximately 95M parameters.

We have experimented with multiple architectures and base models, including wav2vec2-BERT, LSTM, and additional transformer classifier layers. We've found that Wav2Vec2 with a linear classifier gives the best accuracy, however model architecture is still an open area of investigation. 

### Links

- [Meta AI Seamless paper](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)
- [Wav2Vec2 paper](https://arxiv.org/abs/2006.11477)
- [Wav2Vec2 model](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)


## Inference

`predict.py` shows how to pass an audio sample through the model for classification. A small convenience function in `inference.py` wraps the audio preprocessing and PyTorch inference code.

```
# defined in inference.py
def predict_endpoint(audio_array):
    """
    Predict whether an audio segment is complete (turn ended) or incomplete.

    Args:
        audio_array: Numpy array containing audio samples at 16kHz

    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion (sigmoid output)
    """

    # Process audio
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=16000 * 16,  # 16 seconds at 16kHz as specified in training
        return_attention_mask=True,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

        # The model returns sigmoid probabilities directly in the logits field
        probability = outputs["logits"][0].item()

        # Make prediction (1 for Complete, 0 for Incomplete)
        prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }
```

## Training

All training code is defined in `train.py`.

The training code will download datasets from the [pipecat-ai](https://huggingface.co/pipecat-ai) HuggingFace repository. (But of course you can modify it to use your own datasets.)

You can run training locally or using [Modal](https://modal.com). Training runs are logged to [Weights & Biases](https://www.wandb.ai) unless you disable logging.

```
# To run a training job on Modal, run:
modal run --detach train.py
```

### Collecting and contributing data

Currently, the following datasets are used for training and evaluation:

* pipecat-ai/rime_2
* pipecat-ai/human_5_all
* pipecat-ai/human_convcollector_1
* pipecat-ai/orpheus_grammar_1
* pipecat-ai/orpheus_midfiller_1
* pipecat-ai/orpheus_endfiller_1
* pipecat-ai/chirp3_1

The data is split into training, eval, and testing sets by `train.py`.

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

### Optimization

This model will likely perform well in quantized versions. Quantized versions should run significantly faster than the current float32 weights.

The PyTorch inference code is not particularly optimized. We should be able to hand-write inference code that runs substantially faster on both GPU and CPU, for this model architecture.

It would be nice to port inference code to Apple's MLX platform. This would be particular useful for local development and debugging, as well as potentially open up the possibility of running this model locally on iOS devices (in combination with quantization).

## Contributors

- [Marcus](https://github.com/marcus-daily)
- [Eli](https://github.com/ebb351)
- [Mark](https://github.com/markbackman)
- [Kwindla](https://github.com/kwindla)
