# Smart turn detection

This is an open source, community-driven, native audio turn detection model.

HuggingFace page: [pipecat-ai/smart-turn](https://huggingface.co/pipecat-ai/smart-turn)

Turn detection is one of the most important functions of a conversational voice AI technology stack. Turn detection means deciding when a voice agent should respond to human speech.

 Most voice agents today use *voice activity detection (VAD)* as the basis for turn detection. VAD segments audio into "speech" and "non-speech" segments. VAD can't take into account the actual linguistic or acoustic content of the speech. Humans do turn detection based on grammar, tone and pace of speech, and various other complex audio and semantic cues. We want to build a model that matches human expectations more closely than the VAD-based approach can.

This is a truly open model (BSD 2-clause license). Anyone can use, fork, and contribute to this project. This model started its life as a work in progress component of the [Pipecat](https://pipecat.ai) ecosystem. Pipecat is an open source, vendor neutral framework for building voice and multimodal realtime AI agents.

 ## Current state of the model

 This is an initial proof-of-concept model. It handles a small number of common non-completion scenarios. It supports only English. The training data set is relatively small.

 We have experimented with a number of different architectures and approaches to training data, and are releasing this version of the model now because we are confident that performance can be rapidly improved.

 We invite you to try it, and to contribute to model development and experimentation.

 ## Run the proof-of-concept model checkpoint

Set up the environment.

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Installing audio dependencies
You may need to install PortAudio development libraries if not already installed as those are required for PyAudio:
### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
```

### macOS (using Homebrew)
```bash
brew install portaudio
```

Run a command-line utility that streams audio from the system microphone, detects segment start/stop using VAD, and sends each segment to the model for a phrase endpoint prediction.

```
# 
# It will take about 30 seconds to start up the first time.
#

# "Vocabulary" is limited. Try:
#
#   - "I can't seem to, um ..."
#   - "I can't seem to, um, find the return label."

python record_and_predict.py
```

## Project goals

The current version of this model is based on Meta AI's Wav2Vec2-BERT backbone. More on model architecture below.

The high-level goal of this project is to build a state-of-the-art turn detection model that is:
  - Anyone can use,
  - Is easy to deploy in production,
  - Is easy to fine-tune for specific application needs.

Current limitations:
  - English only
  - Relatively slow inference
    - ~150ms on GPU
    - ~1500ms on CPU
  - Training data focused primarily on pause filler words at the end of a segment.

Medium-term goals:
  - Support for a wide range of languages
  - Inference time <50ms on GPU and <500ms on CPU
  - Much wider range of speech nuances captured in training data
  - A completely synthetic training data pipeline
  - Text conditioning of the model, to support "modes" like credit card, telephone number, and address entry.

## Model architecture

Wav2Vec2-BERT is a speech encoder model developed as part of Meta AI's Seamless-M4T project. It is a 580M parameter base model that can leverage both acoustic and linguistic information. The base model is trained on 4.5M hours of unlabeled audio data covering more than 143 languages.

To use Wav2Vec2-BERT, you generally add additional layers to the base model and then train/fine-tune on an application-specific dataset. 

We are currently using a simple, two-layer classification head, conveniently packaged in the Hugging Face Transformers library as `Wav2Vec2BertForSequenceClassification`.

We have experimented with a variety of architectures, including the widely-used predecessor of Wav2Vec2-BERT, Wav2Vec2, and more complex approaches to classification. Some of us who are working on the model think that the simple classification head will work well as we scale the data set to include more complexity. Some of us have the opposite intuition. Time will tell! Experimenting with additions to the model architecture is an excellent learning project if you are just getting into ML engineering. See "Things to do" below.

### Links

- [Meta AI Seamless paper](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)
- [W2v-BERT 2.0 speech encoder README](https://github.com/facebookresearch/seamless_communication?tab=readme-ov-file#w2v-bert-20-speech-encoder)
- [Wav2Vec2BertForSequenceClassification HuggingFace docs](https://huggingface.co/docs/transformers/v4.49.0/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForSequenceClassification)


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
        - probability: Probability of completion class
    """

    # Process audio
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=800,  # Maximum length as specified in training
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        completion_prob = probabilities[0, 1].item()  # Probability of class 1 (Complete)

        # Make prediction (1 for Complete, 0 for Incomplete)
        prediction = 1 if completion_prob > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": completion_prob,
    }
```

## Training

All training code is defined in `train.py`.

You can run training locally or using [Modal](https://modal.com). Training runs are logged to [Weights & Biases](https://www.wandb.ai) unless you disable logging.

```
# To run a training job on Modal, upload training data to a Modal volume,
# set up the Modal environment, then run:
modal run --detach train.py
```

### Collecting and contributing data

Currently, there are two datasets used for training and evaluation:
  - datasets/human_5_all -- segmented speech recorded from human interactions
  - datasets/rime_2 -- synthetic speech generated using [Rime](https://rime.ai/)

Four splits are created [when these two datasets are loaded](https://github.com/pipecat-ai/smart-turn/blob/a9e49f18da2d70dde94477be05405638db9dd8bc/train.py#L188).
  - The train, validate, and test sets are a mix of synthetic and human data
  - The human eval set contains only human data

```
  7 -- TRAIN --
  8   Total samples: 5,694
  9   Positive samples (Complete): 2,733 (48.00%)
 10   Negative samples (Incomplete): 2,961 (52.00%)
 11 
 12 -- VALIDATION --
 13   Total samples: 712
 14   Positive samples (Complete): 352 (49.44%)
 15   Negative samples (Incomplete): 360 (50.56%)
 16 
 17 -- TEST --
 18   Total samples: 712
 19   Positive samples (Complete): 339 (47.61%)
 20   Negative samples (Incomplete): 373 (52.39%)
 21 
 22 -- HUMAN_EVAL --
 23   Total samples: 773
 24   Positive samples (Complete): 372 (48.12%)
 25   Negative samples (Incomplete): 401 (51.88%)
 ```

Our goal for an initial version of this model was to overfit on a non-trivial amount of data, plus exceed a non-quantitative vibes threshold when experimenting interactively. The next step is to broaden the amount of data and move away from overfitting towards more generalization.


![Confusion matrix for test set](docs/static/confusion_matrix_test_1360_b0d85b27b14cc7bd6a0d.png)


[ more notes on data coming soon ]

## Things to do

### More languages

The base Wav2Vec2-BERT model is trained on a large amount of multi-lingual data. Supporting additional languages will require either collecting and cleaning or synthetically generating data for each language.

### More data

The current checkpoint was trained on a dataset of approximately 8,000 samples. These samples mostly focus on filler words that are typical indications of a pause without utterance completion in English-language speech.

Two data sets are used in training: around 4,000 samples collected from human speakers, and around 4,000 synthetic data samples generated using [Rime](https://rime.ai/). 

The biggest short-term data need is to collect, categorize, and clean human data samples that represent a broader range of speech patterns:
  - inflection and pacing that indicates a "thinking" pause rather than a completed speech segment
  - grammatical structures that typically occur in unfinished speech segments (but not in finished segments)
  - more individual speakers represented
  - more regions and accents represented

The synthetic data samples in the `datasets/rime_2` dataset only improve model performance by a small margin, right now. But one possible goal for this project is to work towards a completely synthetic data generation pipeline. The potential advantages of such a pipeline include the ability to support more languages more easily, a better flywheel for building more accurate versions of the model, and the ability to rapidly customize the model for specific use cases.

If you have expertise in steering speech models so that they output specific patterns (or if you want to experiment and learn), please consider contributing synthetic data.

### Architecture experiments

The current model architecture is relatively simple, because the base Wav2Vec2-BERT model is already quite powerful.

However, it would be interesting to experiment with other approaches to classification added on top of the Wav2Vec2-BERT model. This might be particularly useful if we want to move away from binary classification towards an approach that is more customized for this turn detection task.

For example, it would be great to provide the model with additional context to condition the inference. A use case for this would be for the model to "know" that the user is currently reciting a credit card number, or a phone number, or an email address.

Adding additional context to the model is an open-ended research challenge. Some simpler todo list items include:

  - Experimenting with freezing different numbers of layers during training.
  - Hyperparameter tuning.
  - Trying different sizes for the classification head or moderately different classification head designs and loss functions.

### Supporting training on more platforms

We trained early versions of this model on Google Colab. We should support Colab as a training platform, again! It would be great to have quickstarts for training on a wide variety of platforms.

We should alsoport the training code to Apple's MLX platform. A lot of us have MacBooks!

### Optimization

This model will likely perform well in quantized versions. Quantized versions should run significantly faster than the current float32 weights.

The PyTorch inference code is not particularly optimized. We should be able to hand-write inference code that runs substantially faster on both GPU and CPU, for this model architecture.

It would be nice to port inference code to Apple's MLX platform. This would be particular useful for local development and debugging, as well as potentially open up the possibility of running this model locally on iOS devices (in combination with quantization).

## Contributors

- [Marcus](https://github.com/marcus-daily)
- [Eli](https://github.com/ebb351)
- [Mark](https://github.com/markbackman)
- [Kwindla](https://github.com/kwindla)
