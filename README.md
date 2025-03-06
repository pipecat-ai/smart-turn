# Smart turn detection

This is an open source, community-driven, native audio turn detection model.

HuggingFace page: [pipecat-ai/smart-turn](https://huggingface.co/pipecat-ai/smart-turn)

Turn detection is one of the most important functions of a conversational voice AI technology stack. Turn detection means deciding when a voice agent should respond to human speech.

 Most voice agents today use *voice activity detection (VAD)* as the basis for turn detection. VAD segments audio into "speech" and "non-speech" segments. VAD can't take into account the actual linguistic or acoustic content of the speech. Humans do turn detection based on grammar, tone and pace of speech, and various other complex audio and semantic cues. We want to build a model that matches human expectations more closely than the VAD-based approach can.

This is a truly open model. Anyone can use, fork, and contribute to this project.

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

[ notes on data coming soon ]

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

We trained early versions of this model on Google Colab. We should support Colab as a training platform, again!

It would be great to port the training code to Apple's MLX platform as well. (A lot of us have MacBooks!)

### Optimization

This model will likely perform well in quantized versions. Quantized models should run significantly faster.

The PyTorch inference code is not particularly optimized. We should be able to hand-write inference code that runs substantially faster on both GPU and CPU, for this model architecture.

It would be nice to port inference code to Apple's MLX platform. This would be particular useful for local development and debugging, as well as potentially open up the possibility of running this model locally on iOS devices (in combination with quantization).

## Contributors

- [Marcus](https://github.com/marcus-daily)
- [Eli](https://github.com/ebb351)
- [Mark](https://github.com/markbackman)
- [Kwindla](https://github.com/kwindla)
