# checkpoint-250415

* Reduced frozen layer count to 10

* Added new datasets
    * **`pipecat-ai/human_convcollector_1`** (~720 samples) -- conversational data collected using [https://turn-training.pipecat.ai/](https://turn-training.pipecat.ai/)
    * **`pipecat-ai/orpheus_grammar_1`** (~2,000 samples) -- voice samples generated using [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS). These samples primarily rely on grammatical structure to indicate incompleteness, and ellipses are used at the end of incomplete samples to guide the model's intonation.
    * **`pipecat-ai/orpheus_midfiller_1`** (~2,000 samples) -- similar to `orpheus_grammar_1`, but with filler words like "um" and "er" inserted into the _middle_ of sentences. These filler words should be ignored by the model.
    * **`pipecat-ai/orpheus_endfiller_1`** (~2,000 samples) -- voice samples where the incomplete sentences have filler words and the _end_.

# checkpoint-250305

Initial release.

Datasets:

* **`pipecat-ai/human_5_all`** (~4,000 samples) -- a selection of complete and incomplete human voice samples.
* **`pipecat-ai/rime_2`** (~4,000 samples) -- voice data generated using [Rime](https://rime.ai/), with filler words like "um" and "er" at the end of sentences.
