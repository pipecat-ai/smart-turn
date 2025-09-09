# Contributing training data to Smart Turn

Help make Smart Turn better by contributing training and testing data! This guide describes the required format of new Smart Turn datasets.

## File format

**FLAC is the preferred format** for training data. Lossy formats such as MP3 or Opus should be avoided.

**Mono audio** with a **bit depth of 16 bits** is used for training and is strongly preferred. The model works with 16kHz audio, but sample rates higher than this are fine.

We are happy to work with whatever naming scheme and directory structure suits you best. As an example, we use a unique UUID for each filename, and directory names as labels, for example, `eng/incomplete/b3799254-8d6c-11f0-a90e-e7e92780240b.flac`.

Samples must be labelled/grouped by language, and also by whether they are "complete" or "incomplete" (there is more detail on these in the "Labelling" section below).

It's fine to use directories to group samples, or to supply a separate metadata file (such as JSONL).


## Length

Each audio file should contain one speech sample, **no longer than 16 seconds**.

**Variation in length** is good, and samples can range from complex sentences to single word responses such as "yes" or "five".


## License

Smart Turn is a **fully open model**, and we release all datasets publicly: https://huggingface.co/pipecat-ai/datasets

By contributing, you confirm you own the recordings, the speakers consent to public release of their voice, and grant us the rights to redistribute.


## Speech content

Each audio file should contain a **single turn** in the conversation. **Only one person** should speak in each file.

Ideally the spoken words should resemble what someone would say to either a **voice assistant AI**, or a **customer service phone representative**. For example:

* "How tall is the Eiffel Tower?"
* "I'm trying to log into my account, but..."
* "Yes, that's correct."
* "The card number is eight seven one two..."

Please **avoid repeating sentences** in the dataset, and **minimise background noise**. No real PII (such as names or addresses) should be included in samples.


## Labelling

Each audio sample should be labelled **either "complete" or "incomplete"**. More detail is included on these categories below.

To ensure unbiased training, we use a **50:50 split** of complete to incomplete samples, so please aim for an equal number of each type. 

Please note that **prosody (intonation, speed, etc) is just as important as the choice of words**. If the sentence is grammatically complete, but the speaker sounds like they're still thinking, that sample should either be marked "incomplete", or not included. 


### "Complete" samples

A sample is "complete" if it represents a **finished thought**, and it would be natural for someone to respond straight away. For example:

* "How tall is the Eiffel Tower?"
* "Yes."
* "My PIN code is five, three, two, seven."

### "Incomplete" samples

A sample is "incomplete" if the speaker is **likely to continue talking**. For example:

* "How tall is the, um..."
* "Yes, but..."
* "My PIN code is five, *threeee*..."

Each incomplete sample should end in one or more of the following:

* **A filler word**, such as "um", "er", "well"
* **A connective word**, such as "and", "but", "because"
* **Prosody which suggests the speaker is thinking**, for example drawing out the last syllable, or using a pitch contour that leaves the thought hanging

Please note that incomplete samples **must not be cut off in the middle of a word**.

* ❌ "How tall is the Eiff"
* ✅ "How tall is the *Eiffel...*"

To give some background, Smart Turn operates in conjunction with a VAD (Voice Activity Detection) model. Only once the speaker has been silent for 200ms does Smart Turn actually run. So ideally, each incomplete sample (and complete sample) would **end with around 200ms of silence**. More silence than this is also fine.


## Submission

We don't have any specific requirements for how the data is submitted. For example, this could be through a shared cloud storage location (such as S3 or Google Drive), a HuggingFace repository, or similar.

The best way to get in touch with us about submitting a dataset is through [GitHub issues](https://github.com/pipecat-ai/smart-turn/issues).


## Questions

Please let us know if you have any questions or concerns, either through [GitHub](https://github.com/pipecat-ai/smart-turn/issues), or the Pipecat Discord server.