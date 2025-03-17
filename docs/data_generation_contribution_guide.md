# Turn Detection Audio Data Contribution README

## Overview
This guide provides instructions for generating and properly formatting a new dataset that you can use to train the turn detection classification model. Below are guidelines for what **"complete"** and **"incomplete"** samples should look like, how to process and upload raw audio files to Hugging Face Hub for training with our provided `train.py` script, and how to share the dataset publicly.
## Audio Sample Guidelines
- **File Type**: Use a lossless audio format (e.g. FLAC) to ensure high-quality recordings. The samples should be converted to a Hugging Face dataset before submission
- **Duration**: Each audio sample must be shorter than 16 seconds.
- **Content**: Samples should feature a clearly spoken sentence or phrase that is distinctly identifiable as "complete" or "incomplete."
### Complete Sentence Criteria
- **Finished Thought**: The sentence or phrase represents a finished thought or statement.
- **Natural Conclusion**: The sample ends naturally without filler words or trailing intonations.
- **Listener Clarity**: A human listener would confidently identify the sample as complete and ready for a response.
### Incomplete Sentence Criteria
- **Indication of Continuation**: The sample ends with incomplete phrases or filler words (e.g., "but," "um," "er").
- **Prosodic Cues**: The recording includes prosodic patterns (e.g. trailing off) that signal the speaker intends to continue speaking.
- **Listener Expectation**: A human listener would reasonably determine that the thought is not finished.
## Converting Your Raw Audio Files into a Hugging Face Dataset
Hugging Face datasets require specific formatting and labelling. We have provided the `convert_raw_files_to_huggingface_dataset.py` script. The script handles file organization, metadata generation, and dataset conversion. Follow these steps:
1. **Prepare Your Data**  
   - Organize your raw files into two subdirectories, `complete` and `incomplete`. 
   - Ensure each subdirectory contains the appropriate audio files.
   - The script expects FLAC files, but can easily be modified for other lossless audio formats
2. **Configure the Script**  
   - Open the script and replace the placeholder paths with your actual directories:
     - `COMPLETE_DIR`: Directory with complete audio files.  
     - `INCOMPLETE_DIR`: Directory with incomplete audio files.
     - `TMP_OUTPUT_DIR`: Directory where the organized audio files and JSONL metadata will be saved. 
     - `DATASET_SAVE_PATH`: Directory to save the final Hugging Face dataset. 
3. **Run the Script** 
   - In terminal: `python convert_raw_files_to_huggingface_dataset.py`
   - The script will:
     - Copy and rename FLAC files into an `audio` subdirectory.
     - Generate a JSONL metadata file that includes file paths and labels. This is a key intermediary step for creating the final Hugging Face dataset.
     - Convert the JSONL file into a Hugging Face dataset using the `audiofolder` loader.
     - Print dataset information and save the dataset to disk.

For additional guidance, refer to the **[Hugging Face Documentation](https://huggingface.co/docs/datasets/en/audio_dataset)**.
## Upload Your Dataset
After converting your raw audio files into a Hugging Face dataset, upload it to the Hugging Face Hub using the provided `upload-to-hub.py` script located in the `/datasets/scripts` directory. Follow these steps:
1. **Verify Your Dataset**  
   - Ensure the dataset was successfully created and saved using the conversion script. It should look like:
    ```
   final-dataset-name/
    ├── dataset_dict.json
    ├── train/
    │   ├── data-00000-of-00001.arrow
    │   ├── dataset_info.json
    │   └── state.json
   ```
2. **Upload Your Dataset to Hugging Face Hub**
   - Open a terminal and navigate to the `/datasets/scripts` directory.
   - Execute the `upload-to-hub` script, providing the path to your dataset directory. For example:
     `python3 upload-to-hub.py /path/to/your/dataset --upload --hub-id YOUR-DATASET-NAME --token YOUR-HUGGINGFACE-API-TOKEN`
	   - `--hub-id` is optional. It defines the dataset name in Hugging Face Hub. If excluded, the local directory name will be used
	   - `--token` should be used if you have not logged into Hugging Face via the CLI already
   - The script will:
     - Load and display information about your dataset.
     - Authenticate with the Hugging Face Hub (using your provided API token or saved credentials).
     - Upload the dataset to the Hub under the specified dataset ID.
3. **Making Your Dataset Public**  
   - If you want your dataset to be available publicly under the **pipecat-ai** organization, share the Hugging Face download link with us, and we will help with the final steps for publication. Open sourced contributions are encouraged!

**Now your dataset is ready to be used for training!**
## Additional Information
For further details or questions, please refer to the full project documentation or contact the maintainers.
