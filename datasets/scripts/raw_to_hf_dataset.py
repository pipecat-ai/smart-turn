#!/usr/bin/env python3
import os
import shutil
import json
import uuid
from pathlib import Path
from datasets import load_dataset

def create_audio_dataset(complete_dir: str, incomplete_dir: str, output_dir: str):
    """
    Create a dataset from two directories of FLAC files with metadata in JSONL format.

    Args:
        complete_dir (str): Path to the directory containing complete FLAC files.
        incomplete_dir (str): Path to the directory containing incomplete FLAC files.
        output_dir (str): Path to the output directory for the dataset.
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)

    # Open JSONL file for writing metadata
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, "w") as jsonl_file:
        # Process complete FLAC files
        for flac_file in Path(complete_dir).glob("*.flac"):
            new_uuid = uuid.uuid4().hex
            new_filename = f"complete_{new_uuid}.flac"
            new_filepath = os.path.join(output_dir, "audio", new_filename)

            # Copy file to the audio directory with a new name
            shutil.copy2(flac_file, new_filepath)

            # Write metadata for complete file
            metadata = {
                "file_name": f"audio/{new_filename}",  # Reference to file in the audio subfolder
                "endpoint_bool": True
            }
            jsonl_file.write(json.dumps(metadata) + "\n")

        # Process incomplete FLAC files
        for flac_file in Path(incomplete_dir).glob("*.flac"):
            new_uuid = uuid.uuid4().hex
            new_filename = f"incomplete_{new_uuid}.flac"
            new_filepath = os.path.join(output_dir, "audio", new_filename)

            # Copy file to the audio directory with a new name
            shutil.copy2(flac_file, new_filepath)

            # Write metadata for incomplete file
            metadata = {
                "file_name": f"audio/{new_filename}",
                "endpoint_bool": False
            }
            jsonl_file.write(json.dumps(metadata) + "\n")

if __name__ == "__main__":
    # Set placeholders for the required paths
    COMPLETE_DIR = "raw/azure_1/complete"  # Replace with your complete FLAC files directory
    INCOMPLETE_DIR = "raw/azure_1/incomplete"  # Replace with your incomplete FLAC files directory
    TMP_OUTPUT_DIR = "tmp_datasets/azure_1_test3"  # Replace with the desired intermediary output dataset directory
    DATASET_SAVE_PATH = "datasets/azure_1_test3"  # Replace with the directory where you want to save the Hugging Face dataset

    # Create the audio dataset and generate metadata
    create_audio_dataset(COMPLETE_DIR, INCOMPLETE_DIR, TMP_OUTPUT_DIR)

    # Load the dataset using Hugging Face's audiofolder loader
    dataset = load_dataset("audiofolder", data_dir=TMP_OUTPUT_DIR)
    print(dataset)

    # Save the dataset to disk
    dataset.save_to_disk(DATASET_SAVE_PATH)
