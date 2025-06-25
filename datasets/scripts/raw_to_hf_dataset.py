#!/usr/bin/env python3
"""
Audio Dataset Creation Script

Expected Input Directory Structure:
{input_dir}/
├── {language1}/
│   ├── complete-midfiller/
│   │   ├── {uuid1}.flac
│   │   └── {uuid2}.flac
│   ├── incomplete-endfiller/
│   │   ├── {uuid3}.flac
│   │   └── {uuid4}.flac
│   └── complete-midfiller-endfiller/
│       └── {uuid5}.flac
├── {language2}/
│   ├── incomplete-nofiller/
│   │   └── {uuid6}.flac
│   └── complete-midfiller/
│       └── {uuid7}.flac
└── ...

Usage:
    python script.py <base_name> <input_dir> <output_dir> <tmp_dir>

Example:
    python script.py dataset_name raw_audio datasets tmp_processing

Output:
    Creates a single dataset named {output_dir}/{base_name} containing all languages.
    Each item has language, midfiller, endfiller, endpoint_bool, and id labels.
"""

import os
import sys
import shutil
import json
import uuid
from pathlib import Path
from datasets import load_dataset


def parse_directory_suffix(dir_name: str) -> tuple[bool, bool, bool]:
    """
    Parse directory name to extract endpoint_bool, midfiller, and endfiller flags.

    Args:
        dir_name (str): Directory name (e.g., "complete-midfiller", "incomplete-endfiller")

    Returns:
        tuple[bool, bool, bool]: (endpoint_bool, midfiller, endfiller)

    Raises:
        ValueError: If directory name doesn't match expected format
    """
    valid_suffixes = ['-midfiller', '-endfiller', '-midfiller-endfiller', '-nofiller']

    # Check if it starts with complete or incomplete
    if dir_name.startswith('complete-'):
        endpoint_bool = True
        suffix = dir_name[9:]  # Remove 'complete' prefix
    elif dir_name.startswith('incomplete-'):
        endpoint_bool = False
        suffix = dir_name[11:]  # Remove 'incomplete' prefix
    else:
        raise ValueError(f"Directory '{dir_name}' must start with 'complete-' or 'incomplete-'")

    # Check if suffix is valid
    if suffix not in ['midfiller', 'endfiller', 'midfiller-endfiller', 'nofiller']:
        raise ValueError(f"Directory '{dir_name}' has invalid suffix {suffix}. Must be one of: {valid_suffixes}")

    # Determine midfiller and endfiller flags
    midfiller = 'midfiller' in suffix
    endfiller = 'endfiller' in suffix

    return endpoint_bool, midfiller, endfiller


def is_valid_uuid(filename: str) -> bool:
    """Check if filename (without extension) is a valid UUID."""
    name_without_ext = Path(filename).stem
    try:
        uuid.UUID(name_without_ext)
        return True
    except ValueError:
        return False


def process_audio_files(audio_dir: Path, language: str, endpoint_bool: bool, midfiller: bool, endfiller: bool,
                        output_audio_dir: Path, jsonl_file):
    """
    Process audio files from a directory and write metadata.

    Args:
        audio_dir (Path): Directory containing FLAC files
        language (str): Language code
        endpoint_bool (bool): True if complete, False if incomplete
        midfiller (bool): True if midfiller in suffix
        endfiller (bool): True if endfiller in suffix
        output_audio_dir (Path): Output directory for audio files
        jsonl_file: Open file handle for writing JSONL metadata
    """
    for flac_file in audio_dir.glob("*.flac"):
        # Validate that filename is a UUID
        if not is_valid_uuid(flac_file.name):
            raise ValueError(f"Filename '{flac_file.name}' is not a valid UUID")

        file_uuid = Path(flac_file.name).stem
        new_filename = f"{language}_{audio_dir.name}_{file_uuid}.flac"
        new_filepath = output_audio_dir / new_filename

        # Copy file to the audio directory with a new name
        shutil.copy2(flac_file, new_filepath)

        # Write metadata
        metadata = {
            "file_name": f"audio/{new_filename}",
            "id": file_uuid,
            "language": language,
            "endpoint_bool": endpoint_bool,
            "midfiller": midfiller,
            "endfiller": endfiller
        }
        jsonl_file.write(json.dumps(metadata) + "\n")


def create_audio_dataset(input_dir: Path, tmp_output_dir: Path):
    """
    Create a single dataset from all language directories.

    Args:
        input_dir (Path): Path to the input directory containing language subdirectories
        tmp_output_dir (Path): Path to the temporary output directory for the dataset
    """
    # Create output directories
    audio_output_dir = tmp_output_dir / "audio"
    os.makedirs(audio_output_dir, exist_ok=True)

    # Open JSONL file for writing metadata
    metadata_path = tmp_output_dir / "metadata.jsonl"
    with open(metadata_path, "w") as jsonl_file:
        # Process each language directory
        for language_dir in input_dir.iterdir():
            if not language_dir.is_dir():
                continue

            language_name = language_dir.name
            print(f"Processing language: {language_name}")

            # Process each subdirectory in the language directory
            for subdir in language_dir.iterdir():
                if not subdir.is_dir():
                    continue

                try:
                    endpoint_bool, midfiller, endfiller = parse_directory_suffix(subdir.name)
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

                print(f"  Processing subdirectory: {subdir.name}")
                process_audio_files(subdir, language_name, endpoint_bool, midfiller, endfiller, audio_output_dir,
                                    jsonl_file)


def main():
    if len(sys.argv) != 5:
        print("Usage: python raw_to_hf_dataset.py <base_name> <input_dir> <output_dir> <tmp_dir>")
        sys.exit(1)

    base_name = sys.argv[1]
    input_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    tmp_dir = Path(sys.argv[4])

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    # Create output and tmp directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Create temporary directory for the dataset
    tmp_dataset_dir = tmp_dir / base_name

    # Create the audio dataset and generate metadata
    print("Creating unified dataset...")
    create_audio_dataset(input_dir, tmp_dataset_dir)

    # Load the dataset using Hugging Face's audiofolder loader
    dataset = load_dataset("audiofolder", data_dir=str(tmp_dataset_dir))
    print(f"Dataset: {dataset}")

    # Save the dataset to final output directory
    final_dataset_path = output_dir / base_name
    dataset.save_to_disk(str(final_dataset_path))
    print(f"Saved dataset to: {final_dataset_path}")

    # Clean up temporary directory
    shutil.rmtree(tmp_dataset_dir)


if __name__ == "__main__":
    main()