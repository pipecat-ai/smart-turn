#!/usr/bin/env python
import argparse
import os
from datasets import load_from_disk, Dataset, DatasetDict
from huggingface_hub import login


def print_dataset_info(dataset):
    """
    Print information about a dataset.

    Args:
        dataset: A Dataset or DatasetDict object
    """
    if isinstance(dataset, DatasetDict):
        print(f"Dataset type: DatasetDict with {len(dataset)} split(s)")
        print(f"Splits: {', '.join(dataset.keys())}")

        # Print information for each split
        for split_name, split_dataset in dataset.items():
            print(f"\n--- Split: {split_name} ---")
            print(f"Number of examples: {len(split_dataset)}")

            # Print feature information
            print("Features:")
            for feature_name, feature in split_dataset.features.items():
                print(f"  - {feature_name}: {feature}")

            # Print first example
            if len(split_dataset) > 0:
                print("\nFirst example:")
                first_example = split_dataset[0]
                for key, value in first_example.items():
                    # Truncate long values for readability
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"  {key}: {value}")
    elif isinstance(dataset, Dataset):
        print("Dataset type: Dataset")
        print(f"Number of examples: {len(dataset)}")

        # Print feature information
        print("Features:")
        for feature_name, feature in dataset.features.items():
            print(f"  - {feature_name}: {feature}")

        # Print first example
        if len(dataset) > 0:
            print("\nFirst example:")
            first_example = dataset[0]
            for key, value in first_example.items():
                # Truncate long values for readability
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")


def upload_dataset_to_hub(dataset_path, hub_dataset_id=None, token=None, private=False):
    """
    Upload a dataset to the Hugging Face Hub.

    Args:
        dataset_path (str): Path to the dataset directory
        hub_dataset_id (str, optional): ID for the dataset on the Hub. If None, uses basename of dataset_path.
        token (str, optional): Hugging Face API token. If None, uses token from huggingface-cli login.
        private (bool, optional): Whether to make the uploaded dataset private. Default is False.

    Returns:
        bool: True if upload was successful, False otherwise
    """
    try:
        # Load the dataset
        dataset = load_from_disk(dataset_path)
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        print(f"Successfully loaded dataset from {dataset_path}")

        # Login to Hugging Face Hub if token is provided
        if token:
            login(token=token)
            print("Logged in to Hugging Face Hub with provided token")
        else:
            print("Using existing Hugging Face Hub credentials (if available)")

        # Set the repository ID for the dataset
        if hub_dataset_id is None:
            hub_dataset_id = dataset_name

        # Upload the dataset to the Hub
        print(f"Uploading dataset to Hugging Face Hub as '{hub_dataset_id}'...")
        dataset.push_to_hub(hub_dataset_id, private=private)
        print(
            f"Successfully uploaded dataset to Hugging Face Hub: https://huggingface.co/datasets/{hub_dataset_id}"
        )
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Load, print information about, and optionally upload a dataset."
    )
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument(
        "--upload",
        "-u",
        dest="upload",
        action="store_true",
        help="Upload the dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub-id",
        dest="hub_dataset_id",
        help="ID for the dataset on the Hub (defaults to dataset directory name)",
    )
    parser.add_argument("--token", dest="token", help="Hugging Face API token")
    parser.add_argument(
        "--private", dest="private", action="store_true", help="Make the uploaded dataset private"
    )
    args = parser.parse_args()

    try:
        # Load dataset from the exact path provided
        dataset = load_from_disk(args.dataset_path)
        print(f"\n{'=' * 50}")
        print(f"Dataset: {os.path.basename(os.path.normpath(args.dataset_path))}")
        print(f"{'=' * 50}")

        # Print dataset information
        print_dataset_info(dataset)

        # Upload dataset if requested
        if args.upload:
            upload_dataset_to_hub(
                args.dataset_path,
                hub_dataset_id=args.hub_dataset_id,
                token=args.token,
                private=args.private,
            )
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
