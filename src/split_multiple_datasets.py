import json
import os
import random


def split_dataset(
    jsonl_path,
    output_dir,
    prefix,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Split a JSONL dataset into train, validation, and test sets for a specific prefix, using absolute paths.

    Args:
        jsonl_path (str): Path to the input JSONL file.
        output_dir (str): Base directory to save the processed datasets.
        prefix (str): Prefix for the dataset (e.g., 'crypto', 'stock').
        train_ratio (float): Proportion of the dataset for training.
        val_ratio (float): Proportion for validation.
        test_ratio (float): Proportion for testing.
        seed (int): Random seed for reproducibility.
    """
    # Ensure output directories exist
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Read all records from the JSONL file
    with open(jsonl_path, "r") as f:
        records = [json.loads(line) for line in f]

    # Shuffle the records
    random.seed(seed)
    random.shuffle(records)

    # Compute split indices
    total = len(records)
    train_idx = int(total * train_ratio)
    val_idx = train_idx + int(total * val_ratio)

    # Split the data
    train_records = records[:train_idx]
    val_records = records[train_idx:val_idx]
    test_records = records[val_idx:]

    # Get absolute base path for images
    base_path = os.path.abspath("./data/raw/train/images/")  # Adjust path if needed

    # Update the image paths to absolute paths
    for record in train_records:
        record["image"] = os.path.join(base_path, os.path.basename(record["image"]))
    for record in val_records:
        record["image"] = os.path.join(base_path, os.path.basename(record["image"]))
    for record in test_records:
        record["image"] = os.path.join(base_path, os.path.basename(record["image"]))

    # Save each split to separate JSONL files
    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    for split_name, split_records in splits.items():
        output_file = os.path.join(output_dir, split_name, f"{prefix}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(split_records)} records to {output_file}")


def process_all_datasets(datasets, raw_dir, processed_dir):
    """
    Process all datasets by splitting each into train, val, and test sets.

    Args:
        datasets (list): List of datasets with 'dataset_name' and 'prefix'.
        raw_dir (str): Directory where raw JSONL files are stored.
        processed_dir (str): Directory where processed files will be saved.
    """
    for dataset in datasets:
        dataset_name = dataset["dataset_name"]
        prefix = dataset["prefix"]

        print(f"Processing dataset: {dataset_name} with prefix: {prefix}...")
        raw_file_path = os.path.join(raw_dir, "train", f"{prefix}_train.jsonl")

        # Ensure raw file exists
        if not os.path.exists(raw_file_path):
            print(f"Raw file not found: {raw_file_path}. Skipping...")
            continue

        # Split and save processed files
        split_dataset(jsonl_path=raw_file_path, output_dir=processed_dir, prefix=prefix)


# Example usage
datasets_to_process = [
    {"dataset_name": "StephanAkkerman/crypto-charts", "prefix": "crypto"},
    {"dataset_name": "StephanAkkerman/stock-charts", "prefix": "stock"},
    {"dataset_name": "StephanAkkerman/fintwit-images", "prefix": "fintwit"},
]

raw_dir = "./data/raw/"
processed_dir = "./data/processed/"

process_all_datasets(datasets_to_process, raw_dir, processed_dir)
