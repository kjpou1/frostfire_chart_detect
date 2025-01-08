import json
import os

from datasets import load_dataset
from PIL import Image

# Directory structure
RAW_DATA_DIR = "./data/raw/"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)


def save_image(image, save_dir, prefix, image_id):
    """
    Save a PIL image to the specified directory with a unique prefix and ID.
    Removes any incorrect ICC profile in the process.
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Remove ICC profile by re-creating the image without metadata
    image = image.convert("RGB")  # Convert to RGB to ensure compatibility
    image_path = os.path.join(save_dir, f"{prefix}_image_{image_id}.png")

    # Save the image without embedding any ICC profile
    image.save(image_path, format="PNG", icc_profile=None)
    return image_path


def download_and_save(dataset_info, save_dir):
    """
    Download a dataset from Hugging Face and save it to disk.
    Handles image data by saving images with the specified prefix.
    """
    dataset_name = dataset_info["dataset_name"]
    prefix = dataset_info["prefix"]

    print(f"Downloading {dataset_name}...")
    dataset = load_dataset(dataset_name)

    print(f"\n\nProcessing {dataset_name} with prefix '{prefix}'...")
    for split, data in dataset.items():
        split_dir = os.path.join(save_dir, split)
        image_dir = os.path.join(split_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        json_output_path = os.path.join(split_dir, f"{prefix}_{split}.jsonl")
        with open(json_output_path, "w", encoding="utf-8") as f:
            for idx, example in enumerate(data):
                try:
                    # Handle image data
                    if "image" in example and isinstance(example["image"], Image.Image):
                        image_path = save_image(
                            example["image"], image_dir, prefix, idx
                        )
                        example["image"] = (
                            image_path  # Replace image object with file path
                        )

                    # Serialize the rest of the example as JSON
                    json_line = json.dumps(example, ensure_ascii=False)
                    f.write(json_line + "\n")
                except Exception as e:
                    print(f"Skipping problematic record {idx} in {split}: {e}")

        print(f"Saved {split} data to {json_output_path}")


if __name__ == "__main__":
    # Define datasets to download
    datasets_to_download = [
        {"dataset_name": "StephanAkkerman/crypto-charts", "prefix": "crypto"},
        {"dataset_name": "StephanAkkerman/stock-charts", "prefix": "stock"},
        {"dataset_name": "StephanAkkerman/fintwit-images", "prefix": "fintwit"},
    ]

    for dataset in datasets_to_download:
        download_and_save(dataset, RAW_DATA_DIR)
