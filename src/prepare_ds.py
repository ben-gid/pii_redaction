"""
PII Redaction Dataset Preparation Script.

This script fetches, cleans, preprocesses, and splits the ai4privacy/pii-masking-300k
dataset to prepare it for training token classification models for PII redaction.

Key Lessons & Design Decisions:
1. Multilingual vs. Monolingual training:
   The dataset is multilingual, but for our target task, training on the English subset
   reduces noise and training time while maintaining high quality for English pipelines.
2. Handling of rare classes (CARDISSUER):
   Rare classes with extremely low frequency (e.g., CARDISSUER with < 50 training samples)
   cause unreliable evaluation metrics (F1 scores) and are practically impossible for the
   model to generalize on without extensive upsampling. Therefore, they are cleaned and
   removed from the dataset.
3. Custom Validation / Test Splits:
   The original dataset does not contain a test split. To perform unbiased final
   evaluation and benchmarking across model variants, we split the original validation set
   50/50 into distinct validation and test sets.
"""

from typing import Any
import argparse
import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for PII Redaction model training.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ai4privacy/pii-masking-300k",
        help="Hugging Face dataset identifier."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the cleaned dataset and label info. Defaults to project 'data' directory."
    )
    parser.add_argument(
        "--rare_threshold",
        type=int,
        default=50,
        help="Minimum number of entity occurrences required in training split. Entities below this threshold are dropped."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine default paths
    # Assuming this script is run from src/ or project root
    project_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output_dir) if args.output_dir else project_root / "data"
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_path = output_path / "cleaned_ai4privacy_300k_pii"
    label_info_path = output_path / "label_info.json"

    print(f"Loading dataset: {args.dataset_name}...")
    ds = load_dataset(args.dataset_name)

    # Filter for English examples
    print("Filtering dataset for English examples...")
    ds = ds.filter(lambda x: x["language"] == "English")
    print(f"Dataset english subset length: {sum(len(ds[split]) for split in ds):,}")

    # Count entities in training data to find rare labels
    print("Analyzing entity class frequencies in training split...")
    train_entity_counter = Counter()
    for masks in tqdm(ds["train"]["privacy_mask"], desc="Counting train entities"):
        train_entity_counter.update([mask["label"] for mask in masks])

    # Find rare classes below threshold
    keys_to_remove = [key for key, val in train_entity_counter.items() if val < args.rare_threshold]
    for key in keys_to_remove:
        print(f"Excluding rare tag: {key} (frequency = {train_entity_counter[key]})")
        del train_entity_counter[key]

    # Generate labels maps
    # TODO: add sorted back to entity_classes when i can retrain model with new ids
    # it's currently sorted by first appearance
    entity_classes = list(train_entity_counter.keys())
    label2id = {"O": 0}
    for i, entity in enumerate(entity_classes):
        b_id = 1 + i * 2
        i_id = b_id + 1
        label2id[f"B-{entity}"] = b_id
        label2id[f"I-{entity}"] = i_id

    id2label = {i: label for label, i in label2id.items()}

    # Clean privacy masks of deleted tags
    print(f"Removing references to excluded tags {keys_to_remove} from privacy masks...")
    
    def remove_tag_from_pm(example):
        pm = example["privacy_mask"]
        filtered = [d for d in pm if d["label"] not in keys_to_remove]
        return {"privacy_mask": filtered}

    cleaned_ds = ds.map(remove_tag_from_pm)

    # Keep only required columns
    columns_to_keep = ["source_text", "privacy_mask"]
    columns_to_remove = [col for col in cleaned_ds["train"].column_names if col not in columns_to_keep]
    cleaned_ds = cleaned_ds.remove_columns(columns_to_remove)

    # Split validation 50/50 to get a stable test split
    print("Splitting validation set 50/50 into validation and test sets...")
    val_test = cleaned_ds["validation"].train_test_split(test_size=0.5, seed=SEED)
    
    updated_and_cleaned_dataset = DatasetDict({
        "train": cleaned_ds["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    # Validate that rare classes were successfully cleaned
    final_entity_counter = Counter()
    for split in tqdm(updated_and_cleaned_dataset, desc="Validating final dataset"):
        for masks in updated_and_cleaned_dataset[split]["privacy_mask"]:
            final_entity_counter.update([mask["label"] for mask in masks])

    for key in keys_to_remove:
        assert final_entity_counter[key] == 0, f"Exclusion failed: {key} still present in dataset"

    print("Saving processed dataset to disk...")
    updated_and_cleaned_dataset.save_to_disk(str(dataset_path))

    label_info = {
        "entities": entity_classes,
        "label2id": label2id,
        "id2label": id2label,
        "all_entities_counted": dict(final_entity_counter),
        "train_counted_entities": dict(train_entity_counter),
    }

    print(f"Writing label configuration to {label_info_path}...")
    with open(label_info_path, "w") as f:
        json.dump(label_info, f, indent=2)

    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
