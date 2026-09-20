"""
PII Redaction Dataset Preparation Script.

This script fetches, cleans, preprocesses, and splits a source dataset (ai4privacy or
nemotron, see `--dataset`) to prepare it for training token classification models for
PII redaction. Both flows save the same two columns, `source_text` and `privacy_mask`,
so training does not need to know which dataset it is looking at.

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
4. Nemotron span parsing:
   nvidia/Nemotron-PII stores spans as a repr string with single quotes (so `json.loads`
   fails, `ast.literal_eval` is needed), and ~12.5k spans hold the matched text as an int
   (age, cvv, pin), which Arrow cannot reconcile into a single struct type. Both are
   handled while converting spans into the `privacy_mask` column.
"""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm

from . import DATASETS

SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for PII Redaction model training.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ai4privacy",
        choices=list(DATASETS),
        help="Which source dataset to prepare."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset identifier. Defaults to the identifier for --dataset."
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


def build_label_maps(entity_classes: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build BIO label maps: O=0, then B-/I- pairs in the order given."""
    label2id = {"O": 0}
    for i, entity in enumerate(entity_classes):
        b_id = 1 + i * 2
        label2id[f"B-{entity}"] = b_id
        label2id[f"I-{entity}"] = b_id + 1

    return label2id, {i: label for label, i in label2id.items()}


def prepare_nemotron(dataset_name: str, dataset_path: Path, label_info_path: Path):
    """Prepare nvidia/Nemotron-PII into the same shape as the ai4privacy dataset."""
    print(f"Loading dataset: {dataset_name}...")
    ds = load_dataset(dataset_name)
    total_len = sum(len(split) for split in ds.values())
    print(f"Dataset length: {total_len:,}")

    def to_privacy_mask(batch):
        return {"privacy_mask": [
            # some spans store the matched text unquoted (e.g. 44 for age/cvv/pin), so
            # literal_eval yields an int there and arrow can't infer one struct type.
            [span | {"text": str(span["text"])} for span in ast.literal_eval(spans)]
            for spans in batch["spans"]
        ]}

    print("Parsing spans into privacy masks...")
    ds = ds.map(to_privacy_mask, batched=True, batch_size=1_000, desc="Parsing spans")

    # The source is a 50/50 train/test split; move 40% of test into train, then halve
    # the remainder into validation and test -> 70/15/15.
    print("Re-splitting dataset 70/15/15...")
    moved = ds["test"].train_test_split(test_size=0.6, seed=SEED)
    val_test = moved["test"].train_test_split(test_size=0.5, seed=SEED)

    ds = DatasetDict({
        "train": concatenate_datasets([ds["train"], moved["train"]]),
        "validation": val_test["train"],
        "test": val_test["test"],
    })
    assert sum(len(split) for split in ds.values()) == total_len

    ds = ds.rename_column("text", "source_text")
    columns_to_keep = ["source_text", "privacy_mask"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in columns_to_keep])

    print("Analyzing entity class frequencies...")
    entity_counter, train_entity_counter = Counter(), Counter()
    for split_name, split in ds.items():
        counter = Counter()
        for masks in tqdm(split["privacy_mask"], desc=f"Counting {split_name} entities"):
            counter.update([mask["label"] for mask in masks])
        entity_counter.update(counter)
        if split_name == "train":
            train_entity_counter = counter

    label2id, id2label = build_label_maps(sorted(entity_counter))
    print(f"Found {len(entity_counter)} entity classes -> {len(label2id)} BIO labels")

    print("Saving processed dataset to disk...")
    ds.save_to_disk(str(dataset_path))

    label_info = {
        "entities": sorted(entity_counter),
        "label2id": label2id,
        "id2label": id2label,
        "all_entities_counted": dict(entity_counter),
        "train_counted_entities": dict(train_entity_counter),
    }

    print(f"Writing label configuration to {label_info_path}...")
    with open(label_info_path, "w") as f:
        json.dump(label_info, f, indent=2)


def main():
    args = parse_args()

    # Determine default paths
    # Assuming this script is run from src/ or project root
    project_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output_dir) if args.output_dir else project_root / "data"
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = DATASETS[args.dataset]
    dataset_name = args.dataset_name or cfg.hf_name
    dataset_path = output_path / cfg.dir_name
    label_info_path = output_path / cfg.label_info

    if args.dataset == "nemotron":
        prepare_nemotron(dataset_name, dataset_path, label_info_path)
        print("Dataset preparation complete!")
        return

    print(f"Loading dataset: {dataset_name}...")
    ds = load_dataset(dataset_name)

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
    label2id, id2label = build_label_maps(entity_classes)

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
