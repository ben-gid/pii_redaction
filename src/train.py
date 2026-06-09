"""
PII Redaction Model Training Script.

Trains a token classification model targeting 27 PII entity types using DeBERTa-v3
checkpoints. Implements class-weighted loss and two-stage fine-tuning.

Key Lessons & Design Decisions:
1. SentencePiece offset mapping & subword labeling:
   We align annotations using the batched character-union approach implemented in
   train_utils.py. Continuing sub-word tokens within a word are marked with -100 to
   be ignored by CrossEntropyLoss, preventing fragmentation errors during tokenization.
2. Two-Stage Fine-Tuning:
   Fine-tuning DeBERTa-v3 token classifiers from scratch can cause instability.
   We implement a robust two-stage schedule:
   - Stage 1: Freeze the pretrained base backbone and train the custom classification head
     with a higher learning rate (1e-3) for a few epochs.
   - Stage 2: Unfreeze the entire backbone and fine-tune globally with a lower learning
     rate (2e-5) for a longer period.
3. Classifier Head Weight Initialization & NaNs:
   Initializing the classification head can sometimes lead to NaN weights, particularly
   with lower-precision float16 training or when layers are initialized with extreme weights.
   Ensuring proper module casting to float32 before training and using bf16 mixed-precision
   helps mitigate NaN instability.
4. Class Weights computation:
   Calculates class weights as 1.0 / count (normalized to sum to 1.0) based on training tags.
"""

from tqdm import tqdm
import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import wandb
from datasets import load_from_disk
import transformers
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)

from train_utils import (
    WeightedTokenClassificationTrainer,
    DetailedProgressCallback,
    make_compute_metrics_fn,
    tokenize_and_align_labels,
)

# Remove unneeded diagnostic warnings
from transformers.utils import logging
logging.set_verbosity_error()

# Seed for reproducibility
transformers.set_seed(42)
torch.manual_seed(42)

EFFECTIVE_BATCH_SIZE = 16

def parse_args():
    parser = argparse.ArgumentParser(description="Train DeBERTa models for PII Redaction.")
    parser.add_argument(
        "--model_variant",
        type=str,
        default="small",
        choices=["base", "small", "xsmall"],
        help="DeBERTa-v3 model variant to train."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the tokenized dataset. Defaults to 'data/cleaned_ai4privacy_300k_pii'."
    )
    parser.add_argument(
        "--label_info_path",
        type=str,
        default=None,
        help="Path to the label info JSON file. Defaults to 'data/label_info.json'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints. Defaults to 'models/<variant>-bf16-weighted-trainer'."
    )
    parser.add_argument(
        "--stage1_epochs",
        type=int,
        default=2,
        help="Number of epochs to train the classification head while the backbone is frozen."
    )
    parser.add_argument(
        "--stage1_lr",
        type=float,
        default=1e-3,
        help="Learning rate for Stage 1."
    )
    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=10,
        help="Number of epochs to fine-tune the entire model."
    )
    parser.add_argument(
        "--stage2_lr",
        type=float,
        default=2e-5,
        help="Learning rate for Stage 2."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per device batch size for training and evaluation."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="pii-redaction",
        help="Weights & Biases project name."
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging."
    )
    return parser.parse_args()


def log_pii_test_results(trainer, test_dataset, tokenizer, id2label, run=None):
    """
    Generate and log an HTML table showcasing model predictions compared to ground truth labels
    using colorized badges on the W&B dashboard.
    """
    import html as html_lib

    active_run = run or wandb.run
    if active_run is None:
        return

    # Normalize to int keys (JSON loads keys as strings, active config has ints)
    id2label = {int(k): v for k, v in id2label.items()}

    print("Generating prediction visualization for W&B...")
    pred_output = trainer.predict(test_dataset)
    pred_ids = np.argmax(pred_output.predictions, axis=-1)
    true_ids = pred_output.label_ids

    table = wandb.Table(columns=["id", "annotated_sequence", "f1", "precision", "recall", "tp", "fp", "fn"])

    # Log up to 100 sample predictions
    num_samples = min(len(test_dataset), 100)
    for i in range(num_samples):
        tokens, true_labels, pred_labels = [], [], []
        for token_id, true_id, pred_id in zip(test_dataset[i]["input_ids"], true_ids[i], pred_ids[i]):
            if true_id == -100:
                continue
            tokens.append(tokenizer.convert_ids_to_tokens(int(token_id)))
            true_labels.append(id2label[int(true_id)])
            pred_labels.append(id2label[int(pred_id)])

        tp = fp = fn = 0
        for true, pred in zip(true_labels, pred_labels):
            if true != "O" and pred == true:
                tp += 1
            elif true != "O" and pred != true:
                fn += 1
            elif true == "O" and pred != "O":
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        parts = []
        for token, true, pred in zip(tokens, true_labels, pred_labels):
            t = html_lib.escape(token)
            if true != "O" and pred == true:
                badge = f'<sup style="font-size:9px;padding:1px 4px;border-radius:3px;background:#C0DD97;color:#27500A">{html_lib.escape(true)}</sup>'
                parts.append(f'<span style="background:#C0DD97;color:#27500A;font-weight:500;padding:2px 4px;border-radius:3px;margin:1px 2px;display:inline-block">{t}{badge}</span>')
            elif true != "O" and pred != true:
                label = html_lib.escape(f"{true}→{pred}")
                badge = f'<sup style="font-size:9px;padding:1px 4px;border-radius:3px;background:#F7C1C1;color:#791F1F">{label}</sup>'
                parts.append(f'<span style="background:#F7C1C1;color:#791F1F;font-weight:500;padding:2px 4px;border-radius:3px;margin:1px 2px;display:inline-block">{t}{badge}</span>')
            elif true == "O" and pred != "O":
                badge = f'<sup style="font-size:9px;padding:1px 4px;border-radius:3px;background:#FAC775;color:#633806">{html_lib.escape(pred)}</sup>'
                parts.append(f'<span style="background:#FAC775;color:#633806;font-weight:500;padding:2px 4px;border-radius:3px;margin:1px 2px;display:inline-block">{t}{badge}</span>')
            else:
                parts.append(f'<span style="margin:1px 2px;display:inline-block">{t}</span>')

        seq_html = '<div style="font-family:monospace;font-size:12px;line-height:2">' + "".join(parts) + "</div>"
        table.add_data(i, wandb.Html(seq_html), round(f1, 4), round(precision, 4), round(recall, 4), tp, fp, fn)

    active_run.log({"pii_test_results": table})


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    
    # Establish default paths if not provided
    data_dir = project_root / "data"
    dataset_path = Path(args.dataset_path) if args.dataset_path else data_dir / "cleaned_ai4privacy_300k_pii"
    label_info_path = Path(args.label_info_path) if args.label_info_path else data_dir / "label_info.json"

    # Map variant labels to HF paths
    variant_mapping = {
        "base": "microsoft/deberta-v3-base",
        "small": "microsoft/deberta-v3-small",
        "xsmall": "microsoft/deberta-v3-xsmall",
    }
    model_path = variant_mapping[args.model_variant]
    
    # Determine outputs directory
    run_name = f"{args.model_variant}-bf16-weighted-trainer"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading label configuration from {label_info_path}...")
    with open(label_info_path) as f:
        label_info = json.load(f)
    label2id = label_info["label2id"]
    id2label = {int(k): v for k, v in label_info["id2label"].items()}

    print(f"Loading prepared dataset from {dataset_path}...")
    dataset = load_from_disk(str(dataset_path))

    print(f"Initializing tokenizer for model variant: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert tokenizer.is_fast, "A fast tokenizer implementation is required for offset mapping."

    print("Aligning ground truth annotations with token offset spans (batched mapping)...")
    dataset = dataset.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, label2id),
        batched=True,
        desc="Tokenizing and aligning dataset"
    )

    print(f"Initializing model {model_path}...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )

    # Convert model layers to float32 to prevent initial NaN instabilities
    print("Casting model to float32 to prevent initial NaN instabilities...")
    for module in model.modules():
        module.to(torch.float32)

    # Initialize Weights & Biases if enabled
    run = None
    if not args.no_wandb:
        print("Initializing Weights & Biases...")
        run = wandb.init(project=args.wandb_project, name=run_name)

    # Calculate class weights from training tags (inverse frequency)
    print("Calculating class weights from training labels (inverse frequency)...")
    labels_counter = Counter()
    for labels in tqdm(dataset["train"]["labels"], desc="Counting labels"):
        labels_counter.update(labels)
    
    labels_count_tensor = torch.tensor(
        [labels_counter.get(i, 0) for i in sorted(id2label.keys())],
        dtype=torch.float
    )
    weights = 1.0 / labels_count_tensor.clamp(min=1.0)
    weights = weights / weights.sum()

    print("Initializing data collator...")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    print("Initializing compute metrics function...")
    compute_metrics_fn = make_compute_metrics_fn(id2label)

    # Calculate steps for scheduler warmup
    print("Calculating scheduler warmup steps...")
    num_train_samples = len(dataset["train"])
    total_steps = (num_train_samples // args.batch_size) * args.stage2_epochs
    step2_step2_step2_warmup_steps = int(0.1 * total_steps)

    # Setup Stage 1: Freeze backbone and train classifier head
    print("--- Stage 1: Fine-tuning classification head (backbone frozen) ---")
    for param in model.base_model.parameters():
        param.requires_grad = False

    stage1_args = TrainingArguments(
        output_dir=str(output_dir),
        report_to="none" if args.no_wandb else "wandb",
        learning_rate=args.stage1_lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.stage1_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        warmup_steps=int(0.1 * ((num_train_samples // args.batch_size) * args.stage1_epochs)),
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        disable_tqdm=False
    )

    trainer = WeightedTokenClassificationTrainer(
        model=model,
        args=stage1_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[DetailedProgressCallback()],
        class_weights=weights
    )
    trainer.train()

    # Setup Stage 2: Unfreeze backbone and fine-tune globally
    print("--- Stage 2: Full Fine-tuning (backbone unfrozen) ---")
    for param in model.base_model.parameters():
        param.requires_grad = True

    stage2_args = TrainingArguments(
        output_dir=str(output_dir),
        report_to="none" if args.no_wandb else "wandb",
        learning_rate=args.stage2_lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, EFFECTIVE_BATCH_SIZE // args.batch_size),
        num_train_epochs=args.stage2_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        warmup_steps=step2_step2_step2_warmup_steps,
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        disable_tqdm=False
    )

    trainer = WeightedTokenClassificationTrainer(
        model=model,
        args=stage2_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[DetailedProgressCallback()],
        class_weights=weights
    )
    trainer.train()

    # Verify best checkpoint is loaded, compute metrics, and visualize results on W&B
    if not args.no_wandb:
        log_pii_test_results(trainer, dataset["test"], tokenizer, id2label, run=run)
        
        # Log worst F1 performance categories to stdout/wandb logs
        pred_output = trainer.predict(test_dataset=dataset["test"])
        f1s = [(metric, val) for metric, val in pred_output.metrics.items() if metric.endswith("f1")]
        sorted_f1s = sorted(f1s, key=lambda x: x[1])
        print("Worst performing entity groups by F1 score:")
        for metric, val in sorted_f1s[:7]:
            print(f"  {metric:<25} = {val:.4f}")

        wandb.finish()

    print("Model training complete!")


if __name__ == "__main__":
    main()
