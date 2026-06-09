import json
import os
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline,\
    EvalPrediction
from huggingface_hub import ModelCardData, ModelCard, EvalResult
import evaluate

    
with open(Path(os.getcwd()).parent/"data"/"label_info.json") as f:
    label_info: dict = json.load(f)
id2label = label_info["id2label"]
label2id = label_info["label2id"]


ENTITY_LABELS = [
    "BOD", "BUILDING", "CITY", "COUNTRY", "DATE", "DRIVERLICENSE",
    "EMAIL", "GEOCOORD", "GIVENNAME1", "GIVENNAME2", "IDCARD", "IP",
    "LASTNAME1", "LASTNAME2", "LASTNAME3", "PASS", "PASSPORT",
    "POSTCODE", "SECADDRESS", "SEX", "SOCIALNUMBER", "STATE",
    "STREET", "TEL", "TIME", "TITLE", "USERNAME",
]

def benchmark(repo_id, text, n_warmup=5, n_runs=50):
    pipe = pipeline("token-classification", model=repo_id, device=torch.device("cuda"))
    
    # Warmup — GPU needs a few runs to reach steady state
    for _ in range(n_warmup):
        pipe(text)
    
    # CPU timing (works for both CPU and GPU if you sync first)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        pipe(text)
        torch.cuda.synchronize()  # wait for GPU op to finish
    elapsed = time.perf_counter() - start
    
    return (elapsed / n_runs) * 1000  # ms per sample

def get_best_checkpoint(model_dir: Path) -> tuple[Path, dict, dict]:
    """Return (best_checkpoint_path, trainer_state, best_log_entry)."""
    checkpoints = [
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint")
    ]
    latest = max(checkpoints, key=lambda d: int(d.name.split("-")[1]))

    with open(latest / "trainer_state.json") as f:
        state = json.load(f)

    best_ckpt = Path(state["best_model_checkpoint"])
    best_step = state["best_global_step"]

    best_log = {}
    for log in state["log_history"]:
        if log.get("step") == best_step and "eval_f1" in log:
            best_log = log
            break

    return best_ckpt, state, best_log

def model_stats(repo_id):
    model = AutoModelForTokenClassification.from_pretrained(repo_id).to("cuda")
    
    # Params
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # VRAM after loading
    torch.cuda.synchronize()
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    
    print(f"{repo_id}: {params:.0f}M params, {vram_mb:.0f}MB VRAM")
    del model
    torch.cuda.empty_cache()
    

def load_training_args(checkpoint: Path):
    """Load TrainingArguments from the binary saved alongside each checkpoint."""
    return torch.load(checkpoint / "training_args.bin", weights_only=False)


def make_entity_table(log: dict, worst:Optional[int]=None) -> str:
    """Build a markdown table of per-entity F1 scores from the best eval log. 
    if worst, include the worst ones"""
    rows = []
    for label in ENTITY_LABELS:
        f1  = log.get(f"eval_{label}_f1", None)
        sup = log.get(f"eval_{label}_support", None)
        if f1 is not None:
            rows.append((label, f1, sup))
        else:
            print(f"eval_{label}_f1 missing")
    
    if worst is not None:
        rows = sorted(rows, key=lambda x: x[1])[:worst]

    header = "| Entity | F1 | Support |\n|--------|------|---------|\n"
    body   = "".join(f"| {r} | {f:.4f} | {s} |\n" for r, f, s in rows)
    return header + body

def get_first_checkpoint_args(model_dir: Path):
    """Load TrainingArguments from the very first checkpoint (Stage 1)."""
    checkpoints = [
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint")
    ]
    first = min(checkpoints, key=lambda d: int(d.name.split("-")[1]))
    return torch.load(first / "training_args.bin", weights_only=False)

def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    predictions, label_ids = p.predictions, p.label_ids
    predictions = np.argmax(predictions, axis=-1)
    
    true_preds = [
        [id2label[str(pred)] for pred, tgt_id in zip(preds_row, labels_row) if tgt_id != -100]
        for preds_row, labels_row in zip(predictions, label_ids)
    ]
    
    true_labels =[
        [id2label[str(tgt_id)] for tgt_id in labels_row if tgt_id != -100]
        for labels_row in label_ids
    ]
    
    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    
    flat = {}
    if results is not None:
        flat.update({
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        })
        
        for entity, scores in results.items():
            if isinstance(scores, dict): # filter scores for individual labels
                flat[f"{entity}_f1"] = scores["f1"]
                flat[f"{entity}_support"] = scores["number"]
    
    return flat

def tokenize_and_align_labels(batch):
    """Tokenize and align NER labels for a batch of examples.

    Designed for use with ``dataset.map(..., batched=True)``.
    Since the DeBERTa tokenizer marks the space before a word as part of
    that word (SentencePiece-style), label alignment is done via character
    offset mapping rather than naive word-index alignment.

    Args:
        batch: A dict of lists as produced by HuggingFace ``datasets`` in
               batched mode.  Expected keys:
               - ``"source_text"``   – list[str]
               - ``"privacy_mask"``  – list[list[dict]]  (each dict has
                 ``"start"``, ``"end"``, and ``"label"`` keys)

    Returns:
        The encoding dict extended with two new list-of-lists fields:
        - ``"labels"``   – integer label ids (-100 for ignored tokens)
        - ``"ner_tags"`` – human-readable BIO tag strings (None for ignored)
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    encoding = tokenizer(
        batch["source_text"],
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        is_split_into_words=False,
    )

    all_labels: list[list[int]] = []
    all_bio_tags: list[list[Optional[str]]] = []

    for i, masks in enumerate(batch["privacy_mask"]):
        word_ids = encoding.word_ids(batch_index=i)
        offsets  = encoding["offset_mapping"][i]

        # Build per-word character span (union across sub-tokens)
        word_span: dict[int, tuple[int, int]] = {}
        for offset, word_id in zip(offsets, word_ids):
            if word_id is None:
                continue
            if word_id not in word_span:
                word_span[word_id] = (offset[0], offset[1])
            else:
                prev_start, prev_end = word_span[word_id]
                word_span[word_id] = (
                    min(prev_start, offset[0]),
                    max(prev_end,   offset[1]),
                )

        # Map every word_id that overlaps a privacy mask to its entity label
        word_to_ent: dict[int, str] = {}
        for mask in masks:
            for word_id, (w_start, w_end) in word_span.items():
                if w_start < mask["end"] and w_end > mask["start"]:
                    word_to_ent[word_id] = mask["label"]

        # Produce aligned label ids and BIO tag strings
        labels:   list[int]           = []
        bio_tags: list[Optional[str]] = []
        prev_word_id = None
        prev_ent     = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
                bio_tags.append(None)

            elif word_id not in word_to_ent:
                labels.append(label2id["O"])
                bio_tags.append("O")
                prev_ent = None

            else:
                entity = word_to_ent[word_id]

                if word_id != prev_word_id:
                    # First sub-token of this word
                    tag = f"I-{entity}" if prev_ent == entity else f"B-{entity}"
                    labels.append(label2id[tag])
                else:
                    # Continuation sub-token – mask from loss
                    tag = f"I-{entity}"
                    labels.append(-100)

                bio_tags.append(tag)
                prev_ent = entity

            prev_word_id = word_id

        all_labels.append(labels)
        all_bio_tags.append(bio_tags)

    encoding["labels"]   = all_labels
    encoding["ner_tags"] = all_bio_tags
    return encoding