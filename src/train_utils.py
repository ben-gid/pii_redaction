"""
Helper utilities for PII model training.

Contains alignment logic, customized Trainer class with class weighting,
and seqeval evaluation wrappers.

Learnings & Context:
1. SentencePiece whitespace alignment:
   Tokenizers based on SentencePiece (e.g. DeBERTa-v3) include the leading whitespace
   prefix in the first sub-word token of a word. Consequently, standard character mapping
   can cause offset mismatches where the token starts before the annotated mask span.
   To align properly, we map token offsets to their associated word index and compute the
   aggregate character span (union) of each word's constituent tokens before verifying
   overlap with the ground truth PII masks.
2. Loss calculation & class imbalance:
   A standard Token Classification Trainer treats all tokens equally. PII datasets are
   heavily dominated by the 'O' (non-PII) class, causing models to predict only 'O'.
   We mitigate this using `WeightedTokenClassificationTrainer` which overrides `compute_loss`
   to apply normalized inverse-frequency class weights during cross entropy calculation.
3. Ignore token conventions (-100):
   During evaluation, continuation sub-words assigned to -100 are ignored to stay
   consistent with the token-classification training configuration.
"""

import json
from typing import Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import evaluate
from transformers import Trainer, EvalPrediction, TrainerCallback


class WeightedTokenClassificationTrainer(Trainer):
    """
    Custom Trainer that overrides loss computation to support class weights.
    
    This helps balance highly skewed class distributions common in PII datasets where the
    'O' label (non-PII) significantly outnumbers positive PII target tokens.
    """
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(
        self, 
        model: nn.Module, 
        inputs: dict[str, Union[Tensor, Any]], 
        return_outputs: bool = False, 
        num_items_in_batch: Optional[Union[Tensor, int]] = None
    ) -> Union[Tensor, tuple[Tensor, Any]]:
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is None:
            if model.training:
                raise ValueError(
                    "Labels are required during training for WeightedTokenClassificationTrainer."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            return (loss, outputs) if return_outputs else loss
        
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            
        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
            
        return (loss, outputs) if return_outputs else loss


class DetailedProgressCallback(TrainerCallback):
    """Callback to log progress transparently during training steps and evaluations."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_local_process_zero:
            step = state.global_step
            total = state.max_steps
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            eval_loss = logs.get("eval_loss")

            if loss is not None and lr is not None:
                print(f"  Step {step}/{total} | loss: {loss:.4f} | lr: {lr:.2e}")
            elif eval_loss is not None:
                f1 = logs.get("eval_f1", "—")
                f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
                print(f"  Eval step {step} | eval_loss: {eval_loss:.4f} | f1: {f1_str}")


def make_compute_metrics_fn(id2label: dict[int, str]):
    """Returns a compute_metrics function aligned to the specified id2label mapping."""
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p: EvalPrediction) -> dict[str, float]:
        predictions, label_ids = p.predictions, p.label_ids
        predictions = np.argmax(predictions, axis=-1)
        
        # Filter out ignored indices (-100)
        true_preds = [
            [id2label[pred] for pred, tgt_id in zip(preds_row, labels_row) if tgt_id != -100]
            for preds_row, labels_row in zip(predictions, label_ids)
        ]
        
        true_labels = [
            [id2label[tgt_id] for tgt_id in labels_row if tgt_id != -100]
            for labels_row in label_ids
        ]
        
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
                if isinstance(scores, dict):
                    flat[f"{entity}_f1"] = scores["f1"]
                    flat[f"{entity}_support"] = scores["number"]
        
        return flat

    return compute_metrics


def tokenize_and_align_labels(batch, tokenizer, label2id):
    """
    Tokenize and align NER labels for a batch of examples.
    
    Designed to be used with dataset.map(..., batched=True).
    Matches token offsets to ground truth annotation offsets using union logic to
    correctly support SentencePiece tokenizers.
    """
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
        offsets = encoding["offset_mapping"][i]

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
                    max(prev_end, offset[1]),
                )

        # Map every word_id that overlaps a privacy mask to its entity label
        word_to_ent: dict[int, str] = {}
        for mask in masks:
            for word_id, (w_start, w_end) in word_span.items():
                if w_start < mask["end"] and w_end > mask["start"]:
                    word_to_ent[word_id] = mask["label"]

        # Produce aligned label ids and BIO tag strings
        labels: list[int] = []
        bio_tags: list[Optional[str]] = []
        prev_word_id = None
        prev_ent = None

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

    encoding["labels"] = all_labels
    encoding["ner_tags"] = all_bio_tags
    return encoding
