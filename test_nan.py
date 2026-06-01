import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
import json

# Load everything exactly as the notebook does
model_path = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

with open('data/label_info.json', 'r') as f:
    label_info = json.load(f)
label2id = label_info["label2id"]
id2label = {int(k): v for k, v in label_info["id2label"].items()}

model = AutoModelForTokenClassification.from_pretrained(
    model_path, num_labels=len(id2label), id2label=id2label, label2id=label2id
)

dataset = load_dataset("ben-gid/pii_redaction")
dataset = dataset["train"].select(range(16))

def get_ner_labels(batch):
    token_offsets = tokenizer(
        batch["source_text"],
        return_offsets_mapping=True,
        add_special_tokens=False,
    )["offset_mapping"]
    batch_ner_labels = []
    for i, row_masks in enumerate(batch["privacy_mask"]):
        row_ner_labels = []
        for offset in token_offsets[i]:
            if offset == (0, 0): 
                row_ner_labels.append("O")
                continue
            label = "O" 
            for privacy_mask in row_masks:
                if offset[1] > privacy_mask["start"] and offset[0] < privacy_mask["end"]:
                    label = privacy_mask["label"]
                    if offset[0] <= privacy_mask["start"]:
                        label = "B-" + label
                    else:
                        label = "I-" + label
                    break
            row_ner_labels.append(label)
        batch_ner_labels.append(row_ner_labels)
    return {"ner_tags": batch_ner_labels}

dataset = dataset.map(get_ner_labels, batched=True, batch_size=16)

def tokenize_and_align(example):
    tokenized = tokenizer(example["source_text"], truncation=True, add_special_tokens=False)
    labels = []
    word_ids = tokenized.word_ids()
    prev = None
    for i, tag in enumerate(example["ner_tags"]):
        if i >= len(word_ids): break
        w_id = word_ids[i]
        if w_id is None:
            labels.append(-100)
        elif w_id != prev:
            labels.append(label2id.get(tag, 0))
        else:
            labels.append(-100)
        prev = w_id
    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(tokenize_and_align, batched=False)
dataset = dataset.remove_columns(['source_text', 'privacy_mask', 'ner_tags', 'uid'])

collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = collator([dataset[i] for i in range(16)])

print("Input ids shape:", batch["input_ids"].shape)
print("Labels shape:", batch["labels"].shape)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

outputs = model(**batch)
loss = outputs.loss
print("Loss:", loss.item())

loss.backward()
optimizer.step()

has_nan = False
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in: {name}")
        has_nan = True
        break
if not has_nan:
    print("No NaNs found after 1 step!")
