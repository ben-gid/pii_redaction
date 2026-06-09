---
base_model: microsoft/deberta-v3-base
datasets:
- ai4privacy/pii-masking-300k
language: en
library_name: transformers
license: other
metrics:
- f1
- precision
- recall
pipeline_tag: token-classification
tags:
- token-classification
- pii
- ner
- deberta
- privacy
- named-entity-recognition
model-index:
- name: DeBERTa-v3-Base PII Redaction
  results:
  - task:
      type: token_classification
    dataset:
      name: ai4privacy/pii-masking-300k
      type: ai4privacy/pii-masking-300k
      split: validation
    metrics:
    - type: f1
      value: 0.9556626506024097
    - type: precision
      value: 0.9506611801366306
    - type: recall
      value: 0.9607170253139005
---

# PII Redaction Model

Fine-tuned [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) for Named Entity Recognition targeting 27 PII entity types. Trained on the English subset of [ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) with a class-weighted `CrossEntropyLoss`. Achieves **0.9557 macro-F1** on the validation set.

**Recommended when** accuracy is the priority and compute is not a 
constraint. Best overall performance (macro F1: 0.9557), particularly 
on rare entity types. Well-suited for offline batch processing, 
compliance pipelines, or any server-side deployment where an extra 
~5ms of latency is acceptable. Inference: ~11.7ms on RTX 5070.

## Usage

```python
from transformers import pipeline

pipe = pipeline(
    "ner",
    model=bengid/pii-redaction-deberta-base,
    aggregation_strategy="first",
    device=0  # omit for CPU
)

text = "She lives at 742 Evergreen Terrace, Springfield, IL 62704."
entities = pipe(text)
print(entities)
```
```output
[{'entity_group': 'BUILDING', 'score': np.float32(0.96388406), 'word': '742', 'start': 12, 'end': 16}, {'entity_group': 'STREET', 'score': np.float32(0.8134035), 'word': 'EvergreenTerrace,', 'start': 16, 'end': 35}, {'entity_group': 'CITY', 'score': np.float32(0.98855907), 'word': 'Springfield,', 'start': 35, 'end': 48}, {'entity_group': 'STATE', 'score': np.float32(0.9980453), 'word': 'IL', 'start': 48, 'end': 51}, {'entity_group': 'POSTCODE', 'score': np.float32(0.9239461), 'word': '62704.', 'start': 51, 'end': 58}]
```

## Training Data

Filtered subset of [ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k),
restricted to **English-language examples only** (`language == "en"`).  
The full dataset is multilingual; this model targets English text only.

| Split | Full Dataset | English Subset |
|-------|-------------|----------------|
| Train | 177,677 | 29,908 |
| Validation | 47,728 | 3,973 |
| Test | — | 3,973 |

**Preprocessing:**
- Dropped `CARDISSUER` entity class (little support)
- Validation set split 50/50 into validation and test

*Full preprocessing notebook:*
[prepare_ds.ipynb](https://github.com/ben-gid/pii_redaction/blob/main/notebooks/prepare_ds.ipynb)

## Training Procedure

Two-phase Fine-tuned (frozen backbone → unfrozen) from [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) using a weighted token-classification trainer and discriminative LRs.

### Hyperparameters
| Parameter | Stage 1 (frozen backbone) | Stage 2 (full fine-tune) |
|-----------|--------------------------|--------------------------|
| Learning rate         | 0.001           | 2e-05          |
| LR scheduler          | linear    | linear   |
| Warmup steps          | 186       | 186      |
| Batch size (per device) | 32         | 8          |
| Gradient accumulation | 1     | 2    |
| Effective batch size  | 32 | 16|
| Precision             | bf16    | bf16   |
| Weight decay          | 0.01 | 0.01 |
| Seed                  | 42    | 42   |


## Evaluation

Evaluated on the English validation subset (3,973 examples) at the best checkpoint.

| Metric | Value |
|--------|-------|
| **F1 (macro)** | **0.9557** |
| Precision | 0.9507 |
| Recall | 0.9607 |
| Token Accuracy | 0.9944 |

### Per-Entity F1

| Entity | F1 | Support |
|--------|------|---------|
| BOD | 0.9774 | 1124 |
| BUILDING | 0.9829 | 963 |
| CITY | 0.9763 | 989 |
| COUNTRY | 0.9639 | 757 |
| DATE | 0.9295 | 837 |
| DRIVERLICENSE | 0.9563 | 1142 |
| EMAIL | 0.9864 | 1206 |
| GEOCOORD | 0.9806 | 104 |
| GIVENNAME1 | 0.8751 | 904 |
| GIVENNAME2 | 0.8175 | 255 |
| IDCARD | 0.9436 | 1300 |
| IP | 0.9942 | 1028 |
| LASTNAME1 | 0.8497 | 1158 |
| LASTNAME2 | 0.7557 | 313 |
| LASTNAME3 | 0.7302 | 105 |
| PASS | 0.9741 | 784 |
| PASSPORT | 0.9602 | 1173 |
| POSTCODE | 0.9732 | 954 |
| SECADDRESS | 0.9705 | 440 |
| SEX | 0.9635 | 969 |
| SOCIALNUMBER | 0.9677 | 1285 |
| STATE | 0.9840 | 995 |
| STREET | 0.9749 | 967 |
| TEL | 0.9673 | 991 |
| TIME | 0.9792 | 1825 |
| TITLE | 0.9685 | 906 |
| USERNAME | 0.9687 | 1295 |


## Limitations

- **English only** — trained exclusively on English text; performance on other languages is undefined.
- **Max 512 tokens** — inherited from DeBERTa's positional embeddings. Longer documents should be chunked.
- **Name entities are harder** — The model underperforms on `GIVENNAME` and `LASTNAME` entities:

	| Entity | F1 | Support |
	|--------|------|---------|
	| LASTNAME2 | 0.7570 | 313 |
	| LASTNAME3 | 0.7822 | 105 |
	| GIVENNAME2 | 0.8102 | 255 |
	| LASTNAME1 | 0.8501 | 1158 |
	| GIVENNAME1 | 0.8640 | 904 |

	Likely causes: performance correlates strongly with training support — 
	LASTNAME1/GIVENNAME1 (primary occurrences, ~900-1100 examples) score 
	significantly higher than LASTNAME2/3 (secondary/tertiary occurrences, 
	105-313 examples). Additionally, names are inherently context-dependent: 
	without surrounding cues like titles or formal structure, the model has 
	less signal to distinguish them from non-PII tokens — even the 
	best-supported name entities (LASTNAME1, GIVENNAME1) fall notably below 
	the macro F1 of 0.9557, suggesting names are a structurally harder 
	category regardless of support.
- **Not a redaction tool by itself** — this model detects and labels PII spans; downstream redaction/masking logic must be implemented separately.
- **Subword labeling convention** — following the HuggingFace token classification convention, only the first subword of each word was assigned its NER label during training; continuation subwords were assigned `-100` (ignored by the loss). The practical consequence is that the model predicts `O` with high confidence on continuation subwords, which can cause partial detection of multi-subword entities (e.g. `john@example.com` returned as only `john`) when using `aggregation_strategy="simple"`. Use `aggregation_strategy="first"` for inference, which is consistent with this training convention.
- **Not a redaction tool by itself** — this model detects and labels PII spans; downstream redaction/masking logic must be implemented separately.

## Intended Use

**Intended uses:**
- Detecting and labeling PII spans in English text for downstream redaction or pseudonymization pipelines.
- Privacy compliance tooling (GDPR, CCPA, HIPAA).
- Pre-processing step before storing or sharing user-generated content.

**Out-of-scope uses:**
- Non-English text.
- Real-time high-stakes medical or legal decision-making without human review.
- As a sole compliance mechanism — model errors are expected; human auditing is recommended.


## Model Comparison

| Model | Macro F1 | Params (non-embedding) | Inference Speed | Best For |
|-------|----------|------------------------|-----------------|----------|
| [DeBERTa-v3-Base PII Redaction](https://huggingface.co/bengid/pii-redaction-deberta-base) | 0.9557 | Base (86M params) | ~11.7ms on RTX 5070 | Accuracy |
| [DeBERTa-v3-Small PII Redaction](https://huggingface.co/bengid/pii-redaction-deberta-small) | 0.9476 | Small (44M params) | ~6.5ms on RTX 5070 | Latency |
| [DeBERTa-v3-XSmall PII Redaction](https://huggingface.co/bengid/pii-redaction-deberta-xsmall) | 0.9303 | XSmall (22M params) | ~11.6ms on RTX 5070 [1] | Memory |

[1] see x-small card for latency explanation

## License

The model weights are released for research and non-commercial use,
consistent with the training data license
([ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)).
Users should review the dataset license before commercial deployment.

## Citation

If you use this model, please cite the base model architecture and the training dataset:

**Base model (DeBERTa-v3):**
```bibtex
@misc{he2021debertav3,
      title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing}, 
      author={Pengcheng He and Jianfeng Gao and Weizhu Chen},
      year={2021},
      eprint={2111.09543},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

**Training dataset:**
```bibtex
@misc{ai4privacy2023pii,
  title     = {PII Masking 300k},
  author    = {Ai4Privacy},
  year      = {2023},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/1995},
  url       = {https://huggingface.co/datasets/ai4privacy/pii-masking-300k}
}
```

