"""
PII Redaction Model Publishing Script.

benchmarks trained model checkpoints, compiles comprehensive Hugging Face Model Cards
reflecting stage-specific training hyperparameters, and optionally publishes the weights
and configs to the Hugging Face Hub.

Key Lessons & Design Decisions:
1. Dynamic model card generation:
   Instead of copy-pasting training arguments, we dynamically parse hyperparameter tables
   and metrics directly from `trainer_state.json` and saved checkpoints.
2. GPU benchmarking:
   Includes local benchmarking using a standard sample PII sentence to generate comparative
   latency statistics for the registry comparison table.
3. Edge deployment warnings (Latency vs. Depth):
   Highlights the architectural caveat that depth dominates token classification latency
   on GPUs, reminding users that the 22M XSmall model has a similar latency to the 86M Base
   model because both utilize 12 transformer layers.
"""

import argparse
from pathlib import Path
import sys

from transformers import AutoModelForTokenClassification, AutoTokenizer
from huggingface_hub import ModelCardData, ModelCard, EvalResult

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from cli.publish_utils import (  # noqa: E402
    benchmark,
    get_best_checkpoint,
    load_training_args,
    get_first_checkpoint_args,
    make_entity_table,
) 


# Default model configurations registry
MODELS = [
    {
        "dir_name": "base-bf16-weighted-trainer",
        "repo_id": "bengid/pii-redaction-deberta-base",
        "model_name": "DeBERTa-v3-Base PII Redaction",
        "base_model": "microsoft/deberta-v3-base",
        "size_label": "Base (86M params)",
        "macro_f1": 0.9557,
        "inference_speed_label": "~11.7ms on RTX 5070",
        "best_for": "Accuracy",
        "recommendation": """
**Recommended when** accuracy is the priority and compute is not a 
constraint. Best overall performance (macro F1: 0.9557), particularly 
on rare entity types. Well-suited for offline batch processing, 
compliance pipelines, or any server-side deployment where an extra 
~5ms of latency is acceptable.""",
    },
    {
        "dir_name": "small-bf16-weighted-trainer",
        "repo_id": "bengid/pii-redaction-deberta-small",
        "model_name": "DeBERTa-v3-Small PII Redaction",
        "base_model": "microsoft/deberta-v3-small",
        "size_label": "Small (44M params)",
        "macro_f1": 0.9517,
        "inference_speed_label": "~6.5ms on RTX 5070",
        "best_for": "Latency",
        "recommendation": """
**Recommended when** you need the best latency with minimal accuracy 
tradeoff. Despite being 44M parameters, its 6-layer architecture 
makes it the fastest of the three (~6.5ms on RTX 5070) — nearly 2x faster 
than base while retaining strong performance. 
Best fit for real-time APIs or high-throughput services.""",
    },
    {
        "dir_name": "xsmall-bf16-weighted-trainer",
        "repo_id": "bengid/pii-redaction-deberta-xsmall",
        "model_name": "DeBERTa-v3-XSmall PII Redaction",
        "base_model": "microsoft/deberta-v3-xsmall",
        "size_label": "XSmall (22M params)",
        "macro_f1": 0.9424,
        "inference_speed_label": "~11.6ms on RTX 5070 [1]",
        "best_for": "Memory",
        "recommendation": """
**Recommended when** memory footprint is the hard constraint — 
edge deployments, CPU inference, or environments where the 22M 
parameter count matters more than raw latency. 

#### Latency Note
Note that despite being the smallest model, RTX 5070 latency (~11.6ms) 
is comparable to base due to its identical 12-layer depth; sequential 
layer passes dominate GPU latency more than hidden dimension width. The 
advantage over base is memory, not speed.""",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Publish PII Redaction models to Hugging Face Hub.")
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Path containing model run directories. Defaults to project 'models/'."
    )
    parser.add_argument(
        "--model_cards_dir",
        type=str,
        default=None,
        help="Output directory to save compiled readmes locally. Defaults to 'model_cards/'."
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push models and compiled model cards to Hugging Face Hub."
    )
    parser.add_argument(
        "--benchmark_only",
        action="store_true",
        help="Perform speed benchmark locally on available hardware and print statistics."
    )
    return parser.parse_args()


def compile_model_card(model_cfg, model_dir: Path):
    """Load model metrics and arguments dynamically to assemble the Model Card MD."""
    print(f"Loading best checkpoint info from {model_dir}...")
    best_ckpt, state, best_log = get_best_checkpoint(model_dir)
    stage1_args = get_first_checkpoint_args(model_dir)
    stage2_args = load_training_args(best_ckpt)

    f1 = best_log.get("eval_f1", 0.0)
    pre = best_log.get("eval_precision", 0.0)
    rec = best_log.get("eval_recall", 0.0)
    acc = best_log.get("eval_accuracy", 0.0)
    entity_table = make_entity_table(best_log)

    base_kwargs = dict(
        task_type="token-classification",
        dataset_name="ai4privacy/pii-masking-300k",
        dataset_type="ai4privacy/pii-masking-300k",
        dataset_split="validation",
    )

    eval_results = [
        EvalResult(**{**base_kwargs, "metric_type": "f1", "metric_value": f1}),
        EvalResult(**{**base_kwargs, "metric_type": "precision", "metric_value": pre}),
        EvalResult(**{**base_kwargs, "metric_type": "recall", "metric_value": rec}),
    ]

    card_data = ModelCardData(
        language="en",
        license="other",
        library_name="transformers",
        base_model=model_cfg["base_model"],
        tags=["token-classification", "pii", "ner", "deberta", "privacy", "named-entity-recognition"],
        datasets=["ai4privacy/pii-masking-300k"],
        metrics=["f1", "precision", "recall"],
        model_name=model_cfg["model_name"],
        pipeline_tag="token-classification",
        eval_results=eval_results,
    )

    description = (
        f"Fine-tuned [{model_cfg['base_model']}](https://huggingface.co/{model_cfg['base_model']}) for "
        f"Named Entity Recognition targeting 27 PII entity types. "
        f"Trained on the English subset of [ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) "
        f"with a class-weighted `CrossEntropyLoss`. Achieves **{f1:.4f} macro-F1** on the validation set."
    )

    def get_precision(args) -> str:
        if getattr(args, "bf16", False):
            return "bf16"
        if getattr(args, "fp16", False):
            return "fp16"
        return "fp32"

    # Stage 1 parameters
    stage1_lr = stage1_args.learning_rate
    stage1_bs = stage1_args.per_device_train_batch_size
    stage1_grad_acc = stage1_args.gradient_accumulation_steps
    stage1_effective_bs = stage1_bs * stage1_grad_acc
    stage1_precision = get_precision(stage1_args)
    stage1_scheduler = stage1_args.lr_scheduler_type.value
    stage1_warmup = stage1_args.warmup_steps

    # Stage 2 parameters
    stage2_lr = stage2_args.learning_rate
    stage2_bs = stage2_args.per_device_train_batch_size
    stage2_grad_acc = stage2_args.gradient_accumulation_steps
    stage2_effective_bs = stage2_bs * stage2_grad_acc
    stage2_precision = get_precision(stage2_args)
    stage2_scheduler = stage2_args.lr_scheduler_type.value
    stage2_warmup = stage2_args.warmup_steps

    # Compile registry statistics table
    comparison_rows = []
    for m in MODELS:
        row = (
            f"| [{m['model_name']}](https://huggingface.co/{m['repo_id']}) "
            f"| {m['macro_f1']:.4f} "
            f"| {m['size_label']} "
            f"| {m['inference_speed_label']} "
            f"| {m['best_for']} |"
        )
        comparison_rows.append(row)
    if model_cfg['model_name'] == "DeBERTa-v3-XSmall PII Redaction":
        note = "[1] see [Latency Note](#latency-note) for latency explanation"
    else:
        note = "[1] see [DeBERTa-v3-XSmall PII Redaction](https://huggingface.co/bengid/pii-redaction-deberta-xsmall/#latency-note) for latency explanation"

    comparison_rows.append("\n" + note)        
        
    comparison_table = "\n".join(comparison_rows)

    card_content = f"""---
{card_data.to_yaml()}
---

# {model_cfg["model_name"]}

{description}
{model_cfg["recommendation"]}

## Usage

```python
from transformers import pipeline

pipe = pipeline(
    "token-classification",
    model="{model_cfg["repo_id"]}",
    aggregation_strategy="first",
    device=0  # omit for CPU
)

text = "She lives at 742 Evergreen Terrace, Springfield, IL 62704."
entities = pipe(text)
print(entities)
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

## Training Procedure

Two-phase Fine-tuning (frozen backbone → unfrozen) from [`{model_cfg['base_model']}`](https://huggingface.co/{model_cfg['base_model']}) using a weighted token-classification trainer and stage-specific learning rates.

### Hyperparameters
| Parameter | Stage 1 (frozen backbone) | Stage 2 (full fine-tune) |
|-----------|--------------------------|--------------------------|
| Learning rate         | {stage1_lr}           | {stage2_lr}          |
| LR scheduler          | {stage1_scheduler}    | {stage2_scheduler}   |
| Warmup steps          | {stage1_warmup}       | {stage2_warmup}      |
| Batch size (per device) | {stage1_bs}         | {stage2_bs}          |
| Gradient accumulation | {stage1_grad_acc}     | {stage2_grad_acc}    |
| Effective batch size  | {stage1_effective_bs} | {stage2_effective_bs}|
| Precision             | {stage1_precision}    | {stage2_precision}   |
| Weight decay          | {stage1_args.weight_decay} | {stage2_args.weight_decay} |
| Seed                  | {stage1_args.seed}    | {stage2_args.seed}   |

## Evaluation

Evaluated on the English validation subset (3,973 examples) at the best checkpoint.

| Metric | Value |
|--------|-------|
| **F1 (macro)** | **{f1:.4f}** |
| Precision | {pre:.4f} |
| Recall | {rec:.4f} |
| Token Accuracy | {acc:.4f} |

### Per-Entity F1

{entity_table}

## Limitations

- **English only** — trained exclusively on English text; performance on other languages is undefined.
- **Max 512 tokens** — inherited from DeBERTa's positional embeddings. Longer documents should be chunked.
- **Name entities are harder** — The model underperforms on `GIVENNAME` and `LASTNAME` entities:
    
    Likely causes: performance correlates strongly with training support — 
    LASTNAME1/GIVENNAME1 (primary occurrences, ~900-1100 examples) score 
    significantly higher than LASTNAME2/3 (secondary/tertiary occurrences, 
    105-313 examples). Additionally, names are inherently context-dependent: 
    without surrounding cues like titles or formal structure, the model has 
    less signal to distinguish them from non-PII tokens — even the 
    best-supported name entities (LASTNAME1, GIVENNAME1) fall notably below 
    the macro F1 of {f1:.4f}, suggesting names are a structurally harder 
    category regardless of support.
- **Not a redaction tool by itself** — this model detects and labels PII spans; downstream redaction/masking logic must be implemented separately.
- **Subword labeling convention** — following the HuggingFace token classification convention, only the first subword of each word was assigned its NER label during training; continuation subwords were assigned `-100` (ignored by the loss). The practical consequence is that the model predicts `O` with high confidence on continuation subwords, which can cause partial detection of multi-subword entities (e.g. `john@example.com` returned as only `john`) when using `aggregation_strategy="simple"`. Use `aggregation_strategy="first"` for inference, which is consistent with this training convention.

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
{comparison_table}

## License

The model weights are released for research and non-commercial use,
consistent with the training data license
([ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)).
Users should review the dataset license before commercial deployment.

## Citation

If you use this model, please cite the base model architecture and the training dataset:

**Base model (DeBERTa-v3):**
```bibtex
@misc{{he2021debertav3,
      title={{DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing}}, 
      author={{Pengcheng He and Jianfeng Gao and Weizhu Chen}},
      year={{2021}},
      eprint={{2111.09543}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.CL}}
}}
```

**Training dataset:**
```bibtex
@misc{{ai4privacy2023pii,
  title     = {{PII Masking 300k}},
  author    = {{Ai4Privacy}},
  year      = {{2023}},
  publisher = {{Hugging Face}},
  doi       = {{10.57967/hf/1995}},
  url       = {{https://huggingface.co/datasets/ai4privacy/pii-masking-300k}}
}}
```
"""
    return card_content


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    models_dir = Path(args.models_dir) if args.models_dir else project_root / "models"
    model_cards_dir = Path(args.model_cards_dir) if args.model_cards_dir else project_root / "model_cards"
    model_cards_dir.mkdir(exist_ok=True)

    test_sentence = "John Smith's account number is 4111-1111-1111-1111 and his email is john@example.com"

    for model_cfg in MODELS:
        model_dir = models_dir / model_cfg["dir_name"]
        
        # Verify run directory exists before processing
        if not model_dir.exists():
            print(f"Skipping variant {model_cfg['model_name']} - Directory not found: {model_dir}")
            continue

        best_ckpt, _, _ = get_best_checkpoint(model_dir)

        if args.benchmark_only:
            print(f"Benchmarking speed of {model_cfg['model_name']} at {best_ckpt}...")
            latency = benchmark(str(best_ckpt), test_sentence)
            print(f"  Result: {latency:.2f}ms per sample.")
            continue

        print(f"Generating Model Card for {model_cfg['model_name']}...")
        card_content = compile_model_card(model_cfg, model_dir)

        readme_filename = f"{model_cfg['dir_name']}-readme.md"
        card_path = model_cards_dir / readme_filename
        with open(card_path, "w") as f:
            f.write(card_content)
        print(f"  Saved compiled card locally to: {card_path}")

        if args.push:
            print(f"Pushing model variant {model_cfg['model_name']} to HF hub ({model_cfg['repo_id']})...")
            
            # Load assets from local best checkpoint
            model = AutoModelForTokenClassification.from_pretrained(str(best_ckpt))
            tokenizer = AutoTokenizer.from_pretrained(str(best_ckpt))

            model.push_to_hub(model_cfg["repo_id"], commit_message="Push fine-tuned checkpoint")
            tokenizer.push_to_hub(model_cfg["repo_id"])
            
            card = ModelCard(content=card_content)
            card.push_to_hub(model_cfg["repo_id"])
            print(f"  ✓ Model published successfully to Hub: {model_cfg['repo_id']}")


if __name__ == "__main__":
    main()
