# PII Redaction API — Project Plan

**One-line pitch:** A production-ready API that detects and redacts personally identifiable information from text before it enters an LLM pipeline.

**Stack:** DeBERTa-v3 · HuggingFace Trainer · FastAPI · Docker · AWS ECS Fargate · HuggingFace Spaces

---

## Phase 1 — Data & Baseline (Week 1)

### Goal
Understand the data, establish a training baseline, validate the task is working end-to-end before any optimization.

### Steps

**1. Load and explore the dataset**
- Use `ai4privacy/pii-masking-400k` (broader entity coverage, multilingual) or `nvidia/Nemotron-PII` (cleaner, more modern provenance). Either works; ai4privacy has more training data.
- Explore label distribution — which entity types are rare? Which dominate?
- Inspect raw examples to understand how BIO tags map to spans in context.

**2. Tokenization alignment**
- This is the hardest conceptual step in token classification: subword tokenizers split words into pieces, but labels are word-level. You need to align labels to subword tokens correctly (label the first subword, mask the rest with `-100`).
- HuggingFace's `tokenize_and_align_labels` pattern handles this. Get comfortable with it before moving on.

**3. Train a quick baseline**
- Use `distilbert-base-uncased` first — fast to iterate, cheap to train.
- Train for 3 epochs, evaluate with `seqeval` (entity-level F1, not token-level accuracy).
- Don't optimize yet. Just confirm the pipeline works: data loads → model trains → seqeval reports entity F1.

### Deliverable
A working training script and a rough F1 number. Doesn't need to be good yet.

---

## Phase 2 — Model & Training (Week 2)

### Goal
Swap to DeBERTa-v3, tune training, reach competitive performance.

### Steps

**1. Upgrade to DeBERTa-v3-base**
- `microsoft/deberta-v3-base` is the current state-of-the-art encoder for token classification tasks.
- Requires the `sentencepiece` tokenizer — install separately, it's a common gotcha.

**2. Use HuggingFace Trainer API properly**
- `TrainingArguments`: set `evaluation_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="eval_f1"`.
- Log to W&B — you already have this in your stack. Track entity-level F1 per epoch, not just loss.

**3. Training details to get right**
- Learning rate: `2e-5` is a safe default for DeBERTa fine-tuning.
- Batch size: 16 if your RTX 5070 can handle it, else 8 with gradient accumulation steps=2.
- Class imbalance: `O` (outside) tokens vastly outnumber entity tokens. seqeval ignores `O` in F1 calculation, so this matters less than it looks — but worth being aware of.

**4. Evaluate properly**
- Report per-entity-type F1, not just macro F1. If your model is great at detecting `EMAIL` but terrible at `SSN`, you want to know that.
- `seqeval`'s classification report gives you this breakdown.

### Deliverable
A fine-tuned DeBERTa checkpoint pushed to HuggingFace Hub with a model card. The model card is not optional — it's part of the portfolio artifact.

---

## Phase 3 — Post-processing & Redaction Logic (Week 2–3)

### Goal
Turn raw NER output (entity spans + labels) into actual redacted text. This is the engineering layer that makes it a product.

### Steps

**1. Span extraction**
- Raw model output is per-token logits. You need to aggregate these back into word-level spans using the `aggregation_strategy` in HuggingFace pipelines (`"simple"` or `"first"`).
- Output format per entity: `{text, label, start, end, score}`.

**2. Redaction**
- Replace each detected span with a typed placeholder: `[NAME]`, `[EMAIL]`, `[SSN]`, `[PHONE]`, etc.
- Process spans in reverse order (end → start) so character offsets stay valid as you modify the string.

**3. Structured output schema**
Define a clean Pydantic response model:
```python
class Entity(BaseModel):
    text: str          # original text
    label: str         # entity type
    start: int         # char offset
    end: int
    score: float       # model confidence

class RedactionResponse(BaseModel):
    original: str
    redacted: str
    entities: list[Entity]
    entity_count: int
```

**4. Confidence thresholding**
- Add a `threshold` parameter (default `0.85`) — entities below confidence get flagged but not redacted by default. This gives API callers control over precision/recall tradeoff.

### Deliverable
A standalone `redactor.py` module that takes raw text and returns a `RedactionResponse`. Fully testable without the API layer.

---

## Phase 4 — FastAPI Service (Week 3)

### Goal
Wrap the redaction logic in a production-quality API.

### Steps

**1. Endpoints**
```
POST /redact          — main endpoint, text in, redacted text + entities out
GET  /health          — liveness check
GET  /model-info      — returns model name, version, entity types supported
```

**2. Request validation**
- Use Pydantic for request models.
- Set a max input length (e.g. 512 tokens / ~2000 chars) and return a `422` with a clear message if exceeded.
- Add a `threshold` query param with a default.

**3. Model loading**
- Load the model once at startup using FastAPI's lifespan context, not on every request.
- Use `torch.no_grad()` and set `model.eval()`.

**4. Error handling**
- Return structured JSON errors, not raw Python tracebacks.
- Handle empty input, oversized input, and model inference failures separately.

**5. Basic auth**
- Add API key auth via a header (`X-API-Key`). Protects the deployed endpoint and looks production-appropriate.

### Deliverable
A FastAPI app that runs locally with `uvicorn`, passes a Postman/httpx test suite, and has auto-generated `/docs` (Swagger UI) that you can screenshot for the README.

---

## Phase 5 — Docker & Deployment (Week 4)
**remind claude for clarification**
*When you get to Phase 5, flag me — the ECS Fargate path for a CPU inference container has some specific gotchas worth knowing in advance (task memory sizing for DeBERTa, model loading at container startup vs. baking weights into the image, etc.)*
### Goal
Containerize and deploy to ECS Fargate. You've done this path before with the flower classifier.

### Steps

**1. Dockerfile**
- Multi-stage build: install deps in a builder stage, copy artifacts to a slim runtime stage.
- Don't copy the full HuggingFace cache — download the model from Hub at container startup or bake the weights into the image (baking is faster at runtime, bigger image).
- Pin all dependency versions.

**2. ECR → ECS Fargate**
- Same path as your flower classifier. Push image to ECR, define a Fargate task definition, run as a service.
- Task size: at least 4GB memory for DeBERTa-v3. CPU inference only (Fargate doesn't support GPU), which is fine for a portfolio project.

**3. Environment config**
- API key, model name, confidence threshold — all via environment variables, not hardcoded.

### Deliverable
A live HTTPS endpoint. This is the URL that goes in your resume and README.

---

## Phase 6 — HuggingFace Space (Week 4)

### Goal
A public demo that anyone can use in 30 seconds without an API key.

### Steps

**1. Gradio app**
- Text input box, "Redact" button.
- Output: two columns — original text with entity spans highlighted, redacted text below.
- Use `gr.HighlightedText` for the entity visualization — it's built for exactly this.

**2. Pre-loaded examples**
- Include 4–5 example inputs covering different entity types: a fake medical note, a fake customer email, a fake financial record.
- Make them realistic enough to demonstrate breadth.

**3. Deploy to HF Spaces**
- Free tier is fine. CPU-only inference, may be slow on first request (cold start) — that's acceptable for a demo.

### Deliverable
A public HuggingFace Space URL. This + the deployed API + the model card = a complete, three-artifact portfolio project.

---

## Project Artifacts Checklist

| Artifact | Purpose |
|---|---|
| GitHub repo | Code, README, architecture diagram |
| HuggingFace model card | Documents training data, metrics, entity types, intended use |
| FastAPI service on ECS Fargate | Live deployed endpoint |
| HuggingFace Space (Gradio) | Public demo |
| W&B run | Training metrics, loss curves, per-entity F1 |

---

## README Structure (important for portfolio)

1. **What it does** — one paragraph, non-technical
2. **Why it matters** — LLM pipelines, GDPR/HIPAA compliance context
3. **Architecture diagram** — shows the full stack visually
4. **API usage** — `curl` example hitting the live endpoint
5. **Entity types supported** — list all labels the model handles
6. **Model performance** — seqeval F1 table by entity type
7. **Local setup** — Docker one-liner to run locally
8. **Roadmap** — "Planned: RAG pipeline integration as preprocessing layer"

The roadmap bullet is intentional — it signals to a hiring team that this is a living project with a clear next step, not a completed homework assignment.

---

## Future Upgrades (post-RAG learning)

- Add the model as a preprocessing step in a LangChain document ingestion pipeline
- Redact → embed → retrieve, so sensitive documents can be safely indexed
- Add a `/pseudonymize` endpoint that replaces PII with consistent fake values (same name always maps to same fake name) — useful for preserving document coherence in testing