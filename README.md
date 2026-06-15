# PII Redaction API

A production-ready API that detects and redacts personally identifiable information from text before it enters an LLM pipeline.

**Stack:** DeBERTa-v3 · HuggingFace Trainer · FastAPI · Docker · AWS ECS Fargate · HuggingFace Spaces

---

## Roadblocks

- **SentencePiece offset mismatch** — DeBERTa's SentencePiece tokenizer includes leading whitespace in the next token's offset, so the dataset's `privacy_mask` start index (which points to the first non-whitespace character) doesn't match any token offset. Fixed by switching from `offset[0] >= start and offset[1] <= end` to `offset[1] > start and offset[0] < end` (overlap-based check) and using `offset[0] <= start` for the B- prefix instead of `offset[0] == start`.

- **NaN gradients from hidden fp16** — The model loaded weights in fp16 even though `fp16=False` and `bf16=True` were set, causing all weights to become NaN and the model to predict only "O". Had to manually iterate `model.named_modules()` and cast every module to `torch.float32`.

- **Name entities are harder** — GIVENNAME and LASTNAME entities consistently score ~0.10–0.20 below macro F1 across all variants, driven by limited training support for secondary/tertiary occurrences (LASTNAME2/3, GIVENNAME2) and names being inherently context-dependent.

- **Tokenizer differences across architectures** — DistilBERT (WordPiece), DeBERTa (SentencePiece), and RoBERTa (BPE) each tokenize differently. The offset-alignment logic had to be rewritten per tokenizer.

- **Subword labeling convention** — Continuation subwords are masked with `-100` during training, so `aggregation_strategy="simple"` produces partial entity detections. Must use `aggregation_strategy="first"` at inference.

---

## Implementation

- [x] **Phase 1 — Data & Baseline**
  - Loaded and explored `ai4privacy/pii-masking-300k`
  - Filtered to English only (29,908 train / 3,973 val / 3,973 test)
  - Removed CARDISSUER (only 5 examples)
  - Created BIO label mappings for 27 entity types
  - Trained DistilBERT baseline pipeline

- [x] **Phase 2 — Model & Training**
  - DeBERTa-v3-base: macro F1 **0.9564** on test set
  - DeBERTa-v3-small: macro F1 **0.9497** on test set
  - DeBERTa-v3-xsmall: macro F1 **0.9422** on test set
  - RoBERTa-base experiment (0.9553 test F1)
  - Custom `WeightedTokenClassificationTrainer` with class-balanced loss
  - Two-phase training (frozen backbone → unfrozen full fine-tune)
  - W&B logging with per-entity F1 and interactive HTML prediction tables
  - Models published to HF Hub: `bengid/pii-redaction-deberta-{base,small,xsmall}`

- [x] **Phase 3 — Redaction Logic**
  - `PIIRedactor` class with chunking, overlap resolution, and redaction
  - Overlapping token chunks with boundary-aligned splitting (paragraph/sentence-aware)
  - Overlap resolution by max confidence score
  - CLI entry point for file/stdin redaction
  - Structured `RedactionResponse` Pydantic schema

- [x] **Phase 4 — FastAPI Service**
  - `POST /redact` — detect and redact PII, threshold per request
  - `GET /health` — liveness check
  - `GET /model-info` — model metadata and entity types
  - Optional API key auth via `X-API-Key` header
  - Model loaded once at startup via lifespan
  - CORS enabled, structured JSON error handling
  - Swagger UI at `/docs` (and a direct shortcut button in the UI)
  - Interactive dark-mode HTML Demo page (`GET /`) with glassmorphism layout
  - Drag-and-drop plain text file upload to automatically populate the input text
  - Real-time entity classification unpacking with confidence score badges
  - External static styles separated into `static/styles.css`

- [ ] Phase 5 — Docker multi-stage build
- [ ] Phase 5 — ECR push & ECS Fargate deployment
- [ ] Phase 6 — HuggingFace Spaces Gradio demo with `gr.HighlightedText`
- [ ] Architecture diagram for README
- [ ] `/pseudonymize` endpoint (consistent fake value replacement)
