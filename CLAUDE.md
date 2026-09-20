# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install (Python 3.14+)
uv run pytest tests/ -q              # full suite (all model calls are mocked; no weights loaded)
uv run pytest tests/test_redact.py::test_empty_text -q   # single test
uv run ruff check                    # lint
uv run pyright                       # types
uv run python -m api.app.main        # serve on :8000 (loads real weights from HF Hub)
docker build -t pii-redaction-api . && docker run -p 8000:8000 pii-redaction-api
```

CI runs exactly ruff → pyright → pytest. All three must pass.

Training pipeline (needs a GPU; `models/` and `data/` are gitignored):

```bash
uv run python -m cli.prepare_ds --dataset nemotron     # -> data/<dir_name> + data/<label_info>
uv run python -m cli.train --dataset nemotron --model_variant small
uv run python -m cli.sweep --dry_run                   # print the sweep's train commands
uv run python -m cli.publish --push                    # benchmark, build model cards, push to HF
uv run python cli/redact.py --text "..." --model_variant base
```

## Layout

Three layers that deploy separately:

- [pii_redaction/](pii_redaction/) — the inference library. `PIIRedactor` + pydantic `Entity`/`RedactionResponse`. Copied into the Docker image; keep its imports limited to transformers/torch/pydantic.
- [api/](api/) — FastAPI service wrapping `PIIRedactor`. Also in the image.
- [cli/](cli/) — data prep, training, publishing. **Not** in the image, and CD ignores `cli/**`, `notebooks/**`, `**.md` — changes there never trigger a deploy.

The project is not installed as a package (`uv sync --no-install-project` in Docker, `pythonpath = ["."]` for pytest). [api/app/core/config.py](api/app/core/config.py) and [cli/redact.py](cli/redact.py) prepend the repo root to `sys.path` so `pii_redaction` resolves. Keep those shims when touching imports.

## API

Model is loaded once at lifespan startup into the module-level `state.redactor` singleton in [api/app/core/config.py](api/app/core/config.py); `settings` lives beside it. Routers reach it via `run_redaction` in [dependencies.py](api/app/dependencies.py), which 503s when it's `None`.

- `/redact` requires `X-API-Key`; `/demo/redact` is unauthenticated and rate-limited to 10/day per IP.
- Tests patch `api.app.core.config.PIIRedactor` (see [tests/conftest.py](tests/conftest.py)) — patching it anywhere else misses the load path.
- Production (`ENV=production`) reads `SSL_CERT`/`SSL_KEY` env vars injected by ECS, writes them to `/tmp/certs`, and runs uvicorn with TLS directly.

Logging ([api/app/core/logging_setup.py](api/app/core/logging_setup.py)) is privacy-critical: routers never log, they attach metadata to `request.state` and the middleware emits one JSON line per request. `JSONFormatter.ALLOWED_EXTRA` whitelists fields so a stray `extra=` can't leak PII. Never log raw text, entity values, or unhashed IPs.

## Redaction logic

`PIIRedactor.predict` for long inputs: pre-tokenize without special tokens → build overlapping character-span chunks (`stride` fraction of `max_length`, snapped backwards to a paragraph/sentence boundary within 80 chars) → run each chunk → resolve overlapping spans by highest score → replace right-to-left so earlier substitutions don't invalidate later offsets.

The pipeline must use `aggregation_strategy="first"`. Training masks continuation sub-words with `-100`, so `"simple"` produces partial entities.

## Training pipeline

[cli/__init__.py](cli/__init__.py) holds the `DATASETS` registry (`hf_name`, `dir_name`, `label_info`, `run_suffix`). It is the single source of truth linking `prepare_ds` output to `train` input — add a dataset there, not by passing paths. Both dataset flows normalize to the same two columns (`source_text`, `privacy_mask`) so `train.py` is dataset-agnostic.

Non-obvious constraints, each learned the hard way (the module docstrings carry the full reasoning):

- Label alignment ([train_utils.py](cli/train_utils.py) `tokenize_and_align_labels`) matches by **character overlap on the per-word union of sub-token offsets**, not equality — SentencePiece folds leading whitespace into the first sub-token, so exact-offset matching silently labels nothing.
- `train.py` casts every module to float32 before training. DeBERTa-v3 otherwise loads fp16 weights despite `bf16=True, fp16=False` and every weight goes NaN.
- `make_compute_metrics_fn` returns a **stateful** accumulator and requires `batch_eval_metrics=True`. Without it the Trainer concatenates all logits before one call and OOMs a 12GB card on a 30k-row eval split.
- Two stages, both required: frozen backbone at lr 1e-3, then full fine-tune at 2e-5. `WeightedTokenClassificationTrainer` applies normalized inverse-frequency class weights — without them the model predicts only `O`.
- `prepare_ds.py` keeps `entity_classes` in first-appearance order, not sorted, for the ai4privacy flow. Sorting changes label ids and invalidates the published checkpoints (see the TODO there).

Run names double as `models/<name>` output directories, so a duplicate silently overwrites a sibling run.

## Notebooks

[notebooks/](notebooks/) are experimental scratch versions of the `cli/` scripts and are excluded from ruff. The `cli/` scripts are canonical.
