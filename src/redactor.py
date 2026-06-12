"""
PII Redaction Module — extract and redact PII entities from text using
fine-tuned DeBERTa-v3 models.

Supports three local model variants (small / base / xsmall), custom
overlapping token chunking, boundary-aligned splitting, prediction
centering, and structured JSON output.

Typical usage::

    redactor = PIIRedactor(model_id="small", threshold=0.3)
    response = redactor.predict("My email is john@example.com")
    print(response.redacted)          # "My email is [EMAIL]"
    print(response.model_dump_json()) # full structured output

CLI usage::

    python -m src.redactor --text "My email is john@example.com"
    python -m src.redactor --file input.txt --output_path out.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from src.models import Entity, RedactionResponse

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for the PII redactor.

    Accepts input via ``--text`` or ``--file``, supports all model
    variants, threshold tuning, stride / max-length overrides, and
    optional JSON output to a file.

    Examples::

        python -m src.redactor --text "email: test@example.com"
        python -m src.redactor --file input.txt --output_path out.json
        python -m src.redactor --file input.txt --model_variant base \\
            --threshold 0.5 --stride 0.25 --max_length 384
    """
    args = _parse_args()

    text = _validate_args(args)

    redactor = PIIRedactor(
        model_id=args.model_variant,
        stride=args.stride,
        max_length=args.max_length,
    )

    response = redactor.predict(text, threshold=args.threshold)
    output = response.model_dump_json(indent=2)

    if args.output_path:
        Path(args.output_path).write_text(output, encoding="utf-8")
        print(f"Output (json) saved to {args.output_path}")
    else:
        print(output)


MODEL_REGISTRY: dict[str, str] = {
    "small": "bengid/pii-redaction-deberta-small",
    "base": "bengid/pii-redaction-deberta-base",
    "xsmall": "bengid/pii-redaction-deberta-xsmall",
}
"""Maps short variant names to HuggingFace Hub repository IDs."""

# "natural" split point near the end of an oversized chunk.
PARAGRAPH_SEPS: list[str] = ["\n\n", "\n"]
SENTENCE_ENDINGS: list[str] = [". ", "? ", "! "]

# Number of tokens to reserve for [CLS] / [SEP] that the pipeline adds
# automatically.  Chunk boundaries are computed so that the *total* number
# of tokens (real + special) never exceeds ``max_length``.
SPECIAL_TOKENS_RESERVE: int = 2


# ---------------------------------------------------------------------------
# Main redactor class
# ---------------------------------------------------------------------------


class PIIRedactor:
    """Detect and redact PII entities in text using a DeBERTa-v3 model.

    Handles long texts via overlapping token-chunking with
    boundary-aligned splits and overlap-resolution by confidence score.

    Parameters:
        model_id:
            One of ``"small"``, ``"base"``, ``"xsmall"`` (the three
            pre-trained variants), or an arbitrary HuggingFace Hub repo ID.
        threshold:
            Minimum confidence score for an entity to be kept.
            Entities below this threshold are silently dropped.
        stride:
            Overlap ratio between consecutive chunks expressed as a
            fraction of ``max_length``.  A value of 0.5 means each chunk
            overlaps the previous one by 50 %.  Must be in (0, 1).
        max_length:
            Maximum number of tokens the underlying model can process in a
            single forward pass (including special tokens).  Defaults to
            512, the DeBERTa-v3 limit.
        device:
            Torch device index (``0`` for GPU, ``-1`` for CPU).  When
            ``None`` the implementation auto-detects CUDA.
    """

    def __init__(
        self,
        model_id: str = "small",
        stride: float = 0.5,
        max_length: int = 512,
        device: Optional[int] = None,
    ) -> None:
        self.stride = stride
        self.max_length = max_length

        # Resolve short-name to HF Hub ID; pass through arbitrary IDs.
        resolved_id = MODEL_REGISTRY.get(model_id, model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(resolved_id)
        self.model = AutoModelForTokenClassification.from_pretrained(resolved_id)

        self.id2label: dict[int, str] = self.model.config.id2label
        self.label2id: dict[str, int] = self.model.config.label2id

        # Determine device for the HF pipeline wrapper.
        pipe_device = -1
        if device is not None:
            pipe_device = device
        elif torch.cuda.is_available():
            pipe_device = 0

        # Internal pipeline uses "first" aggregation strategy, which is
        # consistent with the subword-labelling convention used during
        # training (only the first subword of each word was labelled;
        # continuation subwords received -100 / ignored).
        self._pipe = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="first",
            device=pipe_device,
        )

        self.model.eval()

    def _pretokenize(self, text: str):
        """Run the fast tokenizer on the full text **without** special tokens.

        We need the raw token count and character-offset mapping to plan
        chunk boundaries.  Special tokens (``[CLS]`` / ``[SEP]``) are
        excluded here and accounted for via ``SPECIAL_TOKENS_RESERVE`` so
        that the per-chunk real-token budget is ``max_length - 2``.
        """
        return self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

    @staticmethod
    def _find_logical_split(text: str, search_end: int, lookback: int = 80) -> int:
        """Search backwards from ``search_end`` for a natural split point.

        Priority order:
        1. Paragraph break (``\\n\\n`` or ``\\n``).
        2. Sentence ending (``. `` / ``? `` / ``! ``).

        If none is found within the ``lookback``-wide window the original
        ``search_end`` is returned unchanged (fall back to token boundary).
        
        Args:
            text (str): text to split
            search_end (int): index in text to search up to
            lookback (int, optional): how far to look back from search_end. Defaults to 80.

        Returns:
            int: index of split point
        """
        search_start = max(0, search_end - lookback)
        segment = text[search_start:search_end]

        for sep in PARAGRAPH_SEPS:
            idx = segment.rfind(sep)
            # if sep is found in segment
            if idx != -1:
                return search_start + idx + len(sep)

        for sep in SENTENCE_ENDINGS:
            idx = segment.rfind(sep)
            if idx != -1:
                return search_start + idx + len(sep)

        # return search end if no split point was found
        return search_end

    def _create_chunks(self, text: str) -> list[tuple[int, int]]:
        """Split ``text`` into overlapping character-span chunks.

        Each chunk is guaranteed to fit within ``max_length`` tokens
        (including the ``[CLS]`` / ``[SEP]`` that the pipeline will add).

        The algorithm:

        1. Pre-tokenize the full text to obtain exact token → character
           offset mappings.
        2. Walk forward in token-space, grouping ``effective_max`` tokens
           per chunk.
        3. Before finalizing a chunk boundary (except for the last chunk),
           attempt to snap it to a nearby sentence or paragraph boundary
           so that entities are not split mid-span.
        4. Chunks overlap according to ``stride`` — the overlap region
           ensures that entities near cut-points are still visible to the
           model in at least one full context window.

        Returns:
            A list of ``(start_char, end_char)`` tuples, each describing
            one chunk.
        """
        encoding = self._pretokenize(text)
        offsets = encoding["offset_mapping"]
        total_tokens = len(encoding["input_ids"])

        if total_tokens <= self.max_length:
            return [(0, len(text))]

        effective_max = self.max_length - SPECIAL_TOKENS_RESERVE
        stride_tokens = int(self.max_length * self.stride)

        chunks: list[tuple[int, int]] = []
        start = 0

        while start < total_tokens:
            # Tentative token boundary for this chunk.
            end = min(start + effective_max, total_tokens)

            chunk_start_char = offsets[start][0]
            chunk_end_char = offsets[end - 1][1] if end > start else len(text)

            # Try to snap the end boundary to a logical split point when
            # there is still text after this chunk.
            if end < total_tokens:
                candidate = self._find_logical_split(text, chunk_end_char)
                # Walk backwards through tokens to find the token whose
                # start offset is ≤ the candidate character position.
                for t in range(end - 1, start - 1, -1):
                    if offsets[t][0] <= candidate:
                        end = t + 1
                        chunk_end_char = (
                            offsets[end - 1][1] if end > start else candidate
                        )
                        break

            chunks.append((chunk_start_char, chunk_end_char))

            # If we have consumed the entire token sequence there is
            # nothing left to chunk — stop.
            if end >= total_tokens:
                break

            # Advance the window by the stride (the non-overlapping
            # portion), ensuring forward progress.
            next_start = end - stride_tokens
            if next_start <= start:
                next_start = start + 1
            start = next_start

        # Guarantee the last chunk reaches the very end of the text.
        if chunks and chunks[-1][1] < len(text):
            chunks[-1] = (chunks[-1][0], len(text))

        return chunks

    # ------------------------------------------------------------------
    # Prediction & overlap resolution
    # ------------------------------------------------------------------

    def _predict_chunk(
        self, text: str, chunk_start: int, chunk_end: int, threshold: float
    ) -> list[Entity]:
        """Run the NER pipeline on a single chunk and return entities.

        Entity offsets are adjusted from chunk-relative to text-absolute.
        Entities whose confidence is below ``threshold`` are dropped.
        """
        chunk_text = text[chunk_start:chunk_end]
        results = self._pipe(chunk_text)

        entities: list[Entity] = []
        for result in results:
            score = result["score"] 
            if float(score) < threshold:
                continue

            # Newer versions of ``transformers`` return ``entity_group``;
            # older ones use ``entity``.  Accept either.
            label = result.get("entity_group") or result.get("entity")
            if label is None:
                continue

            entities.append(
                Entity(
                    text=result["word"],
                    label=label,
                    start=int(result["start"]) + chunk_start,
                    end=int(result["end"]) + chunk_start,
                    score=float(score),
                )
            )

        return entities

    @staticmethod
    def _resolve_overlap(chunk_entities: list[list[Entity]]) -> list[Entity]:
        """Merge overlapping entity predictions across chunks.

        When the same entity appears in the overlap zone of two (or more)
        chunks we keep only the prediction with the highest confidence
        score.

        Overlap is determined by checking whether spans share any
        character positions (i.e. ``ne.start < group_end`` **and**
        ``ne.end > group_start``).
        """
        # Flatten per-chunk lists into a single sorted list.
        flat: list[Entity] = []
        for ents in chunk_entities:
            flat.extend(ents)

        if not flat:
            return []

        flat.sort(key=lambda e: (e.start, e.end))

        resolved: list[Entity] = []
        i = 0
        while i < len(flat):
            entity = flat[i]
            group_start = entity.start
            group_end = entity.end
            group = [entity]

            # Collect every subsequent entity whose span overlaps the
            # current group's aggregate span.
            j = i + 1
            while j < len(flat):
                ne = flat[j]
                if ne.start < group_end and ne.end > group_start:
                    group.append(ne)
                    # Expand the group span to include this entity so
                    # that transitive overlaps are also caught.
                    group_start = min(group_start, ne.start)
                    group_end = max(group_end, ne.end)
                    j += 1
                else:
                    break

            if len(group) == 1:
                resolved.append(entity)
            else:
                best = max(group, key=lambda e: e.score)
                resolved.append(best)

            i = j

        return resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str, threshold: float=0.3) -> RedactionResponse:
        """Detect PII entities in ``text`` and produce a redacted version.

        This is the main entry-point.  It handles chunking, batched
        inference, overlap resolution, and reverse-order redaction
        automatically.

        Returns:
            A :class:`RedactionResponse` with the original text, the
            redacted text, all detected entities, and a count.
        """
        chunks = self._create_chunks(text)

        if len(chunks) == 1:
            entities = self._predict_chunk(text, chunks[0][0], chunks[0][1], threshold)
        else:
            all_entities = [self._predict_chunk(text, s, e, threshold) for s, e in chunks]
            entities = self._resolve_overlap(all_entities)

        entities.sort(key=lambda e: e.start)
        redacted = self._redact(text, entities)

        return RedactionResponse(
            original=text,
            redacted=redacted,
            entities=entities,
            entity_count=len(entities),
        )

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------

    @staticmethod
    def _redact(text: str, entities: list[Entity]) -> str:
        """Replace every detected entity span with a typed placeholder.

        Operates **right-to-left** (reverse order) so that replacing an
        earlier span does not invalidate the character offsets of later
        spans — a crucial concern when multiple entities are present.
        """
        result = list(text)
        # Process in reverse order to preserve offset validity.
        for entity in reversed(entities):
            placeholder = f"[{entity.label}]"
            result[entity.start : entity.end] = list(placeholder)
        return "".join(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and redact PII entities in text using a DeBERTa-v3 model."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to redact (inline).  Mutually exclusive with --file.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a file whose contents should be redacted.",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="small",
        choices=[*MODEL_REGISTRY.keys()],
        help="Pre-trained model variant to use.  Default: %(default)s.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum confidence score for keeping an entity.  Default: %(default)s.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="If provided, the structured JSON response is written to this file "
        "instead of printed to stdout.",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=0.5,
        help="Overlap ratio between consecutive chunks (0-1).  Default: %(default)s.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum model input length in tokens (incl. special tokens).  "
        "Default: %(default)s.",
    )

    return parser.parse_args()

def _validate_args(args: argparse.Namespace) -> str:
    # Validate input source.
    if not args.text and not args.file:
        print("Error: Must provide either --text or --file.", file=sys.stderr)
        sys.exit(1)
    if args.text and args.file:
        print("Error: --text and --file are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = args.text

    if not text:
        print("Error: Input text is empty.", file=sys.stderr)
        sys.exit(1)
        
    return text

if __name__ == "__main__":
    main()
