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

import torch

# Ensure the project root is on sys.path so that pii_redaction is importable
# when running this script directly (e.g. `uv run python cli/redact.py`).
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pii_redaction.redactor import PIIRedactor, MODEL_REGISTRY  # noqa: E402

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
        device= "cuda" if torch.cuda.is_available else -1
    )

    response = redactor.predict(text, threshold=args.threshold)
    output = response.model_dump_json(indent=2)

    if args.output_path:
        Path(args.output_path).write_text(output, encoding="utf-8")
        print(f"Output (json) saved to {args.output_path}")
    else:
        print(output)

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