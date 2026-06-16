"""
Publishing and Benchmarking Utilities.

Helper utilities to evaluate, benchmark, and generate model card metadata for
fine-tuned PII redaction models.

Learnings & Context:
1. GPU vs. CPU timing latency measurements:
   Benchmarking models correctly requires pre-warming the GPU via a small set of warmup
   passes to avoid measuring library load times, followed by calling `torch.cuda.synchronize()`
   prior to logging timestamps to prevent async execution timing skew.
2. Best checkpoint lookup:
   Retrieves model checkpoint paths based on the lowest validation loss or highest F1 score
   recorded inside `trainer_state.json`, rather than simply selecting the final epoch's weights.
3. Parameter-size vs. layer count latency impacts:
   Note that despite having half the parameters of the base model, DeBERTa-v3-XSmall shares
   the same depth (12 layers). sequential forward passes dominate GPU latency, meaning small
   is often faster due to its 6-layer architecture, highlighting that parameter counts
   do not always correlate directly with GPU inference speeds.
"""

import json
import time
from pathlib import Path
from typing import Optional, Tuple, Any

import torch
from transformers import AutoModelForTokenClassification, pipeline


# Supported entities list mapping index to name
ENTITY_LABELS = [
    "BOD", "BUILDING", "CITY", "COUNTRY", "DATE", "DRIVERLICENSE",
    "EMAIL", "GEOCOORD", "GIVENNAME1", "GIVENNAME2", "IDCARD", "IP",
    "LASTNAME1", "LASTNAME2", "LASTNAME3", "PASS", "PASSPORT",
    "POSTCODE", "SECADDRESS", "SEX", "SOCIALNUMBER", "STATE",
    "STREET", "TEL", "TIME", "TITLE", "USERNAME",
]


def benchmark(repo_or_path: str, text: str, n_warmup: int = 5, n_runs: int = 50) -> float:
    """
    Measure model latency (in milliseconds) on GPU if CUDA is available, otherwise CPU.
    
    Includes warmup cycles and synchronization to guarantee stable measurements.
    """
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("token-classification", model=repo_or_path, device=device)
    
    # Warmup
    for _ in range(n_warmup):
        pipe(text)
    
    if device >= 0:
        torch.cuda.synchronize()
        
    start = time.perf_counter()
    for _ in range(n_runs):
        pipe(text)
        if device >= 0:
            torch.cuda.synchronize()
            
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms per sample


def get_best_checkpoint(model_dir: Path) -> Tuple[Path, dict, dict]:
    """
    Parse trainer_state.json inside the checkpoints directory to find the best performing checkpoint.
    
    Returns:
        (best_checkpoint_path, trainer_state_dict, best_log_entry_dict)
    """
    checkpoints = [
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint")
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint directories found in {model_dir}")
        
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


def model_stats(repo_or_path: str) -> Tuple[float, float]:
    """
    Calculate parameters (in millions) and active GPU memory allocations (in MB).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(repo_or_path).to(device)
    
    # Params
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # VRAM allocated
    vram_mb = 0.0
    if device == "cuda":
        torch.cuda.synchronize()
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        
    return params, vram_mb


def load_training_args(checkpoint: Path) -> Any:
    """Load TrainingArguments saved alongside a checkpoint."""
    return torch.load(checkpoint / "training_args.bin", weights_only=False)


def get_first_checkpoint_args(model_dir: Path) -> Any:
    """Load TrainingArguments from the very first checkpoint (Stage 1)."""
    checkpoints = [
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint")
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint directories found in {model_dir}")
        
    first = min(checkpoints, key=lambda d: int(d.name.split("-")[1]))
    return torch.load(first / "training_args.bin", weights_only=False)


def make_entity_table(log: dict, worst: Optional[int] = None) -> str:
    """
    Build a markdown table summarizing F1 scores per target entity.
    
    Optionally filters to the worst-performing entities if `worst` is specified.
    """
    rows = []
    for label in ENTITY_LABELS:
        f1 = log.get(f"eval_{label}_f1", None)
        sup = log.get(f"eval_{label}_support", None)
        if f1 is not None:
            rows.append((label, f1, sup))
            
    if worst is not None:
        rows = sorted(rows, key=lambda x: x[1])[:worst]

    header = "| Entity | F1 | Support |\n|--------|------|---------|\n"
    body = "".join(f"| {r} | {f:.4f} | {s} |\n" for r, f, s in rows)
    return header + body
