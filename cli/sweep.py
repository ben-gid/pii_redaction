"""
Nemotron Training Sweep.

Runs `cli.train` once per configuration below, each in a fresh process — wandb runs and
CUDA allocator state do not survive being reused across trainings.

The `*-old` runs replicate the hyperparameters published in `model_cards/` exactly
(stage 1: lr 1e-3, batch 32; stage 2: lr 2e-5, batch 8 for base / 16 for small+xsmall,
effective batch 16 either way), so the dataset is the only variable versus the released
ai4privacy models. The remaining runs probe knobs the nemotron dataset justifies
revisiting: it has 4.7x the training rows and 111 BIO labels instead of 55.

Epoch counts are not recorded in the model card. Stage 2 is capped at 3 epochs here
because 140k rows at effective batch 16 is already ~26k optimizer steps per epoch;
override with --stage2_epochs if you want longer.

Usage::

    python -m cli.sweep --dry_run              # print the commands, run nothing
    python -m cli.sweep                        # run all configurations in order
    python -m cli.sweep --only xsmall-nemotron-old
"""

import argparse
import subprocess
import sys

from pydantic import BaseModel

WANDB_PROJECT = "pii-redaction-nemotron"


class Run(BaseModel, frozen=True):
    name: str               # W&B run name and models/<name> output directory
    model_variant: str
    stage1_batch_size: int  # cards: 32 for every variant
    batch_size: int         # stage 2 per device; train.py derives accumulation from effective
    note: str
    effective_batch_size: int = 16
    stage1_lr: float = 1e-3
    stage2_lr: float = 2e-5
    stage1_epochs: int = 2
    stage2_epochs: int = 3

    def argv(self, wandb_project: str, no_wandb: bool) -> list[str]:
        cmd = [
            sys.executable, "-m", "cli.train",
            "--dataset", "nemotron",
            "--model_variant", self.model_variant,
            "--run_name", self.name,
            "--stage1_batch_size", str(self.stage1_batch_size),
            "--batch_size", str(self.batch_size),
            "--effective_batch_size", str(self.effective_batch_size),
            "--stage1_lr", str(self.stage1_lr),
            "--stage2_lr", str(self.stage2_lr),
            "--stage1_epochs", str(self.stage1_epochs),
            "--stage2_epochs", str(self.stage2_epochs),
            "--wandb_project", wandb_project,
        ]
        if no_wandb:
            cmd.append("--no_wandb")
        return cmd


# Cheapest first, so a broken sweep fails fast and trends show up early.
RUNS = [
    Run(
        name="xsmall-nemotron-old",
        model_variant="xsmall", stage1_batch_size=32, batch_size=16,
        note="model card hyperparameters, nemotron instead of ai4privacy",
    ),
    Run(
        name="small-nemotron-old",
        model_variant="small", stage1_batch_size=32, batch_size=16,
        note="model card hyperparameters, nemotron instead of ai4privacy",
    ),
    Run(
        name="base-nemotron-old",
        model_variant="base", stage1_batch_size=32, batch_size=8,
        note="model card hyperparameters (batch 8 x accum 2), nemotron instead of ai4privacy",
    ),
    Run(
        name="xsmall-nemotron-eff32-lr3e5",
        model_variant="xsmall", stage1_batch_size=32, batch_size=16, effective_batch_size=32,
        stage2_lr=3e-5,
        note="effective batch 32 with lr scaled to 3e-5 — less gradient noise across 111 labels",
    ),
    Run(
        name="small-nemotron-eff32-lr3e5",
        model_variant="small", stage1_batch_size=32, batch_size=16, effective_batch_size=32,
        stage2_lr=3e-5,
        note="effective batch 32 with lr scaled to 3e-5 — less gradient noise across 111 labels",
    ),
    Run(
        name="small-nemotron-warmhead",
        model_variant="small", stage1_batch_size=32, batch_size=16,
        stage1_lr=5e-4, stage1_epochs=3,
        note="gentler, longer head warmup — 111 classes is 2x the head the 1e-3/2ep recipe was tuned for",
    ),
]

# names double as output directories, so a duplicate silently overwrites a sibling run
assert len({run.name for run in RUNS}) == len(RUNS), "duplicate run name"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PII Redaction models on the nemotron dataset.")
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=[run.name for run in RUNS],
        default=None,
        help="Run only the named configurations. Defaults to all of them, in order."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the training commands without running them."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=WANDB_PROJECT,
        help="Weights & Biases project name."
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging for every run."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runs = [run for run in RUNS if not args.only or run.name in args.only]

    failed = []
    for i, run in enumerate(runs, 1):
        cmd = run.argv(args.wandb_project, args.no_wandb)
        print(f"\n[{i}/{len(runs)}] {run.name} — {run.note}")
        print(f"  {' '.join(cmd)}")

        if args.dry_run:
            continue

        # Keep going after a failure: a sweep that dies overnight on run 1 of 6 is worse
        # than one that reports which configurations broke.
        if subprocess.run(cmd).returncode != 0:
            print(f"  FAILED: {run.name}")
            failed.append(run.name)

    if failed:
        print(f"\n{len(failed)}/{len(runs)} runs failed: {', '.join(failed)}")
        sys.exit(1)

    print(f"\nSweep complete: {len(runs)} runs.")


if __name__ == "__main__":
    main()
