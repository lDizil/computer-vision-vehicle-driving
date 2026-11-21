"""Train the prepared real-data LOC pilot without downloads or driving controls."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.alignment_pilot import PilotRecipe, run_alignment_pilot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "model-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    root, result = run_alignment_pilot(
        args.data_root,
        args.model_root,
        args.output_root,
        revision=args.revision,
        recipe=PilotRecipe(args.epochs, args.batch_size, args.seed),
    )
    print(f"Alignment pilot: {result['status']} — {root / 'report.html'}")
    if result["status"] != "passed":
        print(result.get("error"))
    if result["status"] == "interrupted":
        return 130
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
