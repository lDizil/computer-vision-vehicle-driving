"""Optimize the trainable core through miniature frozen CPU fixture models, without downloads."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.neural_core_smoke import run_neural_core_smoke


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_dir, result = run_neural_core_smoke(
        args.output_root, revision=args.revision, steps=args.steps, seed=args.seed
    )
    print("Neural core report:", run_dir / "report.html")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
