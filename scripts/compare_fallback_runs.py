import argparse
from pathlib import Path

from hybrid_drive.experiments.fallback_comparison import compare_fallback_runs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare fixed-input longitudinal and curved fallback runs")
    for name in ("longitudinal-root", "curved-root", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    run_dir, _ = compare_fallback_runs(args.longitudinal_root, args.curved_root, args.output_root)
    print("Paired report:", run_dir / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
