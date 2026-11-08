"""Write the deterministic slow-path contract report; failure returns nonzero."""

import argparse
import json
from pathlib import Path

from hybrid_drive.slow.replay import run_replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional machine-readable JSON report path")
    args = parser.parse_args(argv)
    report = run_replay()
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
