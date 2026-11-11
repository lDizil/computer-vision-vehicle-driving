"""Prepare a pinned, removable local VLM model directory; no inference or global installs."""

import argparse
import json
from pathlib import Path

from hybrid_drive.assets.slow import prepare_slow
from hybrid_drive.assets.slow_registry import MODEL_SPECS, resolve_model_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rate-limit-mib", type=float, default=4)
    parser.add_argument("--variant", choices=tuple(MODEL_SPECS), default="256m")
    args = parser.parse_args()
    name = resolve_model_key(args.variant).sources_filename
    sources = json.loads(Path(__file__).with_name(name).read_text())
    result = prepare_slow(args.root, sources, rate_limit_mib=args.rate_limit_mib)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
