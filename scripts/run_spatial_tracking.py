"""Compare tracking identities using two label-free captures and frozen ground-truth files."""

import argparse
from pathlib import Path

from hybrid_drive.experiments.spatial_tracking_benchmark import TrackingTruthFile, run_tracking_benchmark
from hybrid_drive.experiments.spatial_tracking_capture import TRACKING_ROLE, reopen_tracking_capture
from hybrid_drive.experiments.temporal_spatial_capture import ROLES
from scripts.capture_spatial_tracking import add_tracking_inputs, tracking_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_tracking_inputs(parser)
    parser.add_argument("--tracking-capture-root", type=Path, required=True)
    parser.add_argument("--tracking-capture-sha256", required=True)
    for role in (*ROLES, TRACKING_ROLE):
        prefix = role.replace("_", "-")
        parser.add_argument(f"--{prefix}-truth", type=Path, required=True)
        parser.add_argument(f"--{prefix}-truth-sha256", required=True)
    cfg = parser.parse_args(argv)
    try:
        past, fresh, protected = tracking_inputs(cfg)
        frozen, _ = reopen_tracking_capture(
            cfg.tracking_capture_root, past, fresh, expected_capture_sha256=cfg.tracking_capture_sha256,
        )
        truth = {
            role: TrackingTruthFile(getattr(cfg, role + "_truth"), getattr(cfg, role + "_truth_sha256"))
            for role in (*ROLES, TRACKING_ROLE)
        }
        run, result, digest = run_tracking_benchmark(
            past, fresh, frozen, truth, cfg.output_root, revision=cfg.revision,
            tracking_capture_sha256=cfg.tracking_capture_sha256,
            protected_roots=(*protected, cfg.tracking_capture_root),
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Spatial tracking failed: {exc}\n")
    print(f"{result['status']}: {run / 'report.html'}")
    print(f"Tracking benchmark SHA256: {digest}")
    if result.get("error"):
        print(result["error"])
    return 130 if result["status"] == "interrupted" else int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
