"""Fit and compare bounded box refiners on captured zoom and global grids."""

import argparse
from pathlib import Path

import torch

from hybrid_drive.experiments.spatial_tracking_benchmark import TrackingTruthFile
from hybrid_drive.experiments.spatial_zoom_benchmark import run_zoom_benchmark
from hybrid_drive.experiments.spatial_zoom_capture import ZOOM_ROLES, reopen_zoom_capture
from scripts.capture_spatial_zoom import add_zoom_inputs, zoom_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_zoom_inputs(parser)
    parser.add_argument("--zoom-capture-root", type=Path, required=True)
    parser.add_argument("--zoom-capture-sha256", required=True)
    for role in ZOOM_ROLES:
        prefix = role.replace("_","-")
        parser.add_argument(f"--{prefix}-truth", type=Path, required=True)
        parser.add_argument(f"--{prefix}-truth-sha256", required=True)
    cfg = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        inputs = zoom_inputs(cfg)
        capture = reopen_zoom_capture(cfg.zoom_capture_root, inputs, expected_sha256=cfg.zoom_capture_sha256)
        truth = {role:TrackingTruthFile(getattr(cfg,role+"_truth"),getattr(cfg,role+"_truth_sha256"))
                 for role in ZOOM_ROLES}
        root, result, sha = run_zoom_benchmark(
            capture, truth, cfg.output_root, revision=cfg.revision, extra_inputs=(cfg.zoom_capture_root,),
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Zoom benchmark failed: {exc}\n")
    print(f"{result['status']}: {root / 'report.html'}")
    print(f"Zoom benchmark SHA256: {sha}")
    if result.get("error"):
        print(result["error"])
    return 130 if result["status"] == "interrupted" else int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
