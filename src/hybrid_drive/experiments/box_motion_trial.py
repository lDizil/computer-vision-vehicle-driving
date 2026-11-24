"""Compare three motion methods on identical frozen detections before opening labels."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from math import atan2
from pathlib import Path
from time import perf_counter

import torch
from safetensors.torch import load_file

from hybrid_drive.contracts.trajectory import Pose2D
from hybrid_drive.experiments.bounded_io import require_hex
from hybrid_drive.experiments.box_motion_inputs import load_motion_inputs, load_motion_snapshot
from hybrid_drive.experiments.fast_capture import artifact_hash, load_json
from hybrid_drive.experiments.io import write_json
from hybrid_drive.fast.forecast_proposals import ForecastObject, propose_from_forecast
from hybrid_drive.fast.metric_bev import BevGrid, past_forward_speed, temporal_bev
from hybrid_drive.fast.metric_forecast import MetricForecastHead, decode_boxes
from hybrid_drive.fast.temporal_box_motion import MotionFrame, associate_box_motion
from hybrid_drive.neural.pretrained_alignment import load_alignment_vision
from hybrid_drive.neural.surround_features import SharedSurroundEncoder


MINI_HEAD_REVISION = "d93ae54ba71d7b8873c6519e6e204a45c72ddfa8"
MINI_VISION_SHA256 = "bb2e0946286a9c12cc67fdec943db76f8854c04e35a38d85f92429639abcff7d"
MINI_VISION_THRESHOLD = .2


@torch.no_grad()
def predict_box_motion(input_root: Path, model_root: Path, frozen_fit: Path, output: Path, *, revision: str) -> None:
    require_hex(revision, 40, "revision")
    if output.exists():
        raise ValueError("Expected a new prediction directory")
    index = load_motion_inputs(input_root)
    fit = _load_frozen_head(frozen_fit)
    vision, source = load_alignment_vision(model_root)
    identity = f'{source["source"]["repository"]}@{source["source"]["revision"]}'
    encoder = SharedSurroundEncoder(vision, encoder_id=identity)
    grid = BevGrid()
    observation = _ObservationCache(input_root, index["frames"], encoder)
    rows = []
    since = perf_counter()
    for anchor in index["anchors"]:
        quartet = [observation(token) for token in (*anchor["history_tokens"], anchor["sample_token"])]
        row = _predict_anchor(anchor["recording"], quartet, fit, grid)
        rows.append(row)
        print(f"Predicted {len(rows)}/{len(index['anchors'])}; {observation.encodes} camera batches", flush=True)
    output.mkdir(parents=True)
    write_json(output / "predictions.json", {
        "schema":"hybrid-drive-box-motion-predictions-v1",
        "revision":revision, "heldout_targets_opened":False,
        "observation_sha256":artifact_hash(input_root / "index.json"),
        "grid":asdict(grid), "vision":source, "source":index["source"],
        "head_revision":MINI_HEAD_REVISION, "head_sha256":MINI_VISION_SHA256,
        "threshold":MINI_VISION_THRESHOLD, "rows":rows,
    })
    print(f"Completed target-free inference in {perf_counter()-since:.2f}s", flush=True)


class _ObservationCache:
    def __init__(self, root, frames, encoder):
        self.root, self.frames, self.encoder = root, frames, encoder
        self.cache = OrderedDict()
        self.encodes = 0

    def __call__(self, token):
        if token not in self.cache:
            self.cache[token] = self.encoder.encode(load_motion_snapshot(self.root, token, self.frames[token]))
            self.encodes += 1
            if len(self.cache) > 5:
                self.cache.popitem(last=False)
        self.cache.move_to_end(token)
        return self.cache[token]


def _predict_anchor(recording, quartet, model, grid):
    detected = []
    for earlier, later in zip(quartet, quartet[1:]):
        values, visible = temporal_bev(later, earlier, grid)
        values = values.to(torch.float16).to(torch.float32)
        boxes = tuple(ForecastObject(**asdict(b)) for b in decode_boxes(
            model(values.unsqueeze(0)), visible, grid, threshold=MINI_VISION_THRESHOLD))
        s = later.snapshot
        detected.append(MotionFrame(recording, s.ref_time_us, s.ego_to_world, boxes))
    past2, past1, current = detected
    tracked = associate_box_motion(current, past1, past2)
    pose = quartet[-1].snapshot.ego_to_world
    current_id = quartet[-1].snapshot.snapshot_id
    speed = past_forward_speed(quartet[-1].snapshot, quartet[-2].snapshot)
    return {
        "recording":recording, "sample_token":current_id, "ref_time_us":current.ref_time_us,
        "snapshot_sha256":quartet[-1].snapshot.content_sha256,
        "history_times_us":[f.snapshot.ref_time_us for f in quartet[:-1]],
        "boxes":[asdict(box) for box in current.boxes],
        "track_velocity_xy":[list(box.velocity_xy) for box in tracked.objects],
        "track_evidence":[asdict(row) for row in tracked.diagnostics],
        "supported_cells":torch.nonzero(visible.flatten()).flatten().tolist(),
        "routing":_routing(current_id, current.ref_time_us, pose, speed, current.boxes, tracked.objects),
    }


def _routing(sample_token, time_us, pose, speed, learned, associated):
    if not -.10 <= speed <= 60:
        return None
    current = Pose2D(float(pose[0,3]),float(pose[1,3]),atan2(pose[1,0],pose[0,0]))
    standstill = tuple(ForecastObject(**(asdict(b) | {"velocity_xy":(0.,0.)})) for b in learned)
    choices = {}
    for name, prediction in (("frozen_learned",learned),("stationary",standstill),("associated",associated)):
        plans = propose_from_forecast(sample_token, time_us, current, max(0,speed), prediction)
        choices[name] = {
            "ordered":[c.candidate_id for c in plans.plan.candidates],
            "all_overlap":plans.all_candidates_overlap,
        }
    return choices


def _load_frozen_head(path):
    receipt = load_json(path / "fit.json")
    vision = receipt["arms"]["vision"]
    weights = path / "vision.safetensors"
    if (receipt["revision"] != MINI_HEAD_REVISION or receipt["channels"] != 773
            or receipt["test_targets_opened"] is not False
            or vision["threshold"] != MINI_VISION_THRESHOLD
            or vision["weights_sha256"] != MINI_VISION_SHA256 or artifact_hash(weights) != MINI_VISION_SHA256):
        raise ValueError("Expected the exact frozen visual head from Mini pilot #42")
    model = MetricForecastHead(773)
    model.load_state_dict(load_file(weights), strict=True)
    return model.eval()
