"""Report end-to-end sharing up to Fast's input and the real ROI tokenizer."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageDraw, ImageOps
import torch

from hybrid_drive.contracts.sensors import CameraSnapshot
from hybrid_drive.experiments.io import write_json
from hybrid_drive.neural.prior_tokenizer import PriorTokenizer
from hybrid_drive.neural.surround_features import SharedSurroundEncoder, route_surround_scene


@torch.no_grad()
def run_surround_probe(snapshot: CameraSnapshot, encoder: SharedSurroundEncoder, output: Path,
                       *, revision: str, encoder_provenance: dict, source: dict) -> dict:
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be the exact source commit")
    fast_tensor = None
    bridge_digest = None
    started = perf_counter()

    def fast_input(features):
        nonlocal fast_tensor
        fast_tensor = features.feature_maps
        return _feature_hash(fast_tensor)

    def bridge_input(features, fast_digest):
        nonlocal bridge_digest
        maps = features.feature_maps
        bridge_digest = _feature_hash(maps)
        if maps is not fast_tensor or bridge_digest != fast_digest:
            raise RuntimeError("Fast and Bridge did not receive the same feature tensor")
        return _tokenize_views(features)

    features, fast_digest, bridge = route_surround_scene(snapshot, encoder, fast_input, bridge_input)
    elapsed = perf_counter() - started
    result = {
        "schema": "hybrid-drive-surround-feature-probe-v1",
        "scope": "stage1_real_rasters_shared_features_not_driving",
        "source_revision": revision,
        "source": source,
        "encoder": encoder_provenance,
        "encoder_id": features.encoder_id,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_sha256": snapshot.content_sha256,
        "rig_id": snapshot.rig_id,
        "reference_time_us": snapshot.ref_time_us,
        "timing_basis": snapshot.timing.value,
        "measured_skew_us": snapshot.measured_skew_us,
        "max_skew_us": snapshot.max_skew_us,
        "ego_motion_available": snapshot.motion is not None,
        "feature_shape": list(features.feature_maps.shape),
        "feature_dtype": str(features.feature_maps.dtype),
        "fast_input_sha256": fast_digest,
        "bridge_input_sha256": bridge_digest,
        "same_tensor_object": True,
        "bridge_token_shape": list(bridge.tokens.shape),
        "bridge_token_sha256": _feature_hash(bridge.tokens),
        "bridge_input_kind": "one_fixed_roi_per_view_with_camera_forward_direction; untrained tokenizer",
        "executed_fast_planner": False,
        "executed_3d_prediction": False,
        "executed_slow_vqa": False,
        "probe_wall_s": elapsed,
        "views": [
            {
                "direction": view.direction.value,
                "source_camera": view.calibration.name,
                "capture_time_us": view.capture_us,
                "source_rgb_sha256": view.pixels_sha256,
                "source_size_wh": [view.calibration.width, view.calibration.height],
                "preprocessed_size_wh": [calibration.width, calibration.height],
                "preprocessed_intrinsic": calibration.intrinsic.tolist(),
                "ego_to_camera": calibration.ego_to_camera.tolist(),
                "feature_sha256": _feature_hash(features.feature_maps[0, i]),
            }
            for i, (view, calibration) in enumerate(zip(snapshot.views, features.calibrations))
        ],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_contact_sheet(output / "cameras.png", snapshot)
    write_json(output / "result.json", result)
    return result


def _tokenize_views(features):
    maps = features.feature_maps
    count = len(features.calibrations)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        tokenizer = PriorTokenizer(maps.shape[2], geometry_dim=2, token_dim=32, hidden_dim=32,
                                   geometry_feature_names=("camera_forward_ego_x", "camera_forward_ego_y"))
    tokenizer = tokenizer.eval().to(device=maps.device, dtype=maps.dtype)
    rois = maps.new_tensor([0.20, 0.20, 0.80, 0.80]).repeat(1, count, 1)
    views = torch.arange(count, dtype=torch.long, device=maps.device).unsqueeze(0)
    # Camera optical +z is the third row of the reference ego -> camera rotation.
    heading = maps.new_tensor([[c.ego_to_camera[2, :2].tolist() for c in features.calibrations]])
    tokens = tokenizer(maps, rois, views, heading, torch.ones_like(views, dtype=torch.bool))
    if not torch.isfinite(tokens.tokens).all():
        raise RuntimeError("Bridge probe produced nonfinite tokens")
    return tokens


def _feature_hash(value):
    array = value.detach().cpu().numpy()
    return sha256(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")).hexdigest()


def _write_contact_sheet(path, snapshot):
    tile_width, tile_height, gap = 384, 244, 12
    sheet = Image.new("RGB", (tile_width * 3 + gap * 4, tile_height * 2 + gap * 3), "#152434")
    draw = ImageDraw.Draw(sheet)
    for i, view in enumerate(snapshot.views):
        x, y = gap + i % 3 * (tile_width + gap), gap + i // 3 * (tile_height + gap)
        draw.text((x + 5, y + 5), f"{i + 1} / {view.direction.value.upper()}  |  {view.calibration.name}", fill="#eef5ff")
        image = ImageOps.contain(Image.fromarray(view.rgb), (tile_width, tile_height - 27))
        sheet.paste(image, (x + (tile_width - image.width) // 2, y + 25))
    sheet.save(path)
