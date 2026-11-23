"""Read-only restoration of the previously validated calibrated spatial head."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from hybrid_drive.assets.files import check_components
from hybrid_drive.neural.spatial_box_refiner import SpatialBoxRefiner
from .bounded_io import decode_json, read_bounded, require_hex
from .neural_checkpoint import load_neural_weights, module_state_hash
from .protocol import content_hash
from .spatial_decision_policy import ZoneDecisionPolicy


_ACCURACY_SCOPE = "openscene_front_spatial_confidence_and_refinement_v1"
_GATE_SCHEMA = "hybrid-drive-spatial-accuracy-gate-v1"
_DOCUMENT_LIMIT = 256 * 1024
_MAX_REFINER_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class FrozenSpatialAccuracy:
    _prior: object
    _refiner: SpatialBoxRefiner | None
    policy: ZoneDecisionPolicy
    _receipt: dict

    @property
    def receipt(self):
        return deepcopy(self._receipt)

    @torch.no_grad()
    def predict(self, maps):
        spatial = self._prior.predict_spatial(maps)
        boxes = spatial.boxes_xyxy
        if self._refiner is not None:
            boxes = self._refiner(maps[:, 0], boxes)
        return spatial.presence_logits, boxes

    def verify_unchanged(self):
        if self._refiner is None:
            return
        if (
            self._refiner.training
            or any(p.requires_grad or p.grad is not None for p in self._refiner.parameters())
            or module_state_hash(self._refiner) != self._receipt["refiner_state_hash"]
        ):
            raise ValueError("Frozen refiner state changed or requires gradients")


def load_frozen_spatial_accuracy(
    root, *, expected_result_sha256, reference_dataset_hash, vision_receipt, visual_dim, prior,
):
    require_hex(expected_result_sha256, 64, "expected_result_sha256")
    require_hex(reference_dataset_hash, 64, "reference_dataset_hash")
    root = Path(root).absolute()
    check_components(root)
    raw = read_bounded(root / "result.json", _DOCUMENT_LIMIT)
    if hashlib.sha256(raw).hexdigest() != expected_result_sha256:
        raise ValueError("Accuracy result byte hash differs")
    result = _object(decode_json(raw), "accuracy result")
    manifest = _bound_json(root / "manifest.json", result.get("manifest_hash"))
    gate = _bound_json(root / "frozen-policy.json", result.get("holdout_gate_hash"))
    _validate_chain(result, manifest, gate, reference_dataset_hash, vision_receipt, prior.receipt)
    policy = _policy(gate["calibrated_policy"])
    selected = gate["selected_refinement_seed"]
    records = _check_refiner_receipts(result, manifest, gate)
    refiner, checkpoint_hash, state_hash = None, None, None
    if selected is not None:
        record = records[selected]
        refiner, checkpoint_hash, state_hash = _restore(root, record, visual_dim)
    return FrozenSpatialAccuracy(
        prior, refiner, policy,
        {
            "schema": "hybrid-drive-frozen-spatial-camera-head-v1",
            "accuracy_result_sha256": expected_result_sha256,
            "accuracy_revision": manifest["revision"],
            "accuracy_gate_hash": result["holdout_gate_hash"],
            "reference_dataset_hash": reference_dataset_hash,
            "accuracy_data_hashes": dict(manifest["data_hashes"]),
            "vision_hash": content_hash(vision_receipt),
            "prior_hash": content_hash(prior.receipt),
            "calibrated_policy": dict(gate["calibrated_policy"]),
            "selected_refinement_seed": selected,
            "refiner_checkpoint_sha256": checkpoint_hash,
            "refiner_state_hash": state_hash,
        },
    )


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"Expected object for {label}")
    return value


def _bound_json(path, expected):
    require_hex(expected, 64, path.name + " hash")
    answer = _object(decode_json(read_bounded(path, _DOCUMENT_LIMIT)), path.name)
    if content_hash(answer) != expected:
        raise ValueError(f"Frozen document hash differs: {path.name}")
    return answer


def _validate_chain(result, manifest, gate, reference, vision, prior):
    require_hex(manifest.get("revision"), 40, "accuracy revision")
    if (
        result.get("scope") != _ACCURACY_SCOPE
        or result.get("status") != "passed"
        or result.get("frozen_vision_unchanged") is not True
        or result.get("revision") != manifest.get("revision")
        or manifest.get("scope") != _ACCURACY_SCOPE
        or manifest.get("runtime_changed") is not False
        or gate.get("schema") != _GATE_SCHEMA
    ):
        raise ValueError("Expected passed offline calibrated spatial result")
    data = _object(manifest.get("data_hashes"), "calibration dataset hashes")
    if (
        data.get("reference") != reference
        or gate.get("data_hashes") != data
        or manifest.get("vision") != vision
        or manifest.get("prior_receipt") != prior
    ):
        raise ValueError("Accuracy weights or thresholds belong to different data, prior or vision")
    if (
        result.get("calibrated_policy") != gate.get("calibrated_policy")
        or result.get("selected_refinement_seed") != gate.get("selected_refinement_seed")
    ):
        raise ValueError("Accuracy result disagrees with frozen thresholds or geometry choice")
    selected = gate.get("selected_refinement_seed")
    if (
        "selected_refinement_seed" not in gate
        or (selected is not None and (type(selected) is not int or selected not in (0, 1)))
    ):
        raise ValueError("Invalid frozen refiner choice")
    if _object(manifest.get("recipe"), "accuracy recipe").get("seeds") != [0, 1]:
        raise ValueError("Calibration artifact does not acknowledge the two fixed refiner seeds")


def _policy(candidate):
    if not isinstance(candidate, dict) or candidate.keys() != {"absent_below", "present_at_or_above"}:
        raise ValueError("Frozen thresholds must include exactly both confidence bounds")
    return ZoneDecisionPolicy(**candidate)


def _check_refiner_receipts(result, manifest, gate):
    rows, frozen = result.get("refiners"), gate.get("refiners")
    if not isinstance(rows, list) or not isinstance(frozen, list) or len(rows) != 2 or len(frozen) != 2:
        raise ValueError("Expected two dev candidate receipts")
    records = {}
    for seed, (row, proof) in enumerate(zip(rows, frozen)):
        row, proof = _object(row, "refiner result"), _object(proof, "refiner gate receipt")
        checkpoint = _object(row.get("checkpoint"), "refiner checkpoint")
        expected_metadata = {
            "scope": _ACCURACY_SCOPE,
            "revision": manifest["revision"],
            "data_hashes": manifest["data_hashes"],
            "prior_receipt_hash": content_hash(manifest["prior_receipt"]),
            "recipe": manifest["recipe"],
            "seed": seed,
        }
        require_hex(proof.get("sha256"), 64, "refiner checkpoint digest")
        require_hex(proof.get("state_hash"), 64, "refiner trained-state digest")
        if (
            type(row.get("seed")) is not int or row["seed"] != seed
            or type(proof.get("seed")) is not int or proof["seed"] != seed
            or row.get("restored_bitwise_before_final") is not True
            or row.get("checkpoint_metadata") != expected_metadata
            or row.get("state_hash") != proof["state_hash"]
            or checkpoint.get("file") != f"refined-boxes-seed-{seed}.safetensors"
            or checkpoint.get("sha256") != proof["sha256"]
            or type(checkpoint.get("size_bytes")) is not int or not 0 < checkpoint["size_bytes"] <= _MAX_REFINER_BYTES
        ):
            raise ValueError("Refiner receipt identity, size or metadata differs")
        records[seed] = (checkpoint, expected_metadata, proof["state_hash"])
    return records


def _restore(root, record, visual_dim):
    checkpoint, metadata, state = record
    path = root / checkpoint["file"]
    check_components(path)
    if not path.is_file() or path.stat().st_size != checkpoint["size_bytes"]:
        raise ValueError("Frozen refiner file size differs")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = SpatialBoxRefiner(visual_dim).float().requires_grad_(False).eval()
    load_neural_weights(
        path, {"refiner": model}, expected_sha256=checkpoint["sha256"], expected_metadata=metadata,
    )
    if module_state_hash(model) != state:
        raise ValueError("Restored refiner differs from the acknowledged trained state")
    return model, checkpoint["sha256"], state
