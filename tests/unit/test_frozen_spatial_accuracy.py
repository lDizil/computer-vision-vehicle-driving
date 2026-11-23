from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from hybrid_drive.experiments.frozen_spatial_accuracy import load_frozen_spatial_accuracy
from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.neural_checkpoint import module_state_hash, save_neural_weights
from hybrid_drive.experiments.protocol import content_hash
from hybrid_drive.neural.spatial_box_refiner import SpatialBoxRefiner


SCOPE = "openscene_front_spatial_confidence_and_refinement_v1"
REVISION = "a" * 40
REFERENCE = "c" * 64


class _Prior:
    receipt = {"vision": "pin", "reference": REFERENCE}

    def predict_spatial(self, maps):
        bbox = torch.tensor([0.03, 0.25, 0.15, 0.6], dtype=torch.float32)
        bands = torch.arange(4.0)[:, None] * torch.tensor([0.25, 0, 0.25, 0])
        return SimpleNamespace(
            boxes_xyxy=(bbox + bands)[None].expand(len(maps), -1, -1),
            presence_logits=maps.mean(dim=(1, 2, 3, 4))[:, None].expand(-1, 4),
        )


class _AccuracyArtifact:
    def __init__(self, root):
        self.root = root
        self.vision = {"vision_model": "fixture-v1"}
        self.prior = _Prior()
        self.manifest = {
            "revision": REVISION, "scope": SCOPE, "runtime_changed": False,
            "data_hashes": {"reference": REFERENCE, "evaluation": "d" * 64, "previous_final_reused_as_dev": "e" * 64},
            "vision": deepcopy(self.vision), "prior_receipt": deepcopy(self.prior.receipt), "recipe": {"seeds": [0, 1]},
        }
        threshold = {"absent_below": -0.5, "present_at_or_above": 0.25}
        self.result = {
            "revision": REVISION, "scope": SCOPE, "status": "passed", "frozen_vision_unchanged": True,
            "selected_refinement_seed": 0, "calibrated_policy": dict(threshold), "refiners": [],
        }
        self.gate = {
            "schema": "hybrid-drive-spatial-accuracy-gate-v1", "data_hashes": dict(self.manifest["data_hashes"]),
            "calibrated_policy": threshold, "selected_refinement_seed": 0, "refiners": [],
        }
        self.states = []
        for seed in range(2):
            model = SpatialBoxRefiner(4).eval()
            with torch.no_grad():
                model.head[-1].bias.copy_(torch.tensor([0.08 * (seed + 1), -0.1, 0.15, 0.1]))
            meta = {
                "scope": SCOPE, "revision": REVISION, "data_hashes": self.manifest["data_hashes"],
                "prior_receipt_hash": content_hash(self.prior.receipt), "recipe": self.manifest["recipe"], "seed": seed,
            }
            filename = f"refined-boxes-seed-{seed}.safetensors"
            checkpoint = save_neural_weights(root / filename, {"refiner": model}, metadata=meta)
            state = module_state_hash(model)
            self.gate["refiners"].append({"seed": seed, "sha256": checkpoint["sha256"], "state_hash": state})
            self.result["refiners"].append({
                "seed": seed, "state_hash": state, "checkpoint_metadata": deepcopy(meta),
                "checkpoint": {**checkpoint, "file": filename}, "restored_bitwise_before_final": True,
            })
            self.states.append(model)

    def publish(self):
        self.result.update(manifest_hash=content_hash(self.manifest), holdout_gate_hash=content_hash(self.gate))
        for name, data in (("result", self.result), ("manifest", self.manifest), ("frozen-policy", self.gate)):
            write_json(self.root / f"{name}.json", data)
        return hashlib.sha256((self.root / "result.json").read_bytes()).hexdigest()

    def load(self, digest=None, **overrides):
        return load_frozen_spatial_accuracy(self.root, **{
            "expected_result_sha256": digest if digest is not None else self.publish(),
            "reference_dataset_hash": REFERENCE, "vision_receipt": self.vision,
            "visual_dim": 4, "prior": self.prior, **overrides,
        })


def test_restores_acknowledged_seed_without_changing_prior_scores(tmp_path):
    old = _AccuracyArtifact(tmp_path)
    model = old.load()
    features = torch.rand(2, 1, 4, 6, 8)
    raw = old.prior.predict_spatial(features)
    scores, boxes = model.predict(features)
    torch.testing.assert_close(scores, raw.presence_logits, atol=0, rtol=0)
    torch.testing.assert_close(boxes, old.states[0](features[:, 0], raw.boxes_xyxy), atol=0, rtol=0)
    assert not boxes.requires_grad and model.policy.present_at_or_above == 0.25
    assert model.receipt["selected_refinement_seed"] == 0
    model.verify_unchanged()


def test_refiner_absence_fallback_keeps_base_geometry(tmp_path):
    old = _AccuracyArtifact(tmp_path)
    old.result["selected_refinement_seed"] = None
    old.gate["selected_refinement_seed"] = None
    model = old.load()
    features = torch.rand(2, 1, 4, 6, 8)
    _, boxes = model.predict(features)
    assert model.receipt["refiner_checkpoint_sha256"] is None
    torch.testing.assert_close(boxes, old.prior.predict_spatial(features).boxes_xyxy, atol=0, rtol=0)


def test_rejects_changed_result_or_bound_policy(tmp_path):
    old = _AccuracyArtifact(tmp_path)
    digest = old.publish()
    with pytest.raises(ValueError, match="result byte hash"):
        old.load("7" * 64)
    old.gate["calibrated_policy"]["absent_below"] = -1.0
    write_json(tmp_path / "frozen-policy.json", old.gate)
    with pytest.raises(ValueError, match="Frozen document hash"):
        old.load(digest)


@pytest.mark.parametrize("failure", ["different_vision", "different_reference", "offline_status", "invalid_thresholds", "filename"])
def test_rejects_inconsistent_trusted_receipt_even_if_updated_hash_is_supplied(tmp_path, failure):
    old = _AccuracyArtifact(tmp_path)
    overrides = {}
    if failure == "different_vision":
        overrides["vision_receipt"] = {"vision_model": "different"}
    elif failure == "different_reference":
        overrides["reference_dataset_hash"] = "0" * 64
    elif failure == "offline_status":
        old.result["status"] = "failed"
    elif failure == "invalid_thresholds":
        old.result["calibrated_policy"] = dict(absent_below=0.75, present_at_or_above=0.25)
        old.gate["calibrated_policy"] = dict(old.result["calibrated_policy"])
    elif failure == "filename":
        old.result["refiners"][0]["checkpoint"]["file"] = "../refined-boxes-seed-0.safetensors"
    with pytest.raises(ValueError):
        old.load(**overrides)


def test_weight_byte_validation_and_runtime_mutation_are_separate_guards(tmp_path):
    old = _AccuracyArtifact(tmp_path)
    digest = old.publish()
    model = old.load(digest)
    with torch.no_grad():
        model._refiner.head[-1].bias[0] += 0.25
    with pytest.raises(ValueError, match="state changed"):
        model.verify_unchanged()
    path = tmp_path / old.result["refiners"][0]["checkpoint"]["file"]
    payload = path.read_bytes()
    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 0x10]))
    with pytest.raises(ValueError, match="expected SHA256"):
        old.load(digest)
