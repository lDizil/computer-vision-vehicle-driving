from copy import deepcopy
import hashlib
import json
from dataclasses import replace

import pytest
import torch

from hybrid_drive.experiments.io import write_json
from hybrid_drive.experiments.neural_checkpoint import save_neural_weights
from hybrid_drive.experiments.proposal_prior_source import load_spatial_prior_source
from hybrid_drive.experiments.protocol import content_hash
from hybrid_drive.experiments.neural_checkpoint import module_state_hash
from hybrid_drive.neural.spatial_loc import SpatialLocProbe


_DATASET_HASH = "d" * 64
_FILE = "image_features-seed-0.safetensors"


class _Archive:
    def __init__(self, root):
        self.root = root
        root.mkdir()
        self.vision = {
            "source": {"repository": "fixture", "revision": "a" * 40},
            "runtime": {
                "torch": str(torch.__version__),
                "device": "cpu",
                "dtype": "float32",
                "attention": "eager",
            },
            "preprocessing": {"image_size": 224, "patch_size": 14},
        }
        self.manifest = {
            "schema": "hybrid-drive-spatial-experiment-v1",
            "scope": "openscene_spatial_loc_four_band_v2",
            "revision": "b" * 40,
            "dataset_hash": _DATASET_HASH,
            "vision": self.vision,
            "torch_version": str(torch.__version__),
            "recipe": {"epochs": 8, "batch_size": 8, "seeds": [0, 1]},
            "conditions": ["image_features", "constant_zero_features"],
            "training_input": "current_frame_features_only",
            "calibration": "dev_classwise_macro_balanced_accuracy_by_recording",
            "holdout": "open_only_after_all_weights_and_dev_thresholds_frozen",
            "split_recordings": {
                "train": ["train-recording"],
                "dev": ["dev-recording"],
                "holdout": ["held-recording"],
            },
        }
        metadata = {
            "scope": self.manifest["scope"],
            "dataset_hash": _DATASET_HASH,
            "revision": self.manifest["revision"],
            "vision_hash": content_hash(self.vision),
            "seed": 0,
            "condition": "image_features",
            "recipe": deepcopy(self.manifest["recipe"]),
            "train_order_hash": "c" * 64,
        }
        self.probe = SpatialLocProbe(384).eval()
        initial = module_state_hash(self.probe)
        with torch.no_grad():
            self.probe.extent.bias.add_(0.2)
        self.run = {
            "seed": 0,
            "condition": "image_features",
            "train_order_hash": metadata["train_order_hash"],
            "initial_weights_hash": initial,
            "checkpoint_state_bitwise_equal": True,
            "checkpoint_metadata": metadata,
            "dev_threshold": {"weighting": "classwise_macro_by_recording", "threshold": 100.0},
        }
        self.freeze = {
            "seed": 0,
            "condition": "image_features",
            "threshold": 100.0,
            "train_order_hash": "c" * 64,
        }
        self.result = {
            "scope": self.manifest["scope"],
            "revision": self.manifest["revision"],
            "status": "passed",
            "frozen_vision_unchanged": True,
            "runs": [self.run],
        }
        self.calibration = {
            "schema": "hybrid-drive-spatial-calibration-v1",
            "dataset_hash": _DATASET_HASH,
            "calibration_recordings": ["dev-recording"],
            "runs": [self.freeze],
        }
        self.write_checkpoint(self.probe)
        self.write_documents()

    def write_checkpoint(self, probe):
        path = self.root / _FILE
        path.unlink(missing_ok=True)
        record = save_neural_weights(path, {"spatial": probe}, metadata=self.run["checkpoint_metadata"])
        self.run.update(checkpoint=dict(record, file=_FILE), final_weights_hash=module_state_hash(probe))
        self.freeze["checkpoint_sha256"] = record["sha256"]

    def write_documents(self):
        for name, value in (("manifest", self.manifest), ("calibration", self.calibration)):
            write_json(self.root / (name + ".json"), value)
            self.result[name + "_hash"] = content_hash(value)
        path = self.root / "result.json"
        write_json(path, self.result)
        self.result_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def load(self, *, root=None, **overrides):
        args = {
            "expected_result_sha256": self.result_sha256,
            "dataset_hash": _DATASET_HASH,
            "vision_provenance": self.vision,
        }
        return load_spatial_prior_source(self.root if root is None else root, **(args | overrides))


@pytest.fixture
def archive(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return _Archive(tmp_path.resolve() / "run")


def test_pinned_prior_preserves_frames_and_quarters_without_creating_a_grad_path(archive):
    rng_state = torch.get_rng_state()
    source = archive.load()
    assert torch.equal(torch.get_rng_state(), rng_state)
    receipt = source.receipt
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt
    assert receipt["result_sha256"] == archive.result_sha256
    assert receipt["manifest_hash"] == archive.result["manifest_hash"]
    assert receipt["calibration_hash"] == archive.result["calibration_hash"]
    assert receipt["vision_hash"] == content_hash(archive.vision)
    assert receipt["source_revision"] == archive.manifest["revision"]
    assert receipt["checkpoint"] == archive.run["checkpoint"]
    assert (receipt["condition"], receipt["seed"]) == ("image_features", 0)
    assert receipt["calibration"]["applied_to_proposals"] is False
    receipt["checkpoint"]["file"] = "changed"
    assert source.receipt["checkpoint"]["file"] == _FILE

    features = torch.rand(2, 1, 384, 3, 8, generator=torch.Generator().manual_seed(11), requires_grad=True)
    boxes = source.predict(features)
    with torch.no_grad():
        expected = archive.probe(features[:, 0]).boxes_xyxy
    assert boxes.shape == (2, 4, 4) and not boxes.requires_grad
    assert torch.isfinite(boxes).all() and ((boxes >= 0) & (boxes <= 1)).all()
    assert (boxes[..., 2:] > boxes[..., :2]).all()
    assert torch.equal(((boxes[..., 0] + boxes[..., 2]) * 2).long(), torch.arange(4).expand(2, -1))
    torch.testing.assert_close(boxes, expected, atol=0, rtol=0)
    assert not torch.equal(boxes[0], boxes[1])
    torch.testing.assert_close(source.predict(features.flip(0)), boxes.flip(0))

    downstream = torch.nn.Linear(4, 1)
    downstream(boxes).sum().backward()
    assert downstream.weight.grad is not None and features.grad is None
    assert not source._probe.training
    assert all(not p.requires_grad and p.grad is None for p in source._probe.parameters())
    assert module_state_hash(source._probe) == archive.run["final_weights_hash"]


@pytest.mark.parametrize(
    ("filename", "error"),
    [
        ("result.json", "result differs from expected SHA256"),
        ("manifest.json", "manifest differs from the recorded content hash"),
        ("calibration.json", "calibration differs from the recorded content hash"),
        (_FILE, "Neural weights differ from expected SHA256"),
    ],
)
def test_tampered_artifact_is_rejected_under_the_same_result_anchor(archive, filename, error):
    path = archive.root / filename
    if filename == _FILE:
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    else:
        write_json(path, {**json.loads(path.read_bytes()), "changed": True})
    with pytest.raises(ValueError, match=error):
        archive.load()


def test_dataset_and_exact_active_vision_receipt_are_required(archive):
    with pytest.raises(ValueError, match="dataset hash differs"):
        archive.load(dataset_hash="e" * 64)
    different_vision = deepcopy(archive.vision)
    different_vision["preprocessing"]["patch_size"] = 16
    with pytest.raises(ValueError, match="vision provenance differs"):
        archive.load(vision_provenance=different_vision)


@pytest.mark.parametrize(
    ("document", "changes", "error"),
    [
        ("result", {"frozen_vision_unchanged": False}, "unchanged frozen vision"),
        ("result", {"revision": "e" * 40}, "source revision differs"),
        ("manifest", {"torch_version": "0.0.1"}, "runtime differ"),
        ("manifest", {"holdout": "not_frozen_before_holdout"}, "frozen before holdout"),
        ("manifest", {"recipe": {"epochs": 3, "batch_size": 8, "seeds": [0]}}, "checkpoint metadata differs"),
        ("run", {"seed": 1}, "one image_features seed 0 record"),
        ("run", {"condition": "constant_zero_features"}, "one image_features seed 0 record"),
        ("run", {"final_weights_hash": "f" * 64}, "differs from the recorded trained state"),
        ("calibration", {"calibration_recordings": ["held-recording"]}, "bind to its dev recordings"),
        ("freeze", {"threshold": 101.0}, "calibration differs from the pinned result run"),
    ],
)
def test_authenticated_records_still_need_consistent_training_provenance(archive, document, changes, error):
    getattr(archive, document).update(changes)
    archive.write_documents()
    with pytest.raises(ValueError, match=error):
        archive.load()


def test_wrong_head_is_rejected_even_with_internally_rehashed_metadata(archive):
    archive.write_checkpoint(SpatialLocProbe(384, hidden_dim=8))
    archive.write_documents()
    with pytest.raises(ValueError, match="checkpoint tensor differs"):
        archive.load()


@pytest.mark.parametrize(
    "filename", [f"../{_FILE}", f"nested/{_FILE}", f"/{_FILE}", "image_features-seed-1.safetensors"]
)
def test_result_cannot_select_another_checkpoint_path(archive, filename):
    archive.run["checkpoint"]["file"] = filename
    archive.write_documents()
    with pytest.raises(ValueError, match="pinned filename"):
        archive.load()


@pytest.mark.parametrize("name", ["result.json", "manifest.json", "calibration.json", _FILE])
def test_symlinked_artifact_is_rejected_even_with_matching_content(archive, name):
    path = archive.root / name
    target = archive.root / (name + ".original")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        archive.load()


def test_symlinked_run_directory_is_rejected(archive):
    alias = archive.root.parent / "alias"
    alias.symlink_to(archive.root, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        archive.load(root=alias)


def test_invalid_feature_values_fail_closed(archive):
    source = archive.load()
    with pytest.raises(ValueError, match="Expected CPU float32 feature maps"):
        source.predict(torch.zeros(1, 384, 2, 4))
    with pytest.raises(ValueError, match="Patch features must be finite"):
        source.predict(torch.full((1, 1, 384, 2, 4), float("nan")))


@pytest.mark.parametrize(
    ("corrupt", "error"),
    [
        (lambda boxes: boxes.flip(1), "quarter order"),
        (lambda boxes: torch.full_like(boxes, float("nan")), "finite positive-area"),
        (lambda boxes: boxes[..., [0, 1, 0, 3]], "finite positive-area"),
    ],
)
def test_invalid_probe_boxes_fail_closed(archive, monkeypatch, corrupt, error):
    source = archive.load()
    features = torch.zeros(1, 1, 384, 2, 4)
    reference = source.predict_spatial(features)
    invalid = replace(reference, boxes_xyxy=corrupt(reference.boxes_xyxy))
    monkeypatch.setattr(source._probe, "forward", lambda _: invalid)
    with pytest.raises(ValueError, match=error):
        source.predict(features)



def test_direct_spatial_prediction_requires_finite_presence_logits(archive, monkeypatch):
    prior = archive.load()
    features = torch.zeros(1, 1, 384, 2, 4)
    original = prior.predict_spatial(features)
    torch.testing.assert_close(original.boxes_xyxy, prior.predict(features), atol=0, rtol=0)
    invalid = replace(original, presence_logits=torch.full_like(original.presence_logits, float("nan")))
    monkeypatch.setattr(prior._probe, "forward", lambda _: invalid)
    with pytest.raises(ValueError, match="finite direct presence"):
        prior.predict_spatial(features)
