from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_drive.experiments import vru_sam_mask as sam
from hybrid_drive.experiments.vru_metric_detector import depth_sample


HW = (432, 768)
BOX = (100, 100, 200, 200)
BOXES = (BOX, (250, 100, 270, 160), (20, 20, 40, 60))


class _Tensor:
    def __init__(self, values):
        self.values = values

    @property
    def shape(self):
        return self.values.shape

    def float(self):
        return _Tensor(self.values.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Batch(dict):
    def to(self, device):
        assert device == "cpu"
        return self


def _fake_sam(monkeypatch, *, wrong_shape=False):
    sizes = _Tensor(np.asarray([HW]))
    logits = _Tensor(np.zeros((1, len(BOXES), 1, 256, 256)))
    scores = _Tensor(np.asarray([[[.95], [.79], [np.nan]]]))
    masks = np.full((len(BOXES), 1, *HW), True, dtype=bool)

    class Processor:
        @classmethod
        def from_pretrained(cls, root, **kwargs):
            assert root == Path("known-local")
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()

        def __call__(self, *, images, input_boxes, return_tensors):
            assert images.size == HW[::-1]
            assert input_boxes == [[list(box) for box in BOXES]]
            assert return_tensors == "pt"
            return _Batch(original_sizes=sizes)

        def post_process_masks(self, prediction, original_sizes, *, binarize):
            assert prediction.shape == logits.shape
            assert original_sizes is sizes and binarize
            return [_Tensor(masks[:, 0] if wrong_shape else masks)]

    class Model:
        @classmethod
        def from_pretrained(cls, root, **kwargs):
            assert root == Path("known-local")
            assert kwargs == {
                "local_files_only": True, "trust_remote_code": False, "use_safetensors": True,
                "output_loading_info": True,
            }
            return cls(), {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, *, original_sizes, multimask_output):
            assert original_sizes is sizes and multimask_output is False
            return SimpleNamespace(iou_scores=scores, pred_masks=logits)

    monkeypatch.setattr(sam, "version", lambda name: "4.57.3")
    monkeypatch.setattr(sam, "_verified_model", lambda root, manifest: (Path("known-local"), {}))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(Sam2Processor=Processor, Sam2Model=Model))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    return sam.SamBoxMasks("known-local", "manifest", device="cpu")


def test_predict_runs_single_image_boxes_and_preserves_order_when_abstaining(monkeypatch):
    adapter = _fake_sam(monkeypatch)
    image = np.zeros((*HW, 3), dtype=np.uint8)
    assert adapter.predict(image, []) == ()
    observed, below_floor, missing_score = adapter.predict(image, BOXES)
    assert observed.shape == HW and observed.dtype == np.bool_ and observed.all()
    assert below_floor is missing_score is None
    with pytest.raises(ValueError, match="outside"):
        adapter.predict(image, [(-1, 1, 3, 4)])
    with pytest.raises(ValueError, match="rectified uint8"):
        adapter.predict(image[:, :-1], BOXES)


def test_predict_refuses_missing_single_mask_axis(monkeypatch):
    adapter = _fake_sam(monkeypatch, wrong_shape=True)
    with pytest.raises(ValueError, match="boolean SAM 2 masks"):
        adapter.predict(np.zeros((*HW, 3), dtype=np.uint8), BOXES)


def test_same_depth_statistics_as_no_mask_if_all_interior_is_foreground():
    ys, xs = np.indices(HW)
    depth = (6 + xs / 100 + ys / 1000).astype(np.float32)
    assert sam.masked_depth_sample(depth, BOX, np.ones(HW, dtype=bool)) == depth_sample(depth, BOX)


def test_mask_selects_foreground_and_uses_box_interior_only():
    depth = np.full(HW, 28., dtype=np.float32)
    depth[122:170, 147:162] = 11.
    depth[100:110, 100:200] = 2.
    mask = np.zeros(HW, dtype=bool)
    mask[122:170, 147:162] = mask[100:110, 100:200] = True

    measured = sam.masked_depth_sample(depth, BOX, mask)
    assert measured.meters_z == 11. and measured.spread_z == 0.
    assert depth_sample(depth, BOX) is None  # foreground + background spread exceeds limit


def test_masked_sample_abstains_on_tiny_support_invalid_or_mixed_depth():
    depth = np.full(HW, 10., dtype=np.float32)
    mask = np.zeros(HW, dtype=bool)
    assert sam.masked_depth_sample(depth, BOX, None) is None
    mask[122:124, 130:134] = True
    assert sam.masked_depth_sample(depth, BOX, mask) is None

    mask[:] = True
    depth[123:167, 131:165] = np.nan
    assert sam.masked_depth_sample(depth, BOX, mask) is None
    depth[123:167, 131:165] = 30.
    assert sam.masked_depth_sample(depth, BOX, mask) is None
    with pytest.raises(ValueError, match="boolean mask"):
        sam.masked_depth_sample(depth, BOX, mask.astype(np.uint8))


def test_modified_manifest_rejected_before_reading_model_files(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{\n}\n")
    with pytest.raises(ValueError, match="pinned SAM 2"):
        sam._verified_model(tmp_path / "model-does-not-exist", manifest)
