import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hybrid_drive.neural.spatial_zoom_crops import encode_zoom_pair, make_zoom_windows, zoom_protocol


_SIDE = 224


class _FixtureVision(nn.Module):
    visual_dim = 3

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1), requires_grad=False)
        self.finish = nn.Identity()
        self.calls = []
        self.eval()

    def forward(self, images):
        self.calls.append(images.clone())
        # Known patch-center affine map, with a frame-level context term.
        # The term differs between a full-frame and a crop encoder call.
        pixels = images[:, 0]
        pooled = F.adaptive_avg_pool2d(pixels, (16, 16))
        return self.finish(pooled + pixels[:, :, :1, :1]).unsqueeze(1) * self.gain


def _centers(size):
    return (torch.arange(size, dtype=torch.float32) + 0.5) / size


def _plane(xs, ys, offset):
    x = xs.expand(len(ys), -1)
    y = ys[:, None].expand(-1, len(xs))
    return torch.stack((x * 0.30 + offset, y * 0.40 + offset, x * 0.25 + y * 0.20 + offset))


def _images(*offsets):
    grid = _centers(_SIDE)
    return torch.stack([_plane(grid, grid, o) for o in offsets]).unsqueeze(1)


def _one_visible():
    boxes = torch.zeros(1, 4, 4)
    boxes[0, 0] = torch.tensor([0.00, 0.02, 0.08, 0.10])
    return boxes, torch.tensor([[True, False, False, False]])


def test_square_window_covers_original_boxes_and_shifts_at_both_edges():
    boxes = torch.tensor([[
        [0.33, 0.20, 0.49, 0.26],
        [0.00, 0.02, 0.08, 0.10],
        [0.90, 0.89, 0.98, 1.00],
        [0.05, 0.10, 0.93, 0.90],
    ]])
    active = torch.ones(1, 4, dtype=torch.bool)
    window = make_zoom_windows(boxes, active)
    torch.testing.assert_close(window, torch.tensor([[
        [0.25, 0.07, 0.57, 0.39],
        [0.00, 0.00, 0.18, 0.18],
        [0.78, 0.78, 1.00, 1.00],
        [0.00, 0.00, 1.00, 1.00],
    ]]))
    assert (window[..., :2] <= boxes[..., :2]).all()
    assert (window[..., 2:] >= boxes[..., 2:]).all()
    active[0, 3] = False
    assert not make_zoom_windows(boxes, active)[0, 3].any()


def test_geometrically_aligned_features_use_distinct_encoder_inputs_in_exact_slot_order():
    offsets = (0.03, 0.17, 0.32)
    images = _images(*offsets)
    untouched = images.clone()
    boxes = torch.zeros(3, 4, 4)
    boxes[0, 2] = torch.tensor([0.48, 0.35, 0.60, 0.45])
    boxes[2, 0] = torch.tensor([0.15, 0.48, 0.24, 0.61])
    boxes[2, 3] = torch.tensor([0.68, 0.47, 0.78, 0.57])
    active = (boxes[..., 2:] > boxes[..., :2]).all(dim=-1)
    vision = _FixtureVision()

    windows, full, zoom = encode_zoom_pair(vision, images, boxes, active)

    assert full.shape == zoom.shape == (3, 4, 3, 4, 4)
    assert full.dtype == zoom.dtype == torch.float32
    assert not full.requires_grad and not zoom.requires_grad
    assert torch.equal(images, untouched)
    assert not full[~active].any() and not zoom[~active].any() and not windows[~active].any()
    assert [len(x) for x in vision.calls] == [2, 3]
    assert torch.equal(vision.calls[0], images[[0, 2]])
    for (frame, band), region in zip(((0, 2), (2, 0), (2, 3)), (
        (0.42, 0.28, 0.66, 0.52), (0.065, 0.415, 0.325, 0.675), (0.63, 0.42, 0.83, 0.62),
    )):
        left, top, right, bottom = region
        torch.testing.assert_close(windows[frame, band], torch.tensor(region))
        xs = left + (right - left) * _centers(4)
        ys = top + (bottom - top) * _centers(4)
        ideal = _plane(xs, ys, offsets[frame])
        original_anchor = _centers(_SIDE)[:1]
        crop_anchor_x = left + (right - left) * original_anchor
        crop_anchor_y = top + (bottom - top) * original_anchor
        torch.testing.assert_close(
            full[frame, band], ideal + _plane(original_anchor, original_anchor, offsets[frame]), atol=2e-6, rtol=0,
        )
        torch.testing.assert_close(
            zoom[frame, band], ideal + _plane(crop_anchor_x, crop_anchor_y, offsets[frame]), atol=2e-6, rtol=0,
        )
        assert not torch.equal(full[frame, band], zoom[frame, band])


def test_rgb_crop_samples_pixel_centers_with_edge_border_not_zero_padding():
    image = _images(0.2)
    boxes, active = _one_visible()
    vision = _FixtureVision()
    encode_zoom_pair(vision, image, boxes, active)
    crop = vision.calls[1][0, 0]
    # At .18 * .5/224 the interpolation lies before the first source pixel
    # center, so the first pixel is copied with border extension, not darkened.
    torch.testing.assert_close(crop[:, 0, 0], image[0, 0, :, 0, 0], atol=1e-7, rtol=0)
    edge = 0.18 * _centers(_SIDE)[-1:]
    torch.testing.assert_close(crop[:, -1, -1], _plane(edge, edge, 0.2)[:, 0, 0], atol=3e-7, rtol=0)


def test_no_active_slots_never_invoke_encoder_and_keep_exact_float32_zeros():
    vision = _FixtureVision()
    outputs = encode_zoom_pair(vision, _images(0.0, 0.1), torch.zeros(2, 4, 4), torch.zeros(2, 4, dtype=torch.bool))
    assert [tuple(t.shape) for t in outputs] == [(2, 4, 4), (2, 4, 3, 4, 4), (2, 4, 3, 4, 4)]
    assert all(t.dtype == torch.float32 and not t.any() for t in outputs)
    assert not vision.calls


def test_microbatch_bound_does_not_reorder_visible_frame_inputs():
    # Both global and crop batches exceed the per-call bound, while there
    # remains only one selected box per frame. The constants reveal mixups.
    offsets = tuple(i / 32 for i in range(9))
    rgb = torch.tensor(offsets).reshape(-1, 1, 1, 1, 1).expand(-1, 1, 3, _SIDE, _SIDE)
    boxes, active = _one_visible()
    vision = _FixtureVision()
    _, full, zoom = encode_zoom_pair(vision, rgb, boxes.repeat(9, 1, 1), active.repeat(9, 1))
    assert [len(batch) for batch in vision.calls] == [8, 1, 8, 1]
    expected = 2 * torch.tensor(offsets)
    torch.testing.assert_close(full[:, 0, 0, 0, 0], expected)
    torch.testing.assert_close(zoom[:, 0, 0, 0, 0], expected)


@pytest.mark.parametrize("problem", ["dtype", "grad", "outside", "nan", "reversed", "flat", "mask", "too_many"])
def test_invalid_box_or_mask_cannot_reach_encoding(problem):
    boxes, active = _one_visible()
    if problem == "dtype":
        boxes = boxes.double()
    elif problem == "grad":
        boxes.requires_grad_(True)
    elif problem == "outside":
        boxes[0, 0, 2] = 1.001
    elif problem == "nan":
        boxes[0, 3, 2] = float("nan")
    elif problem == "reversed":
        boxes[0, 0] = torch.tensor([0.1, 0.0, 0.09, 0.1])
    elif problem == "flat":
        boxes[0, 0, 3] = boxes[0, 0, 1]
    elif problem == "mask":
        active = active.long()
    else:
        boxes, active = boxes.repeat(17, 1, 1), active.repeat(17, 1)
    with pytest.raises(ValueError, match="boxes|Boxes|active mask"):
        make_zoom_windows(boxes, active)


@pytest.mark.parametrize("problem", ["training", "nested_training", "trainable", "bad_rgb"])
def test_unfrozen_encoder_or_invalid_rgb_rejected_before_first_call(problem):
    image = _images(0.0)
    boxes, active = _one_visible()
    vision = _FixtureVision()
    if problem == "training":
        vision.train()
    elif problem == "nested_training":
        vision.finish.train()
    elif problem == "trainable":
        vision.gain.requires_grad_(True)
    else:
        image[0, 0, 2, 0, 0] = 1.01
    with pytest.raises(ValueError, match="eval mode|frozen|RGB"):
        encode_zoom_pair(vision, image, boxes, active)
    assert not vision.calls


def test_zoom_and_global_require_same_patch_grid_shape():
    class _InconsistentVision(_FixtureVision):
        def forward(self, images):
            out = super().forward(images)
            return out if len(self.calls) == 1 else out[..., :-1, :]

    boxes, active = _one_visible()
    with pytest.raises(ValueError, match="matching vision patch grids"):
        encode_zoom_pair(_InconsistentVision(), _images(0.0), boxes, active)


def test_protocol_describes_fresh_serializable_settings():
    protocol = zoom_protocol()
    assert protocol["features"]["pool_size"] == [4, 4]
    protocol["features"]["pool_size"][0] = 200
    assert zoom_protocol()["features"]["pool_size"] == [4, 4]
