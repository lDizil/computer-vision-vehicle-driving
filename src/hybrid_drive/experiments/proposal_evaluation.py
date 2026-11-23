"""Encode spatial evidence and evaluate exact restored proposal alignment states."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from hybrid_drive.neural.grounded_loc_probe import GroundedLocProbe, fixed_quarter_boxes, grounded_presence_scores
from hybrid_drive.neural.spatial_data import collate_spatial_alignment
from .io import write_json
from .localization_gallery import GalleryPredictions
from .neural_checkpoint import load_neural_weights, module_state_hash
from .spatial_metrics import SpatialPredictions, calibrate_spatial_threshold, shuffled_by_recording, spatial_metrics


CONDITIONS = ("fixed_quarters", "spatial_proposals")


@dataclass(frozen=True)
class _Encoded:
    features: Tensor
    proposals: Tensor
    presence: Tensor
    target_boxes: Tensor
    recordings: tuple[str, ...]
    prior_scores: Tensor | None = None

    def __len__(self):
        return len(self.recordings)

    def batch(self, indices, condition, proposal_order=None):
        features = self.features[indices]
        if condition == "fixed_quarters":
            boxes = fixed_quarter_boxes(features)
        elif condition == "spatial_proposals":
            boxes = self.proposals[indices if proposal_order is None else proposal_order[indices]]
        else:
            raise ValueError("Unknown proposal condition")
        return features, boxes, self.presence[indices], self.target_boxes[indices]


@torch.no_grad()
def encode(dataset, vision, prior, batch_size, *, capture_direct=False):
    features, proposals, present, targets, seen, recordings = [], [], [], [], [], []
    direct_scores = []
    for batch in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_spatial_alignment
    ):
        patches = vision(batch.images)
        if (
            patches.ndim != 5
            or patches.shape[:3] != (len(batch.sample_ids), 1, vision.visual_dim)
            or not torch.isfinite(patches).all()
        ):
            raise ValueError("Expected finite patch grids with unchanged sample order")
        if capture_direct:
            output = prior.predict_spatial(patches)
            predicted = output.boxes_xyxy
            direct_scores.append(output.presence_logits)
        else:
            predicted = prior.predict(patches)
        if (
            predicted.shape != batch.boxes_2d.shape
            or predicted.requires_grad
            or not torch.isfinite(predicted).all()
        ):
            raise ValueError("Prior source must provide one finite box per ordered band without gradients")
        features.append(patches)
        proposals.append(predicted)
        present.append(batch.presence)
        targets.append(batch.boxes_2d)
        seen.extend(batch.sample_ids)
        recordings.extend(batch.recording_ids)
    if tuple(seen) != dataset.sample_ids or tuple(recordings) != dataset.recording_ids:
        raise ValueError("Data loader changed the prepared sample or recording order")
    return _Encoded(
        torch.cat(features), torch.cat(proposals), torch.cat(present), torch.cat(targets), tuple(recordings),
        torch.cat(direct_scores) if capture_direct else None,
    )


@torch.no_grad()
def predict(model, language, encoded, condition, batch_size, proposal_order=None):
    model.eval()
    decoder_scores, language_scores, boxes = [], [], []
    for indices in torch.arange(len(encoded)).split(batch_size):
        features, rois, _, _ = encoded.batch(indices, condition, proposal_order)
        validity, margin, decoded = grounded_presence_scores(model(features, rois, language), language)
        decoder_scores.append(validity)
        language_scores.append(margin)
        boxes.append(decoded)
    present, targets, boxes = encoded.presence.numpy(), encoded.target_boxes.numpy(), torch.cat(boxes).numpy()
    return {
        "decoder": SpatialPredictions(
            torch.cat(decoder_scores).numpy(), present, boxes, targets, encoded.recordings
        ),
        "text": SpatialPredictions(
            torch.cat(language_scores).numpy(), present, boxes, targets, encoded.recordings
        ),
    }


def _metrics(predictions, thresholds):
    output = {
        key: spatial_metrics(values, thresholds[key]["threshold"]) for key, values in predictions.items()
    }
    for metrics in (output["text"], *output["text"]["per_recording"].values()):
        for key in tuple(metrics):
            if "iou" in key or "l2_image_fraction" in key:
                del metrics[key]
    return output


def calibrate(predictions):
    thresholds = {key: calibrate_spatial_threshold(value) for key, value in predictions.items()}
    return _metrics(predictions, thresholds), thresholds


def evaluate_frozen(
    root, result, visual_dim, language, holdout, batch_size, thresholds, *, memory_mode, gallery_seed=None,
):
    gallery = {}
    order = torch.as_tensor(shuffled_by_recording(holdout.recordings), dtype=torch.long)
    result["mismatched_proposal_fraction"] = float(
        (holdout.proposals != holdout.proposals[order]).any(dim=-1).float().mean()
    )
    for row in result["runs"]:
        restored = GroundedLocProbe(language, visual_dim=visual_dim, memory_mode=memory_mode).eval()
        checkpoint = row["checkpoint"]
        load_neural_weights(
            root / checkpoint["file"],
            restored.weight_modules(),
            expected_sha256=checkpoint["sha256"],
            expected_metadata=row["checkpoint_metadata"],
        )
        if module_state_hash(restored) != row["final_weights_hash"]:
            raise ValueError("Reloaded alignment checkpoint differs from the frozen state")
        row["checkpoint_state_bitwise_equal"] = True
        condition = row["condition"]
        cutoff = thresholds[(row["seed"], condition)]
        predictions = predict(restored, language, holdout, condition, batch_size)
        row["holdout"] = _metrics(predictions, cutoff)
        if row["seed"] == gallery_seed:
            decoded = predictions["decoder"]
            gallery[condition] = GalleryPredictions(decoded.scores, decoded.boxes, cutoff["decoder"]["threshold"])
        if condition == "spatial_proposals":
            wrong_boxes = predict(restored, language, holdout, condition, batch_size, order)
            row["holdout_shuffled_rois_same_frame"] = _metrics(wrong_boxes, cutoff)
            row["mean_absolute_score_change_on_proposal_shuffle"] = {
                key: float(np.abs(values.scores - wrong_boxes[key].scores).mean())
                for key, values in predictions.items()
            }
        write_json(root / "result.json", result)
    return gallery


def evaluate_direct_spatial(holdout, threshold):
    """Evaluate the already frozen spatial head with its original dev threshold."""
    scores = holdout.prior_scores.numpy()
    boxes = holdout.proposals.numpy()
    present, targets = holdout.presence.numpy(), holdout.target_boxes.numpy()
    order = shuffled_by_recording(holdout.recordings)
    original = SpatialPredictions(scores, present, boxes, targets, holdout.recordings)
    shuffled = SpatialPredictions(scores[order], present, boxes[order], targets, holdout.recordings)
    return {
        "dev_threshold": threshold,
        "holdout": spatial_metrics(original, threshold),
        "holdout_shuffled_images_same_labels": spatial_metrics(shuffled, threshold),
    }, GalleryPredictions(scores, boxes, threshold)


def paired_deltas(rows, seeds):
    def compare(left, right):
        return {
            section: {
                key: (
                    left[section][key] - right[section][key]
                    if left[section][key] is not None and right[section][key] is not None
                    else None
                )
                for key in keys
            }
            for section, keys in (
                ("decoder", ("macro_balanced_accuracy", "macro_roc_auc", "macro_positive_iou")),
                ("text", ("macro_balanced_accuracy", "macro_roc_auc")),
            )
        }

    output = []
    for seed in seeds:
        fixed, proposals = [
            next(row for row in rows if row["seed"] == seed and row["condition"] == condition)
            for condition in CONDITIONS
        ]
        output.append(
            {
                "seed": seed,
                "proposals_minus_fixed": compare(proposals["holdout"], fixed["holdout"]),
                "proposals_minus_shuffled_rois": compare(
                    proposals["holdout"], proposals["holdout_shuffled_rois_same_frame"]
                ),
            }
        )
    return output
