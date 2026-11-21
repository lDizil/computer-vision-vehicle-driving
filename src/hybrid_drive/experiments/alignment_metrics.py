"""Count-weighted LOC metrics; missing class/box denominators remain unknown."""

import torch


class LocMetricAccumulator:
    def __init__(self):
        self.loss_sums = dict.fromkeys(("text", "classification", "boxes_3d", "giou_2d"), 0.0)
        self.loss_counts = dict.fromkeys(self.loss_sums, 0)
        self.positives = self.negatives = 0
        self.true_positives = self.true_negatives = self.text_correct = 0
        self.center_sum = self.size_sum = self.yaw_sum = self.iou_sum = 0.0

    def add(self, prediction, loss, language, presence, boxes_3d, boxes_2d):
        n = int(presence.sum())
        self.positives += n
        self.negatives += presence.numel() - n
        for key, count in (
            ("text", loss.supervised_text_tokens),
            ("classification", loss.supervised_events),
            ("boxes_3d", loss.supervised_box_coordinates),
            ("giou_2d", loss.supervised_image_boxes),
        ):
            self.loss_sums[key] += float(getattr(loss, key)) * count
            self.loss_counts[key] += count
        logits = prediction.perception.validity_logits
        selected = logits >= 0
        self.true_positives += int((selected & presence).sum())
        self.true_negatives += int((~selected & ~presence).sum())
        text = (
            prediction.logits[:, -1, language.yes_token_id] > prediction.logits[:, -1, language.no_token_id]
        )
        self.text_correct += int((text == presence).sum())
        if not n:
            return
        residual = prediction.perception.boxes_3d[presence] - boxes_3d[presence]
        self.center_sum += float(residual[:, :3].norm(dim=1).sum())
        self.size_sum += float(residual[:, 3:6].abs().mean(dim=1).sum())
        yaw = residual[:, 6]
        self.yaw_sum += float(torch.atan2(yaw.sin(), yaw.cos()).abs().sum())
        self.iou_sum += float(_paired_iou(prediction.perception.boxes_2d[presence], boxes_2d[presence]).sum())

    def summary(self):
        count = self.positives + self.negatives
        if count == 0:
            raise ValueError("Cannot summarize an empty evaluation")
        terms = {key: self.loss_sums[key] / n if n else None for key, n in self.loss_counts.items()}
        recall = self.true_positives / self.positives if self.positives else None
        specificity = self.true_negatives / self.negatives if self.negatives else None
        return {
            "examples": count,
            "positives": self.positives,
            "negatives": self.negatives,
            "loss_terms": terms,
            "loss_denominators": self.loss_counts.copy(),
            # The pilot uses unit weights and no MOT targets.
            "loss": sum(value for value in terms.values() if value is not None),
            "presence_accuracy": (self.true_positives + self.true_negatives) / count,
            "presence_balanced_accuracy": (
                (recall + specificity) / 2 if recall is not None and specificity is not None else None
            ),
            "positive_recall": recall,
            "negative_specificity": specificity,
            "forced_choice_yes_no_accuracy": self.text_correct / count,
            "positive_center_l2_m": self.center_sum / self.positives if self.positives else None,
            "positive_mean_size_mae_m": self.size_sum / self.positives if self.positives else None,
            "positive_yaw_mae_rad": self.yaw_sum / self.positives if self.positives else None,
            "positive_projected_box_iou": self.iou_sum / self.positives if self.positives else None,
        }


def _paired_iou(prediction, target):
    extent = (
        torch.minimum(prediction[:, 2:], target[:, 2:]) - torch.maximum(prediction[:, :2], target[:, :2])
    ).clamp_min(0)
    intersection = extent.prod(dim=1)
    union = (
        (prediction[:, 2:] - prediction[:, :2]).prod(dim=1)
        + (target[:, 2:] - target[:, :2]).prod(dim=1)
        - intersection
    )
    return intersection / union
