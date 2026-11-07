import pytest

from hybrid_drive.experiments.protocol import validate_splits
from hybrid_drive.experiments.splits import recording_group, split_training_logs


def test_split_is_stable_under_input_order_and_keeps_recording_segments_together():
    logs = [f"2021.01.{day:02d}.10.00.00_veh-1_{start:05d}_{start+50:05d}"
            for day in range(1, 11) for start in (0, 50, 100)]
    split = split_training_logs(logs)
    assert split == split_training_logs(reversed(logs))
    assert len(split["validation"]) == 6 and len(split["train"]) == 24
    assert set(split["train"]) | set(split["validation"]) == set(logs)
    assert not ({recording_group(log) for log in split["train"]}
                & {recording_group(log) for log in split["validation"]})


def test_adjacent_segments_with_distinct_names_and_frames_cannot_cross_roles():
    with pytest.raises(ValueError, match="Log crosses"):
        validate_splits({
            "train": [{"scene_token": "one", "log_name": "2021.01.01_veh-1_00000_00050"}],
            "test": [{"scene_token": "two", "log_name": "2021.01.01_veh-1_00050_00100"}],
        })


@pytest.mark.parametrize("logs", [[], ["one", "one"], ["single"], ["a", None]])
def test_degenerate_or_ambiguous_split_is_not_accepted(logs):
    with pytest.raises(ValueError):
        split_training_logs(logs)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan"), True])
def test_invalid_validation_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        split_training_logs(["a", "b", "c"], validation_fraction=fraction)
