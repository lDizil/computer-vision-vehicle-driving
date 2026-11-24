"""Immutable identity and reporting thresholds for the frozen #47 transfer trial."""

PREDICTION_SCHEMA = "hybrid-drive-frozen-slow-transfer-predictions-v1"
SCORE_SCHEMA = "hybrid-drive-frozen-slow-transfer-score-v1"
FIT_REVISION = "37f27c77858f7441fb4fc6a359a787fea7c68fd9"
FIT_SHA256 = "a3c74bab6471f8b838e83534559f7bfb985a924c16f667b7453e98350f075a57"
CHECKPOINT_SHA256 = "d909060f1162cac4629563169e1241c9424114fbe9292120d0f36c344d013b29"
SCORE_ARMS = ("trained", "untrained", "no_camera", "no_motion", "shuffled_camera")
METHODS = ("fast", "fast_gap", *SCORE_ARMS)
MIN_RECORDINGS = 3
MIN_WINDOWS = 6
MIN_PAIRS = 12
MIN_VISUAL_MACRO_GAIN = .08


def donor_anchors(anchors):
    """One label-free donor per recording, taken from the next sorted recording."""
    first = {}
    for row in anchors:
        first.setdefault(row["recording"], row)
    ordered = sorted(first)
    if len(ordered) < 2:
        raise ValueError("Camera swap needs a different recording")
    donors = ordered[1:] + ordered[:1]
    return {source: first[donor] for source, donor in zip(ordered, donors, strict=True)}
