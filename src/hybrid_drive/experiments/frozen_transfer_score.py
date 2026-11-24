"""Score blinded predictions and report support per window and recording."""

from collections import Counter, defaultdict
from hashlib import sha256
from math import fsum

from hybrid_drive.experiments.bounded_io import decode_json, read_bounded, require_hex
from hybrid_drive.experiments.frozen_transfer_contacts import measure_recording
from hybrid_drive.experiments.frozen_transfer_data import (
    INDEX_LIMIT, SELECTION_SCHEMA, SOURCES_SHA256, _select_windows, load_transfer_inputs, load_transfer_source,
    read_transfer_recordings,
)
from hybrid_drive.experiments.frozen_transfer_protocol import (
    CHECKPOINT_SHA256, FIT_REVISION, FIT_SHA256, METHODS, MIN_PAIRS, MIN_RECORDINGS, MIN_VISUAL_MACRO_GAIN,
    MIN_WINDOWS, PREDICTION_SCHEMA, SCORE_ARMS, SCORE_SCHEMA, donor_anchors,
)
from hybrid_drive.experiments.io import write_json


def score_transfer(predictions, sealed_sha256, input_root, selection_path, sources_path, metadata_path, output):
    require_hex(sealed_sha256, 64, "transfer prediction seal")
    if output.exists() or output.is_symlink():
        raise ValueError("Transfer score output must be new")
    raw = read_bounded(predictions, 4 * 1024**2)
    if sha256(raw).hexdigest() != sealed_sha256:
        raise ValueError("Transfer predictions do not match the recorded seal")
    predicted = decode_json(raw)
    _verify_prediction_header(predicted)
    observed = load_transfer_inputs(input_root)
    observed_sha = sha256(read_bounded(input_root / "index.json", INDEX_LIMIT)).hexdigest()
    selection_data = read_bounded(selection_path, 512 * 1024)
    selection = decode_json(selection_data)
    if (predicted["input_sha256"] != observed_sha or selection["input_sha256"] != observed_sha
            or selection["source_sha256"] != SOURCES_SHA256 or selection["schema"] != SELECTION_SCHEMA):
        raise ValueError("Selection, observations and predictions must come from the same prepared cohort")
    indexed = bind_transfer_rows(observed, selection["rows"], predicted["rows"])
    source = load_transfer_source(sources_path)
    if set(indexed) != {log["recording"] for log in source["logs"]}:
        raise ValueError("Wrong set of physical recordings for transfer scoring")

    # Only this stage receives the source metadata archive. The prediction
    # bytes, cohort identities and the frozen checkpoint are already verified.
    rows = []
    for record, frames in read_transfer_recordings(metadata_path, source):
        selection_rows, prediction_rows = indexed[record["recording"]]
        _verify_selection(frames, selection_rows)
        rows.extend(measure_recording(frames, selection_rows, prediction_rows))
    summary = summarize_transfer(rows)
    result = {
        "schema": SCORE_SCHEMA, "revision": predicted["revision"], "prediction_sha256": sealed_sha256,
        "sources_sha256": SOURCES_SHA256, "input_sha256": observed_sha,
        "selection_sha256": sha256(selection_data).hexdigest(), "checkpoint_sha256": CHECKPOINT_SHA256,
        "summary": summary, "rows": rows,
    }
    output.mkdir(parents=True)
    write_json(output / "transfer-score.json", result)
    return result


def _verify_prediction_header(predicted):
    if (predicted.get("schema") != PREDICTION_SCHEMA or predicted.get("sources_sha256") != SOURCES_SHA256
            or predicted.get("checkpoint_sha256") != CHECKPOINT_SHA256 or predicted.get("original_fit_sha256") != FIT_SHA256
            or predicted.get("original_fit_revision") != FIT_REVISION or predicted.get("future_annotations_mounted") is not False
            or predicted.get("checkpoint_changed") is not False or predicted.get("arms") != list(SCORE_ARMS)):
        raise ValueError("Expected sealed observations evaluated by the unmodified #47 checkpoint")
    require_hex(predicted.get("revision"), 40, "transfer revision")


def bind_transfer_rows(observed, selected, predicted):
    anchors = observed["anchors"]
    if len(anchors) != len(selected) or len(anchors) != len(predicted):
        raise ValueError("Each preselected observation must appear exactly once")
    by_recording, indexed = {}, {}
    for anchor, choice, prediction in zip(anchors, selected, predicted, strict=True):
        token, recording = anchor["sample_token"], anchor["recording"]
        stamp = observed["frames"][token]["ref_time_us"]
        input_id = (token, recording, stamp)
        if token in indexed or _identity(choice) != input_id or _identity(prediction) != input_id:
            raise ValueError("Missing, reordered or mismatched transfer observation identity")
        require_hex(prediction.get("snapshot_sha256"), 64, "camera snapshot")
        if choice["population"] not in ("uniform", "vulnerable") or type(choice["frame_index"]) is not int or choice["frame_index"] < 1:
            raise ValueError("Invalid preselected population or source frame index")
        if prediction["status"] not in ("ok", "out_of_envelope"):
            raise ValueError("Unknown frozen Fast observation status")
        if prediction["status"] != "ok" and prediction["candidates"] != []:
            raise ValueError("Unsupported observations must not have invented candidates")
        group = by_recording.setdefault(recording, ([], []))
        group[0].append(choice)
        group[1].append(prediction)
        indexed[token] = prediction
    donors = donor_anchors(anchors)
    for row in predicted:
        if row["status"] == "ok":
            donor = indexed.get(row.get("donor_snapshot_id"))
            if (donor is None or donor["sample_token"] != donors[row["recording"]]["sample_token"]
                    or donor["snapshot_sha256"] != row.get("donor_snapshot_sha256")):
                raise ValueError("Camera donor must be a bound observation from a different recording")
    return by_recording


def _identity(row):
    return row["sample_token"], row["recording"], row["ref_time_us"]


def _verify_selection(frames, rows):
    frozen_choices, _ = _select_windows(frames)
    if [(row["frame_index"], row["population"]) for row in rows] != frozen_choices:
        raise ValueError("Selection differs from the fixed observation-only sampling protocol")


def summarize_transfer(rows):
    profiles = {}
    slices = (
        ("vehicle_uniform", "vehicle_outcome", lambda row, pair: row["population"] == "uniform"),
        ("vehicle_uniform_fast_tied", "vehicle_outcome", lambda row, pair: row["population"] == "uniform" and pair["fast_binary_tied"]),
        ("visible_vulnerable_only", "vulnerable_outcome", lambda row, pair: pair["vulnerable_only"]),
        ("visible_vulnerable_only_fast_tied", "vulnerable_outcome", lambda row, pair: pair["vulnerable_only"] and pair["fast_binary_tied"]),
    )
    for name, outcome, condition in slices:
        profiles[name] = _summarize_pairs(rows, outcome, condition)
    return {
        "windows": len(rows), "recordings": sorted({row["recording"] for row in rows}),
        "populations": Counter(row["population"] for row in rows),
        "window_statuses": Counter(row["window_status"] for row in rows),
        "windows_with_candidate_pairs": sum(bool(row["pairs"]) for row in rows),
        "windows_with_vulnerable_centroids_in_frustum": sum(row.get("eligible_vulnerable_anchor_count", 0) > 0 for row in rows),
        "slices": profiles,
        "vehicle_transfer": _transfer_gate(profiles["vehicle_uniform"]),
        "visual_contribution_vehicle": _visual_gate(profiles["vehicle_uniform_fast_tied"]),
        "visual_contribution_vulnerable": _visual_gate(profiles["visible_vulnerable_only_fast_tied"]),
    }


def _summarize_pairs(rows, target, condition):
    gathered, statuses = [], Counter()
    for row in rows:
        pairs = []
        for pair in row["pairs"]:
            if not condition(row, pair):
                continue
            outcome = pair[target]
            statuses[outcome] += 1
            if outcome in ("left_contact", "right_contact"):
                without_contact = pair["candidate_ids"][int(outcome == "left_contact")]
                pairs.append((without_contact, pair["preferred"]))
        if pairs:
            gathered.append((row["recording"], pairs))
    by_recording = defaultdict(list)
    for recording, pairs in gathered:
        by_recording[recording].append(pairs)
    return {"pairs": sum(len(pairs) for _, pairs in gathered), "windows": len(gathered),
            "recordings": len(by_recording), "outcomes": statuses,
            "methods": _method_counts([pairs for _, pairs in gathered]),
            "by_recording": {recording: {"windows": len(windows), "pairs": sum(map(len, windows)),
                                        "methods": _method_counts(windows)}
                             for recording, windows in sorted(by_recording.items())}}


def _method_counts(windows):
    result = {}
    for arm in METHODS:
        outcomes, accuracy = Counter(correct=0, incorrect=0, tied=0), []
        for pairs in windows:
            current = Counter()
            for truth, preferred in pairs:
                choice = preferred[arm]
                current["tied" if choice is None else ("correct" if choice == truth else "incorrect")] += 1
            outcomes.update(current)
            accuracy.append(current["correct"] / len(pairs))
        result[arm] = {**outcomes, "window_macro_accuracy": fsum(accuracy) / len(accuracy) if accuracy else None}
    return result


def _meets_support(profile):
    return (profile["recordings"] >= MIN_RECORDINGS and profile["windows"] >= MIN_WINDOWS and profile["pairs"] >= MIN_PAIRS)


def _transfer_gate(profile):
    policy = {"minimum_recordings": MIN_RECORDINGS, "minimum_windows": MIN_WINDOWS, "minimum_pairs": MIN_PAIRS}
    if not _meets_support(profile):
        return {"status": "insufficient_support", **policy}
    arms = profile["methods"]
    trained = arms["trained"]["window_macro_accuracy"]
    strongest = max(arms[name]["window_macro_accuracy"] for name in ("fast", "fast_gap", "untrained"))
    wins = sum(r["methods"]["trained"]["window_macro_accuracy"] > max(
        r["methods"][k]["window_macro_accuracy"] for k in ("fast", "fast_gap")) for r in profile["by_recording"].values())
    passed = trained > strongest and 2 * wins > profile["recordings"]
    return {"status": "passed_transfer_probe" if passed else "advantage_not_reproduced", **policy,
            "gain_over_strongest_baseline": trained - strongest, "recordings_beating_both_fast_baselines": wins}


def _visual_gate(profile):
    policy = {"minimum_recordings": MIN_RECORDINGS, "minimum_windows": MIN_WINDOWS, "minimum_pairs": MIN_PAIRS,
              "min_gain_over_both_camera_controls": MIN_VISUAL_MACRO_GAIN}
    if not _meets_support(profile):
        return {"status": "insufficient_support", **policy}
    controls = ("no_camera", "shuffled_camera")
    def gain(methods):
        return methods["trained"]["window_macro_accuracy"] - max(methods[k]["window_macro_accuracy"] for k in controls)
    overall = gain(profile["methods"])
    wins = sum(gain(r["methods"]) > 0 for r in profile["by_recording"].values())
    passed = overall >= MIN_VISUAL_MACRO_GAIN and 2 * wins > profile["recordings"]
    return {"status": "visual_signal_detected" if passed else "direct_camera_contribution_not_established", **policy,
            "gain_over_best_camera_control": overall, "recordings_beating_both_camera_controls": wins}
