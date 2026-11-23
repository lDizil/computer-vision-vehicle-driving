from types import SimpleNamespace
import json
import re

import numpy as np
import pytest

from hybrid_drive.experiments import temporal_spatial_gallery as gallery
from hybrid_drive.experiments.spatial_track_metrics import TrackingTarget
from hybrid_drive.experiments.spatial_tracking_report import write_tracking_gallery
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation


def _inputs(tmp_path):
    root = tmp_path / "pack"
    root.mkdir()
    frame_refs, results, targets = [], [], {}
    key = "private-source-track-alias"
    boxes = ((.02,.1,.19,.6),(.28,.1,.4,.6),(.56,.1,.65,.6),(.77,.1,.92,.6))
    for index in range(4):
        clip = f"clip-{index // 2}"
        sid = f"frame-{index}"
        row = FrameObservation("recording", clip, sid, index, 1_500_000 + index * 500_000,
                               (1.,0.,0.,0.), boxes)
        frame_refs.append(SimpleNamespace(recording_id=row.recording_id, clip_id=clip, sample_id=sid,
                                           source_index=index, timestamp_us=row.timestamp_us, rgb={}))
        decision = FrameDecision(row, (1,0,-1,-1), row.scores, row.boxes,
                                 ("new_track","uncertain","absent","absent"), "initial" if index%2==0 else None)
        # One reused synthetic token and changing predictor ID; neither
        # source token nor any per-frame metadata token should appear in HTML.
        targets[sid] = TrackingTarget((key,None,None,None),(boxes[0],None,None,None))
        results.append(SimpleNamespace(decision=decision, track_ids=(1+index%2,None,None,None),
                                       reasons=("new_track","uncertain","absent","absent")))
    return SimpleNamespace(root=root, frames=tuple(frame_refs)), targets, tuple(results), key


def test_gallery_adds_only_short_display_ids_for_same_verified_frames(tmp_path, monkeypatch):
    pack, truth, tracked, original = _inputs(tmp_path)
    monkeypatch.setattr(gallery, "read_spatial_rgb", lambda *args: np.zeros((224,224,3), dtype=np.uint8))
    result = tmp_path / "tracking.html"
    write_tracking_gallery(result, pack, truth, {"zone": tracked, "motion": tracked}, selected="motion")
    html = result.read_text()
    assert original not in html and "trackMark(" in html
    payload = json.loads(re.search(r'id="video-data">(.*?)</script>', html, flags=re.S).group(1))
    assert payload["tracking"] is True and payload["baseline"] == "zone" and payload["selected"] == "motion"
    for clip in payload["clips"]:
        assert [row["track_truth"][0] for row in clip["frames"]] == ["G1","G1"]
        assert [row["predictions"][1]["track_labels"][0] for row in clip["frames"]] == ["P1","P2"]
    assert all(set(x["predictions"][0]["track_labels"]) == {"P1",None}
               for x in (clip["frames"][0] for clip in payload["clips"]))


def test_gallery_rejects_ids_without_a_visible_box_before_reading_any_pixels(tmp_path, monkeypatch):
    pack, truth, tracked, _ = _inputs(tmp_path)
    valid = tracked[0]
    bad = (SimpleNamespace(decision=valid.decision, track_ids=(1,2,None,None), reasons=valid.reasons), *tracked[1:])
    monkeypatch.setattr(gallery, "read_spatial_rgb", lambda *args: pytest.fail("Bad identity overlay read image"))
    with pytest.raises(ValueError, match="own tracking label"):
        write_tracking_gallery(tmp_path / "out.html", pack, truth, {"zone": bad}, selected="zone")
    assert not (tmp_path / "out.html").exists()
