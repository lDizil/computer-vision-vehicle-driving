from collections import Counter
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from hybrid_drive.experiments import spatial_appearance_benchmark as bench
from hybrid_drive.experiments.spatial_appearance_features import FrameAppearance, descriptor_protocol
from hybrid_drive.experiments.spatial_track_metrics import TrackingTarget
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation


def _signals(a):
    rgb = tuple(1.0 if i == a else 0.0 for i in range(24))
    dino = tuple(1.0 if i == a else 0.0 for i in range(4))
    return rgb, dino


def _seq(role):
    decisions, features, truth = [], [], {}
    for i, (a, b) in enumerate(((.213,.287),(.231,.269),(.271,.229),(.296,.207))):
        sid = f"{role}-sample-{i}"
        centers = sorted(((a, "A",0), (b, "B",1)))
        declared = tuple((center-.057, .18, center+.057, .44) for center, _, _ in centers)
        boxes = (*declared, (.54,.12,.64,.5), (.78,.12,.91,.5))
        observation = FrameObservation(role, f"{role}-clip", sid, i, 1_000_000+500_000*i, (.9,.9,-.8,-.8), boxes)
        decision = FrameDecision(observation,(1,1,-1,-1),observation.scores,boxes,
                                 ("raw",)*4,"initial" if i==0 else None)
        decisions.append(decision)
        color, patch = zip(*(_signals(which) for _, _, which in centers))
        features.append(FrameAppearance(sid,(*color,None,None),(*patch,None,None),(*patch,None,None)))
        truth[sid] = TrackingTarget(tuple(token for _,token,_ in centers)+(None,None), (*declared,None,None))
    return tuple(decisions), tuple(features), truth


def _environment(tmp_path, monkeypatch):
    packs, decisions, records, truth, refs = {}, {}, {}, {}, {}
    for i, role in enumerate(bench.INPUT_ROLES,1):
        packs[role] = SimpleNamespace(root=tmp_path/role, role=role)
        decisions[role], records[role], truth[role] = _seq(role)
        refs[role] = bench.TrackingTruthFile(tmp_path/f"{role}.json",str(i)*64)
    cap = SimpleNamespace(inputs=SimpleNamespace(model_hash="a"*64, protected_roots=tuple(p.root for p in packs.values()),
                                                packs=packs),
        decisions=decisions,appearances=records,sha256="b"*64,
        manifest={"revision":"c"*40,"observation_anchors":{},"descriptor_protocol":descriptor_protocol()})
    asked = []
    out = tmp_path/"out"

    def load(pack, ref):
        assert ref == refs[pack.role]
        asked.append(pack.role)
        if pack.role == bench.APPEARANCE_ROLE:
            gate, = out.glob("*/development-gate.json")
            frozen = json.loads(gate.read_text())
            assert frozen["selection"]["name"] in ("zone", *bench.APPEARANCE_KINDS)
        return truth[pack.role]

    def render(path, pack, gt, rows, *, selected):
        assert pack is packs[bench.APPEARANCE_ROLE] and gt == truth[bench.APPEARANCE_ROLE]
        assert len(rows) == 8 and selected in rows
        assert all(t.decision == d for mode in rows.values() for t,d in zip(mode,decisions[pack.role]))
        path.write_text("identity comparison")

    monkeypatch.setattr(bench, "_targets", load)
    monkeypatch.setattr(bench, "write_appearance_gallery", render)
    return cap,refs,out,asked


def test_full_matrix_and_signal_on_unchanged_frames_with_dev_gate(tmp_path, monkeypatch):
    capture, truth, out, asked = _environment(tmp_path, monkeypatch)
    path, r, _ = bench.run_appearance_benchmark(capture,truth,out,revision="d"*40)
    assert r["status"] == "passed", r.get("error")
    assert asked == list(bench.INPUT_ROLES)
    matrix = r["matrix"]
    assert matrix["configurations"] == 16 and matrix["frame_counts"] == {"development":16,"holdout":4}
    assert matrix["raw_prediction_inputs_identical"] is True
    assert matrix["holdout_diagnostic"]["evaluated_recipe"] == matrix["selection"]["name"]
    assert len({x["evaluation_identity"] for x in matrix["holdout"]["tracking"].values()}) == 1
    assert matrix["holdout"]["signal"]["eligible_geometry"] > 0
    assert (path/"appearance.html").exists() and 'href="appearance.html"' in (path/"report.html").read_text()


def test_changed_dev_receipt_stops_before_fresh_label_read(tmp_path, monkeypatch):
    capture, truth, out, asked = _environment(tmp_path, monkeypatch)
    original = bench.write_json

    def change_gate(path, value):
        if path.name == "development-gate.json":
            value = {**value,"development_hash":"corrupt"}
        original(path,value)

    monkeypatch.setattr(bench, "write_json",change_gate)
    _, r, _ = bench.run_appearance_benchmark(capture,truth,out,revision="d"*40)
    assert r["status"] == "failed" and "did not survive" in r["error"]
    assert asked == [r for r in bench.INPUT_ROLES if r!=bench.APPEARANCE_ROLE]


def test_frame_control_does_not_move_missing_slots_or_borrow_from_other_time():
    _, rows, _ = _seq("toy")
    other = tuple(replace(rows[0],sample_id=f"toy-{i}") for i in range(32))
    singleton = tuple(_signals(0)[i] for i in (0,1))
    rows = (*other,FrameAppearance("single",(singleton[0],None,None,None),
                                   (singleton[1],None,None,None),(None,None,None,None)))
    shuffled, changed = bench.permute_frame_appearance(rows)
    assert bench.permute_frame_appearance(rows) == (shuffled,changed)
    assert 0 < changed["rgb"]["changed_slots"] < 2*len(other)
    for a,b in zip(rows,shuffled):
        assert a.sample_id == b.sample_id
        for kind in bench.APPEARANCE_KINDS:
            p,q = getattr(a,kind),getattr(b,kind)
            assert tuple(x is None for x in p) == tuple(x is None for x in q)
            assert Counter(p) == Counter(q)
    assert rows[-1] == shuffled[-1]


def test_useful_looking_choice_is_rejected_if_shuffle_carries_same_score():
    def m(score,wrong=0):
        return {"fixed_match_idf1":score,"wrong_links":wrong,"adjacent_switches":1,"evaluation_identity":"same",
                **dict.fromkeys(("frames","labeled_boxes","output_boxes","detections_matched",
                                "adjacent_links","cross_zone_links"),12)}
    methods = {"zone":m(.40),"motion_strict":m(.41),"dino":m(.43),"dino_shuffled":m(.43),
               "rgb":m(.42,1),"rgb_shuffled":m(.39),"contrast":m(.42),"contrast_shuffled":m(.417)}
    control={name:{"changed_slots":8} for name in bench.APPEARANCE_KINDS}
    pick=bench._pick(methods,control)
    assert pick["name"] == "zone"
    assert "insufficient_gain_vs_shuffled" in pick["candidates"]["dino"]["reasons"]
    assert "motion_strict:more_wrong_links" in pick["candidates"]["rgb"]["reasons"]
    methods["dino_shuffled"]=m(.415)
    assert bench._pick(methods,control)["name"] == "dino"
    # Final alternatives may not become the default after a failed frozen candidate.
    methods["dino"]=m(.39)
    methods["contrast"]=m(.49)
    assert bench._final_check("dino",methods,control)["outcome"] == "no_replacement"
