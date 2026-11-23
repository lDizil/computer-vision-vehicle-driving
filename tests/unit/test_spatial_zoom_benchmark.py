import json
from types import SimpleNamespace

import torch

from hybrid_drive.experiments import spatial_zoom_benchmark as bench
from hybrid_drive.experiments.protocol import content_hash
from hybrid_drive.experiments.spatial_track_metrics import TrackingTarget
from hybrid_drive.experiments.spatial_zoom_capture import ZoomGrids
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation
from hybrid_drive.neural.spatial_zoom_crops import make_zoom_windows


def _rows(role):
    frames, labels = [], {}
    for i in range(4):
        sid = f"{role}-frame-{i}"
        boxes = tuple((b*.25+.05+i*.004,.27,b*.25+.18+i*.004,.52) for b in range(4))
        states = (1,-1,1,0)
        observation = FrameObservation(role, role+"-clip", sid, i, 1_000_000+500_000*i,(2.,-4.,2.,0.),boxes)
        frames.append(FrameDecision(
            observation,states,observation.scores,boxes,("raw",)*4,"initial" if i==0 else None,
        ))
        observed = list(boxes)
        lo, y1, hi, y2 = boxes[0 if i % 2 == 0 else 2]
        observed[0 if i % 2 == 0 else 2] = (lo+.015,y1+.007,hi+.018,y2+.01)
        labels[sid] = TrackingTarget((role+"A",None,role+"C",role+"D"),
                                     (observed[0],None,observed[2],observed[3]))
    active = torch.tensor([[True,False,True,False]]*len(frames))
    original = torch.tensor([f.boxes for f in frames],dtype=torch.float32)
    g = torch.zeros(4,4,4,4,4)
    c = torch.zeros_like(g)
    g[active] = torch.arange(8*4*4*4,dtype=torch.float32).reshape(8,4,4,4) / 64
    c[active] = g[active].flip(-1)
    return tuple(frames), labels, ZoomGrids(make_zoom_windows(original,active),g,c)


def _arrange(tmp_path, monkeypatch):
    packs, decisions, truth, grids, refs = {}, {}, {}, {}, {}
    output = tmp_path/"output"
    for i,role in enumerate(bench.ZOOM_ROLES,1):
        packs[role] = SimpleNamespace(root=tmp_path/role,role=role)
        decisions[role],truth[role],grids[role] = _rows(role)
        refs[role] = bench.TrackingTruthFile(tmp_path/f"{role}.json",str(i)*64)
    capture = SimpleNamespace(inputs=SimpleNamespace(
        packs=packs, decisions=decisions, protected_roots=tuple(p.root for p in packs.values()),
        source_model_hash="a"*64,source_capture_sha256="b"*64),
        grids=grids, manifest={"protocol":{"fixed":"mock"},"feature_dim":4}, sha256="c"*64)
    opened, reloaded = [], []
    restore = bench.load_neural_weights

    def load_weights(path, modules, **kwargs):
        out = restore(path,modules,**kwargs)
        reloaded.append(path.stem)
        return out

    def load_targets(pack, ref):
        assert ref == refs[pack.role]
        opened.append(pack.role)
        if pack.role in bench.REFERENCE_ROLES:
            f, = output.glob("*/development-gate.json")
            gate = json.loads(f.read_text())
            assert set(reloaded) == set(gate["trained"]) and len(reloaded) == 6
        return truth[pack.role]

    def gallery(path, pack, truth, predictions, *, selected):
        assert pack.role == "appearance_holdout" and len(truth) == 4
        assert len(predictions) == 7 and selected in predictions
        for outcomes in predictions.values():
            assert all(x.observation is y.observation and x.states == y.states
                       for x,y in zip(outcomes,decisions[pack.role]))
        path.write_text("local box comparison")

    monkeypatch.setattr(bench,"_targets",load_targets)
    monkeypatch.setattr(bench,"load_neural_weights",load_weights)
    monkeypatch.setattr(bench,"write_zoom_gallery",gallery)
    return capture,refs,output,opened,reloaded


def test_eval_gate_has_all_restored_weights_before_first_reference_labels(tmp_path, monkeypatch):
    capture,refs,output,opened,reloaded = _arrange(tmp_path,monkeypatch)
    path,r,_ = bench.run_zoom_benchmark(capture,refs,output,revision="d"*40)
    assert r["status"] == "passed",r.get("error")
    assert opened == list(bench.ZOOM_ROLES) and len(reloaded) == len(r["trained"]) == 6
    assert r["protocol"]["reference_previously_observed"] and not r["protocol"]["fresh_blind_test"]
    matrix = r["matrix"]
    assert matrix["predictions_only_move_original_visible_boxes"]
    assert matrix["frame_counts"] == {"train":8,"development":4,
                                       "references":{role:4 for role in bench.REFERENCE_ROLES}}
    assert r["matrix_hash"] == content_hash(matrix) and (path/"zoom.html").exists()
    for scored in matrix["references"].values():
        assert len({m["evaluation_identity"] for m in scored.values()})==1
        assert len({m["output_boxes"] for m in scored.values()})==1
        assert all(set(m["tracking"])=={"zone","motion_strict"} for m in scored.values())


def test_rehashed_gate_mismatch_prevents_opening_reference_labels(tmp_path, monkeypatch):
    capture,refs,output,opened,reloaded = _arrange(tmp_path,monkeypatch)
    write = bench.write_json
    def corrupt(path, data):
        if path.name=="development-gate.json":
            data={**data,"development_hash":"tampered"}
        write(path,data)
    monkeypatch.setattr(bench,"write_json",corrupt)
    _,r,_=bench.run_zoom_benchmark(capture,refs,output,revision="d"*40)
    assert r["status"]=="failed" and "differs from the frozen gate" in r["error"]
    assert opened==list((*bench.TRAIN_ROLES,*bench.DEV_ROLES))
    assert not reloaded



def test_gallery_failure_retains_trained_checkpoints_and_completed_scores(tmp_path, monkeypatch):
    capture, refs, output, opened, reloaded = _arrange(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise OSError("injected gallery failure")
    monkeypatch.setattr(bench, "write_zoom_gallery", fail)
    path, result, _ = bench.run_zoom_benchmark(capture, refs, output, revision="d"*40)
    assert result["status"] == "failed" and "injected gallery failure" in result["error"]
    assert len(result["trained"]) == len(reloaded) == 6 and opened == list(bench.ZOOM_ROLES)
    assert set(result["matrix"]["references"]) == set(bench.REFERENCE_ROLES)
    assert content_hash(result["matrix"]) == result["matrix_hash"]
    assert "gallery" not in result and (path/"result.json").exists()
