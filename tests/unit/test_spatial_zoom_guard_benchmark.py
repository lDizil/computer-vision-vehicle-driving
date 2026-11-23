import json
from types import SimpleNamespace

import torch

from hybrid_drive.experiments import spatial_zoom_guard_benchmark as bench
from hybrid_drive.experiments.protocol import content_hash
from hybrid_drive.experiments.spatial_track_metrics import TrackingTarget
from hybrid_drive.experiments.spatial_zoom_capture import ZoomGrids
from hybrid_drive.experiments.temporal_spatial_types import FrameDecision, FrameObservation
from hybrid_drive.neural.spatial_zoom_crops import make_zoom_windows


def _make(role, marker):
    decisions, targets = [], {}
    for i in range(3):
        name=f"{role}-{i}"
        boxes=tuple((q*.25+.04+i*.005,.22,q*.25+.17+i*.005,.51) for q in range(4))
        obs=FrameObservation(role,role+"-clip",name,i,1_000_000+500_000*i,(2.,-4.,2.,0.),boxes)
        decisions.append(FrameDecision(obs,(1,-1,1,0),obs.scores,boxes,("raw",)*4,"initial" if i==0 else None))
        targets[name]=TrackingTarget((role+"a",None,role+"b",None),(boxes[0],None,boxes[2],None))
    active=torch.tensor([[True,False,True,False]]*len(decisions))
    f=torch.zeros(3,4,4,4,4)
    f[active]=float(marker)
    windows=make_zoom_windows(torch.tensor([d.boxes for d in decisions],dtype=torch.float32),active)
    return tuple(decisions),targets,ZoomGrids(windows,f,f.clone())


def _arrange(tmp_path,monkeypatch):
    out=tmp_path/"result"
    roles=(*bench.FOLD_ROLES,*bench.DEV_ROLES,bench.FRESH_ROLE)
    marker={role:i for i,role in enumerate(roles,1)}
    content={role:_make(role,i) for role,i in marker.items()}
    packs={role:SimpleNamespace(role=role) for role in roles}
    known=SimpleNamespace(sha256="a"*64,
        inputs=SimpleNamespace(source_model_hash="b"*64,packs={r:packs[r] for r in roles[:-1]},
                               decisions={r:content[r][0] for r in roles[:-1]}),
        manifest={"feature_dim":4,"protocol":{"fixture":True}},grids={r:content[r][2] for r in roles[:-1]})
    inputs=SimpleNamespace(known=known,fresh_pack=packs[bench.FRESH_ROLE],protected_roots=())
    fresh=SimpleNamespace(sha256="c"*64,decisions=content[bench.FRESH_ROLE][0],grids=content[bench.FRESH_ROLE][2])
    truth={role:bench.TrackingTruthFile(tmp_path/f"{role}.json",str(i)*64) for role,i in marker.items()}
    called,fit_groups,restored=[],[],[]
    old_load=bench.load_neural_weights

    def targets(pack, ref):
        assert ref==truth[pack.role]
        called.append(pack.role)
        if pack.role==bench.FRESH_ROLE:
            gate,=out.glob("*/development-gate.json")
            saved=json.loads(gate.read_text())
            assert saved["selection"]["name"] in bench.GUARD_MODES
            assert restored==["full-zoom.safetensors"]
        return content[pack.role][1]

    def fake_fit(boxes,windows,features,active,labels,present,**kwargs):
        fit_groups.append(set(int(i) for i in features[active][:,0,0,0].tolist()))
        return bench.ZoomBoxRefiner(4).eval(),[{"fixture":True}]

    def load(path,modules,**kwargs):
        r=old_load(path,modules,**kwargs)
        restored.append(path.name)
        return r

    def gallery(path, pack, labels, results, *,selected):
        assert pack.role==bench.FRESH_ROLE and selected in results and len(results)==5
        assert all(x.observation is y.observation and x.states==y.states
                   for f in results.values() for x,y in zip(f,fresh.decisions))
        path.write_text("fixed visibility")

    monkeypatch.setattr(bench,"_targets",targets)
    monkeypatch.setattr(bench,"fit_zoom_refiner",fake_fit)
    monkeypatch.setattr(bench,"load_neural_weights",load)
    monkeypatch.setattr(bench,"write_guard_gallery",gallery)
    return inputs,fresh,truth,out,called,fit_groups


def test_oof_groups_never_train_on_their_held_frames_and_blind_opens_after_restore(tmp_path,monkeypatch):
    args=_arrange(tmp_path,monkeypatch)
    root, r, _ = bench.run_guarded_zoom(*args[:4],revision="d"*40)
    assert r["status"]=="passed",r.get("error")
    assert args[4]==[*bench.FOLD_ROLES,*bench.DEV_ROLES,bench.FRESH_ROLE]
    assert args[5]==[{2,3},{1,3},{1,2},{1,2,3}]
    assert len(r["training"]["folds"])==3
    assert all(k not in v["trained_roles"] for k,v in r["training"]["folds"].items())
    assert r["matrix"]["frame_counts"]=={"oof":9,"development":dict.fromkeys(bench.DEV_ROLES,3),"holdout":3}
    assert r["matrix"]["detector_and_visibility_frozen"]
    assert r["matrix"]["holdout_diagnostic"]["evaluated_recipe"]==r["matrix"]["selection"]["name"]
    assert content_hash(r["matrix"])==r["matrix_hash"] and (root/"guard.html").exists()


def test_tampering_with_gate_never_reads_fresh_targets(tmp_path,monkeypatch):
    args=_arrange(tmp_path,monkeypatch)
    original=bench.write_json
    def corrupt(path,value):
        if path.name=="development-gate.json":
            value={**value,"development_hash":"tampered"}
        return original(path,value)
    monkeypatch.setattr(bench,"write_json",corrupt)
    _,r,_=bench.run_guarded_zoom(*args[:4],revision="d"*40)
    assert r["status"]=="failed" and "frozen development gate" in r["error"]
    assert args[4]==[*bench.FOLD_ROLES,*bench.DEV_ROLES]


def test_completed_holdout_and_training_survive_player_write_failure(tmp_path,monkeypatch):
    args=_arrange(tmp_path,monkeypatch)
    def cannot_render(*args,**kwargs):
        raise OSError("cannot write player")
    monkeypatch.setattr(bench,"write_guard_gallery",cannot_render)
    _,r,_=bench.run_guarded_zoom(*args[:4],revision="d"*40)
    assert r["status"]=="failed" and "cannot write player" in r["error"]
    assert "gallery" not in r and len(r["training"]["folds"])==3
    assert len(r["matrix"]["holdout"])==5 and content_hash(r["matrix"])==r["matrix_hash"]
