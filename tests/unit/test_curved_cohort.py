from dataclasses import replace

import numpy as np
import pytest

from test_cohort_score import bundle, evidence_for
from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.experiments import cohort_score as scoring
from hybrid_drive.experiments.navsim import sample_arbiter
from hybrid_drive.experiments.real_replay import plan_for_bundle
from hybrid_drive.experiments.real_bundle import RealSceneBundle
from hybrid_drive.safety.fallback_policy import FallbackModel, FallbackObservation


def lateral_bundle(bundle):
    return RealSceneBundle(bundle.scene_token, bundle.frame_timestamp_us, bundle.trajectory,
                           EgoMotion(2, .1, 0, .2), bundle.views, bundle.provenance)


def test_policy_changes_only_fallback_and_keeps_original_evidence_clock(bundle):
    bundle = lateral_bundle(bundle)
    frozen = evidence_for(bundle, state="blocked", checkpoint=102, received=101)
    plan = plan_for_bundle(bundle)
    observation = FallbackObservation(plan.snapshot_id, plan.t_ref_s, bundle.motion, .03, "f"*64)
    old_outputs,old_details = scoring._freeze_predictions(bundle,frozen,sample_arbiter())
    calls=[]
    engine=replace(sample_arbiter(),fallback_model=FallbackModel.CURVED)
    class RecordingArbiter:
        def decide(self, plan, motion, **kwargs):
            calls.append(kwargs)
            return engine.decide(plan,motion,**kwargs)
    new_outputs,new_details=scoring._freeze_predictions(bundle,frozen,RecordingArbiter(),fallback_observation=observation)
    assert old_details["combined"]["failure"] == "unsupported_motion"
    assert new_details["combined"]["failure"] is None
    assert new_details["combined"]["fallback_model_report"]["heading_rate_radps"] == .03
    assert new_details["combined"]["motion_observation_hash"] == "f"*64
    for mode in ("baseline","checks_only"):
        np.testing.assert_array_equal(old_outputs[mode].poses,new_outputs[mode].poses)
    assert [value["now_monotonic_s"] for value in calls] == [102,102]
    assert calls[1]["received_at_monotonic_s"] == 101
    assert calls[1]["request"] is frozen.context.request
    assert all(value["fallback_observation"] is observation for value in calls)


def test_curved_unknown_slow_keeps_fast_candidate(bundle):
    bundle=lateral_bundle(bundle)
    frozen=evidence_for(bundle,state="unknown")
    plan=plan_for_bundle(bundle)
    observation=FallbackObservation(plan.snapshot_id,plan.t_ref_s,bundle.motion,.03,"f"*64)
    outputs,details=scoring._freeze_predictions(bundle,frozen,replace(sample_arbiter(),fallback_model=FallbackModel.CURVED),fallback_observation=observation)
    assert outputs["combined"] is outputs["checks_only"] is bundle.trajectory
    assert details["combined"]["slow_disposition"] == "unknown_result"
    assert not details["combined"]["fallback_invoked"]


@pytest.mark.parametrize("model,root", [("curved",None),("longitudinal","sidecars"),("other",None)])
def test_invalid_comparison_composition_fails_before_scoring(model,root):
    with pytest.raises(ValueError): scoring._fallback_composition(model,root,{})


def test_invalid_sidecar_does_not_fabricate_baseline_planning_failure(bundle, monkeypatch):
    bundle = lateral_bundle(bundle)
    frozen = evidence_for(bundle, state="blocked")
    cohort = {"source_revision":bundle.provenance["source_revision"], "sources_hash":"f"*64,
              "model":{"checkpoint_sha256":bundle.provenance["checkpoint_sha256"]}}
    case = {"case_id":"r00", "status":"success", "scene_token":bundle.scene_token,
            "bundle_path":"r00/bundle", "bundle_hash":bundle.bundle_hash,
            "raw_candidate_hash":bundle.provenance["raw_candidate_hash"]}
    manifest = dict(raw_candidate_hash=case["raw_candidate_hash"], model=cohort["model"],
                    source_revision=cohort["source_revision"], dataset_source_hash=cohort["sources_hash"],
                    manifest_hash=bundle.provenance["source_experiment_manifest_hash"])
    monkeypatch.setattr(scoring,"load_real_bundle",lambda path:bundle)
    monkeypatch.setattr(scoring,"_read_bounded",lambda *args:b"{}")
    monkeypatch.setattr(scoring,"_decode_json",lambda data:manifest)
    monkeypatch.setattr(scoring,"verify_manifest",lambda value:None)
    def invalid(*args): raise ValueError("invalid observation")
    monkeypatch.setattr(scoring,"_case_observation",invalid)
    monkeypatch.setattr(scoring,"_case_evidence",lambda *args:frozen)
    engine=replace(sample_arbiter(),fallback_model=FallbackModel.CURVED)
    item=scoring._freeze_case(case,cohort,"fast","evidence",{},None,engine,
                             motion_observation_root="motion",motion_index={})
    assert item["outputs"]["baseline"] is item["outputs"]["checks_only"] is bundle.trajectory
    assert item["details"]["baseline"]["failure"] is None
    assert item["details"]["checks_only"]["failure"] is None
    assert item["details"]["combined"]["failure"] == "missing_heading_rate"
    assert "invalid observation" in item["details"]["combined"]["motion_observation_error"]
