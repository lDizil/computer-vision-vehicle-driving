from dataclasses import replace

import pytest

from hybrid_drive.experiments.object_vru_association import associate_objects
from hybrid_drive.experiments.object_vru_geometry import GroundHit, GroundObject
from hybrid_drive.experiments.triple_vru_confirmation import confirm_objects, frozen_two_frame_velocity
from hybrid_drive.experiments.triple_vru_trial import score_triple


_A = (1.,) + (0.,) * 35
_B = (.60, .40) + (0.,) * 34
_C = (.20, .80) + (0.,) * 34


def seen(key, y, *, eyes=("front",), descriptor=_A):
    center = (10., y)
    return GroundObject(key, "person", center, .8,
                        tuple(GroundHit(key+eye, "person", eye, center, (0.,0.,1.5), .8, .2, descriptor)
                              for eye in eyes))


def check(current, mid, old, *, latest=.5, earlier=.5):
    first = associate_objects((current,), (mid,), latest, mode="appearance")
    item, = confirm_objects((current,), (mid,), None if old is None else (old,), first,
                            latest, None if old is None else earlier)
    return item


def test_consistent_motion_uses_actual_uneven_frame_times_and_one_fixed_tangent():
    answer = check(seen("now",0), seen("mid",-.4), seen("old",-.95), latest=.4, earlier=.55)
    assert answer.status == "confirmed" and answer.older_object_id == "old" and answer.common_cameras == ("front",)
    assert answer.trend_velocity_xy_mps == pytest.approx((0.,1.))


def test_missing_sensor_observation_and_empty_detection_bundle_are_distinct_abstentions():
    n,m=seen("now",0),seen("mid",-.4)
    link=associate_objects((n,),(m,),.5,mode="appearance")
    missing,=confirm_objects((n,),(m,),None,link,.5,None)
    empty,=confirm_objects((n,),(m,),(),link,.5,.5)
    assert (missing.status,empty.status)==("no_older_observation","no_older_link")
    assert missing.older_object_id is None and empty.trend_velocity_xy_mps==(0.,0.)


def test_motion_floor_rejects_small_drift_even_with_two_consistent_links():
    result=check(seen("now",0),seen("mid",-.08),seen("old",-.16))
    assert result.status=="below_motion_floor" and result.trend_velocity_xy_mps==(0.,0.)


def test_opposite_legs_with_enough_distance_do_not_produce_a_velocity():
    result=check(seen("now",0),seen("mid",-.7),seen("old",-.35))
    assert result.status=="inconsistent_motion" and result.older_object_id is None


def test_two_pairs_in_different_camera_sets_are_not_a_three_way_observation():
    result=check(seen("now",0,eyes=("left",)),seen("mid",-.4,eyes=("left","right")),
                 seen("old",-.8,eyes=("right",)))
    assert result.status=="no_common_camera"


def test_first_to_third_crop_can_contradict_two_individually_plausible_links():
    result=check(seen("now",0,descriptor=_A),seen("mid",-.4,descriptor=_B),seen("old",-.8,descriptor=_C))
    assert result.status=="inconsistent_appearance"


def test_initial_history_identity_must_refer_to_the_supplied_middle_observation():
    n,m,o=seen("now",0),seen("mid",-.4),seen("old",-.8)
    first,=associate_objects((n,),(m,),.5,mode="appearance")
    with pytest.raises(ValueError, match="middle observation"):
        confirm_objects((n,),(m,),(o,),(replace(first, previous_object_id="other"),),.5,.5)


def test_frozen_pair_gain_removes_radial_component_and_keeps_tangential_measurement():
    n,m=seen("now",0),seen("mid",-.4)
    first,=associate_objects((n,),(m,),.5,mode="appearance")
    first=replace(first,velocity_xy_mps=(2.,.8))
    assert frozen_two_frame_velocity(first,radial_gain=0.,tangential_gain=1.)==(0.,.8)


def test_wrong_prediction_seal_does_not_touch_labels_or_input_roots(tmp_path):
    forecast=tmp_path/"predictions.json"
    forecast.write_bytes(b"{}")
    invalid=tmp_path/"not-mounted"
    with pytest.raises(ValueError,match="recorded seal"):
        score_triple(invalid,invalid,forecast,invalid,tmp_path/"score",sealed_sha256="0"*64)
