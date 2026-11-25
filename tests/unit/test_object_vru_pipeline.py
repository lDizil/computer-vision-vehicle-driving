from dataclasses import replace

import pytest

from hybrid_drive.experiments.box_motion_inputs import load_motion_snapshot
from hybrid_drive.experiments.object_vru_association import associate_objects
from hybrid_drive.experiments.object_vru_detector import DetectedBox, _Tile, deduplicate_boxes, remap_box
from hybrid_drive.experiments.object_vru_geometry import GroundHit, GroundObject
from hybrid_drive.experiments.object_vru_trial import score_object_motion


_DESCRIPTOR = (1.,) + (0.,) * 35
_OTHER = (.55, .45) + (0.,) * 34


def hit(key, xy, *, kind="person", camera="front", signature=_DESCRIPTOR, sigma=.5):
    return GroundHit(key, kind, camera, xy, (0.,0.,1.8), .8, sigma, signature)


def obj(key, xy, *, kind="person", camera="front", signature=_DESCRIPTOR):
    return GroundObject(key, kind, xy, .8, (hit(key, xy, kind=kind, camera=camera, signature=signature),))


def test_same_camera_local_speed_and_radial_unit_come_from_measured_points():
    current=GroundObject("c", "person", (10.,0.), .9,(hit("a", (10.,1.), sigma=.2),))
    old=GroundObject("p", "person", (9.,0.), .9,(hit("b", (10.,0.), sigma=.3),))
    motion,=associate_objects((current,), (old,), .5, mode="appearance")
    assert motion.status=="estimated" and motion.previous_object_id=="p"
    assert motion.velocity_xy_mps==(0.,2.) and motion.past_xy_m==old.xy
    assert motion.common_cameras==("front",) and motion.radial_unit_xy[1] > 0


def test_camera_colour_breaks_spatial_tie_instead_of_declaring_an_identity_by_position():
    current=(obj("c",(10.,0.)),)
    old=(obj("colour-match",(9.,.12)),obj("colour-conflict",(9.,-.12),signature=_OTHER))
    geometry,=associate_objects(current,old,.5,mode="geometry")
    colour,=associate_objects(current,old,.5,mode="appearance")
    assert geometry.status=="ambiguous_history" and geometry.velocity_xy_mps==(0.,0.)
    assert colour.status=="estimated" and colour.previous_object_id=="colour-match"


def test_no_link_for_disjoint_views_classes_or_out_of_envelope_velocity():
    current=(obj("c",(10.,0.)),)
    for bad in (obj("b",(9.5,0.),camera="right"), obj("b",(9.8,0.),kind="bicycle"),obj("b",(6.,0.))):
        estimated,=associate_objects(current,(bad,),.5,mode="appearance")
        assert estimated.status=="no_reliable_history" and estimated.previous_object_id is None
        assert estimated.radial_unit_xy is None and estimated.velocity_xy_mps==(0.,0.)


def test_same_historical_detection_cannot_assign_motion_to_two_people():
    now=(obj("near",(10.2,0.)),obj("far",(11.,0.)))
    past=(obj("p",(10.,0.)),)
    a,b=associate_objects(now,past,.5,mode="appearance")
    assert a.status=="estimated" and b.status=="nonmutual_history"
    assert b.previous_object_id is None


def test_disagreeing_camera_relative_motion_cannot_create_a_velocity():
    a=GroundObject("now","person",(10.,0.),.8,(hit("a",(10.,0.)),hit("b",(10.,0.),camera="left")))
    b=GroundObject("old","person",(10.,0.),.8,(hit("d",(9.3,0.)),hit("e",(10.8,0.),camera="left")))
    motion,=associate_objects((a,),(b,),.5,mode="geometry")
    assert motion.status=="no_reliable_history"


def test_detection_tile_remaps_coordinates_and_suppresses_clipped_or_repeated_evidence():
    crop=_Tile(0,2,(200,80,760,408))
    mapped=remap_box(crop,1,.8,[20.,10.,42.,80.])
    assert mapped.xyxy==(220.,90.,242.,160.)
    assert remap_box(crop,1,.8,[20.,10.,42.,328.]) is None
    assert remap_box(crop,3,.8,[20.,10.,42.,80.]) is None
    duplicate=replace(mapped,confidence=.7,tile=0)
    nearby=DetectedBox("person",.6,(244.,90.,260.,160.),0)
    different=replace(mapped,kind="bicycle")
    assert set(deduplicate_boxes((duplicate,nearby,mapped,different)))=={nearby,mapped,different}


def test_rectified_camera_does_not_allow_resize_above_native_or_bool(tmp_path):
    for size in ((1921,1080),(True,216),[768,432]):
        with pytest.raises(ValueError, match="fit the native"):
            load_motion_snapshot(tmp_path,"none",{},rectified_wh=size)


def test_object_holdout_rejects_wrong_prediction_seal_before_label_access(tmp_path):
    raw=tmp_path/"predictions.json"
    raw.write_bytes(b"{}")
    missing=tmp_path/"not-mounted"
    with pytest.raises(ValueError, match="prediction seal"):
        score_object_motion(missing, missing, raw, missing, tmp_path/"score", sealed_sha256="0"*64)
