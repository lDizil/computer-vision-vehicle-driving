from dataclasses import replace

import pytest

from hybrid_drive.experiments.object_vru_association import associate_objects
from hybrid_drive.experiments.object_vru_geometry import GroundHit, GroundObject
from hybrid_drive.experiments.vru_observation_fusion import fuse_vru_observations


def seen(key, x, y, *, kind="person", confidence=.5):
    hit = GroundHit(f"hit-{key}", kind, "front", (x,y), (0.,0.,1.5), confidence, .25, (1.,) + (0.,)*35)
    return GroundObject(key, kind, (x,y), confidence, (hit,))


def test_fusion_preserves_primary_coordinate_and_rejects_nearby_even_if_other_class():
    s = seen("d0", 8., 0.)
    near = seen("d0", 8.9, 0., kind="bicycle", confidence=.99)
    far = seen("d1", 9., 0.)
    a,b = fuse_vru_observations((s,), (near,far))
    assert (a.object_id,b.object_id) == ("ssd:d0","rcnn:d1")
    assert a.xy == s.xy and a.confidence == s.confidence and b.xy == far.xy
    assert a.hits[0].camera_name == s.hits[0].camera_name and a.hits[0].appearance == s.hits[0].appearance


def test_empty_primary_retains_secondary_and_object_and_hit_ids_cannot_collide():
    a = seen("d", 10., 0.)
    b = seen("d", 11., 0.)
    first,second = fuse_vru_observations((a,), (b,))
    assert first.object_id != second.object_id
    assert first.hits[0].detection_id != second.hits[0].detection_id
    assert fuse_vru_observations((), (b,)) == (second,)
    assert fuse_vru_observations((a,), ()) == (first,)
    assert fuse_vru_observations((), ()) == ()


def test_duplicates_inside_one_profile_are_rejected_before_tracking():
    a = seen("d", 10., 0.)
    with pytest.raises(ValueError, match="object identity"):
        fuse_vru_observations((), (a,a))
    with pytest.raises(ValueError, match="hit identity"):
        fuse_vru_observations((a,replace(a,object_id="another")), ())


def test_unchanged_association_consumes_namespaced_ground_objects():
    now = fuse_vru_observations((seen("n",10.,.0),), (seen("n",12.,0.),))
    past = fuse_vru_observations((seen("p",10.,-.3),), (seen("p",12.,-.3),))
    links = associate_objects(now, past, .5, mode="appearance")
    assert [r.previous_object_id for r in links]==["ssd:p","rcnn:p"]
    assert [r.status for r in links]==["estimated","estimated"]
