"""Frozen synthetic validation geometry, not a real-road or held-out benchmark."""

import numpy as np

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory

from .cases import (CounterfactualPair, QualificationCase, QualificationSuite,
                    _digest, _images_hash, _json, _rig_manifest)
from .render import Box, render_cameras


VALIDATION_VERSION = "synthetic-validation-v1"
VALIDATION_CASE_IDS = tuple(f"v{index:02d}" for index in range(30))
_GEOMETRIES = (
    ("near", (1.5, 3, 4.5, 6, 7.5), (.35, .8, 1.45, 2, 2.45),
     (-.25, -.7, -1.3, -1.85, -2.4), 3, (1.0, .8, 1.2)),
    ("middle", (2, 4.5, 7, 9.5, 12), (.15, .6, 1.4, 2.6, 3.2),
     (-.45, -1.0, -1.55, -2.1, -2.8), 3, (1.8, 1.3, 1.8)),
    ("far", (3, 6, 9, 12, 15), (.5, .95, 1.8, 3.1, 4.0),
     (-.2, -.65, -1.1, -1.9, -3.0), 3, (2.2, 1.6, 2.1)),
)


def build_validation_suite() -> QualificationSuite:
    ids = tuple("c-" + _digest({"suite": VALIDATION_VERSION, "candidate_slot": slot})[:16]
                for slot in range(2))
    motion = EgoMotion(4, 0, 0, 0)
    cases, pairs, geometry = [], [], []
    for family, (name, xs, left_y, right_y, centre_index, dimensions) in enumerate(_GEOMETRIES):
        paths = tuple(np.column_stack((xs, ys)) for ys in (left_y, right_y))
        trajectories = tuple(_trajectory(path) for path in paths)
        boxes = tuple(Box(*path[centre_index], *dimensions) for path in paths)
        layouts = ((boxes[0],), (boxes[1],), (), boxes)
        offset = family * 10
        family_ids = VALIDATION_CASE_IDS[offset:offset + 10]
        for index, case_id in enumerate(family_ids):
            blank, swapped = index >= 8, index in (4, 5, 6, 7, 9)
            layout = index % 4
            candidates = tuple(Candidate(candidate_id, trajectories[slot ^ swapped])
                               for slot, candidate_id in enumerate(ids))
            payload = {"ids": list(ids), "poses": [candidate.trajectory.poses.tolist() for candidate in candidates]}
            plan_id = "p-" + _digest(payload)[:24]
            draft = CandidateSet(plan_id, "pending", 0, Pose2D(0, 0, 0), candidates)
            raw, cameras = render_cameras(draft, () if blank else layouts[layout], blank=blank)
            plan = CandidateSet(plan_id, "s-" + _images_hash(raw)[:24], 0, Pose2D(0, 0, 0), candidates)
            # Literal labels follow the fixed centre-crossing construction, never projection output.
            slots = () if blank else ((0,), (1,), (), (0, 1))[layout]
            vetoed = tuple(ids[slot ^ swapped] for slot in slots)
            status = RiskStatus.UNKNOWN if blank else (RiskStatus.VETO if vetoed else RiskStatus.NO_VETO)
            title = "Нет наблюдаемой сцены" if blank else ("Препятствие слева", "Препятствие справа",
                                                         "Без препятствий", "Два препятствия")[layout]
            cases.append(QualificationCase(case_id, f"{name}: {title}", plan, motion, cameras, raw, status, vetoed))
        pairs.extend(CounterfactualPair(f"image-{family_ids[a]}-{family_ids[b]}", "image_change",
                                        family_ids[a], family_ids[b])
                     for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)))
        pairs.extend(CounterfactualPair(f"binding-{family_ids[a]}-{family_ids[b]}", "candidate_change",
                                        family_ids[a], family_ids[b])
                     for a, b in ((0, 4), (1, 5), (2, 6), (3, 7), (8, 9)))
        geometry.append({"family": name, "case_ids": list(family_ids),
                         "left_path_xy_m": paths[0].tolist(), "right_path_xy_m": paths[1].tolist(),
                         "boxes_centres_xy_m": [path[centre_index].tolist() for path in paths],
                         "box_dimensions_lwh_m": list(dimensions)})
    definition = {
        "full_case_ids": list(VALIDATION_CASE_IDS), "suite_role": "synthetic_validation",
        "hand_checked_geometry": {
            "families": geometry,
            "check": "Each matching future centreline passes through the box footprint centre; "
                     "the opposite path stays on the other side of y=0 throughout the box x interval. "
                     "Labels are declared independently of rendering and projection.",
        },
        "rig": _rig_manifest(),
    }
    return QualificationSuite(tuple(cases), tuple(pairs), VALIDATION_VERSION, _json(definition))


def _trajectory(xy):
    headings = np.arctan2(np.diff(xy[:, 1], prepend=0), np.diff(xy[:, 0], prepend=0))
    return Trajectory(np.column_stack((xy, headings)), .5)
