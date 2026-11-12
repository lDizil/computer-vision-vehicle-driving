from dataclasses import dataclass
import hashlib
import json

import numpy as np

from hybrid_drive.contracts.motion import EgoMotion
from hybrid_drive.contracts.planning import Candidate, CandidateSet
from hybrid_drive.contracts.slow import RiskStatus, SlowRequest
from hybrid_drive.contracts.trajectory import Pose2D, Trajectory
from hybrid_drive.slow.context import CAMERA_NAMES, CameraFrame, SlowContext

from .render import Box, CANDIDATE_COLORS, render_cameras, schematic_calibrations


VERSION = "controlled-schematic-v1"
FULL_CASE_IDS = tuple(f"q{index:02d}" for index in range(10))
_MOTION_FIELDS = ("vx_mps", "vy_mps", "ax_mps2", "ay_mps2")


@dataclass(frozen=True, eq=False)
class QualificationCase:
    case_id: str
    title: str
    plan: CandidateSet
    motion: EgoMotion
    cameras: tuple[CameraFrame, ...]
    raw_cameras: tuple[CameraFrame, ...]
    expected_status: RiskStatus
    expected_veto_ids: tuple[str, ...]

    def __post_init__(self):
        _identifier(self.case_id, "case_id")
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("title must be nonempty text")
        if not isinstance(self.plan, CandidateSet) or not isinstance(self.motion, EgoMotion):
            raise ValueError("A case requires CandidateSet and EgoMotion")
        for name in ("cameras", "raw_cameras"):
            frames = _frames(getattr(self, name))
            object.__setattr__(self, name, frames)
        if any((a.width, a.height) != (b.width, b.height)
               for a, b in zip(self.cameras, self.raw_cameras)):
            raise ValueError("Raw and overlaid camera dimensions must match")
        if not isinstance(self.expected_status, RiskStatus):
            raise ValueError("expected_status must be RiskStatus")
        vetoed = _tuple(self.expected_veto_ids, "expected_veto_ids")
        for value in vetoed:
            _identifier(value, "expected veto ID")
        known = {candidate.candidate_id for candidate in self.plan.candidates}
        if len(set(vetoed)) != len(vetoed) or any(value not in known for value in vetoed):
            raise ValueError("Expected veto IDs must be unique known candidate IDs")
        if (self.expected_status is RiskStatus.VETO) != bool(vetoed):
            raise ValueError("Only expected VETO must name one or more candidate IDs")
        object.__setattr__(self, "expected_veto_ids", vetoed)

    @property
    def scene_hash(self) -> str:
        return _images_hash(self.raw_cameras)

    @property
    def candidate_hash(self) -> str:
        return _digest(_candidate_payload(self.plan))

    @property
    def input_hash(self) -> str:
        return _digest({"schema": "qualification-input-v1", "candidates": _candidate_payload(self.plan),
                        "plan_id": self.plan.plan_id, "snapshot_id": self.plan.snapshot_id,
                        "motion": _motion(self), "images": _images_hash(self.cameras)})

    def to_context(self, model_version: str, *, request_id: str,
                   submitted_at_monotonic_s: float, deadline_monotonic_s: float) -> SlowContext:
        request = SlowRequest(
            request_id, self.plan, submitted_at_monotonic_s, deadline_monotonic_s,
            self.plan.t_ref_s + self.plan.candidates[0].trajectory.horizon_s, model_version,
        )
        return SlowContext(request, self.motion, self.cameras)


@dataclass(frozen=True)
class CounterfactualPair:
    pair_id: str
    kind: str
    first_case_id: str
    second_case_id: str

    def __post_init__(self):
        for name in ("pair_id", "first_case_id", "second_case_id"):
            _identifier(getattr(self, name), name)
        if self.kind not in ("image_change", "candidate_change"):
            raise ValueError("Unknown counterfactual kind")
        if self.first_case_id == self.second_case_id:
            raise ValueError("Counterfactual cases must be distinct")


@dataclass(frozen=True)
class QualificationSuite:
    cases: tuple[QualificationCase, ...]
    pairs: tuple[CounterfactualPair, ...]
    version: str
    definition_json: str | None = None

    def __post_init__(self):
        _identifier(self.version, "version")
        cases, pairs = _tuple(self.cases, "cases"), _tuple(self.pairs, "pairs")
        if not cases or any(not isinstance(case, QualificationCase) for case in cases):
            raise ValueError("Suite must contain QualificationCase values")
        if any(not isinstance(pair, CounterfactualPair) for pair in pairs):
            raise ValueError("Suite pairs must be CounterfactualPair values")
        by_id = {case.case_id: case for case in cases}
        if len(by_id) != len(cases) or len({pair.pair_id for pair in pairs}) != len(pairs):
            raise ValueError("Case and pair IDs must be unique")
        definition = _suite_definition(self.version, self.definition_json)
        if definition is not None and not set(by_id) <= set(definition["full_case_ids"]):
            raise ValueError("Suite contains cases outside its definition")
        if self.definition_json is not None:
            object.__setattr__(self, "definition_json", _json(definition))
        edges = set()
        for pair in pairs:
            edge = (pair.kind, frozenset((pair.first_case_id, pair.second_case_id)))
            if edge in edges:
                raise ValueError("Duplicate counterfactual pair")
            edges.add(edge)
            if pair.first_case_id not in by_id or pair.second_case_id not in by_id:
                raise ValueError("Counterfactual pair references a missing case")
            _validate_pair(pair, by_id[pair.first_case_id], by_id[pair.second_case_id])
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "pairs", pairs)

    def select(self, case_ids) -> "QualificationSuite":
        requested = _tuple(case_ids, "case_ids")
        if len(set(requested)) != len(requested):
            raise ValueError("Selected case IDs must be unique")
        known = {case.case_id for case in self.cases}
        if not set(requested) <= known:
            raise ValueError("Unknown selected case ID")
        selected = set(requested)
        return QualificationSuite(tuple(case for case in self.cases if case.case_id in selected),
                                  tuple(pair for pair in self.pairs if
                                        {pair.first_case_id, pair.second_case_id} <= selected),
                                  self.version, self.definition_json)

    def manifest(self) -> dict:
        definition = _suite_definition(self.version, self.definition_json)
        full_ids = definition["full_case_ids"] if definition is not None else ()
        full = bool(full_ids) and {case.case_id for case in self.cases} == set(full_ids)
        known = [case for case in self.cases if case.expected_status is not RiskStatus.UNKNOWN]
        positives = sum(len(case.expected_veto_ids) for case in known)
        total = sum(len(case.plan.candidates) for case in known)
        manifest = {
            "version": self.version, "scope": "controlled_suite" if full else "smoke_subset",
            "case_count": len(self.cases), "full_suite_case_count": len(full_ids),
            "label_source": "analytic_controlled_scene", "real_street_or_holdout": False,
            "label_contract": "Future candidate centreline intersects the solid box ground footprint; "
                              "not swept-vehicle collision, dynamics or road-safety qualification.",
            "known_candidate_units": total, "risk_units": positives, "clear_units": total - positives,
            "unknown_controls": len(self.cases) - len(known),
            "blank_control": "Raw views are black; projected candidate overlays are retained. "
                             "No observed world evidence means UNKNOWN, not inferred clear road.",
            "overlay_colors_by_candidate_order_rgb": [list(color) for color in CANDIDATE_COLORS],
            "cases": [_case_manifest(case) for case in self.cases],
            "pairs": [dict(pair_id=pair.pair_id, kind=pair.kind, first_case_id=pair.first_case_id,
                           second_case_id=pair.second_case_id) for pair in self.pairs],
        }
        if definition is not None:
            manifest["hand_checked_geometry"] = definition["hand_checked_geometry"]
            manifest["rig"] = definition["rig"]
            if self.definition_json is not None:
                manifest["suite_role"] = definition["suite_role"]
        else:
            manifest["label_source"] = "unspecified"
            manifest["label_contract"] = "unspecified"
            manifest["blank_control"] = "unspecified"
        return manifest


def build_suite() -> QualificationSuite:
    ids = tuple("c-" + _digest({"candidate_slot": slot})[:16] for slot in range(2))
    xy = np.array([[2, .2], [4, .5], [6, 1], [8, 1.5], [10, 2]])
    left = Trajectory(np.column_stack((xy, np.arctan([.1, .175, .25, .25, .25]))), .5)
    right_poses = left.poses.copy()
    right_poses[:, 1:] *= -1
    trajectories = (left, Trajectory(right_poses, .5))
    layouts = ((Box(8, 1.5),), (Box(8, -1.5),), (), (Box(8, 1.5), Box(8, -1.5)))
    titles = ("Препятствие слева", "Препятствие справа", "Без препятствий", "Два препятствия")
    motion = EgoMotion(4, 0, 0, 0)
    cases = []
    for index in range(10):
        blank, swapped = index >= 8, index in (4, 5, 6, 7, 9)
        layout = index % 4
        candidates = tuple(Candidate(candidate_id, trajectories[slot ^ swapped])
                           for slot, candidate_id in enumerate(ids))
        geometry = {"ids": list(ids), "poses": [candidate.trajectory.poses.tolist() for candidate in candidates]}
        plan_id = "p-" + _digest(geometry)[:24]
        draft = CandidateSet(plan_id, "pending", 0, Pose2D(0, 0, 0), candidates)
        raw, cameras = render_cameras(draft, () if blank else layouts[layout], blank=blank)
        plan = CandidateSet(plan_id, "s-" + _images_hash(raw)[:24], 0, Pose2D(0, 0, 0), candidates)
        # These labels are declared from the hand-checked x=8 intersection, not renderer/model output.
        slots = () if blank or layout == 2 else ((0, 1) if layout == 3 else (layout,))
        vetoed = tuple(ids[slot ^ swapped] for slot in slots)
        status = RiskStatus.UNKNOWN if blank else (RiskStatus.VETO if vetoed else RiskStatus.NO_VETO)
        title = "Нет наблюдаемой сцены" if blank else titles[layout]
        cases.append(QualificationCase(f"q{index:02d}", title, plan, motion, cameras, raw, status, vetoed))
    pairs = [CounterfactualPair(f"image-{a:02d}-{b:02d}", "image_change", f"q{a:02d}", f"q{b:02d}")
             for a, b in ((0, 1), (2, 3), (4, 5), (6, 7))]
    pairs += [CounterfactualPair(f"binding-{a:02d}-{b:02d}", "candidate_change", f"q{a:02d}", f"q{b:02d}")
              for a, b in ((0, 4), (1, 5), (2, 6), (3, 7), (8, 9))]
    return QualificationSuite(tuple(cases), tuple(pairs), VERSION)


def _suite_definition(version, definition_json):
    if definition_json is None:
        return _legacy_definition() if version == VERSION else None
    if not isinstance(definition_json, str):
        raise ValueError("Suite definition must be a JSON string")
    try:
        definition = json.loads(definition_json, object_pairs_hook=_unique_json_object)
        _json(definition)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Suite definition must be finite JSON with unique keys") from exc
    fields = {"full_case_ids", "suite_role", "hand_checked_geometry", "rig"}
    if not isinstance(definition, dict) or set(definition) != fields:
        raise ValueError("Suite definition has an invalid schema")
    ids = definition["full_case_ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError("Suite definition requires full_case_ids")
    for case_id in ids:
        _identifier(case_id, "full case ID")
    if len(set(ids)) != len(ids):
        raise ValueError("Suite definition full_case_ids must be unique")
    if definition["suite_role"] not in ("controlled_diagnostic", "synthetic_validation"):
        raise ValueError("Suite definition has an invalid role")
    for name in ("hand_checked_geometry", "rig"):
        if not isinstance(definition[name], dict) or not definition[name]:
            raise ValueError(f"Suite definition {name} must be a nonempty object")
    return definition


def _unique_json_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate suite definition key")
        result[key] = value
    return result


def _legacy_definition():
    return {
        "full_case_ids": list(FULL_CASE_IDS), "suite_role": "controlled_diagnostic",
        "hand_checked_geometry": {"left_path_xy_m": [[2, .2], [4, .5], [6, 1], [8, 1.5], [10, 2]],
                                  "right_path_y": "negative of left path y",
                                  "boxes_centres_xy_m": [[8, 1.5], [8, -1.5]],
                                  "box_dimensions_lwh_m": [1.3, 1.1, 1.4],
                                  "check": "At x=8 the matching path goes through the footprint centre; "
                                           "the opposite path stays on the other side of y=0."},
        "rig": _rig_manifest(),
    }


def _rig_manifest():
    return {"provenance": "synthetic pinhole, not NAVSIM calibration", "ego_axes": "x forward, y left, z up",
            "cameras": [{"name": camera.name, "width": camera.width, "height": camera.height,
                         "intrinsic": camera.intrinsic.tolist(),
                         "ego_to_camera": camera.ego_to_camera.tolist(), "near_m": camera.near_m}
                        for camera in schematic_calibrations()]}


def _validate_pair(pair, first, second):
    if _motion(first) != _motion(second):
        raise ValueError("Counterfactual motion must be unchanged")
    if pair.kind == "image_change":
        if first.candidate_hash != second.candidate_hash or first.scene_hash == second.scene_hash:
            raise ValueError("Image counterfactual must change only scene, not candidate geometry")
        if (first.expected_status, set(first.expected_veto_ids)) == (second.expected_status, set(second.expected_veto_ids)):
            raise ValueError("Image counterfactual must have different expected outputs")
    else:
        if first.scene_hash != second.scene_hash or first.candidate_hash == second.candidate_hash:
            raise ValueError("Binding counterfactual must preserve raw scene and change candidates")
        if _candidate_payload(first.plan)["frame"] != _candidate_payload(second.plan)["frame"]:
            raise ValueError("Binding counterfactual must preserve reference frame and time")
        a, b = _bindings(first), _bindings(second)
        if set(a) != set(b) or sorted(a.values()) != sorted(b.values()) or a == b:
            raise ValueError("Binding counterfactual must permute the same candidate IDs and geometries")
        if first.expected_status != second.expected_status or {a[value] for value in first.expected_veto_ids} != {
            b[value] for value in second.expected_veto_ids
        }:
            raise ValueError("Expected labels must follow candidate geometry when bindings change")
    if first.input_hash == second.input_hash:
        raise ValueError("Counterfactual must change actual model input")


def _bindings(case):
    return {candidate.candidate_id: _digest({"poses": candidate.trajectory.poses.tolist(),
                                           "interval_s": candidate.trajectory.interval_s})
            for candidate in case.plan.candidates}


def _candidate_payload(plan):
    return {"frame": {"t_ref_s": plan.t_ref_s,
                      "reference_pose": [plan.reference_pose.x, plan.reference_pose.y, plan.reference_pose.heading]},
            "candidates": [{"candidate_id": candidate.candidate_id, "poses": candidate.trajectory.poses.tolist(),
                            "interval_s": candidate.trajectory.interval_s, "times_s": candidate.trajectory.times_s.tolist()}
                           for candidate in plan.candidates]}


def _case_manifest(case):
    return {"case_id": case.case_id, "title": case.title, "input_hash": case.input_hash,
            "scene_hash": case.scene_hash, "candidate_hash": case.candidate_hash,
            "expected_status": case.expected_status.value, "expected_veto_ids": list(case.expected_veto_ids),
            "candidate_data": _candidate_payload(case.plan), "motion": _motion(case)}


def _motion(case):
    return {name: getattr(case.motion, name) for name in _MOTION_FIELDS}


def _images_hash(frames):
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(_json({"name": frame.name, "width": frame.width, "height": frame.height}).encode("ascii"))
        digest.update(frame.rgb.tobytes())
    return digest.hexdigest()


def _frames(value):
    frames = _tuple(value, "cameras")
    if len(frames) != 3 or any(not isinstance(frame, CameraFrame) for frame in frames):
        raise ValueError("Exactly three CameraFrame values are required")
    if tuple(frame.name for frame in frames) != CAMERA_NAMES:
        raise ValueError("Camera frames must be in L0/F0/R0 order")
    return frames


def _tuple(value, name):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc


def _identifier(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty identifier")


def _digest(value):
    return hashlib.sha256(_json(value).encode("ascii")).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
