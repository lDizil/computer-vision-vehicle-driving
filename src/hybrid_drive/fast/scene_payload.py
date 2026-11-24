"""Immutable, snapshot-bound geometry handed from Fast to the experimental Bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import floor
from numbers import Integral

import numpy as np

from hybrid_drive.contracts._validation import finite_real, identifier, tuple_snapshot
from hybrid_drive.contracts.planning import CandidateSet
from hybrid_drive.contracts.sensors import timestamp_us
from hybrid_drive.fast.forecast_proposals import ForecastObject


@dataclass(frozen=True)
class SceneGrid:
    x_min: float
    y_min: float
    step_m: float
    rows: int
    cols: int

    def __post_init__(self):
        for field_name in ("x_min", "y_min", "step_m"):
            v=finite_real(getattr(self,field_name),field_name)
            object.__setattr__(self,field_name,v)
        if not .05 <= self.step_m <= 10 or max(abs(self.x_min),abs(self.y_min)) > 1000:
            raise ValueError("Unbounded or degenerate BEV grid")
        for name in ("rows", "cols"):
            n=getattr(self,name)
            if isinstance(n,(bool,np.bool_)) or not isinstance(n,Integral) or not 1 <= n <= 128:
                raise ValueError("BEV grid axes must have between 1 and 128 cells")
            object.__setattr__(self,name,int(n))

    def index(self,x,y):
        if not (self.x_min <= x < self.x_min+self.rows*self.step_m
                and self.y_min <= y < self.y_min+self.cols*self.step_m):
            return None
        return (min(self.rows-1,floor((x-self.x_min)/self.step_m)),
                min(self.cols-1,floor((y-self.y_min)/self.step_m)))

    def center(self,row,col):
        return self.x_min+(row+.5)*self.step_m,self.y_min+(col+.5)*self.step_m

    def xy(self):
        row=self.x_min+(np.arange(self.rows)+.5)*self.step_m
        col=self.y_min+(np.arange(self.cols)+.5)*self.step_m
        return np.meshgrid(row,col,indexing="ij")


@dataclass(frozen=True, eq=False, init=False)
class RoadEvidence:
    """Camera-derived evidence; pixels outside `known` are not confirmed road."""
    grid: SceneGrid
    content_sha256: str
    _drivable: bytes = field(repr=False)
    _divider: bytes = field(repr=False)
    _known: bytes = field(repr=False)

    def __init__(self,grid,drivable,lane_divider,known):
        stable=SceneGrid(grid.x_min,grid.y_min,grid.step_m,grid.rows,grid.cols)
        shape=(stable.rows,stable.cols)
        first,second=(_probabilities(d,shape) for d in (drivable,lane_divider))
        active=np.asarray(known)
        if active.shape!=shape or active.dtype!=np.bool_:
            raise ValueError("Road support must be a boolean array matching the grid")
        coverage=active.tobytes(order="C")
        digest=sha256(_json_bytes({"schema":"hybrid-drive-scene-road-v1","grid":asdict(stable)}))
        for value in (first,second,coverage):
            digest.update(value)
        for key,value in (("grid",stable),("content_sha256",digest.hexdigest()),
                          ("_drivable",first),("_divider",second),("_known",coverage)):
            object.__setattr__(self,key,value)

    @property
    def drivable(self):
        return self._unpack(self._drivable,"<f4")

    @property
    def lane_divider(self):
        return self._unpack(self._divider,"<f4")

    @property
    def known(self):
        return self._unpack(self._known,np.bool_)

    def _unpack(self,data,dtype):
        return np.frombuffer(data,dtype=dtype).reshape(self.grid.rows,self.grid.cols)


def _probabilities(value,shape):
    array=np.asarray(value)
    if (array.shape!=shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all()
            or (array < 0).any() or (array > 1).any()):
        raise ValueError("Road probabilities must be finite numbers in [0,1] matching the grid")
    return array.astype("<f4",copy=False).tobytes(order="C")


@dataclass(frozen=True, eq=False)
class FastScene:
    snapshot_id: str
    snapshot_sha256: str
    ref_time_us: int
    boxes: tuple[ForecastObject,...]
    plan: CandidateSet
    road: RoadEvidence
    content_sha256: str = field(init=False)

    def __post_init__(self):
        snapshot_id=identifier(self.snapshot_id,"snapshot_id")
        fingerprint=self.snapshot_sha256
        if (not isinstance(fingerprint,str) or len(fingerprint)!=64
                or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError("snapshot_sha256 must identify an exact captured observation")
        stamp=timestamp_us(self.ref_time_us,"ref_time_us")
        boxes=tuple_snapshot(self.boxes,"boxes")
        if len(boxes)>128 or any(not isinstance(box,ForecastObject) for box in boxes):
            raise ValueError("Expected at most 128 metric forecast boxes")
        if len({b.object_id for b in boxes})!=len(boxes):
            raise ValueError("Duplicate object ID in one snapshot")
        if (not isinstance(self.plan,CandidateSet) or self.plan.snapshot_id!=snapshot_id
                or self.plan.t_ref_s!=stamp/1_000_000):
            raise ValueError("Candidate set belongs to another snapshot/time")
        if not isinstance(self.road,RoadEvidence):
            raise ValueError("Expected immutable road evidence")
        object.__setattr__(self,"boxes",boxes)
        object.__setattr__(self,"ref_time_us",stamp)
        body={
            "schema":"hybrid-drive-fast-scene-v1","snapshot_id":snapshot_id,"snapshot_sha256":fingerprint,
            "ref_time_us":stamp,"boxes":[asdict(b) for b in boxes],"candidates_hash":self.plan.candidates_hash,
            "road_sha256":self.road.content_sha256,
        }
        object.__setattr__(self,"content_sha256",sha256(_json_bytes(body)).hexdigest())


def _json_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
