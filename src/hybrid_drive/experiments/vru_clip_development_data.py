"""Plan a balanced development set from prior, already cached recording groups."""

from hashlib import sha256
from pathlib import Path

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments.bounded_io import canonical_json
from hybrid_drive.experiments.vru_clip_data import (
    _Selection,
    _group,
    _new_plan_path,
    _plan,
    _publish_plan,
    _require_linux,
)
from hybrid_drive.experiments.vru_fusion_holdout_data import (
    CHAIN_CONTROL_SOURCE_SHA256,
    SOURCE_SHA256 as REPLAY_SOURCE_SHA256,
    _load_source as _load_prior_source,
)


DEVELOPMENT_PLAN_SCHEMA = "hybrid-drive-vru-clip-development-plan-v1"
_SOURCE_SCHEMA = "hybrid-drive-vru-clip-development-sources-v1"
_SELECTION = _Selection(per_band=20, min_groups=6, max_per_group=16)
_SHARED_SOURCE_FIELDS = ("dataset", "dataset_revision", "publisher_mapping_url",
                         "publisher_mapping_sha256", "metadata_archive")


def development_plan(metadata_archive: Path, replay_manifest: Path, chain_manifest: Path, output: Path) -> dict:
    """Plan sixty old-group clips using only metadata, never the fixed control inputs."""
    _require_linux()
    output = _new_plan_path(output)
    source = _development_source(replay_manifest, chain_manifest)
    digest = sha256(canonical_json(source)).hexdigest()
    plan = _plan(metadata_archive, source, selection=_SELECTION, schema=DEVELOPMENT_PLAN_SCHEMA, source_sha256=digest)
    return _publish_plan(plan, output)


def _development_source(replay_manifest, chain_manifest):
    replay = _load_prior_source(replay_manifest)
    chain = _load_prior_source(chain_manifest)
    expected = (REPLAY_SOURCE_SHA256, CHAIN_CONTROL_SOURCE_SHA256)
    if tuple(sha256(canonical_json(source)).hexdigest() for source in (replay, chain)) != expected:
        raise PreparationError("Expected original replay and chain-control sources in that order")
    shared = {key: replay[key] for key in _SHARED_SOURCE_FIELDS}
    if any(chain[key] != value for key, value in shared.items()):
        raise PreparationError("Development sources do not share the pinned dataset revision")
    archives = replay["camera_archives"]
    if archives.keys() & chain["camera_archives"].keys():
        raise PreparationError("Development source archives overlap")
    logs = replay["logs"] + chain["logs"]
    if len(logs) != 10 or len({_group(log["recording"]) for log in logs}) != len(logs):
        raise PreparationError("Development sources must cover ten distinct prior recording groups")
    return {"schema": _SOURCE_SCHEMA, **shared,
            "parent_source_sha256": list(expected),
            "camera_archives": {**archives, **chain["camera_archives"]},
            "logs": logs}
