"""Plan a balanced development set from prior, already cached recording groups."""

from hashlib import sha256
from pathlib import Path

from hybrid_drive.assets.files import PreparationError
from hybrid_drive.experiments.bounded_io import canonical_json
from hybrid_drive.experiments.vru_clip_data import (
    _CorpusProfile,
    _Selection,
    _group,
    _load_labels,
    _new_plan_path,
    _plan,
    _prepare_corpus,
    _publish_plan,
    _read_index,
    _require_linux,
)
from hybrid_drive.experiments.vru_fusion_holdout_data import (
    CHAIN_CONTROL_SOURCE_SHA256,
    SOURCE_SHA256 as REPLAY_SOURCE_SHA256,
    _CONTROL_RECORDINGS as _CHAIN_RECORDINGS,
    _RECORDINGS as _REPLAY_RECORDINGS,
    _load_source as _load_prior_source,
)


DEVELOPMENT_PLAN_SCHEMA = "hybrid-drive-vru-clip-development-plan-v1"
DEVELOPMENT_INPUT_SCHEMA = "hybrid-drive-vru-clip-development-inputs-v1"
DEVELOPMENT_LABEL_SCHEMA = "hybrid-drive-vru-clip-development-targets-v1"
DEVELOPMENT_SOURCE_SHA256 = "d534e69a426cfeab6bac03b73c130bff2c0cba1e110273bfd4f1e2f6b819543e"
_SOURCE_SCHEMA = "hybrid-drive-vru-clip-development-sources-v1"
_PROFILE = _CorpusProfile(DEVELOPMENT_PLAN_SCHEMA, DEVELOPMENT_INPUT_SCHEMA, DEVELOPMENT_LABEL_SCHEMA,
                          DEVELOPMENT_SOURCE_SHA256, _Selection(per_band=20, min_groups=6, max_per_group=16),
                          recordings=_REPLAY_RECORDINGS | _CHAIN_RECORDINGS)
_SHARED_SOURCE_FIELDS = ("dataset", "dataset_revision", "publisher_mapping_url",
                         "publisher_mapping_sha256", "metadata_archive")


def development_plan(metadata_archive: Path, replay_manifest: Path, chain_manifest: Path, output: Path) -> dict:
    """Plan sixty old-group clips using only metadata, never the fixed control inputs."""
    _require_linux()
    output = _new_plan_path(output)
    source = _development_source(replay_manifest, chain_manifest)
    plan = _plan(metadata_archive, source, selection=_PROFILE.selection, schema=_PROFILE.plan_schema,
                 source_sha256=_PROFILE.source_sha256)
    return _publish_plan(plan, output)


def prepare_development_corpus(metadata_archive: Path, camera_root: Path, replay_manifest: Path,
                               chain_manifest: Path, plan_path: Path, output: Path) -> dict:
    """Rebuild the pinned sixty-clip development plan and its isolated corpus."""
    _require_linux()
    source = _development_source(replay_manifest, chain_manifest)
    return _prepare_corpus(metadata_archive, camera_root, source, plan_path, output, profile=_PROFILE)


def load_development_inputs(input_root: Path) -> dict:
    """Read only development camera observations; reject a control index."""
    return _read_index(input_root, profile=_PROFILE)[1]


def load_development_labels(input_root: Path, labels_path: Path) -> dict:
    """Open labels only after validating their exact development inputs."""
    return _load_labels(input_root, labels_path, profile=_PROFILE)


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
    source = {"schema": _SOURCE_SCHEMA, **shared,
              "parent_source_sha256": list(expected),
              "camera_archives": {**archives, **chain["camera_archives"]},
              "logs": logs}
    if sha256(canonical_json(source)).hexdigest() != DEVELOPMENT_SOURCE_SHA256:
        raise PreparationError("Unexpected merged development source digest")
    return source
