from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from hybrid_drive.assets import cohort
from hybrid_drive.assets.files import PreparationError


SOURCES = Path(__file__).resolve().parents[2] / "scripts" / "real-cohort-sources.json"


def jpeg_bytes():
    out = io.BytesIO()
    Image.new("RGB", (4, 3), (20, 80, 140)).save(out, format="JPEG")
    return out.getvalue()


def archive_bytes(entries):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data, kind in entries:
            member = tarfile.TarInfo(name)
            member.type, member.size = kind, len(data)
            archive.addfile(member, io.BytesIO(data))
    return gzip.compress(out.getvalue(), mtime=0)


class CohortAssetsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "cohort"
        self.archive = self.base / "cameras.tgz"
        self.sources = cohort.load_cohort_sources(SOURCES)
        self.entries = [entry for case in self.sources["cases"] for entry in case["cameras"]]
        self.tar_entries = [(entry["member"], jpeg_bytes(), tarfile.REGTYPE) for entry in self.entries]

    def pin_fixture(self, entries=None, body=None):
        body = archive_bytes(self.tar_entries if entries is None else entries) if body is None else body
        self.archive.write_bytes(body)
        spec = {**self.sources["camera_archive"], "size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        self.sources["camera_archive"] = spec
        pinned = patch.object(cohort, "_PINNED_ARCHIVE", spec)
        pinned.start()
        self.addCleanup(pinned.stop)
        return body

    def prepare(self):
        return cohort.prepare_cohort(self.root, self.sources, self.archive)

    def rewrite_manifest(self, manifest):
        manifest.pop("manifest_hash", None)
        manifest["manifest_hash"] = cohort._hash_json(manifest)
        (self.root / cohort.MANIFEST_NAME).write_text(json.dumps(manifest))

    def assert_unpublished(self):
        self.assertFalse(self.root.exists())
        self.assertEqual(list(self.base.glob(".cohort-*")), [])

    def test_frozen_sources_and_three_current_only_cameras(self):
        self.assertEqual(self.sources["selection"]["selected_indices"], [3, 147, 292, 437, 587, 727])
        self.assertEqual(self.sources["selection"]["anchors"], [3, 147, 292, 437, 582, 727])
        self.assertEqual(len(self.entries), 18)
        self.assertEqual([case["case_id"] for case in self.sources["cases"]], [f"r{i:02d}" for i in range(6)])

    def test_prepare_exact_allowlist_and_stable_readonly_reuse(self):
        self.pin_fixture(self.tar_entries + [("unused/extra.jpg", b"not decoded", tarfile.REGTYPE)])
        self.root.mkdir()
        manifest = self.prepare()
        self.assertEqual(len(list(self.root.rglob("*.jpg"))), 18)
        self.assertEqual([record["case_id"] for record in manifest["files"]], [f"r{i:02d}" for i in range(6) for _ in range(3)])
        self.assertEqual([(record["width"], record["height"]) for record in manifest["files"]], [(4, 3)] * 18)
        self.assertTrue(manifest["archive"]["full_archive_sha256_verified"])
        self.assertEqual(self.prepare(), manifest)
        with patch.object(cohort, "AtomicFile", side_effect=AssertionError("readonly verification wrote a file")):
            self.assertEqual(cohort.verify_cohort(self.root, self.sources), manifest)

    def test_full_archive_hash_checked_before_extraction(self):
        body = self.pin_fixture()
        self.archive.write_bytes(body[:-1] + bytes([body[-1] ^ 1]))
        with patch.object(cohort, "_extract_cameras", side_effect=AssertionError("parsed before verification")):
            with self.assertRaisesRegex(PreparationError, "SHA256"):
                self.prepare()
        self.assert_unpublished()

    def test_wrong_archive_size_rejected(self):
        body = self.pin_fixture()
        self.archive.write_bytes(body[:-1])
        with self.assertRaisesRegex(PreparationError, "size"):
            self.prepare()
        self.assert_unpublished()

    def test_verified_but_truncated_gzip_rejected(self):
        body = archive_bytes(self.tar_entries)
        self.pin_fixture(body=body[:-5])
        with self.assertRaises(PreparationError):
            self.prepare()
        self.assert_unpublished()

    def test_duplicate_member_rejected_even_after_all_selected_files(self):
        self.pin_fixture(self.tar_entries + [self.tar_entries[0]])
        with self.assertRaisesRegex(PreparationError, "Duplicate"):
            self.prepare()
        self.assert_unpublished()

    def test_unsafe_archive_paths_and_special_members_rejected(self):
        bad = [
            ("../escape.jpg", b"x", tarfile.REGTYPE),
            ("/absolute.jpg", b"x", tarfile.REGTYPE),
            ("a\\b.jpg", b"x", tarfile.REGTYPE),
            ("link.jpg", b"", tarfile.SYMTYPE),
            ("hard.jpg", b"", tarfile.LNKTYPE),
        ]
        for entry in bad:
            with self.subTest(entry=entry[0]):
                self.pin_fixture(self.tar_entries + [entry])
                with self.assertRaises(PreparationError):
                    self.prepare()
                self.assert_unpublished()

    def test_missing_camera_and_invalid_jpeg_rejected(self):
        for entries in (self.tar_entries[:-1], [(self.entries[0]["member"], b"invalid", tarfile.REGTYPE)] + self.tar_entries[1:]):
            with self.subTest(missing=len(entries) != 18):
                self.pin_fixture(entries)
                with self.assertRaises(PreparationError):
                    self.prepare()
                self.assert_unpublished()

    def test_nonzero_tar_trailer_rejected(self):
        body = archive_bytes(self.tar_entries)
        self.pin_fixture(body=gzip.compress(gzip.decompress(body) + b"untrusted trailing bytes", mtime=0))
        with self.assertRaisesRegex(PreparationError, "padding"):
            self.prepare()
        self.assert_unpublished()

    def test_archive_and_destination_symlinks_rejected(self):
        self.pin_fixture()
        link = self.base / "link.tgz"
        link.symlink_to(self.archive)
        with self.assertRaises(PreparationError):
            cohort.prepare_cohort(self.root, self.sources, link)
        self.root.symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(PreparationError):
            self.prepare()

    def test_existing_legacy_root_unchanged(self):
        self.pin_fixture()
        self.root.mkdir()
        legacy = self.root / "ltf-manifest.json"
        legacy.write_bytes(b"legacy bytes")
        with self.assertRaisesRegex(PreparationError, "empty root"):
            self.prepare()
        self.assertEqual(legacy.read_bytes(), b"legacy bytes")

    def test_manifest_failure_does_not_publish_jpegs(self):
        self.pin_fixture()
        with patch.object(cohort, "write_manifest", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.prepare()
        self.assert_unpublished()

    def test_file_content_hash_and_missing_camera_verified(self):
        self.pin_fixture()
        manifest = self.prepare()
        path = self.root / manifest["files"][0]["path"]
        original = path.read_bytes()
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaises(PreparationError):
            cohort.verify_cohort(self.root, self.sources)
        path.unlink()
        with self.assertRaisesRegex(PreparationError, "missing"):
            cohort.verify_cohort(self.root, self.sources)

    def test_manifest_tampering_record_order_and_dimensions_rejected(self):
        self.pin_fixture()
        original = self.prepare()
        changes = [
            lambda value: value["files"].reverse(),
            lambda value: value["files"][0].update(width=99),
            lambda value: value.update(sources_hash="0" * 64),
            lambda value: value["files"][0].update(case_id="r05"),
        ]
        for change in changes:
            with self.subTest(change=change):
                manifest = deepcopy(original)
                change(manifest)
                self.rewrite_manifest(manifest)
                with self.assertRaises(PreparationError):
                    cohort.verify_cohort(self.root, self.sources)
        manifest = deepcopy(original)
        manifest["files"][0]["height"] = 88
        (self.root / cohort.MANIFEST_NAME).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(PreparationError, "manifest"):
            cohort.verify_cohort(self.root, self.sources)

    def test_unexpected_file_and_symlink_rejected(self):
        self.pin_fixture()
        manifest = self.prepare()
        extra = self.root / "untracked.jpg"
        extra.write_bytes(b"x")
        with self.assertRaisesRegex(PreparationError, "Unexpected"):
            cohort.verify_cohort(self.root, self.sources)
        extra.unlink()
        first = self.root / manifest["files"][0]["path"]
        first.unlink()
        first.symlink_to(self.root / manifest["files"][1]["path"])
        with self.assertRaisesRegex(PreparationError, "Symlink"):
            cohort.verify_cohort(self.root, self.sources)

    def test_source_tampering_rejected(self):
        changes = [
            lambda value: value["log"].update(sha256="0" * 64),
            lambda value: value["cases"][1].update(scene_token=value["cases"][0]["scene_token"]),
            lambda value: value["cases"][0]["cameras"].reverse(),
            lambda value: value["cases"][0]["cameras"][0].update(path="../escape.jpg"),
            lambda value: value["selection"]["selected_indices"].__setitem__(4, 582),
            lambda value: value["cases"][0].update(source_list_index=4),
        ]
        for change in changes:
            with self.subTest(change=change):
                value = deepcopy(self.sources)
                change(value)
                with self.assertRaises(PreparationError):
                    cohort.validate_cohort_sources(value)
