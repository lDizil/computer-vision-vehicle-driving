import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zlib


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_navsim_sample.py"
SPEC = importlib.util.spec_from_file_location("prepare_navsim_sample", SCRIPT)
sample = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sample)


def deflate(data):
    encoder = zlib.compressobj(wbits=-15)
    return encoder.compress(data) + encoder.flush()


def tar_prefix(member_name, payload, extra=None, member_type=tarfile.REGTYPE):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        if extra is not None:
            entry = tarfile.TarInfo(extra)
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b"x"))
        entry = tarfile.TarInfo(member_name)
        entry.type = member_type
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    return gzip.compress(buffer.getvalue(), mtime=0)


class Response(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status = status
        self.headers = headers


class FakeHTTP:
    def __init__(self, bodies):
        self.bodies = bodies
        self.requests = []
        self.status = 206
        self.wrong_range = False
        self.truncate = False
        self.extra_byte = False

    def __call__(self, request, timeout):
        self.requests.append(request)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        start, end = map(int, match.groups())
        body = self.bodies[request.full_url]
        data = body[start:end + 1]
        headers = {"Content-Range": f"bytes {start}-{end}/{len(body)}", "Content-Length": str(end - start + 1)}
        if request.get_header("If-match"):
            headers["ETag"] = request.get_header("If-match")
        if self.wrong_range:
            headers["Content-Range"] = f"bytes 0-{end}/{len(body)}-wrong"
        if self.truncate:
            data = data[:-1]
        if self.extra_byte:
            data += b"!"
        return Response(data, self.status, headers)


class PrepareSampleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "sample"
        self.log_payload = b"opaque log bytes; never unpickle"
        self.member = "openscene-v1.1/meta_datas/mini/example.pkl"
        prefix = tar_prefix(self.member, self.log_payload)
        self.sources = {
            "navsim_revision": "navsim-sha",
            "nuplan_revision": "nuplan-sha",
            "log": {
                "url": "https://example.invalid/log.tgz",
                "revision": "dataset-sha",
                "archive_size_bytes": len(prefix),
                "prefix_size_bytes": len(prefix),
                "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "member": self.member,
                "path": "navsim_logs/mini/example.pkl",
                "size_bytes": len(self.log_payload),
                "sha256": hashlib.sha256(self.log_payload).hexdigest(),
            },
            "maps": {"url": "https://example.invalid/maps.zip", "etag": "pinned-etag", "files": []},
        }
        archive = bytearray(b"padding")
        self.expected = {"navsim_logs/mini/example.pkl": self.log_payload}
        for path in ["maps/city-a/map.gpkg", "maps/city-b/map.gpkg", "maps/city-c/map.gpkg", "maps/city-d/map.gpkg", "maps/nuplan-maps-v1.0.json"]:
            payload = b"map bytes: " + path.encode()
            compressed = deflate(payload)
            spec = {"member": "nuplan-maps-v1.0/" + path[5:], "path": path, "offset": len(archive),
                    "compressed_size_bytes": len(compressed), "size_bytes": len(payload),
                    "crc32": f"{zlib.crc32(payload) & 0xffffffff:08x}"}
            if path.endswith(".json"):
                spec["sha256"] = hashlib.sha256(payload).hexdigest()
            self.sources["maps"]["files"].append(spec)
            self.expected[path] = payload
            archive.extend(compressed)
        self.sources["maps"]["archive_size_bytes"] = len(archive)
        self.http = FakeHTTP({self.sources["log"]["url"]: prefix, self.sources["maps"]["url"]: bytes(archive)})
        self.print_patch = patch("builtins.print")
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def prepare(self, sources=None):
        return sample.prepare(self.root, sources or self.sources, rate_limit_mib=1000000, opener=self.http)

    def assert_no_partial(self):
        self.assertEqual(list(self.root.rglob("*.partial")), [])

    def replace_prefix(self, prefix):
        log = self.sources["log"]
        self.http.bodies[log["url"]] = prefix
        log["archive_size_bytes"] = len(prefix)
        log["prefix_size_bytes"] = len(prefix)
        log["prefix_sha256"] = hashlib.sha256(prefix).hexdigest()

    def test_valid_subset_and_manifest(self):
        result = self.prepare()
        self.assertEqual(len(self.http.requests), 6)
        for path, payload in self.expected.items():
            self.assertEqual((self.root / path).read_bytes(), payload)
        saved = json.loads((self.root / "sample-manifest.json").read_text())
        self.assertEqual(result, saved)
        self.assertFalse(saved["full_archive_hashes_verified"])
        self.assertFalse(saved["pickle_deserialized"])
        self.assertEqual(saved["sources"]["log"]["revision"], "dataset-sha")
        for record in saved["files"]:
            self.assertEqual(record["sha256"], hashlib.sha256(self.expected[record["path"]]).hexdigest())
        self.assert_no_partial()

    def test_verified_files_are_reused_without_network(self):
        first = self.prepare()
        requests = len(self.http.requests)
        self.assertEqual(self.prepare(), first)
        self.assertEqual(len(self.http.requests), requests)

    def test_complete_log_in_incomplete_gzip_prefix(self):
        prefix = self.http.bodies[self.sources["log"]["url"]][:-8]
        self.replace_prefix(prefix)
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        decoder.decompress(prefix)
        self.assertFalse(decoder.eof)
        self.prepare()
        self.assertEqual((self.root / self.sources["log"]["path"]).read_bytes(), self.log_payload)

    def test_corrupt_existing_file_fails_without_overwrite(self):
        target = self.root / self.sources["log"]["path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"corrupt")
        with self.assertRaisesRegex(sample.PreparationError, "Existing file is corrupt"):
            self.prepare()
        self.assertEqual(target.read_bytes(), b"corrupt")
        self.assertEqual(self.http.requests, [])

    def test_server_ignoring_range_is_rejected(self):
        self.http.status = 200
        with self.assertRaisesRegex(sample.PreparationError, "exact HTTP Range"):
            self.prepare()
        self.assertFalse((self.root / self.sources["log"]["path"]).exists())
        self.assert_no_partial()

    def test_wrong_content_range_is_rejected(self):
        self.http.wrong_range = True
        with self.assertRaisesRegex(sample.PreparationError, "exact HTTP Range"):
            self.prepare()

    def test_truncated_http_payload_is_rejected(self):
        self.http.truncate = True
        with self.assertRaisesRegex(sample.PreparationError, "Truncated HTTP"):
            self.prepare()

    def test_extra_http_byte_is_rejected(self):
        self.http.extra_byte = True
        with self.assertRaisesRegex(sample.PreparationError, "exceeds its declared length"):
            self.prepare()

    def test_prefix_checksum_mismatch_is_rejected(self):
        url = self.sources["log"]["url"]
        prefix = self.http.bodies[url]
        self.http.bodies[url] = bytes([prefix[0] ^ 1]) + prefix[1:]
        with self.assertRaisesRegex(sample.PreparationError, "prefix SHA256"):
            self.prepare()

    def test_stale_partial_is_not_overwritten_or_deleted(self):
        partial = self.root / (self.sources["log"]["path"] + ".partial")
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"interrupted")
        with self.assertRaisesRegex(sample.PreparationError, "Partial file already exists"):
            self.prepare()
        self.assertEqual(partial.read_bytes(), b"interrupted")

    def test_tar_decompression_budget_is_enforced(self):
        with patch.object(sample, "MAX_TAR_BYTES", 1024):
            with self.assertRaisesRegex(sample.PreparationError, "Tar prefix exceeds"):
                self.prepare()
        self.assert_no_partial()

    def test_selected_log_checksum_mismatch_removes_partial(self):
        self.sources["log"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(sample.PreparationError, "File SHA256"):
            self.prepare()
        self.assert_no_partial()

    def test_corrupt_map_crc_removes_partial(self):
        spec = self.sources["maps"]["files"][0]
        payload = self.expected[spec["path"]]
        replacement = deflate(b"!" + payload[1:])
        self.assertEqual(len(replacement), spec["compressed_size_bytes"])
        url = self.sources["maps"]["url"]
        archive = bytearray(self.http.bodies[url])
        start = spec["offset"]
        archive[start:start + len(replacement)] = replacement
        self.http.bodies[url] = bytes(archive)
        with self.assertRaisesRegex(sample.PreparationError, "CRC32"):
            self.prepare()
        self.assertFalse((self.root / spec["path"]).exists())
        self.assert_no_partial()

    def test_map_metadata_sha_mismatch_is_rejected(self):
        self.sources["maps"]["files"][-1]["sha256"] = "0" * 64
        with self.assertRaisesRegex(sample.PreparationError, "File SHA256"):
            self.prepare()
        self.assert_no_partial()

    def test_archive_etag_mismatch_is_rejected(self):
        opener = self.http

        def wrong_etag(request, timeout):
            response = opener(request, timeout)
            response.headers["ETag"] = '"unexpected"'
            return response

        with self.assertRaisesRegex(sample.PreparationError, "ETag"):
            sample.prepare(self.root, self.sources, 1000000, wrong_etag)
        self.assert_no_partial()

    def test_truncated_deflate_is_rejected(self):
        spec = self.sources["maps"]["files"][0]
        spec["compressed_size_bytes"] -= 1
        with self.assertRaisesRegex(sample.PreparationError, "Truncated ZIP"):
            self.prepare()
        self.assert_no_partial()

    def test_deflate_cannot_exceed_declared_size(self):
        self.sources["maps"]["files"][0]["size_bytes"] -= 1
        with self.assertRaisesRegex(sample.PreparationError, "uncompressed size limit"):
            self.prepare()
        self.assert_no_partial()

    def test_deflate_streams_multiple_output_chunks(self):
        payload = b"map geometry " * 50000
        compressed = deflate(payload)
        output = io.BytesIO()
        chunks = (compressed[i:i + 7] for i in range(0, len(compressed), 7))
        sample.inflate_member(chunks, len(payload), output)
        self.assertEqual(output.getvalue(), payload)

    def test_declared_size_budget_checked_before_network(self):
        self.sources["maps"]["files"][0]["size_bytes"] = sample.MAX_MAP_BYTES + 1
        with self.assertRaisesRegex(sample.PreparationError, "size limit"):
            self.prepare()
        self.assertEqual(self.http.requests, [])

    def test_range_bounds_checked_before_network(self):
        self.sources["maps"]["files"][0]["offset"] = self.sources["maps"]["archive_size_bytes"]
        with self.assertRaisesRegex(sample.PreparationError, "range exceeds"):
            self.prepare()
        self.assertEqual(self.http.requests, [])

    def test_total_budget_checked_before_network(self):
        with patch.object(sample, "MAX_DOWNLOAD_BYTES", 1024 * 1024):
            sources = copy.deepcopy(self.sources)
            sources["log"]["archive_size_bytes"] = sample.MAX_PREFIX_BYTES
            sources["log"]["prefix_size_bytes"] = 1024 * 1024
            with self.assertRaisesRegex(sample.PreparationError, "total byte budget"):
                self.prepare(sources)
        self.assertEqual(self.http.requests, [])

    def test_destination_traversal_checked_before_network(self):
        self.sources["log"]["path"] = "../escaped.pkl"
        with self.assertRaisesRegex(sample.PreparationError, "Unsafe relative path"):
            self.prepare()
        self.assertEqual(self.http.requests, [])
        self.assertFalse(self.root.parent.joinpath("escaped.pkl").exists())

    def test_destination_symlink_outside_root_is_rejected(self):
        self.root.mkdir()
        self.root.joinpath("navsim_logs").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(sample.PreparationError, "escapes root"):
            self.prepare()
        self.assertEqual(self.http.requests, [])

    def test_tar_traversal_is_rejected_without_extraction(self):
        self.replace_prefix(tar_prefix(self.member, self.log_payload, extra="../escaped"))
        with self.assertRaisesRegex(sample.PreparationError, "Unsafe relative path"):
            self.prepare()
        self.assert_no_partial()
        self.assertFalse(self.root.parent.joinpath("escaped").exists())

    def test_selected_tar_symlink_is_rejected(self):
        self.replace_prefix(tar_prefix(self.member, self.log_payload, member_type=tarfile.SYMTYPE))
        with self.assertRaisesRegex(sample.PreparationError, "not a regular file"):
            self.prepare()
        self.assert_no_partial()

    def test_selected_tar_size_mismatch_is_rejected(self):
        self.replace_prefix(tar_prefix(self.member, self.log_payload + b"extra"))
        with self.assertRaisesRegex(sample.PreparationError, "expected size"):
            self.prepare()
        self.assert_no_partial()

    def test_invalid_rate_is_rejected_before_network(self):
        for value in [0, -1, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaisesRegex(sample.PreparationError, "Rate limit"):
                sample.prepare(self.root, self.sources, value, self.http)
        self.assertEqual(self.http.requests, [])


if __name__ == "__main__":
    unittest.main()
