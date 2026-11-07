import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from hybrid_drive.assets.cameras import MAX_MEMBER_BYTES, reuse_cameras, stream_cameras, validate_jpeg
from hybrid_drive.assets.fetch import AssetSpec, TransferBudget, download_verified_asset
from hybrid_drive.assets.files import PreparationError, checked_root, destination, read_manifest, write_manifest
from hybrid_drive.assets.prepare import MODEL_PATHS, validate_sources, verify_prepared


JPEG = base64.b64decode(
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAx'
    'NDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy'
    'MjIyMjIyMjIyMjIyMjL/wAARCAADAAQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIE'
    'AwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpT'
    'VFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ'
    '2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC'
    'AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hp'
    'anN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP0'
    '9fb3+Pn6/9oADAMBAAIRAxEAPwDy+iiitjI//9k='
)
REVISION = 'a' * 40
URL = f'https://huggingface.co/datasets/example/public/resolve/{REVISION}/archive.tgz'


def tar_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, payload, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return gzip.compress(buffer.getvalue(), mtime=0)


def make_spec(body):
    return AssetSpec(URL, REVISION, len(body), hashlib.sha256(body).hexdigest(), 'b' * 64)


def decoded(_):
    return {'width': 4, 'height': 3, 'format': 'JPEG', 'mode': 'RGB'}


class Response(io.BytesIO):
    def __init__(self, body, headers):
        super().__init__(body)
        self.status = 200
        self.headers = headers


class FakeHTTP:
    def __init__(self, body):
        self.body = body
        self.requests = []
        self.headers = {'Content-Length': str(len(body)), 'ETag': '"' + 'b' * 64 + '"'}
        self.payload = body

    def __call__(self, request, timeout):
        self.requests.append(request)
        return Response(b'' if request.get_method() == 'HEAD' else self.payload, self.headers.copy())


class AssetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        output = patch("builtins.print")
        output.start()
        self.addCleanup(output.stop)
        self.root = Path(self.temp.name).resolve()
        self.entries = [dict(camera=c, member=f'archive/{c}/image.jpg', path=f'sensor_blobs/{c}/image.jpg')
                        for c in ['CAM_L0', 'CAM_F0', 'CAM_R0']]
        self.tar_entries = [(entry['member'], JPEG, tarfile.REGTYPE) for entry in self.entries]

    def camera_source(self, entries=None):
        body = tar_bytes(entries or self.tar_entries)
        spec = make_spec(body)
        return {**spec.__dict__, 'files': self.entries}, FakeHTTP(body)

    def extract(self, source=None, http=None, **kwargs):
        if source is None:
            source, http = self.camera_source()
        return stream_cameras(self.root, source, TransferBudget(128), opener=http, validator=decoded, **kwargs)

    def assert_no_outputs(self):
        self.assertEqual(list(self.root.rglob('*.jpg')), [])
        self.assertEqual(list(self.root.rglob('*.partial')), [])

    def test_exact_asset_download_and_reuse(self):
        body = b'opaque model bytes, never unpickle'
        http = FakeHTTP(body)
        target = self.root / 'model.ckpt'
        record = download_verified_asset(make_spec(body), target, rate_limit_mib=128, opener=http)
        self.assertEqual(target.read_bytes(), body)
        self.assertEqual(record['sha256'], hashlib.sha256(body).hexdigest())
        self.assertEqual([r.get_method() for r in http.requests], ['HEAD', 'GET'])
        reused = download_verified_asset(make_spec(body), target, opener=http)
        self.assertTrue(reused['reused'])
        self.assertEqual(reused['bytes_transferred'], 0)
        self.assertEqual(len(http.requests), 2)

    def test_corrupt_existing_asset_never_overwritten(self):
        target = self.root / 'model.ckpt'
        target.write_bytes(b'corrupt')
        http = FakeHTTP(b'expected')
        with self.assertRaisesRegex(PreparationError, 'corrupt'):
            download_verified_asset(make_spec(http.body), target, opener=http)
        self.assertEqual(target.read_bytes(), b'corrupt')
        self.assertEqual(http.requests, [])

    def test_pin_mismatches_fail_before_get(self):
        for field, value in [('Content-Length', '1'), ('ETag', 'wrong'), ('Content-Encoding', 'gzip'), ('X-Repo-Commit', 'wrong')]:
            with self.subTest(field=field):
                http = FakeHTTP(b'expected')
                http.headers[field] = value
                with self.assertRaises(PreparationError):
                    download_verified_asset(make_spec(http.body), self.root / 'missing', opener=http)
                self.assertEqual(len(http.requests), 1)
                self.assert_no_outputs()

    def test_get_pin_is_rechecked_after_head(self):
        body = b'expected'
        http = FakeHTTP(body)
        def changed_get(request, timeout):
            response = http(request, timeout)
            if request.get_method() == 'GET':
                response.headers['ETag'] = '"' + 'c' * 64 + '"'
            return response
        with self.assertRaisesRegex(PreparationError, 'ETag'):
            download_verified_asset(make_spec(body), self.root / 'missing', opener=changed_get)
        self.assertEqual([request.get_method() for request in http.requests], ['HEAD', 'GET'])
        self.assertFalse((self.root / 'missing').exists())
        self.assert_no_outputs()

    def test_get_sha_truncation_and_overrun_are_rejected(self):
        for payload in [b'same len', b'short', b'expected-extra']:
            with self.subTest(payload=payload):
                http = FakeHTTP(b'expected')
                http.payload = payload
                with self.assertRaises(PreparationError):
                    download_verified_asset(make_spec(http.body), self.root / 'missing', rate_limit_mib=128, opener=http)
                self.assertFalse((self.root / 'missing').exists())
                self.assert_no_outputs()

    def test_three_cameras_in_model_order_and_manifest_reuse(self):
        source, http = self.camera_source(list(reversed(self.tar_entries)))
        record = self.extract(source, http)
        self.assertEqual([r['camera'] for r in record['files']], ['CAM_L0', 'CAM_F0', 'CAM_R0'])
        self.assertFalse(record['full_archive_sha256_verified'])
        self.assertTrue(record['stopped_after_allowlist'])
        self.assertEqual(record['bytes_transferred'], len(http.body))
        for entry in record['files']:
            self.assertEqual((self.root / entry['path']).read_bytes(), JPEG)
            self.assertEqual(entry['sha256'], hashlib.sha256(JPEG).hexdigest())
        self.assertEqual(reuse_cameras(self.root, source, record, decoded), record['files'])
        (self.root / self.entries[0]['path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(PreparationError, 'corrupt'):
            reuse_cameras(self.root, source, record, decoded)

    def test_stops_before_unneeded_archive_suffix(self):
        # Deterministic high-entropy suffix is larger than gzip's read-ahead buffer.
        suffix = b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(20000))
        source, http = self.camera_source(self.tar_entries + [('archive/unneeded.bin', suffix, tarfile.REGTYPE)])
        record = self.extract(source, http)
        self.assertLess(record['bytes_transferred'], len(http.body))
        self.assertFalse((self.root / 'archive/unneeded.bin').exists())

    def test_links_traversal_duplicates_and_pax_are_rejected(self):
        bad = [('../outside.jpg', b'x', tarfile.REGTYPE), ('/absolute', b'x', tarfile.REGTYPE),
               ('archive/link', b'', tarfile.SYMTYPE), ('archive/hard', b'', tarfile.LNKTYPE),
               ('archive/pax', b'x', tarfile.XHDTYPE)]
        for entry in bad:
            with self.subTest(entry=entry):
                source, http = self.camera_source([entry] + self.tar_entries)
                with self.assertRaises(PreparationError):
                    self.extract(source, http)
                self.assert_no_outputs()
        source, http = self.camera_source([self.tar_entries[0], self.tar_entries[0], *self.tar_entries[1:]])
        with self.assertRaisesRegex(PreparationError, 'Duplicate'):
            self.extract(source, http)
        self.assert_no_outputs()

    def test_missing_and_truncated_tar_leave_no_final_or_partial_files(self):
        source, http = self.camera_source(self.tar_entries[:2])
        with self.assertRaisesRegex(PreparationError, 'before all'):
            self.extract(source, http)
        self.assert_no_outputs()
        source, http = self.camera_source()
        http.payload = http.body[:60]
        with self.assertRaises(PreparationError):
            self.extract(source, http)
        self.assert_no_outputs()

    def test_member_size_limit_checked_before_payload(self):
        header = tarfile.TarInfo('archive/huge.jpg')
        header.size = MAX_MEMBER_BYTES + 1
        body = gzip.compress(header.tobuf())
        source = {**make_spec(body).__dict__, 'files': self.entries}
        with self.assertRaisesRegex(PreparationError, 'size limit'):
            self.extract(source, FakeHTTP(body))
        self.assert_no_outputs()

    def test_jpeg_decoder_failure_is_atomic(self):
        source, http = self.camera_source()
        def reject(_):
            raise PreparationError('bad JPEG')
        with self.assertRaisesRegex(PreparationError, 'bad JPEG'):
            stream_cameras(self.root, source, TransferBudget(128), opener=http, validator=reject)
        self.assert_no_outputs()

    def test_symlinks_and_existing_unprovenanced_cameras_rejected(self):
        (self.root / 'link').symlink_to(self.root, target_is_directory=True)
        for operation in [lambda: checked_root(self.root / 'link'), lambda: destination(self.root, 'link/x')]:
            with self.assertRaisesRegex(PreparationError, 'Symlink'):
                operation()
        source, _ = self.camera_source()
        path = self.root / self.entries[0]['path']
        path.parent.mkdir(parents=True)
        path.write_bytes(JPEG)
        with self.assertRaisesRegex(PreparationError, 'provenance'):
            reuse_cameras(self.root, source, None, decoded)
        self.assertEqual(path.read_bytes(), JPEG)

    def test_invalid_limits_urls_paths_and_manifest(self):
        for rate in [0, -1, float('nan'), float('inf'), 129]:
            with self.assertRaises(PreparationError):
                TransferBudget(rate)
        for url in [URL.replace('https:', 'http:'), URL.replace('huggingface.co', 'evil.invalid'), URL.replace(REVISION, 'main')]:
            with self.assertRaises(PreparationError):
                AssetSpec(url, REVISION, 1, 'a'*64, 'b'*64).validate()
        for path in ['../out', '/out', 'a/../b', 'a\\b', 'a//b']:
            with self.assertRaises(PreparationError):
                destination(self.root, path)
        manifest = self.root / 'manifest.json'
        manifest.write_text('[]')
        with self.assertRaises(PreparationError):
            read_manifest(manifest)

    def test_compressed_corruption_and_bad_tar_header_are_atomic(self):
        for body in [b'not gzip', gzip.compress(b'broken header'.ljust(512, b'x'))]:
            with self.subTest(body=body[:12]):
                source = {**make_spec(body).__dict__, 'files': self.entries}
                with self.assertRaises(PreparationError):
                    self.extract(source, FakeHTTP(body))
                self.assert_no_outputs()

    def test_transfer_budget_and_deadline(self):
        budget = TransferBudget(128, max_bytes=2)
        with self.assertRaisesRegex(PreparationError, 'byte budget'):
            budget.consume(3)
        with patch('hybrid_drive.assets.fetch.time.monotonic', return_value=10):
            budget = TransferBudget(128, timeout_seconds=1)
        with patch('hybrid_drive.assets.fetch.time.monotonic', return_value=12):
            with self.assertRaisesRegex(PreparationError, 'deadline'):
                budget.check()

    def test_checked_in_sources_match_selected_frame_and_budget(self):
        source = json.loads((Path(__file__).resolve().parents[2] / 'scripts/ltf-sample-sources.json').read_text())
        validate_sources(source)
        source['cameras']['files'][0]['member'] += '.changed'
        with self.assertRaisesRegex(PreparationError, 'allowlist'):
            validate_sources(source)

    def prepared_fixture(self):
        sources = json.loads((Path(__file__).resolve().parents[2] / 'scripts/ltf-sample-sources.json').read_text())
        body = tar_bytes([(entry['member'], JPEG, tarfile.REGTYPE) for entry in sources['cameras']['files']])
        sources['cameras'].update(size_bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), etag='b'*64)
        cameras = stream_cameras(self.root, sources['cameras'], TransferBudget(128), opener=FakeHTTP(body), validator=decoded)
        models = []
        for key, path in MODEL_PATHS:
            source = sources
            for part in key.split('.'):
                source = source[part]
            payload = key.encode()
            source.update(size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            models.append(dict(path=path, source=source.copy(), size_bytes=len(payload), sha256=source['sha256']))
        manifest = dict(sources=sources, scene_token=sources['scene_token'], frame_index=3,
                        camera_order=sources['camera_order'], models=models, cameras=cameras)
        write_manifest(self.root / 'ltf-camera-manifest.json', cameras)
        write_manifest(self.root / 'ltf-manifest.json', manifest)
        return sources, manifest

    def test_read_only_verification_has_no_write_or_network(self):
        sources, manifest = self.prepared_fixture()
        with patch('pathlib.Path.mkdir', side_effect=AssertionError('must not mkdir')), \
                patch('hybrid_drive.assets.prepare.download_verified_asset', side_effect=AssertionError('must not download')), \
                patch('hybrid_drive.assets.prepare.write_manifest', side_effect=AssertionError('must not write')):
            self.assertEqual(verify_prepared(self.root, sources, validator=decoded), manifest)

    def test_read_only_verification_rejects_missing_or_tampered_assets(self):
        for kind in ['model-corrupt', 'model-missing', 'camera-corrupt', 'camera-missing', 'manifest-mismatch', 'manifest-missing', 'camera-identity']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                original = self.root
                self.root = Path(folder).resolve()
                try:
                    sources, manifest = self.prepared_fixture()
                    if kind.startswith('model'):
                        target = self.root / MODEL_PATHS[0][1]
                    elif kind.startswith('camera') and kind != 'camera-identity':
                        target = self.root / sources['cameras']['files'][0]['path']
                    else:
                        target = self.root / 'ltf-camera-manifest.json'
                    if kind == 'camera-identity':
                        changed = manifest['cameras']
                        changed['files'][0]['camera'] = 'CAM_F0'
                        target.write_text(json.dumps(changed))
                        (self.root / 'ltf-manifest.json').write_text(json.dumps(manifest))
                    elif kind.endswith('missing'):
                        target.unlink()
                    elif kind == 'manifest-mismatch':
                        changed = json.loads(target.read_text())
                        changed['bytes_transferred'] += 1
                        target.write_text(json.dumps(changed))
                    else:
                        target.write_bytes(b'corrupt')
                    with self.assertRaises(PreparationError):
                        verify_prepared(self.root, sources, validator=decoded)
                finally:
                    self.root = original

    def test_real_jpeg_decoding_and_fake_jpeg_rejection(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest('Pillow supplied by neural runtime, not host stdlib')
        path = self.root / 'image.jpg'
        path.write_bytes(JPEG)
        self.assertEqual(validate_jpeg(path), decoded(path))
        for payload in [b'not a JPEG', JPEG[:50], JPEG[:-10]]:
            path.write_bytes(payload)
            with self.assertRaises(PreparationError):
                validate_jpeg(path)


if __name__ == '__main__':
    unittest.main()
