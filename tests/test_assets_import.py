"""The handoff must not overwrite Git code or silently accept changed input data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('import_assets', Path(__file__).resolve().parents[1] / 'scripts/import_assets.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AssetImportTests(unittest.TestCase):
    def fixture(self, root, data=b'accepted input', member='assets/input.json'):
        expected = hashlib.sha256(b'accepted input').hexdigest()
        (root / 'INPUTS.sha256.json').write_text(json.dumps({member: expected}))
        archive = root / 'inputs.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('covermem_selector/' + member, data)
            z.writestr('covermem_selector/README.md', 'old code must not overwrite checkout')
        return archive

    def test_import_allowlisted_inputs_only_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.fixture(root)
            (root / 'README.md').write_text('current checkout')
            self.assertEqual(module.import_inputs(root, archive)['imported'], 1)
            self.assertEqual((root / 'README.md').read_text(), 'current checkout')
            self.assertEqual(module.import_inputs(root)['imported'], 0)

    def test_reject_tampered_archive_without_installing_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.fixture(root, data=b'wrong input')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                module.import_inputs(root, archive)
            self.assertFalse((root / 'assets/input.json').exists())

    def test_reject_changed_existing_input_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.fixture(root)
            (root / 'assets').mkdir()
            (root / 'assets/input.json').write_bytes(b'local changes')
            with self.assertRaisesRegex(ValueError, 'Existing input differs'):
                module.import_inputs(root, archive)
            self.assertEqual((root / 'assets/input.json').read_bytes(), b'local changes')

    def test_reject_input_path_outside_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.fixture(root, member='assets/../../outside.json')
            with self.assertRaisesRegex(ValueError, 'manifest path'):
                module.import_inputs(root, archive)
