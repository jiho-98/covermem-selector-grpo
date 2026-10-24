"""Import only hash-locked inputs from the full handoff ZIP; no dependencies/GPU.

The archive's code and manifests never overwrite the checked-out repository.
Existing mismatched files are an error, not permission to replace local work.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def import_inputs(root, archive=None):
    root = Path(root).resolve()
    expected = json.loads((root / 'INPUTS.sha256.json').read_text())
    missing = []
    for name, sha in expected.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in {'assets', 'adapters'}:
            raise ValueError('Invalid input manifest path: ' + name)
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid input SHA256: ' + name)
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError('Input path escapes repository: ' + name)
        if target.exists():
            if not target.is_file() or digest(target) != sha:
                raise ValueError('Existing input differs; preserve it and use a fresh checkout: ' + name)
        else:
            missing.append(name)
    if missing and archive is None:
        raise ValueError('Missing inputs; import the full ZIP first: ' + ', '.join(missing))
    if missing:
        # Verify the entire import before placing any file into assets/adapters.
        with tempfile.TemporaryDirectory(prefix='.asset-import-', dir=root) as temp:
            staged = Path(temp)
            with zipfile.ZipFile(archive) as source:
                members = source.namelist()
                for name in missing:
                    member = 'covermem_selector/' + name
                    if members.count(member) != 1:
                        raise ValueError('Expected exactly one ZIP member: ' + member)
                    target = staged / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
                    if digest(target) != expected[name]:
                        raise ValueError('ZIP input SHA256 mismatch: ' + name)
            for name in missing:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation protects against an accidental concurrent import.
                with (staged / name).open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
    return {'status': 'inputs_verified', 'files': len(expected), 'imported': len(missing), 'gpu_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='CoverMem_selector_GRPO_AB_full.zip')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only == (args.archive is not None):
        parser.error('Choose exactly one of --archive PATH or --check-only')
    print(json.dumps(import_inputs(ROOT, args.archive), indent=2))


if __name__ == '__main__':
    main()
