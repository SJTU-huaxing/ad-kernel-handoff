"""Verify a moved handoff using Python's standard library; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Invalid relative manifest path: {relative}')
    return root / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-checkpoints', action='store_true')
    parser.add_argument('--project-root', type=Path,
                        help='Optional relocated kan_attention_theory root, to verify external causal_direction/data')
    parser.add_argument('--model-root', type=Path,
                        help='Optional relocated Qwen snapshot directory, to verify base weights/tokenizer')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'BUNDLE_MANIFEST.json').read_text())
    checkpoints_present = any(safe_path(root, r['path']).is_file()
                              for r in manifest['files'] if r['group'] == 'checkpoints')
    failures, verified, skipped = [], {}, {}

    def check(path, row, group):
        if not path.is_file():
            failures.append({'path': str(path), 'reason': 'missing'})
        elif path.stat().st_size != row['bytes']:
            failures.append({'path': str(path), 'reason': 'size mismatch'})
        elif sha256(path) != row['sha256']:
            failures.append({'path': str(path), 'reason': 'SHA256 mismatch'})
        else:
            verified[group] = verified.get(group, 0) + 1

    for row in manifest['files']:
        group = row['group']
        if group == 'checkpoints' and not (args.require_checkpoints or checkpoints_present):
            skipped[group] = skipped.get(group, 0) + 1
            continue
        check(safe_path(root, row['path']), row, group)

    for supplied_root, filename, group in [
        (args.project_root, 'EXTERNAL_DATA_MANIFEST.json', 'external_data'),
        (args.model_root, 'EXTERNAL_MODEL_MANIFEST.json', 'external_model'),
    ]:
        data = json.loads((root / filename).read_text())
        if supplied_root is None:
            skipped[group] = len(data['files'])
            continue
        for row in data['files']:
            check(safe_path(supplied_root, row['path']), row, group)

    print(json.dumps({'passed': not failures, 'verified': verified, 'skipped': skipped,
                      'failures': failures,
                      'scope': 'File integrity only. No NPU reproduction, training, or statistical certification.'},
                     indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
