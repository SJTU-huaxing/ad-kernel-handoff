"""Download only the original base-model revision, then check all asset hashes."""
import argparse
import json
from pathlib import Path
from verify_bundle import sha256


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true')
    p.add_argument('--destination', type=Path, default=root / 'work' / 'models' / 'Qwen2.5-1.5B')
    args = p.parse_args()
    manifest = json.loads((root / 'EXTERNAL_MODEL_MANIFEST.json').read_text())
    print(json.dumps({'model': manifest['model'], 'revision': manifest['revision'],
                      'bytes': manifest['bytes'], 'destination': str(args.destination),
                      'download_requested': args.download}, indent=2))
    if not args.download:
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=manifest['model'], revision=manifest['revision'],
                      allow_patterns=[r['path'] for r in manifest['files']],
                      local_dir=str(args.destination))
    for row in manifest['files']:
        path = args.destination / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise SystemExit(f'Base-model verification failed: {row["path"]}')
    print(json.dumps({'passed': True, 'verified_files': len(manifest['files'])}))


if __name__ == '__main__':
    main()
