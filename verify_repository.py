"""Verify the combined repository or its downloaded ZIP, using the standard library."""
import json
from pathlib import Path
from verify_bundle import safe_path, sha256


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'REPOSITORY_MANIFEST.json').read_text())
    errors = []
    for row in manifest['files']:
        path = safe_path(root, row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            errors.append(row['path'])
    print(json.dumps({'passed': not errors, 'checked_files': len(manifest['files']),
                      'failures': errors, 'raw_data_included': False, 'base_model_included': False},
                     indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
