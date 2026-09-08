"""Make an editable source/checkpoint copy without touching the archived evidence."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    target = args.destination or root / 'work' / 'kan_attention_theory'
    if target.exists():
        raise SystemExit(f'Destination already exists; nothing overwritten: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / 'snapshot' / 'kan_attention_theory', target)
    shutil.copytree(root / 'checkpoints' / 'kan_attention_theory', target, dirs_exist_ok=True)
    print(json.dumps({'workspace': str(target), 'data_included': False,
                      'base_model_included': False, 'npu_port_completed': False}, indent=2))


if __name__ == '__main__':
    main()
