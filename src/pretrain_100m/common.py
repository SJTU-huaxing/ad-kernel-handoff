import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/pretrain_100m_2b.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(cpu_tree(v) for v in obj)
    return obj


def load_checkpoint(path):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def source_binding():
    paths = list((ROOT / "src/pretrain_100m").glob("*.py"))
    paths += [ROOT / "src/ad_kernel" / x for x in ["features.py", "baselines.py", "attention.py"]]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
