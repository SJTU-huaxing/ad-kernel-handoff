"""Freeze coherent long-document evaluation without accessing model outputs."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from prepare_pretraining_data import sketch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work/pretraining/data/fineweb_edu_v1"
OUT = ROOT / "work/pretraining/evaluation/long_v1"
OUT.mkdir(parents=True, exist_ok=True)
protocol = ROOT / "configs/long_context_evaluation_v1.json"
binding = hashlib.sha256(protocol.read_bytes()).hexdigest()
if (OUT / "manifest.json").exists():
    assert json.loads((OUT / "manifest.json").read_text())["protocol_sha256"] == binding
    raise SystemExit("Verified existing long-context manifest")
train_hashes = set()
with (DATA / "train.documents.jsonl").open() as source:
    for line in source:
        train_hashes.add(json.loads(line)["text_sha256"])
tokenizer = Tokenizer.from_file(str(ROOT / "work/tokenizers/gpt2/tokenizer.json"))
records, arrays, signatures, seen = [], [], [], set()
for path in sorted((ROOT / "work/data/fineweb-edu/sample/10BT").glob("*.parquet")):
    row_offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text", "id", "url", "language", "language_score", "int_score", "token_count"]):
        for ordinal, row in enumerate(batch.to_pylist()):
            if row["token_count"] < 8193 or row["language"] != "en" or row["language_score"] < .8 or row["int_score"] < 3:
                continue
            try:
                hostname = (urlsplit(row["url"]).hostname or "").lower().removeprefix("www.")
            except ValueError:
                continue
            bucket = int.from_bytes(hashlib.blake2b(hostname.encode(), digest_size=8, person=b"ad-split-v1").digest(), "little") % 1000
            if not 5 <= bucket < 10:
                continue
            normalized = " ".join(unicodedata.normalize("NFKC", row["text"]).lower().split())
            text_hash = hashlib.sha256(normalized.encode()).hexdigest()
            if text_hash in train_hashes or text_hash in seen:
                continue
            signature = set(sketch(re.findall(r"\w+", normalized)))
            duplicate = False
            for other in signatures:
                union = set(sorted(signature | other)[:32])
                if len(signature & other & union) / len(union) >= .8:
                    duplicate = True
                    break
            if duplicate:
                continue
            ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
            if len(ids) < 8193:
                continue
            array = np.asarray(ids[:8193], dtype="<u2")
            records.append({"index": len(records), "source_file": path.name, "source_row": row_offset + ordinal,
                            "source_id": row["id"], "hostname": hostname, "text_sha256": text_hash,
                            "token_sha256": hashlib.sha256(array.tobytes()).hexdigest(), "tokens": 8193})
            arrays.append(array)
            signatures.append(signature)
            seen.add(text_hash)
            if len(records) == 64:
                break
        row_offset += len(batch)
        if len(records) == 64:
            break
    if len(records) == 64:
        break
assert len(records) == 64, len(records)
with (OUT / "tokens.npy.partial").open("wb") as out:
    np.save(out, np.stack(arrays), allow_pickle=False)
(OUT / "tokens.npy.partial").replace(OUT / "tokens.npy")
manifest = {"protocol_sha256": binding, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
            "token_file_sha256": hashlib.sha256((OUT / "tokens.npy").read_bytes()).hexdigest(), "records": records}
(OUT / "manifest.json.partial").write_text(json.dumps(manifest, indent=2) + "\n")
(OUT / "manifest.json.partial").replace(OUT / "manifest.json")
print(json.dumps({"event": "complete", "documents": 64, "hostnames": len({r['hostname'] for r in records}),
                  "manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest()}), flush=True)
