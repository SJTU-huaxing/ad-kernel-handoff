"""Prepare a reproducible FineWeb-Edu stream with document-level audits.

CPU worker processes tokenize and sketch; the parent deduplicates in stable
source order. Publication of a manifest marks completion; partial output is
never accepted as a usable training dataset.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/pretraining_data_v1.json").read_text())
TOKENIZER = None


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def initialize_worker():
    global TOKENIZER
    from tokenizers import Tokenizer
    TOKENIZER = Tokenizer.from_file(str(ROOT / "work/tokenizers/gpt2/tokenizer.json"))


@lru_cache(maxsize=262144)
def word_hash(word):
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8, person=b"ad-word-v1").digest(), "little")


def sketch(words):
    h = np.fromiter((word_hash(w) for w in words), dtype=np.uint64, count=len(words))
    s = np.zeros(len(h) - 4, dtype=np.uint64)
    for i, shift in enumerate([0, 13, 26, 39, 52]):
        x = h[i:i + len(s)]
        s ^= x if shift == 0 else (x << np.uint64(shift)) | (x >> np.uint64(64 - shift))
    # SplitMix avalanche prevents systematic low hashes of repeated words.
    s ^= s >> np.uint64(30)
    s *= np.uint64(0xbf58476d1ce4e5b9)
    s ^= s >> np.uint64(27)
    s *= np.uint64(0x94d049bb133111eb)
    s ^= s >> np.uint64(31)
    s = np.unique(s)
    return s[:32].tolist()


def process_batch(batch):
    accepted, rejected = [], Counter()
    for row in batch:
        text = row["text"]
        quality = CONFIG["quality"]
        if row["language"] != quality["language"] or row["language_score"] < quality["min_language_score"] or row["int_score"] < quality["min_int_score"]:
            rejected["quality"] += 1
            continue
        normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
        words = re.findall(r"\w+", normalized)
        if len(normalized) < quality["min_characters"] or len(words) < quality["min_words"]:
            rejected["short"] += 1
            continue
        signature = sketch(words)
        if len(signature) < 16:
            rejected["repetitive"] += 1
            continue
        try:
            hostname = (urlsplit(row["url"]).hostname or "").lower().removeprefix("www.")
        except ValueError:
            hostname = ""
        if not hostname:
            rejected["invalid_hostname"] += 1
            continue
        bucket = int.from_bytes(hashlib.blake2b(hostname.encode(), digest_size=8, person=b"ad-split-v1").digest(), "little") % 1000
        split = "validation" if bucket < 5 else "test" if bucket < 10 else "train"
        accepted.append({"text": text, "text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                         "sketch": signature, "split": split, "hostname": hostname,
                         "source_file": row["source_file"], "source_row": row["source_row"],
                         "source_id": row["id"], "int_score": row["int_score"]})
    if accepted:
        encodings = TOKENIZER.encode_batch([row.pop("text") for row in accepted], add_special_tokens=False)
        for row, encoded in zip(accepted, encodings):
            row["tokens"] = np.asarray(encoded.ids + [CONFIG["eos_token_id"]], dtype="<u2").tobytes()
    return accepted, dict(rejected), len(batch)


def batches(paths):
    for path in paths:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text", "id", "url", "language", "language_score", "int_score"]):
            rows = batch.to_pylist()
            for i, row in enumerate(rows):
                row.update(source_file=path.name, source_row=offset + i)
            offset += len(rows)
            yield rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    target = ROOT / "work/pretraining/data/fineweb_edu_v1"
    target.mkdir(parents=True, exist_ok=True)
    config_hash = digest(ROOT / "configs/pretraining_data_v1.json")
    manifest = target / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert old["config_sha256"] == config_hash
        for name, meta in old["artifacts"].items():
            assert digest(target / name) == meta["sha256"]
        print(json.dumps({"event": "verified_existing", "manifest": str(manifest)}), flush=True)
        return
    if any(target.glob("*.partial")):
        raise RuntimeError(f"Unfinished preparation at {target}; inspect before restarting")
    paths = [ROOT / "work/data/fineweb-edu/sample/10BT" / name for name in CONFIG["files"]]
    for path in paths:
        assert digest(path) == CONFIG["files"][path.name], path
    tokenizer = ROOT / "work/tokenizers/gpt2/tokenizer.json"
    # Verify the Hugging Face Git blob ID of this small (non-LFS) asset.
    raw = tokenizer.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == "4b988bccc9dc5adacd403c00b4704976196548f8"
    tokens, documents, rejected, processed = Counter(), Counter(), Counter(), 0
    exact, signatures, index = set(), [], defaultdict(list)
    seen_hosts = {split: set() for split in CONFIG["maximum_tokens"]}
    handles = {split: ((target / f"{split}.bin.partial").open("wb"),
                       (target / f"{split}.documents.jsonl.partial").open("w")) for split in CONFIG["maximum_tokens"]}
    start = time.perf_counter()
    try:
        with mp.get_context("spawn").Pool(args.workers, initializer=initialize_worker) as pool:
            for rows, counts, batch_count in pool.imap(process_batch, batches(paths), chunksize=1):
                rejected.update(counts)
                processed += batch_count
                for row in rows:
                    text_hash, signature = row["text_sha256"], row.pop("sketch")
                    if text_hash in exact:
                        rejected["exact_duplicate"] += 1
                        continue
                    exact.add(text_hash)
                    keys = signature[:4]
                    if any(len(index[key]) >= CONFIG["deduplication"]["max_bucket"] for key in keys):
                        rejected["large_near_bucket"] += 1
                        continue
                    candidates = set(candidate for key in keys for candidate in index[key])
                    sigset = set(signature)
                    duplicate = False
                    for candidate in candidates:
                        other = signatures[candidate]
                        union = set(sorted(sigset | other)[:32])
                        if len(sigset & other & union) / len(union) >= .8:
                            duplicate = True
                            break
                    if duplicate:
                        rejected["near_duplicate"] += 1
                        continue
                    for key in keys:
                        index[key].append(len(signatures))
                    signatures.append(sigset)
                    split = row.pop("split")
                    available = CONFIG["maximum_tokens"][split] - tokens[split]
                    if available <= 0:
                        rejected[split + "_beyond_budget"] += 1
                        continue
                    ids = row.pop("tokens")[:available * 2]
                    count = len(ids) // 2
                    row.update(offset=tokens[split], tokens=count,
                               token_sha256=hashlib.sha256(ids).hexdigest())
                    handles[split][0].write(ids)
                    handles[split][1].write(json.dumps(row, separators=(",", ":")) + "\n")
                    tokens[split] += count
                    documents[split] += 1
                    seen_hosts[split].add(row["hostname"])
                if processed % 16384 < 256:
                    for pair in handles.values():
                        for handle in pair:
                            handle.flush()
                    print(json.dumps({"event": "prepare", "source_documents": processed,
                                      "tokens": dict(tokens), "documents": dict(documents),
                                      "rejected": dict(rejected), "seconds": time.perf_counter() - start}), flush=True)
                if all(tokens[s] >= maximum for s, maximum in CONFIG["maximum_tokens"].items()):
                    break
    finally:
        for pair in handles.values():
            for handle in pair:
                handle.close()
    assert tokens["train"] >= 100_000_000
    assert min(tokens["validation"], tokens["test"]) >= 500_000
    assert not (seen_hosts["train"] & seen_hosts["validation"] or seen_hosts["train"] & seen_hosts["test"] or seen_hosts["validation"] & seen_hosts["test"])
    artifacts = {}
    for partial in sorted(target.glob("*.partial")):
        path = partial.with_suffix("")
        partial.replace(path)
        artifacts[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    result = {"config": CONFIG, "config_sha256": config_hash, "artifacts": artifacts,
              "tokenizer_sha256": digest(tokenizer), "preparation_source_sha256": digest(Path(__file__)),
              "tokens": dict(tokens), "documents": dict(documents), "source_documents": processed,
              "hostnames": {s: len(h) for s, h in seen_hosts.items()}, "rejected": dict(rejected),
              "seconds": time.perf_counter() - start, "workers": args.workers}
    temporary = manifest.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(manifest)
    print(json.dumps({"event": "complete", "tokens": dict(tokens), "manifest_sha256": digest(manifest)}), flush=True)


if __name__ == "__main__":
    main()
