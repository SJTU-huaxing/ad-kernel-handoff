"""Freeze WikiText transfer and synthetic recall inputs before scoring models."""
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/pretraining/evaluation/transfer_recall_v1"
PROTOCOL = ROOT / "configs/pretrained_evaluation_v1.json"
DATA = ROOT / "work/pretraining/data/fineweb_edu_v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized_hash(text):
    return hashlib.sha256(" ".join(unicodedata.normalize("NFKC", text).lower().split()).encode()).hexdigest()


def main():
    protocol = json.loads(PROTOCOL.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        old = json.loads((OUT / "manifest.json").read_text())
        assert old["protocol_sha256"] == digest(PROTOCOL)
        for name, expected in old["artifacts"].items():
            assert digest(OUT / name) == expected
        print("Verified existing evaluation manifest", flush=True)
        return
    tokenizer = Tokenizer.from_file(str(ROOT / "work/tokenizers/gpt2/tokenizer.json"))
    source = ROOT / "work/data/wikitext-document/wikitext-103-raw-v1/wikitext-103-raw-v1-test.parquet"
    assert digest(source) == "e7fc7d4c385d027e0360f7eb320866df61adb68f385986e5c7f4f3911d62f4eb"
    train_hashes = set()
    with (DATA / "train.documents.jsonl").open() as f:
        for line in f:
            train_hashes.add(json.loads(line)["text_sha256"])
    wiki_records, wiki_arrays, offset, excluded = [], [], 0, 0
    for index, row in enumerate(pq.read_table(source).to_pylist()):
        text_hash = normalized_hash(row["page"])
        if text_hash in train_hashes:
            excluded += 1
            continue
        ids = np.asarray(tokenizer.encode(row["page"], add_special_tokens=False).ids, dtype="<u2")
        if len(ids) < 2:
            continue
        wiki_records.append({"source_row": index, "offset": offset, "tokens": len(ids),
                             "text_sha256": text_hash, "token_sha256": hashlib.sha256(ids.tobytes()).hexdigest()})
        wiki_arrays.append(ids)
        offset += len(ids)
    np.concatenate(wiki_arrays).tofile(OUT / "wiki.bin")
    (OUT / "wiki.documents.json").write_text(json.dumps(wiki_records, indent=2) + "\n")
    del train_hashes

    words = ("red blue green yellow black white orange purple silver gold pink brown "
             "apple banana cherry lemon grape peach pear plum cat dog horse sheep tiger lion bear wolf "
             "chair table clock glass shirt shoe hat coat tree flower grass leaf stone sand rain snow "
             "sun moon star cloud wind fire water earth book paper pen pencil road river lake sea "
             "bread milk cheese rice").split()
    assert len(words) == 64 and len(set(words)) == 64
    encoded = [tokenizer.encode(" " + word, add_special_tokens=False).ids for word in words]
    assert all(len(ids) == 1 for ids in encoded), list(zip(words, encoded))
    value_ids = np.asarray([ids[0] for ids in encoded])
    settings = protocol["associative_recall"]
    rng = np.random.default_rng(settings["seed"])
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
    filler = encode(" This paragraph provides background information unrelated to the list. Please keep reading until the question appears. The following passage continues with general discussion and contains no additional entries.\n")
    arrays = {str(length): [] for length in settings["lengths"]}
    cases = []
    for count in settings["pairs"]:
        for trial in range(settings["trials_per_pair_count"]):
            selection = rng.choice(len(words), count, replace=False)
            target = [0, count // 2, count - 1][trial % 3]
            table = "Read the list and repeat the value for the requested item.\n"
            table += "".join(f"Item {i + 1:03d}: {words[int(j)]}\n" for i, j in enumerate(selection))
            prefix = encode(table)
            suffix = encode(f"\nRepeat the value from the list.\nItem {target + 1:03d}:")
            index = len(cases)
            for length in settings["lengths"]:
                gap = length - len(prefix) - len(suffix)
                assert gap >= 0
                tokens = prefix + (filler * ((gap + len(filler) - 1) // len(filler)))[:gap] + suffix
                assert len(tokens) == length
                arrays[str(length)].append(np.asarray(tokens, dtype="<u2"))
            cases.append({"index": index, "pairs": count, "trial": trial, "target_position": target,
                          "target_position_group": settings["target_positions"][trial % 3],
                          "candidate_token_ids": value_ids[selection].tolist(), "gold_candidate_index": target,
                          "gold_token_id": int(value_ids[selection[target]]), "gold_text": " " + words[int(selection[target])],
                          "prefix_tokens": len(prefix), "query_tokens": len(suffix)})
    np.savez(OUT / "recall.npz", **{k: np.stack(v) for k, v in arrays.items()})
    (OUT / "recall.cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    files = ["wiki.bin", "wiki.documents.json", "recall.npz", "recall.cases.json"]
    manifest = {"protocol_sha256": digest(PROTOCOL), "source_sha256": digest(Path(__file__)),
                "training_manifest_sha256": digest(DATA / "manifest.json"), "tokenizer_sha256": digest(ROOT / "work/tokenizers/gpt2/tokenizer.json"),
                "wiki_source_sha256": digest(source), "wiki_documents": len(wiki_records), "wiki_tokens": offset,
                "wiki_exact_training_matches_excluded": excluded, "recall_tables": len(cases),
                "recall_prompts": len(cases) * len(settings["lengths"]), "artifacts": {name: digest(OUT / name) for name in files}}
    temporary = OUT / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(OUT / "manifest.json")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
