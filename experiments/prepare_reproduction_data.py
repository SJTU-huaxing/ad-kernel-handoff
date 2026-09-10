"""Freeze a fresh public-text token manifest without reusing old result files."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"work/reproduction/public_wikitext_v1/data"


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while block:=f.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    protocol_path=ROOT/"configs/reproduction_public_v1.json"
    config=json.loads(protocol_path.read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    destination=OUT/"manifest.json"
    if destination.exists():
        old=json.loads(destination.read_text())
        assert old["protocol_sha256"]==digest(protocol_path), "Existing manifest has a different protocol"
        print(json.dumps({"event":"manifest_already_exists","sha256":digest(destination)}),flush=True)
        return
    source=ROOT/"work/data/wikitext-document/wikitext-103-raw-v1/wikitext-103-raw-v1-train.parquet"
    assert digest(source)=="09410d6abb38a3dc4ef52da1803fdc8db3cae343ac9c40ea2a8d047c7892376c"
    pages=pq.read_table(source,columns=["page"]).column("page").to_pylist()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/"work/models/Qwen2.5-1.5B",local_files_only=True)
    order=list(range(len(pages)))
    random.Random(config["selection_seed"]).shuffle(order)
    seen_text,seen_tokens=set(),set()
    records=[]
    for split,spec in config["splits"].items():
        count=0
        for index in order:
            text=pages[index]
            text_hash=hashlib.sha256(text.encode()).hexdigest()
            if text_hash in seen_text:
                continue
            ids=tokenizer(text,add_special_tokens=False,truncation=True,max_length=spec["tokens"])["input_ids"]
            if len(ids)!=spec["tokens"]:
                continue
            token_hash=hashlib.sha256(np.asarray(ids,dtype="<u4").tobytes()).hexdigest()
            if token_hash in seen_tokens:
                continue
            qpos=torch.randperm(len(ids),generator=torch.Generator().manual_seed(config["query_position_seed"]+count))[:64].sort().values.tolist()
            records.append({"split":split,"index":count,"source_index":index,
                            "input_ids":ids,"query_positions":qpos,
                            "text_sha256":text_hash,"token_sha256":token_hash})
            seen_text.add(text_hash)
            seen_tokens.add(token_hash)
            count+=1
            if count%512==0:
                print(json.dumps({"event":"select","split":split,"documents":count}),flush=True)
            if count==spec["count"]:
                break
        assert count==spec["count"],(split,count)
    result={"protocol":config,"protocol_sha256":digest(protocol_path),
            "records":records,"source_parquet_sha256":digest(source),
            "model":"Qwen/Qwen2.5-1.5B","revision":config["model_revision"],
            "scope":config["scope"],"heads":[[l,h] for l in [14,27] for h in range(12)],
            "key_group_indices":[i//6 for i in range(24)]}
    temporary=destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result,ensure_ascii=False,separators=(",",":"))+"\n")
    temporary.replace(destination)
    print(json.dumps({"event":"manifest_frozen","path":str(destination),"sha256":digest(destination),
                      "counts":{s:sum(r["split"]==s for r in records) for s in config["splits"]},
                      "train_tokens":4096*1024}),flush=True)


if __name__=="__main__":
    main()
