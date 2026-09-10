"""Tail rank-zero JSONL metrics into TensorBoard without importing torch.

Training and its checkpoint source binding remain untouched. JSONL is the
source of truth; startup/corrected history uses TensorBoard restart markers to
replace old curves, so reattaching does not duplicate steps.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import time

from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretrain_100m_2b"
TRAIN_TAGS = {"loss":"train/loss", "lr":"train/learning_rate", "grad_norm":"train/grad_norm",
              "tokens":"progress/trained_tokens", "tokens_per_second":"performance/tokens_per_second",
              "step_seconds":"performance/step_seconds", "micro_batch":"batch/micro_batch_per_npu",
              "accumulation":"batch/gradient_accumulation",
              "numerical_splits_rank0":"numerics/splits_rank0"}
VAL_TAGS = {"nll":"validation/loss", "ppl":"validation/perplexity", "seconds":"validation/seconds"}


def read_rows(path):
    """Ignore an unfinished last line and keep the latest resumed trajectory."""
    if not path.exists(): return {}
    rows, previous = {}, -1
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.endswith("\n"): break
            if not line.strip(): continue
            try:
                row = json.loads(line)
                step = int(row["step"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid complete metric record: {path}:{number}") from exc
            if step <= previous:
                rows = {s:r for s,r in rows.items() if s < step}
            rows[step] = row
            previous = step
    return rows


def values(row, mapping):
    return [Summary.Value(tag=tag, simple_value=float(row[key])) for key,tag in mapping.items()
            if isinstance(row.get(key), (int,float)) and math.isfinite(row[key])]


class RunWriter:
    def __init__(self, run, logdir):
        self.run = run
        self.steps = EventFileWriter(str(logdir/"steps"/run.name), flush_secs=2)
        self.tokens = EventFileWriter(str(logdir/"tokens"/run.name), flush_secs=2)
        self.previous = None
        self.signature = None

    def update(self):
        paths = [self.run/"train.jsonl", self.run/"validation.jsonl"]
        signature = tuple((p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns) if p.exists() else None for p in paths)
        if signature == self.signature: return 0
        train, validation = map(read_rows, paths)
        if train:
            validation = {s:r for s,r in validation.items() if s <= max(train)}
        else:
            validation = {}
        current = {"train":train, "validation":validation}
        reset = self.previous is None or any(
            rows.get(step) != row
            for kind,rows in current.items() for step,row in (self.previous or {}).get(kind,{}).items())
        now = time.time()
        if reset:
            for writer in [self.steps,self.tokens]:
                writer.add_event(Event(wall_time=now, step=0, session_log=SessionLog(status=SessionLog.START)))
            previous = {"train":{}, "validation":{}}
        else: previous = self.previous
        count = 0
        for kind, rows in current.items():
            for step, row in sorted(rows.items()):
                if step in previous[kind]: continue
                mapping = TRAIN_TAGS if kind == "train" else VAL_TAGS
                scalar_values = values(row, mapping)
                if scalar_values:
                    self.steps.add_event(Event(wall_time=now, step=step, summary=Summary(value=scalar_values)))
                token_step = row.get("tokens") if kind == "train" else row.get("trained_tokens")
                mapping = {"loss":"train/loss_by_tokens"} if kind == "train" else {
                    "nll":"validation/loss_by_tokens", "ppl":"validation/perplexity_by_tokens"}
                if isinstance(token_step,int):
                    self.tokens.add_event(Event(wall_time=now, step=token_step, summary=Summary(value=values(row,mapping))))
                count += 1
        self.steps.flush()
        self.tokens.flush()
        self.previous, self.signature = current, signature
        return count

    def close(self):
        self.steps.close()
        self.tokens.close()


class Bridge:
    def __init__(self, runs, logdir):
        self.runs, self.logdir, self.writers = runs, logdir, {}

    def poll(self):
        count = 0
        for file in sorted(self.runs.glob("*/train.jsonl")):
            name = file.parent.name
            if name not in self.writers:
                self.writers[name] = RunWriter(file.parent, self.logdir)
                print(f"TensorBoard: following {file.parent}", flush=True)
            count += self.writers[name].update()
        return count

    def close(self):
        for writer in self.writers.values(): writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=BASE/"runs")
    parser.add_argument("--logdir", type=Path, default=BASE/"tensorboard")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    assert args.interval > 0
    args.logdir.mkdir(parents=True, exist_ok=True)
    lock = (args.logdir/".bridge.lock").open("a")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit(f"A metrics bridge already owns {args.logdir}")
    bridge, stopping = Bridge(args.runs,args.logdir), [False]
    def stop(signum,frame): stopping[0] = True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    print(f"Watching {args.runs}; TensorBoard events: {args.logdir}; poll {args.interval:g}s. CPU only.",flush=True)
    try:
        while True:
            count = bridge.poll()
            if count: print(f"TensorBoard: imported {count} new metric rows",flush=True)
            if args.once or stopping[0]: break
            time.sleep(args.interval)
    finally: bridge.close()


if __name__ == "__main__": main()
