"""Report actual logged training invocations without double-counting resumes."""
import fcntl
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work/pretraining"


def training_segments(events, tokens_per_step):
    segments, current = [], None

    def finish(status, terminal=None):
        rows = current["train"]
        start, target = current["start"]["step"], current["start"]["target_step"]
        end = rows[-1]["step"] if rows else start
        elapsed = terminal["seconds"] if terminal else rows[-1]["elapsed_seconds"] if rows else None
        assert elapsed is None or math.isfinite(elapsed) and elapsed > 0
        if terminal:
            assert terminal["step"] == end == target
            assert terminal["tokens"] == end * tokens_per_step and not terminal["synthetic"]
            assert elapsed >= rows[-1]["elapsed_seconds"]
        tokens = (end - start) * tokens_per_step
        segments.append({"status": status, "start_step": start, "last_logged_step": end, "target_step": target,
                         "logged_update_tokens": tokens, "seconds": elapsed,
                         "tokens_per_second": tokens / elapsed if elapsed else None,
                         "rank0_logged_peak_allocated_gib": max((r["peak_allocated_gib"] for r in rows), default=None),
                         "rank0_numerical_splits_at_last_train_log": rows[-1]["numerical_splits_this_process_rank0"] if rows else None})

    for event in events:
        if event["event"] in {"start", "resume"}:
            if current is not None:
                finish("interrupted_before_restart")
            assert not event["synthetic"] and 0 <= event["step"] < event["target_step"]
            current = {"start": event, "train": []}
        elif event["event"] == "train":
            assert current is not None
            previous = current["train"][-1]["step"] if current["train"] else current["start"]["step"]
            assert previous < event["step"] <= current["start"]["target_step"]
            assert event["tokens"] == event["step"] * tokens_per_step
            current["train"].append(event)
        elif event["event"] == "run_complete":
            assert current is not None and current["train"]
            finish("complete", event)
            current = None
    if current is not None:
        finish("open_at_snapshot")
    return segments


def main():
    protocol_path = ROOT / "configs/pretraining_protocol_v2.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    tokens_per_step = protocol["global_batch_sequences"] * protocol["sequence_length"]
    quality_path = BASE / "reports/summary.json"
    quality = json.loads(quality_path.read_text())
    assert quality["protocol_sha256"] == protocol_hash
    evaluations = {(r["method"], r["seed"], r["step"]): r for r in quality["runs"]}
    invocations, sources = [], {}
    for method in protocol["job_order"]:
        for seed in [protocol["primary_seed"], *protocol["additional_seeds"]]:
            directory = BASE / "runs" / f"{method}_s{seed}"
            if not (directory / "run.json").exists():
                continue
            metadata = json.loads((directory / "run.json").read_text())
            binding = metadata["binding"]
            assert binding["protocol_sha256"] == protocol_hash and not binding["synthetic"]
            assert binding["data_manifest_sha256"] == protocol["data_manifest_sha256"]
            assert binding["method"] == method and binding["seed"] == seed
            for source, expected in binding["code_sha256"].items():
                assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected
            assert metadata["world_size"] == protocol["world_size"] == 2
            log = directory / "train.jsonl"
            if not log.exists():
                continue
            raw = log.read_bytes()
            lines, events, partial = raw.decode().splitlines(), [], False
            for index, line in enumerate(lines):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    assert index == len(lines) - 1
                    partial = True
            source = str(log.relative_to(ROOT))
            sources[source] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "partial_final_line_ignored": partial}
            for number, segment in enumerate(training_segments(events, tokens_per_step)):
                row = {"method": method, "seed": seed, "invocation": number, "log": source, **segment}
                evaluation = evaluations.get((method, seed, row["last_logged_step"]))
                if row["status"] == "complete" and evaluation:
                    row["evaluated_checkpoint_sha256"] = evaluation["checkpoint_sha256"]
                    row["packed_test_perplexity"] = evaluation["packed_test"]["perplexity"]
                invocations.append(row)
    output = {"training_protocol_sha256": protocol_hash, "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "log_snapshots": sources, "invocations": invocations,
              "scope": "Each row is one start/resume invocation. Complete duration includes training, in-loop validation and checkpointing, excluding initialization/data verification before the logged start and separate quality evaluation. Incomplete duration ends at the last training log and is a lower bound on spent wall time. These are observed training costs, not exclusive inference or operator microbenchmarks.",
              "counting": "Tokens are (last_logged_step - start_step) * global_batch * sequence_length. Interrupted/replayed updates may overlap across rows; rows must not be added and called distinct data exposure. Open rows do not prove a process is alive.",
              "memory_and_splits": "Peak allocated memory and numerical subdivisions are rank0 observations at training log points. Subdivisions may include in-loop validation before those points and reset on process restart. Final validation after the last training log is not included in these counters."}
    target = BASE / "reports/training_cost.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    lines = ["# 实际训练阶段成本", "", "每行对应一次实际 start/resume 调用。表中仅列正常结束的阶段；运行中或被后续重启打断的阶段另存 JSON，不能当作完成结果。", "",
             "时间从训练循环前的 start/resume 日志之后开始，到 run_complete 为止，包含数据取 batch、HCCL、AdamW、中途验证和 checkpoint 保存；不含此前的环境/模型初始化、语料哈希核验和随后独立的质量评测。这是本次实现的训练观测，不是独占设备推理基准。", "",
             "| 方法 | seed | 起止 step | 本次更新 tokens | 阶段分钟 | 阶段平均 tokens/s | rank0 训练日志峰值 GiB | rank0 末次日志分段计数 |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in invocations:
        if row["status"] != "complete":
            continue
        lines.append(f"| {row['method']} | {row['seed']} | {row['start_step']}→{row['last_logged_step']} | {row['logged_update_tokens']:,} | "
                     f"{row['seconds']/60:.3f} | {row['tokens_per_second']:.1f} | {row['rank0_logged_peak_allocated_gib']:.3f} | {row['rank0_numerical_splits_at_last_train_log']:,} |")
    lines += ["", "恢复段只计算本次实际更新的 tokens，例如从 1600 到 16384 是两者之差，不再次计入前 1600 步。若有故障后重放，跨调用的更新 tokens 会重叠，不能相加称为不同训练数据。未结束调用的耗时仅覆盖到末次训练日志，实际花费可能更多。", "",
              "峰值和分段计数均来自 rank0 的训练日志采样；计数可能包含此前中途验证，进程重启时归零，不包含末次训练日志之后的最终验证。它们不是两张卡合计值。softmax 使用 BF16 原生 SDPA，线性 feature/attention 使用 FP32；所有方法骨干均为 BF16 autocast 和 FP32 主参数。", "",
              "逐调用记录、完整/未结束状态和读取时的日志字节哈希在 `work/pretraining/reports/training_cost.json`。进程存活情况另用 `python experiments/status.py` 检查。", ""]
    target = ROOT / "reports/TRAINING_COST.zh.md"
    temporary = target.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines))
    temporary.replace(target)
    print(json.dumps({"event": "training_cost_summary", "complete_invocations": sum(r["status"] == "complete" for r in invocations),
                      "other_invocations": sum(r["status"] != "complete" for r in invocations)}), flush=True)


if __name__ == "__main__":
    (BASE / "reports").mkdir(exist_ok=True)
    with (BASE / "reports/.training_cost.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
