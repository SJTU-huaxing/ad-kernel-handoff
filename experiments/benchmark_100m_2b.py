"""Bounded real two-card probes, with physical-device utilization samples."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cases", nargs="+", default=["ad64:16:64", "softmax:48:64", "favor64:32:64", "hedgehog:24:64"])
    a = p.parse_args()
    assert 4 <= a.steps <= 16
    output = ROOT/"work/pretrain_100m_2b/validation"/a.tag
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH", ""), OMP_NUM_THREADS="4", HCCL_EXEC_TIMEOUT="900")
    results = []
    for case in a.cases:
        method, micro, chunk = case.split(":")
        tag = f"{a.tag}_m{micro}_c{chunk}"
        command = [sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",
                   "-m","pretrain_100m.train","--method",method,"--probe-steps",str(a.steps),"--probe-tag",tag,
                   "--micro-batch",micro,"--chunk-size",chunk]
        log = output/f"{method}_m{micro}_c{chunk}.log"
        with log.open("w") as f, log.with_suffix(".utilization.jsonl").open("w") as monitor:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            while child.poll() is None:
                # One read per second per physical card; no accelerator work is launched.
                for dev in [2,3]:
                    r = subprocess.run(["npu-smi","info","-t","usages","-i",str(dev)],capture_output=True,text=True,timeout=10)
                    record = {"time":time.time(),"physical_npu":dev}
                    for line in r.stdout.splitlines():
                        if ":" not in line: continue
                        key,value = [x.strip() for x in line.split(":",1)]
                        if any(s in key for s in ["Aicore","Aicube","Aivector","HBM Usage","HBM Bandwidth"]):
                            record[key]=value
                    monitor.write(json.dumps(record)+"\n");monitor.flush()
                time.sleep(1)
        result = {"case":case,"exit_code":child.returncode,"log":str(log)}
        path = ROOT/"work/pretrain_100m_2b/probes"/f"{tag}_{method}_s11/probe_result.json"
        if path.exists(): result["measurement"] = json.loads(path.read_text())
        results.append(result)
        (output/"results.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps({"case":case,"exit_code":child.returncode,"tokens_per_second":result.get("measurement",{}).get("measured_tokens_per_second")}),flush=True)
        if child.returncode: break


if __name__ == "__main__": main()
