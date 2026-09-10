"""CPU-only integration checks; fixtures never enter the real dashboard."""
import importlib.util
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tb_bridge",ROOT/"scripts/tensorboard_metrics_100m.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def append(path,row,newline=True):
    with path.open("a") as f: f.write(json.dumps(row)+("\n" if newline else ""))


def points(logdir,tag):
    reader = EventAccumulator(str(logdir),size_guidance={"scalars":0}).Reload()
    return [(p.step,p.value) for p in reader.Scalars(tag)]


def main():
    proof = {}
    with tempfile.TemporaryDirectory(prefix="tensorboard-100m-fixture-") as temp:
        base=Path(temp)
        runs,logs=base/"runs",base/"events"
        run=runs/"ad64_s11"
        run.mkdir(parents=True)
        append(run/"train.jsonl",{"step":1,"tokens":100,"loss":3.,"lr":.001})
        append(run/"train.jsonl",{"step":2,"tokens":200,"loss":2.,"lr":.001})
        append(run/"validation.jsonl",{"step":2,"tokens":40,"trained_tokens":200,"nll":2.,"ppl":math.exp(2)})
        bridge=module.Bridge(runs,logs)
        assert bridge.poll()==3 and bridge.poll()==0
        assert points(logs/"steps/ad64_s11","train/loss")==[(1,3.),(2,2.)]
        assert points(logs/"tokens/ad64_s11","validation/loss_by_tokens")==[(200,2.)]
        proof["training_scalars_and_validation_token_axis"]=True
        append(run/"train.jsonl",{"step":3,"tokens":300,"loss":1.75},newline=False)
        assert bridge.poll()==0
        with (run/"train.jsonl").open("a") as f:f.write("\n")
        assert bridge.poll()==1
        assert points(logs/"steps/ad64_s11","train/loss")[-1]==(3,1.75)
        proof["partial_last_line_waited_then_imported"]=True
        bridge.close()
        bridge=module.Bridge(runs,logs)
        assert bridge.poll()==4
        assert points(logs/"steps/ad64_s11","train/loss")==[(1,3.),(2,2.),(3,1.75)]
        proof["bridge_restart_without_duplicate_steps"]=True
        append(run/"train.jsonl",{"step":2,"tokens":200,"loss":1.5})
        append(run/"validation.jsonl",{"step":2,"tokens":40,"trained_tokens":200,"nll":1.5,"ppl":math.exp(1.5)})
        bridge.poll()
        assert points(logs/"steps/ad64_s11","train/loss")==[(1,3.),(2,1.5)]
        assert points(logs/"tokens/ad64_s11","train/loss_by_tokens")==[(100,3.),(200,1.5)]
        proof["resume_rollback_purges_abandoned_future_steps"]=True
        bridge.close()
        with socket.socket() as s:s.bind(("127.0.0.1",0));port=s.getsockname()[1]
        with (base/"server.log").open("w") as log:
            child=subprocess.Popen([sys.executable,"-m","tensorboard.main","--logdir",str(logs),"--host","127.0.0.1","--port",str(port),"--load_fast=false","--reload_interval","1"],stdout=log,stderr=subprocess.STDOUT)
            try:
                url=f"http://127.0.0.1:{port}/data/plugin/scalars/scalars?"+urllib.parse.urlencode({"run":"steps/ad64_s11","tag":"train/loss"})
                actual=None
                for _ in range(40):
                    try:
                        with urllib.request.urlopen(url,timeout=1) as response:actual=json.load(response)
                        if len(actual)==2:break
                    except (OSError,ValueError):pass
                    assert child.poll() is None,(base/"server.log").read_text()
                    time.sleep(.25)
                assert actual is not None and [(p[1],p[2]) for p in actual]==[(1,3.),(2,1.5)],actual
                proof["tensorboard_http_scalar_api_reads_correct_curve"]=True
            finally:
                child.terminate()
                child.wait(timeout=10)
        assert "torch" not in sys.modules
        proof["no_torch_import_or_npu_context"]=True
    proof["fixtures_removed_no_synthetic_production_curves"]=True
    proof["formal_training_started"]=False
    proof["passed"]=True
    out=ROOT/"work/pretrain_100m_2b/validation/tensorboard.json"
    out.write_text(json.dumps(proof,indent=2)+"\n")
    print(json.dumps(proof,indent=2))


if __name__ == "__main__": main()
