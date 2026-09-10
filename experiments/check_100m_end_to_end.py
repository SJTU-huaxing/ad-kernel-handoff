"""Final full-architecture two-update smoke followed by all evaluation entrypoints."""
import json
import os
import subprocess
import sys
from pretrain_100m.common import ROOT,atomic_json,load_checkpoint,source_binding,digest,DEFAULT_CONFIG


def main():
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH", ""),OMP_NUM_THREADS="4",HCCL_EXEC_TIMEOUT="900")
    launch=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2"]
    run=ROOT/"work/pretrain_100m_2b/probes/final_config_ad64_s29"
    with (ROOT/"work/logs/100m-final-config-train.log").open("w") as f:
        subprocess.run(launch+["-m","pretrain_100m.train","--method","ad64","--seed","29","--probe-tag","final_config",
                              "--probe-steps","2","--probe-save","--probe-eval"],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (ROOT/"work/logs/100m-final-config-eval.log").open("w") as f:
        subprocess.run(launch+["-m","pretrain_100m.evaluate",str(run/"final.pt"),"--allow-probe","--limit","2"],
                       cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    box=load_checkpoint(run/"final.pt")
    evaluation=json.loads((run/"evaluation_probe/summary.json").read_text())
    assert evaluation["complete"] and evaluation["probe"] and box["binding"]["probe"]
    assert set(evaluation["metrics"])=={"validation","test","wiki","long","recall"}
    assert box["binding"]["source_sha256"]==source_binding()
    assert box["binding"]["protocol_sha256"]==digest(DEFAULT_CONFIG)
    assert len(set(box["probe_replica_sha256"]))==1
    assert box["tokens"]<2000000000
    result={"passed":True,"probe":True,"method":"ad64","seed":29,"optimizer_updates":2,"parameters":101964800,
            "tokens":box["tokens"],"all_five_evaluation_entrypoints":True,"two_replicas_bitwise_equal":True,
            "checkpoint":str((run/"final.pt").relative_to(ROOT)),"evaluation":str((run/"evaluation_probe/summary.json").relative_to(ROOT)),
            "source_sha256":source_binding(),"config_sha256":digest(DEFAULT_CONFIG),
            "scope":"Short execution test only; no formal 2B training launched and no final-budget quality claim"}
    atomic_json(ROOT/"work/pretrain_100m_2b/validation/end_to_end.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
