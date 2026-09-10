"""Real two-NPU checkpoint/resume and uneven-final-batch regression."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from pretrain_100m.common import ROOT,load_checkpoint,atomic_json


def main():
    config=ROOT/"work/pretrain_100m_2b/validation/resume_test_config.json"
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH", ""),OMP_NUM_THREADS="4",HCCL_EXEC_TIMEOUT="900")
    base=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2","-m","pretrain_100m.train",
          "--config",str(config),"--method","ad64","--probe-save"]
    for label,args in [
        ("split2",["--probe-tag","resume_split_v2","--probe-steps","2"]),
        ("resume3",["--probe-tag","resume_split_v2","--probe-steps","3","--resume","--probe-eval"]),
        ("direct3",["--probe-tag","resume_direct_v2","--probe-steps","3","--probe-eval"])]:
        path=ROOT/f"work/logs/100m-resume-{label}.log"
        with path.open("w") as f:
            subprocess.run(base+args,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(label,'passed',flush=True)
    p=ROOT/"work/pretrain_100m_2b/probes"
    a=load_checkpoint(p/"resume_split_v2_ad64_s11/checkpoint_000003.pt")
    b=load_checkpoint(p/"resume_direct_v2_ad64_s11/checkpoint_000003.pt")
    assert a["tokens"]==b["tokens"]==4097 and a["step"]==b["step"]==3
    assert a["binding"]==b["binding"]
    comparisons=[]
    def compare(a,b,path):
        if isinstance(a,torch.Tensor):
            assert a.dtype==b.dtype and a.shape==b.shape,path
            equal=torch.equal(a,b)
            error=(a.float()-b.float()).abs().max().item() if a.numel() else 0.
            relative=((a.double()-b.double()).square().sum().sqrt()/b.double().square().sum().sqrt().clamp_min(1e-30)).item()
            comparisons.append({"path":path,"bitwise_equal":equal,"max_absolute":error,"relative_l2":relative})
            assert error <= 1e-7 and relative <= 1e-6,(path,error,relative)
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a: compare(a[k],b[k],path+"/"+str(k))
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+"/"+str(i))
        else: assert a==b,(path,a,b)
    compare(a["model"],b["model"],"model")
    compare(a["optimizer"],b["optimizer"],"optimizer")
    assert len(set(a["probe_replica_sha256"]))==len(set(b["probe_replica_sha256"]))==1
    for name in ["resume_split_v2_ad64_s11","resume_direct_v2_ad64_s11"]:
        val=json.loads((p/name/"validation.jsonl").read_text().splitlines()[-1])
        assert val["tokens"]==8192 and val["trained_tokens"]==4097
    result={"passed":True,"tokens":4097,"updates":3,"split":"2 updates + save/load + final update versus uninterrupted 3 updates",
            "two_replicas_identical":True,"model_and_optimizer_bitwise_equal":all(r['bitwise_equal'] for r in comparisons),
            "max_absolute_error":max(r['max_absolute'] for r in comparisons),
            "max_relative_l2_error":max(r['relative_l2'] for r in comparisons),
            "tolerance":"per tensor max_abs <= 1e-7 AND relative L2 <= 1e-6; scalar counters exact",
            "tensors_compared":len(comparisons),
            "final_partial_batch":True,"comparisons":comparisons}
    atomic_json(ROOT/"work/pretrain_100m_2b/validation/resume.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))


if __name__=="__main__":main()
