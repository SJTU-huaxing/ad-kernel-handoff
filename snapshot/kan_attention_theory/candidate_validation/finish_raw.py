"""Sequential follow-up for the unrestricted nonnegative basis ablation."""
import json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
    methods=['cone_raw_mlp','cone_raw_kan','cone_raw_mulkan']
    stages=[
        ('raw_prepare','raw_basis.py',[]),
        ('raw_verify','verify.py',['--methods',*methods,'--output','raw_implementation.json']),
        ('raw_screen','evaluate.py',['--datasets','validation','moment_train','official','--methods',*methods]),
        ('raw_full','evaluate.py',['--datasets','internal','--n','131072','--methods',*methods]),
        ('raw_ppl','ppl.py',['--methods',*methods]),
        ('raw_ppl_fp64','ppl.py',['--fp64-check','--methods',*methods]),
        ('raw_benchmark','benchmark_full.py',['--output','model_benchmark.json','--methods',*methods]),
        ('raw_features','benchmark_features.py',['--methods',*methods]),
    ]
    rows=[]
    for name,script,args in stages:
        started=time.perf_counter();print(json.dumps(dict(event='stage_start',stage=name)),flush=True)
        with (P/(name+'.log')).open('w') as stream:
            proc=subprocess.run([sys.executable,'-u',str(P/script),*args],stdout=stream,stderr=subprocess.STDOUT)
        row=dict(stage=name,returncode=proc.returncode,seconds=time.perf_counter()-started);rows.append(row)
        (P/'results/raw_stages.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
        if proc.returncode:raise RuntimeError(name)

if __name__=='__main__':main()
