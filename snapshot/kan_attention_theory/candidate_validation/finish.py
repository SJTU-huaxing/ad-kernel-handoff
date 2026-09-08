"""Run remaining GPU stages sequentially; logs preserve each stage separately."""
import json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
    stages=[
        ('ppl_other_seeds','ppl.py',['--methods',*[f'nn_{m}_{s}' for m in ['mlp','kan','mulkan'] for s in [29,47]]]),
        ('ppl_fp64','ppl.py',['--fp64-check','--methods','cone_mulkan','cone_mlp']),
        ('benchmark_full','benchmark_full.py',[]),
        ('benchmark_features','benchmark_features.py',[]),
    ]
    rows=[]
    for name,script,args in stages:
        started=time.perf_counter();print(json.dumps(dict(event='stage_start',stage=name)),flush=True)
        with (P/(name+'.log')).open('w') as stream:
            proc=subprocess.run([sys.executable,'-u',str(P/script),*args],stdout=stream,stderr=subprocess.STDOUT)
        row=dict(stage=name,returncode=proc.returncode,seconds=time.perf_counter()-started);rows.append(row)
        (P/'results/stages.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
        if proc.returncode:raise RuntimeError(name)

if __name__=='__main__':main()
