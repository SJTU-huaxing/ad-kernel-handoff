"""Sequential GPU completion after extend.py: no overlapping GPU jobs."""
import subprocess
from core import *

stages=[]
for script,args in [('value_experiment.py',['extract']),('value_experiment.py',['fit']),
                    ('information.py',[]),
                    ('assess.py',['scenarios']),('fresh_data.py',['extract']),
                    ('assess.py',['kernels']),('assess.py',['ppl'])]:
    start=time.perf_counter();log=P/(script.removesuffix('.py')+'_'+(args[0] if args else 'run')+'.log')
    with log.open('w') as f:r=subprocess.run([sys.executable,str(P/script),*args],stdout=f,stderr=subprocess.STDOUT)
    row=dict(script=script,args=args,exit_code=r.returncode,seconds=time.perf_counter()-start,log=str(log));stages.append(row)
    save(P/'results/stages.json',stages);print(json.dumps(row),flush=True)
    if r.returncode:raise RuntimeError(row)
