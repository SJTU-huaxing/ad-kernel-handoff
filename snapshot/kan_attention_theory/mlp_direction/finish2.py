import subprocess
from core import *
stages=[]
for script,args in [('assess.py',['scenarios']),('fresh_data.py',['extract','--round2']),('verify.py',[]),('assess.py',['kernels']),('assess.py',['ppl']),('benchmark.py',[])]:
    start=time.perf_counter();log=P/('round2_'+script.removesuffix('.py')+'_'+(args[0] if args else 'run')+'.log')
    with log.open('w') as f:r=subprocess.run([sys.executable,str(P/script),*args],stdout=f,stderr=subprocess.STDOUT)
    row=dict(script=script,args=args,exit_code=r.returncode,seconds=time.perf_counter()-start,log=str(log));stages.append(row)
    save(P/'results/stages2.json',stages);print(json.dumps(row),flush=True)
    if r.returncode:raise RuntimeError(row)
