import subprocess
from core import *
stages=[]
for script,args in [('calibrate.py',[]),('fresh_data.py',['extract','--round3']),('gauge_assess.py',['kernels']),('product.py',[]),('gauge_assess.py',['ppl'])]:
    start=time.perf_counter();log=P/('round3_'+script.removesuffix('.py')+'_'+(args[0] if args else 'run')+'.log')
    with log.open('w') as f:r=subprocess.run([sys.executable,str(P/script),*args],stdout=f,stderr=subprocess.STDOUT)
    row=dict(script=script,args=args,exit_code=r.returncode,seconds=time.perf_counter()-start,log=str(log));stages.append(row)
    save(P/'results/stages3.json',stages);print(json.dumps(row),flush=True)
    if r.returncode:raise RuntimeError(row)
