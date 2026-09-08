import subprocess,time,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
stages=[['data.py','extract','--confirm'],['evaluate.py','kernels'],['theory_check.py'],['evaluate.py','ppl']]
for args in stages:
 start=time.perf_counter();name='_'.join(x.replace('--','') for x in args).replace('.py','');log=P/(name+'.log')
 with log.open('w') as f:r=subprocess.run([sys.executable,str(P/args[0]),*args[1:]],stdout=f,stderr=subprocess.STDOUT)
 result=dict(stage=args,exit_code=r.returncode,seconds=time.perf_counter()-start);(P/'results'/f'stage_{name}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
