import sys,json,time,subprocess
from pathlib import Path
O=Path(__file__).resolve().parent;C=O.parent/'causal_direction';stage=C/'results/stage_final_benchmark.json'
while not stage.exists():time.sleep(2)
assert json.loads(stage.read_text())['exit_code']==0
for args in [['train.py'],['collect.py','extract'],['assess.py','kernels'],['assess.py','ppl'],['verify.py']]:
 start=time.perf_counter();name='_'.join(args).replace('.py','')
 with (O/(name+'.log')).open('w') as f:r=subprocess.run([sys.executable,str(O/args[0]),*args[1:]],stdout=f,stderr=subprocess.STDOUT)
 out=dict(stage=args,exit_code=r.returncode,seconds=time.perf_counter()-start);(O/'results'/f'stage_{name}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
