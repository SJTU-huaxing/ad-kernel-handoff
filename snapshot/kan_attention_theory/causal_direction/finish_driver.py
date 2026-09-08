import subprocess,time,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
stage=P/'results/stage_evaluate_ppl.json'
while not stage.exists():time.sleep(2)
assert json.loads(stage.read_text())['exit_code']==0
stages=[['causal_witness.py'],['rotation_diagnostic.py'],['evaluate.py','ppl'],['benchmark.py']]
for args in stages:
 start=time.perf_counter();name='final_'+'_'.join(args).replace('.py','');log=P/(name+'.log')
 with log.open('w') as f:r=subprocess.run([sys.executable,str(P/args[0]),*args[1:]],stdout=f,stderr=subprocess.STDOUT)
 row=dict(stage=args,exit_code=r.returncode,seconds=time.perf_counter()-start);(P/'results'/f'stage_{name}.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
