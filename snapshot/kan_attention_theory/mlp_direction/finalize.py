import subprocess
from core import *

# Preserve the first, algebraically equivalent but separately folded FP32
# deployment measurements. Re-evaluate only the calibrated cases using the
# original parent factors, which avoid this gratuitous floating-point change.
pp=P/'results/ppl_gauge.json';archive=P/'results/ppl_gauge_folded_fp32.json'
if not archive.exists():
    box=json.loads(pp.read_text());save(archive,box)
    box['results']=[r for r in box['results'] if not r['case'].startswith('gauge_calibrated')]
    box['deployment_note']='Gauge calibration executes the exactly equivalent parent feature factors; folded-FP32 results preserved separately.'
    save(pp,box)

# Preserve the completed six-method computation, then add the fixed FAVOR+
# control. It was appended while the initial process had already loaded code.
for ds in ['fresh_wiki3','swde3']:
    path=P/'results'/f'product_{ds}.json'
    if path.exists() and len(json.loads(path.read_text())['results'])==6:
        path.rename(path.with_name(path.stem+'_before_rf.json'))
stages=[]
for script,args in [('gauge_assess.py',['ppl'])]+[(s,[]) for s in ['baselines.py','product.py','benchmark_gauge.py','verify.py','finalaudit.py']]:
    start=time.perf_counter();log=P/('final_'+script.removesuffix('.py')+'.log')
    with log.open('w') as f:r=subprocess.run([sys.executable,str(P/script),*args],stdout=f,stderr=subprocess.STDOUT)
    row=dict(script=script,args=args,exit_code=r.returncode,seconds=time.perf_counter()-start,log=str(log));stages.append(row)
    save(P/'results/stages_final.json',stages);print(json.dumps(row),flush=True)
    if r.returncode:raise RuntimeError(row)
