import subprocess,sys,json,time
from pathlib import Path
p=Path(__file__).resolve().parent
assert (p/'data/manifest.json').exists()
start=time.perf_counter()
with (p/'extract.log').open('w') as f:r=subprocess.run([sys.executable,str(p/'data.py'),'extract'],stdout=f,stderr=subprocess.STDOUT)
(p/'results/extraction_stage.json').write_text(json.dumps(dict(exit_code=r.returncode,seconds=time.perf_counter()-start)))
if r.returncode:raise RuntimeError('Extraction failed')
