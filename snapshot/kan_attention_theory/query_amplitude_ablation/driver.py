import subprocess
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parent
for action in ['verify','precision','kernels','products','ppl']:
    started = time.perf_counter()
    with (P/'logs'/f'{action}.log').open('w') as log:
        subprocess.run([sys.executable,str(P/'eval_run.py'),action],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(action,time.perf_counter()-started,flush=True)
with (P/'logs'/'benchmark.log').open('w') as log:
    subprocess.run([sys.executable,str(P/'benchmark_run.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
print('evaluation complete',flush=True)
