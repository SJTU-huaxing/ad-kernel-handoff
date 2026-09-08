import subprocess
import sys
import time
from pathlib import Path
P=Path(__file__).resolve().parent
for action in ['verify','precision','kernels','ppl']:
    start=time.perf_counter()
    with (P/'logs'/f'favor_{action}.log').open('w') as f:
        subprocess.run([sys.executable,str(P/'favor_recheck.py'),action],stdout=f,stderr=subprocess.STDOUT,check=True)
    print(action,time.perf_counter()-start,flush=True)
