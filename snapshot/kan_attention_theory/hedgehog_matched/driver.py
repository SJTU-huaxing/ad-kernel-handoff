"""Sequential evaluation driver, to be started after train.py finishes."""
import subprocess
import sys
import time
import os
from pathlib import Path

P = Path(__file__).resolve().parent
product = os.environ.get('MATCHED_PLAN') == 'product_plan.json'
prefix = 'product_' if product else ''
for action in ['verify', 'precision', 'kernels', 'products', 'ppl']:
    start = time.perf_counter()
    with (P/f'{prefix}{action}.log').open('w') as out:
        subprocess.run([sys.executable, str(P/'assess.py'), action],
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    print(action, time.perf_counter()-start, flush=True)
if not product:
    with (P/'benchmark.log').open('w') as out:
        subprocess.run([sys.executable, str(P/'benchmark.py')], stdout=out,
                       stderr=subprocess.STDOUT, check=True)
print('complete', flush=True)
