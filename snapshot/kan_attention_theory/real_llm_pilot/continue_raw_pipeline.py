"""Run the risk-preserving sampling control after the uniform GPU jobs finish."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int);args=parser.parse_args()
if args.wait_pid:
    while True:
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        time.sleep(2)
    if len(list((p/'raw_longer').glob('*.json')))!=6:
        raise RuntimeError('Uniform raw runs did not complete')
base=['--data',str(p/'data_qwen25_1p5b'),'--analysis',str(p/'analysis_qwen25_1p5b'),
      '--objective','raw','--seeds','11','29','47','--out',str(p/'raw_importance'),
      '--steps','10000','--m','64','--sampling','energy','--methods','mlp_positive','kan_positive']
stages=[('fit_features.py',base,'raw_importance.log'),
        ('model_replacement.py',['--data',str(p/'data_qwen25_1p5b'),'--fits',str(p/'raw_importance'),
            '--out',str(p/'replacement_raw_importance'),'--objectives','raw'],'replacement_raw_importance.log'),
        ('audit_raw.py',[],'raw_audit.log')]
for script,arguments,log in stages:
    print('START',log,flush=True)
    with (p/log).open('w') as f:
        subprocess.run([sys.executable,str(p/script),*arguments],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',log,flush=True)
