"""Primary experiment: raw exponential target only, with pooled-L2 validation."""
from pathlib import Path
import subprocess
import sys

p=Path(__file__).resolve().parent
base=['--data',str(p/'data_qwen25_1p5b'),'--analysis',str(p/'analysis_qwen25_1p5b'),
      '--objective','raw','--seeds','11','29','47']
stages=[
    ('fit_features.py',base+['--out',str(p/'raw_main'),'--steps','2000','--m','64',
        '--methods','mlp_positive','kan_positive','mlp_signed','kan_signed'],'raw_main.log'),
    ('fit_features.py',base+['--out',str(p/'raw_dimensions'),'--steps','2000','--m','32','128',
        '--methods','mlp_positive','kan_positive'],'raw_dimensions.log'),
    ('fit_features.py',base+['--out',str(p/'raw_longer'),'--steps','10000','--m','64',
        '--methods','mlp_positive','kan_positive'],'raw_longer.log'),
    ('model_replacement.py',['--data',str(p/'data_qwen25_1p5b'),'--fits',str(p/'raw_longer'),
        '--out',str(p/'replacement_raw_longer'),'--objectives','raw'],'replacement_raw_longer.log'),
]
for script,args,log in stages:
    print('START',script,log,flush=True)
    with (p/log).open('w') as f:
        subprocess.run([sys.executable,str(p/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',log,flush=True)
