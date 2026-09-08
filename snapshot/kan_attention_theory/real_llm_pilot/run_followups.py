"""Sequential GPU follow-ups; stop on errors and preserve individual logs."""
from pathlib import Path
import subprocess
import sys

p=Path(__file__).resolve().parent
base=['--data',str(p/'data_qwen25_1p5b')]
stages=[
    ('extra_diagnostics.py',base+['--out',str(p/'extra')],'extra.log'),
    ('model_replacement.py',base+['--fits',str(p/'fits'),'--out',str(p/'replacement')],'replacement.log'),
    ('fit_features.py',base+['--analysis',str(p/'analysis_qwen25_1p5b'),'--out',str(p/'fits_dimensions'),
        '--steps','2000','--seeds','11','29','47','--m','32','128','--methods','mlp_positive','kan_positive'],'fit_dimensions.log'),
    ('fit_features.py',base+['--analysis',str(p/'analysis_qwen25_1p5b'),'--out',str(p/'fits_longer'),
        '--steps','10000','--seeds','11','29','47','--methods','mlp_positive','kan_positive'],'fit_longer.log'),
]
for script,args,log in stages:
    print('START',script,log,flush=True)
    with (p/log).open('w') as f:
        subprocess.run([sys.executable,str(p/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',log,flush=True)
