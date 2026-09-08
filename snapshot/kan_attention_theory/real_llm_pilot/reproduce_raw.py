"""Reproduce the primary experiment with the existing local model/dataset caches."""
from pathlib import Path
import subprocess

p=Path(__file__).resolve().parent
experiment_python='/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python'
plot_python='/root/miniconda3/bin/python'
def run(script,*args,python=experiment_python):
    subprocess.run([python,str(p/script),*map(str,args)],check=True)

if not (p/'data_qwen25_1p5b/manifest.json').exists():
    run('extract_qkv.py','--out',p/'data_qwen25_1p5b')
if not (p/'analysis_qwen25_1p5b/spectra.json').exists():
    run('analyze_distribution.py','--data',p/'data_qwen25_1p5b','--out',p/'analysis_qwen25_1p5b')
run('run_raw_followups.py')
run('continue_raw_pipeline.py')
run('raw_random_baseline.py')
run('exact_raw_evaluation.py')
run('model_replacement.py','--data',p/'data_qwen25_1p5b','--fits',p/'raw_importance',
    '--out',p/'replacement_raw_importance_fp64','--objectives','raw','--feature-dtype','float64')
run('model_replacement.py','--data',p/'data_qwen25_1p5b','--fits',p/'raw_longer',
    '--out',p/'replacement_raw_longer_fp64','--objectives','raw','--feature-dtype','float64')
run('summarize_raw.py',python=plot_python)
