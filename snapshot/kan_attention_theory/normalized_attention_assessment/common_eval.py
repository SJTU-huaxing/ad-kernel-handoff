"""Read-only assessment of the selected pure-deletion KL models and baselines."""
import importlib.util
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parent
sys.path.insert(0, str(ROOT/'query_amplitude_ablation'))
import ablation as ab
from ablation import torch, nn, F, json, math, source, parent, H, D, HEADS, KI, save

SEEDS = [11,29,47]
GROUPS = ['ad_plain','hh_exp','hh_softmax','favor']

def model_name(group, seed):
    return {'ad_plain':f'causal_kl_reduced_plain_s{seed}_lr0.002',
            'hh_exp':f'hh_kl_s{seed}_lr0.002',
            'hh_softmax':f'hh_softmax_kl_s{seed}_lr0.002',
            'favor':f'favor_s{seed}'}[group]

def load_model(name, dtype=torch.float64, device='cuda'):
    if name.startswith('causal_kl_reduced_plain'):
        net, meta = ab.load_fit(name,dtype,device)
        net.runtime_raw = True
    elif name.startswith('favor'):
        net, meta = source.load(name,dtype)
        net = net.to(device)
        meta = dict(meta, parameters_per_head=0, stored_random_coefficients_per_head=8256,m=64)
    else:
        net, meta = parent.load_fit(name,dtype,device)
    return net,meta

def parent_file(group, seed, split, metric):
    name = model_name(group,seed)
    root = ROOT/('query_amplitude_ablation' if group=='ad_plain' else 'hedgehog_matched')
    if group=='favor': root = P
    return root/'results'/f'{metric}_{split}_{name}.json'

def read(path): return json.loads(path.read_text())

def base_assess():
    spec=importlib.util.spec_from_file_location('matched_assess_normalized',ROOT/'hedgehog_matched/assess.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.P=P
    module.load_fit=load_model
    module.plan=lambda:{'models':[model_name('favor',s) for s in SEEDS]}
    return module
