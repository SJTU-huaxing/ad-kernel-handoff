import hashlib
import importlib.util
import sys
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parent
OLD=ROOT/'query_amplitude_ablation'
sys.path.insert(0,str(OLD))
import ablation as ab
from ablation import torch,nn,F,json,math,source,parent,H,D,HEADS,KI,save

SEEDS=[11,29,47]
CONFIGS=[('standard','exp'),('matched_zero','ad'),('matched_zero','exp')]

def read(p):return json.loads(p.read_text())

def fingerprint(tensors):
    h=hashlib.sha256()
    for key,t in sorted(tensors.items()):
        h.update(key.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class Features(ab.Reduced):
    def __init__(self,norm,kind,init,scale):
        super().__init__(norm,'reduced_plain','causal_kl',scale)
        self.kind,self.initialization=kind,init
        if init=='matched_zero':
            with torch.no_grad():self.knet.w2.zero_()
        assert sum(p.numel() for p in self.parameters())//self.heads==73536

    def raw_log_feature(self,x,side):
        if side=='k' and self.kind=='exp':
            x=(x-self.k_mean[:,None])/self.k_std[:,None]
            return self.knet(x)
        return super().raw_log_feature(x,side)

def new_name(init,kind,seed,lr):return f'{init}_{kind}_s{seed}_lr{lr:g}'

def old_name(seed):return f'causal_kl_reduced_plain_s{seed}_lr0.002'

def plan():return read(P/'results/plan.json')

def load_fit(name,dtype=torch.float64,device='cuda'):
    if name.startswith('causal_kl_reduced_plain'):
        return ab.load_fit(name,dtype,device)
    box=torch.load(P/'fits'/f'{name}.pt',weights_only=True)
    sd,meta=box['state_dict'],box['metadata']
    norm={key:sd[key] for key in ['q_mean','q_std','k_mean','k_std']}
    net=Features(norm,meta['map_kind'],meta['initialization'],sd['numerical_scale']).to(dtype=dtype,device=device)
    net.load_state_dict(sd)
    return net.eval(),meta

def file_for(name,prefix,extension='json'):
    root=OLD if name.startswith('causal_kl_reduced_plain') else P
    folder='fits' if prefix=='fit' else 'results'
    stem=name if prefix=='fit' else f'{prefix}_{name}'
    return root/folder/f'{stem}.{extension}'

def all_groups():
    out={'standard_ad':[old_name(s) for s in SEEDS]}
    for init,kind in CONFIGS:
        key=f'{init}_{kind}';lr=plan()['learning_rates'][key]
        out[key]=[new_name(init,kind,s,lr) for s in SEEDS]
    return out

def base_assess():
    spec=importlib.util.spec_from_file_location('matched_assess_attribution',ROOT/'hedgehog_matched/assess.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.Replacement
    class Replacement(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            for net in self.maps.values():net.runtime_raw=True
    module.P=P;module.load_fit=load_fit;module.plan=plan;module.Replacement=Replacement
    return module
