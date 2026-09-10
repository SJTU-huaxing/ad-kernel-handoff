"""Bounded, fixed-weight AD/Hedgehog microbenchmarks. No optimizer or training queue.

Historical decode uses retained original seed11 weights and new public Qwen QKV.
Current cases use step1600 seed11 feature weights and common first-layer QKV.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch
import torch_npu
import fla
from ad_kernel.features import ADFeatures
from ad_kernel.baselines import HedgehogFeatures
from ad_kernel.attention import LinearState, chunk_attention, dense_attention, recurrent_attention
from ad_kernel.model import rotate
from fla.modules.feature_map import HedgehogFeatureMap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/reproduction/feature_speed_npu_v1"


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location='cpu', weights_only=True)


def errors(actual, expected):
    a, e = actual.detach().cpu().double(), expected.detach().cpu().double()
    return {'max_abs': (a-e).abs().max().item(), 'relative_rms': ((a-e).square().mean() / e.square().mean().clamp_min(1e-60)).sqrt().item()}


def measure(fn, *, samples, inner=1, graph=False):
    for _ in range(8):
        result = fn()
    torch.npu.synchronize()
    captured = None
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            for _ in range(inner):
                result = fn()
        def execute():
            captured.replay()
    else:
        def execute():
            for _ in range(inner):
                fn()
    for _ in range(3):
        execute()
    torch.npu.synchronize()
    device_us, wall_us = [], []
    for _ in range(samples):
        a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        start = time.perf_counter()
        a.record()
        execute()
        b.record()
        b.synchronize()
        wall_us.append((time.perf_counter()-start)*1e6/inner)
        device_us.append(a.elapsed_time(b)*1000/inner)
    # Keep captured outputs and graph alive until timing and synchronization end.
    del result, captured
    return {'median_us': statistics.median(device_us), 'p25_us': float(np.quantile(device_us,.25)),
            'p75_us': float(np.quantile(device_us,.75)), 'samples_us': device_us,
            'synchronized_wall_samples_us': wall_us, 'inner_repeats': inner,
            'execution': 'NPU graph replay' if graph else 'eager; event interval can include host launch gaps'}


def historical_step(lq, lk, vg, state):
    """Pure-PyTorch branch of original causal_direction/operators.py:step.

    Same GQA expansion, in-place S/z updates and normalized readout. No CUDA code.
    """
    v = vg.repeat_interleave(6, 0)
    g = torch.maximum(state['g'], lk[:, 0])
    fac = (state['g']-g).exp()
    q, k = (lq[:, 0]+g).softmax(-1), (lk[:, 0]-g).exp()
    state['s'].mul_(fac[:, :, None]).add_(k[:, :, None]*v[:, 0, None, :])
    state['z'].mul_(fac).add_(k)
    state['g'] = g
    den = (q*state['z']).sum(-1)
    return ((q[:, :, None]*state['s']).sum(1)/den[:, None])[:, None], den[:, None]


def verify(net, q, k, v):
    # Small CPU FP64 forward/input/parameter-gradient reference, from measured tensors.
    q = q[:1, :, :17].detach().clone().requires_grad_()
    k = k[:1, :, :17].detach().clone().requires_grad_()
    v = v[:1, :, :17].detach().clone().requires_grad_()
    ref_net = copy.deepcopy(net).cpu().double()
    qr, kr, vr = [x.detach().cpu().double().requires_grad_() for x in (q,k,v)]
    a = (net.log_feature(q,'q'), net.log_feature(k,'k'))
    r = (ref_net.log_feature(qr,'q'), ref_net.log_feature(kr,'k'))
    map_errors = [errors(x,y) for x,y in zip(a,r)]
    out, state = chunk_attention(*a,v,backend='torch')
    truth = dense_attention(*r,vr)
    torch.manual_seed(931)
    dy = torch.randn_like(truth).float()
    actual_grad = torch.autograd.grad(out,(q,k,v,*net.parameters()),dy.to(q.device))
    ref_grad = torch.autograd.grad(truth,(qr,kr,vr,*ref_net.parameters()),dy.double())
    checks = {'features': map_errors, 'attention': errors(out,truth),
              'gradients': [errors(x,y) for x,y in zip(actual_grad,ref_grad)]}
    assert max(x['relative_rms'] for x in [*map_errors,checks['attention'],*checks['gradients']]) < 5e-4, checks
    return checks


def historical_models():
    specs = [
        ('ad64','query_amplitude_ablation/fits/causal_kl_reduced_plain_s11_lr0.002.pt'),
        ('hh_exp576','hedgehog_matched/fits/hh_kl_s11_lr0.002.pt'),
        ('hh_softmax576','hedgehog_matched/fits/hh_softmax_kl_s11_lr0.002.pt')]
    models, sources = {}, {}
    for name, relative in specs:
        path = ROOT/'checkpoints/kan_attention_theory'/relative
        state = {k:v[:12].clone() for k,v in load(path)['state_dict'].items()}
        net = ADFeatures(heads=12,dim=128,hidden=192,m=64) if name=='ad64' else HedgehogFeatures(heads=12,dim=128,m=576,softmax='softmax' in name)
        net.load_state_dict(state)
        models[name] = net
        sources[name] = {'path':str(path.relative_to(ROOT)),'sha256':digest(path)}
    return models,sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device',default='npu:0')
    ap.add_argument('--samples',type=int,default=15)
    ap.add_argument('--profile',choices=['historical','current','all'],default='all')
    args = ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.matmul.allow_hf32=False
    before = subprocess.check_output(['npu-smi','info'],text=True)
    (OUT/'npu_before.txt').write_text(before)
    rows,checks,sources = [],[],{}
    report = {'device':args.device,'dtype':'float32','matmul_allow_hf32':False,'samples_per_round':args.samples,
              'rounds':2,'torch':str(torch.__version__),'torch_npu':str(torch_npu.__version__),
              'fla':importlib.metadata.version('flash-linear-attention'),
              'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/ad_kernel/features.py',ROOT/'src/ad_kernel/baselines.py',ROOT/'src/ad_kernel/attention.py']},
              'scope':'Fixed-weight one-layer feature/kernel microbenchmarks; no optimizer, training, quality selection or full-model timing. Both rounds use the same inputs and reverse method order.',
              'rows':rows,'checks':checks,'inputs_and_weights':sources}
    destination=OUT/f'{args.profile}.json'
    def save():
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def record(profile,method,b,t,scope,round_index,fn,graph=False,inner=1,net=None):
        measurement=measure(fn,samples=args.samples,inner=inner,graph=graph)
        row={'profile':profile,'method':method,'batch':b,'length':t,'heads':12,'scope':scope,'round':round_index,
             'graph':graph,'parameters_per_layer':sum(p.numel() for p in net.parameters()) if net else None,**measurement}
        rows.append(row);save()
        print(json.dumps({k:v for k,v in row.items() if not k.endswith('samples_us')}),flush=True)

    if args.profile in ['historical','all']:
        models,weights=historical_models()
        sources['historical_weights']=weights
        path=ROOT/'work/reproduction/public_wikitext_v1/data/confirm_wiki_000.pt'
        data=load(path)
        sources['historical_input']={'path':str(path.relative_to(ROOT)),'sha256':digest(path),
            'selection':'document0, recorded query0, key/value0, Qwen layer14 heads0..11; newly extracted public Wiki QKV, original GPU QKV unavailable'}
        q=data['q'][0,:12,:1].float().to(args.device)
        k=data['k'][0,:2,:1].float().to(args.device)
        v=data['v'][0,:2,:1].float().to(args.device)
        for name,net in models.items():
            net.to(args.device)
            cq=data['q'][0:1,:12,:17].float().to(args.device)
            ck=data['k'][0:1,:2,:17].repeat_interleave(6,1).float().to(args.device)
            cv=data['v'][0:1,:2,:17].repeat_interleave(6,1).float().to(args.device)
            checks.append({'profile':'historical','method':name,**verify(net,cq,ck,cv)})
        for ri in range(2):
            for name in (list(models) if ri==0 else list(reversed(models))):
                net=models[name]
                with torch.no_grad():
                    def feature():return net.log_feature(q,'q'),net.log_feature(k.repeat_interleave(6,0),'k')
                    lq,lk=feature()
                    state={'s':torch.zeros(12,net.m,128,device=args.device),'z':torch.zeros(12,net.m,device=args.device),'g':lk[:,0].clone()}
                    # Check original in-place decode against the independent current recurrence.
                    expected,_=recurrent_attention(lq[None],lk[None],v.repeat_interleave(6,0)[None])
                    actual,_=historical_step(lq,lk,v,state)
                    err=errors(actual,expected[0]);assert err['relative_rms']<1e-5,err
                    checks.append({'profile':'historical_decode','method':name,'round':ri,**err})
                    for graph in [False,True]:
                        record('historical',name,1,1,'features_forward',ri,feature,graph,32,net)
                        record('historical',name,1,1,'state_only',ri,lambda:historical_step(lq,lk,v,state),graph,32,net)
                        record('historical',name,1,1,'features_and_state',ri,lambda:historical_step(*feature(),v,state),graph,32,net)
        del models,data,q,k,v,net,state
        gc.collect();torch.npu.empty_cache()

    if args.profile in ['current','all']:
        models={}
        for name,m in [('ad64',64),('hh_softmax128',128),('hh_exp128',128),('hh_exp384',384)]:
            path=ROOT/f'work/pretraining/runs/{name}_s11/model_step001600.pt'
            box=load(path)
            state=box['model']
            prefix='blocks.0.attention.features.'
            feature_state={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            net=ADFeatures(heads=12,dim=64,hidden=96,m=64) if name=='ad64' else HedgehogFeatures(heads=12,dim=64,m=m,bias=True,softmax='softmax' in name)
            net.load_state_dict(feature_state);models[name]=net.to(args.device)
            sources[name]={'path':str(path.relative_to(ROOT)),'sha256':digest(path),'layer':0}
            if name=='ad64':
                backbone={k:state[k].to(args.device) for k in ['embedding.weight','blocks.0.attention_norm.weight','blocks.0.attention.qkv.weight']}
            del box,state,feature_state
        stream=np.memmap(ROOT/'work/pretraining/data/fineweb_edu_v1/test.bin',mode='r',dtype='<u2')
        sources['current_input']={'path':'work/pretraining/data/fineweb_edu_v1/test.bin','sha256':digest(ROOT/'work/pretraining/data/fineweb_edu_v1/test.bin'),
            'selection':'First B*T test tokens; common QKV from trained AD seed11 layer0; BF16 QKV projection, FP32 RoPE and d^-1/4 scale; preprocessing outside timing'}
        for b,t in [(1,1),(1,1024),(16,1024),(1,8192)]:
            ids=torch.from_numpy(np.array(stream[:b*t],dtype=np.int64).reshape(b,t)).to(args.device)
            with torch.no_grad(),torch.autocast('npu',dtype=torch.bfloat16):
                hidden=torch.nn.functional.embedding(ids,backbone['embedding.weight'])
                x=torch.nn.functional.rms_norm(hidden,(768,),backbone['blocks.0.attention_norm.weight'],1e-6)
                q,k,v=torch.nn.functional.linear(x,backbone['blocks.0.attention.qkv.weight']).reshape(b,t,3,12,64).permute(2,0,3,1,4).unbind(0)
            with torch.no_grad():
                angles=torch.arange(t,device=args.device).float()[:,None]*(10000.**(-torch.arange(0,64,2,device=args.device).float()/64))
                phase=torch.cat([angles,angles],-1)[None,None]
                q=(rotate(q,phase.cos(),phase.sin())*64**(-.25)).contiguous()
                k=(rotate(k,phase.cos(),phase.sin())*64**(-.25)).contiguous()
                v=v.float().contiguous()
            if t==1024 and b==1:
                for name,net in models.items():checks.append({'profile':'current','method':name,**verify(net,q,k,v)})
            for ri in range(2):
                for name in (list(models) if ri==0 else list(reversed(models))):
                    net=models[name]
                    with torch.no_grad():
                        def feature():return net.log_feature(q,'q'),net.log_feature(k,'k')
                        lq,lk=feature()
                        if t==1:
                            state=LinearState(torch.zeros(b,12,net.m,64,device=args.device),torch.zeros(b,12,net.m,device=args.device),lk[:,:,0].clone())
                            for graph in [False,True]:
                                record('current',name,b,t,'features_forward',ri,feature,graph,32,net)
                                record('current',name,b,t,'features_and_state',ri,lambda:recurrent_attention(*feature(),v,initial_state=state),graph,32,net)
                        else:
                            record('current',name,b,t,'features_forward',ri,feature,False,1,net)
                            record('current',name,b,t,'features_and_attention_forward',ri,lambda:chunk_attention(*feature(),v,backend='torch'),False,1,net)
                    if t>1:
                        qi=q.detach().requires_grad_();ki=k.detach().requires_grad_();vi=v.detach().requires_grad_()
                        gq=torch.ones_like(lq);gk=torch.ones_like(lk);go=torch.ones_like(v)
                        def feature_backward():
                            logs=(net.log_feature(qi,'q'),net.log_feature(ki,'k'))
                            return torch.autograd.grad(logs,(qi,ki,*net.parameters()),(gq,gk))
                        def attention_backward():
                            out,_=chunk_attention(net.log_feature(qi,'q'),net.log_feature(ki,'k'),vi,backend='torch')
                            return torch.autograd.grad(out,(qi,ki,vi,*net.parameters()),go)
                        record('current',name,b,t,'features_forward_backward',ri,feature_backward,False,1,net)
                        record('current',name,b,t,'features_and_attention_forward_backward',ri,attention_backward,False,1,net)
        # Verify the unmodified installed FLA feature module separately: different map and sharing.
        official=HedgehogFeatureMap(64).to(args.device)
        z=torch.randn(1,12,17,64,device=args.device,requires_grad=True)
        truth=copy.deepcopy(official).cpu().double()
        zr=z.detach().cpu().double().requires_grad_()
        actual,expected=official(z),truth(zr)
        dy=torch.randn_like(actual)
        grads=torch.autograd.grad(actual,(z,*official.parameters()),dy)
        refgrads=torch.autograd.grad(expected,(zr,*truth.parameters()),dy.cpu().double())
        gradient_checks=[errors(a,b) for a,b in zip(grads,refgrads)]
        assert max(e['relative_rms'] for e in gradient_checks)<5e-4,gradient_checks
        checks.append({'profile':'unmodified_fla_HedgehogFeatureMap','forward':errors(actual,expected),'gradients':gradient_checks,'finite_gradients':all(torch.isfinite(g).all().item() for g in grads),
            'definition':'softmax(concat(2*Wx+2*b,-2*Wx-2*b)); same module weights shared over heads in FLA LinearAttention; different from local per-head HH variants'})
    report['complete']=True
    save()
    print(json.dumps({'event':'complete','rows':len(rows),'checks':len(checks),'path':str(destination)}),flush=True)


if __name__=='__main__':main()
