"""Frozen teacher activations; diagnostics, never used for fitting or selection."""
from pathlib import Path
from core_attribution import *
from safetensors import safe_open

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest=read(ROOT/'causal_direction/data/manifest.json')
    snapshot=Path('/root/autodl-tmp/hf-cache/models--Qwen--Qwen2.5-1.5B/snapshots')/manifest['revision']
    with safe_open(snapshot/'model.safetensors',framework='pt',device='cpu') as f:
        wo=torch.stack([f.get_tensor(f'model.layers.{layer}.self_attn.o_proj.weight').double() for layer in [14,27]]).cuda()
    nets={name:load_fit(name)[0] for name in plan()['models']}
    bounds=[0,1024,2048,4096,8192]
    for split in ['confirm_wiki','confirm_long']:
        path=P/'results'/f'diagnostics_{split}.json'
        if path.exists():continue
        ds=source.data(split);results={name:[] for name in nets};max_prior_error=0.
        for i in range(len(ds['q'])):
            q=ds['q'][i].cuda().double();k=ds['k'][i].cuda().double()[KI.cuda()]
            v=ds['v'][i].cuda().double()[KI.cuda()];pos=ds['query_positions'][i].cuda()
            mask=(torch.arange(k.shape[1],device='cuda')[None]<=pos[:,None])[None]
            lt=(q@k.transpose(-1,-2)/math.sqrt(D)).masked_fill(~mask,-torch.inf).log_softmax(-1)
            a=lt.exp();y=a@v
            yl=y.reshape(2,12,-1,D).permute(0,2,1,3).flatten(-2)@wo.transpose(-1,-2)
            for name,net in nets.items():
                lp=net.log_matrix(q,k).masked_fill(~mask,-torch.inf).log_softmax(-1)
                b=lp.exp();delta=b-a;e=delta@v;ratio=(lt-lp).masked_fill(~mask,0)
                kl=(a*ratio).sum(-1);tv=delta.abs().sum(-1)/2;l2=delta.square().sum(-1)
                sse=e.square().sum(-1);energy=y.square().sum(-1)
                el=e.reshape(2,12,-1,D).permute(0,2,1,3).flatten(-2)@wo.transpose(-1,-2)
                pe=el.square().sum(-1);py=yl.square().sum(-1)
                rare=(ratio>math.log(10000))&mask
                row=dict(ordinal=i,head_kl=kl.mean(-1).tolist(),head_tv=tv.mean(-1).tolist(),
                    head_coeff_l2=l2.mean(-1).tolist(),head_output_sse=sse.sum(-1).tolist(),
                    head_output_energy=energy.sum(-1).tolist(),
                    head_output_nmse=(sse.sum(-1)/energy.sum(-1).clamp_min(1e-30)).tolist(),
                    layer_projected_sse=pe.sum(-1).tolist(),layer_projected_energy=py.sum(-1).tolist(),
                    layer_projected_nmse=(pe.sum(-1)/py.sum(-1).clamp_min(1e-30)).tolist(),
                    severe_underestimate_mass=(a*rare).sum(-1).mean(-1).tolist(),
                    severe_positive_kl=(a*ratio*rare).sum(-1).mean(-1).tolist(),positions=[])
                for start,end in zip(bounds[:-1],bounds[1:]):
                    selected=(pos>=start)&(pos<end)
                    if not selected.any():continue
                    row['positions'].append(dict(start=start,end=end,queries=int(selected.sum()),
                        kl=kl[:,selected].sum(-1).tolist(),tv=tv[:,selected].sum(-1).tolist(),
                        coeff_l2=l2[:,selected].sum(-1).tolist(),
                        output_sse=sse[:,selected].sum(-1).tolist(),output_energy=energy[:,selected].sum(-1).tolist()))
                results[name].append(row)
            print(split,i+1,len(ds['q']),flush=True)
        for name in nets:
            ref=read(file_for(name,f'kernel_{split}'))
            for row,old in zip(results[name],ref['documents']):
                assert row['ordinal']==old['ordinal']
                for a,b in [('head_kl','kl'),('head_output_nmse','output_nmse')]:
                    err=max(abs(x-y) for x,y in zip(row[a],old[b]));max_prior_error=max(max_prior_error,err)
                    assert err<1e-9,(name,split,a,err)
        # Existing standard-AD diagnostics used exactly the same frozen activations.
        old=read(ROOT/'normalized_attention_assessment/results'/f'diagnostics_{split}.json')['models']
        tail=read(ROOT/'normalized_attention_assessment/results/tail_diagnostics.json')['models']
        for name in all_groups()['standard_ad']:
            rows=old[name]
            if split=='confirm_long':
                for row,t in zip(rows,tail[name]):
                    assert row['ordinal']==t['ordinal']
                    threshold=t['thresholds'][2]
                    assert abs(threshold['log_ratio_threshold']-math.log(10000))<1e-12
                    row['severe_underestimate_mass']=threshold['teacher_mass']
                    row['severe_positive_kl']=threshold['positive_kl']
            results[name]=rows
        save(path,dict(split=split,models=results,head_order=HEADS,
            max_discrepancy_with_parent_metrics=max_prior_error,
            scope='FP64 cached teacher QKV,64queries/document; standardAD diagnostic reused from exact same data; no training or selection. Severe underestimation means predicted probability <teacher probability/10000.'))

if __name__=='__main__':main()
