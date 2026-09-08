"""Actual teacher-QKV diagnostics; no fitting or model selection."""
from common_eval import *
from safetensors import safe_open

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    manifest=read(ROOT/'causal_direction/data/manifest.json')
    snapshot=Path('/root/autodl-tmp/hf-cache/models--Qwen--Qwen2.5-1.5B/snapshots')/manifest['revision']
    with safe_open(snapshot/'model.safetensors',framework='pt',device='cpu') as f:
        wo=torch.stack([f.get_tensor(f'model.layers.{layer}.self_attn.o_proj.weight').double() for layer in [14,27]]).cuda()
    nets={model_name(g,s):load_model(model_name(g,s))[0] for g in GROUPS for s in SEEDS}
    bounds=[0,1024,2048,4096,8192]
    for split in ['confirm_wiki','confirm_long']:
        path=P/'results'/f'diagnostics_{split}.json'
        if path.exists():continue
        ds=source.data(split)
        results={name:[] for name in nets}
        max_prior_error=0.
        for i in range(len(ds['q'])):
            q=ds['q'][i].cuda().double();k=ds['k'][i].cuda().double()[KI.cuda()]
            v=ds['v'][i].cuda().double()[KI.cuda()];pos=ds['query_positions'][i].cuda()
            mask=(torch.arange(k.shape[1],device='cuda')[None]<=pos[:,None])[None]
            logt=(q@k.transpose(-1,-2)/math.sqrt(D)).masked_fill(~mask,-torch.inf)
            lt=logt.log_softmax(-1);a=lt.exp();y=a@v
            yl=y.reshape(2,12,-1,D).permute(0,2,1,3).flatten(-2)@wo.transpose(-1,-2)
            target_local=(a*(((pos[:,None]-torch.arange(k.shape[1],device='cuda')[None])<64)[None]&mask)).sum(-1)
            for name,net in nets.items():
                lp=net.log_matrix(q,k).masked_fill(~mask,-torch.inf).log_softmax(-1)
                b=lp.exp();delta=b-a;e=delta@v
                kl=(a*(lt-lp).masked_fill(~mask,0)).sum(-1)
                tv=delta.abs().sum(-1)/2
                l2=delta.square().sum(-1)
                sse=e.square().sum(-1);energy=y.square().sum(-1)
                el=e.reshape(2,12,-1,D).permute(0,2,1,3).flatten(-2)@wo.transpose(-1,-2)
                projected_sse=el.square().sum(-1);projected_energy=yl.square().sum(-1)
                row=dict(ordinal=i,head_kl=kl.mean(-1).tolist(),head_tv=tv.mean(-1).tolist(),
                    head_coeff_l2=l2.mean(-1).tolist(),head_output_sse=sse.sum(-1).tolist(),
                    head_output_energy=energy.sum(-1).tolist(),
                    head_output_nmse=(sse.sum(-1)/energy.sum(-1).clamp_min(1e-30)).tolist(),
                    layer_projected_sse=projected_sse.sum(-1).tolist(),
                    layer_projected_energy=projected_energy.sum(-1).tolist(),
                    layer_projected_nmse=(projected_sse.sum(-1)/projected_energy.sum(-1).clamp_min(1e-30)).tolist(),
                    target_local64_mass=target_local.mean(-1).tolist(),positions=[])
                for start,end in zip(bounds[:-1],bounds[1:]):
                    selected=(pos>=start)&(pos<end)
                    if not selected.any():continue
                    row['positions'].append(dict(start=start,end=end,queries=int(selected.sum()),
                        kl=kl[:,selected].sum(-1).tolist(),tv=tv[:,selected].sum(-1).tolist(),
                        coeff_l2=l2[:,selected].sum(-1).tolist(),
                        output_sse=sse[:,selected].sum(-1).tolist(),
                        output_energy=energy[:,selected].sum(-1).tolist()))
                results[name].append(row)
            print(split,i+1,len(ds['q']),flush=True)
        for g in GROUPS:
            for seed in SEEDS:
                name=model_name(g,seed)
                old=read(parent_file(g,seed,split,'kernel'))
                for row,ref in zip(results[name],old['documents']):
                    assert row['ordinal']==ref['ordinal']
                    for key,refkey in [('head_kl','kl'),('head_output_nmse','output_nmse')]:
                        err=max(abs(x-y) for x,y in zip(row[key],ref[refkey]))
                        max_prior_error=max(max_prior_error,err)
                        assert err<1e-9,(name,split,key,err)
        save(path,dict(split=split,models=results,head_order=HEADS,
            max_discrepancy_with_parent_metrics=max_prior_error,
            scope='FP64 frozen teacher QKV,64 selected queries/document; W_O includes cross-head output interactions. Not on-policy downstream activations.'))

if __name__=='__main__':main()
