"""Four-head causal linear-attention replacement in the full frozen Qwen model."""
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2
from validate import load_model

P=Path(__file__).resolve().parent


def stable_features(module,q,k):
    lf=module.log_feature(q,'q');lg=module.log_feature(k,'k')
    # Global per-feature reweighting cancels exactly for every causal prefix.
    # No future key contribution enters the prefix state or the result.
    ks=lg.amax(1,keepdim=True);qc=lf+ks;qs=qc.amax(-1,keepdim=True)
    return (qc-qs).exp(),(lg-ks).exp()


def linear_scan(q,k,v):
    output=[];state=torch.zeros_like(k[:,0,:,None]*v[:,0,None,:]);key_sum=torch.zeros_like(k[:,0])
    for start in range(0,q.shape[1],64):
        end=start+64;kk=k[:,start:end];vv=v[:,start:end];qq=q[:,start:end]
        states=(kk[:,:,:,None]*vv[:,:,None,:]).cumsum(1)+state[:,None]
        keys=kk.cumsum(1)+key_sum[:,None]
        denominator=(qq*keys).sum(-1,keepdim=True)
        assert (denominator>0).all() and torch.isfinite(denominator).all()
        output.append((qq[:,:,:,None]*states).sum(2)/denominator)
        state=states[:,-1];key_sum=keys[:,-1]
    return torch.cat(output,1)


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((P/'data/manifest.json').read_text())
    batch=torch.load(P/'data/test_000.pt',weights_only=True,map_location='cpu')
    ids=batch['input_ids'][:32].cuda();del batch
    active={};original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];checks=[];checked=set();scenario='teacher'
    def replaced(module,query,key,value,attention_mask,**kwargs):
        out,weights=original(module,query,key,value,attention_mask,**kwargs)
        if module.layer_idx in active:
            fmap,heads=active[module.layer_idx];kv=[h//module.num_key_value_groups for h in heads]
            q,k=stable_features(fmap,query[0,heads].double(),key[0,kv].double())
            v=value[0,kv].double();y=linear_scan(q,k,v)
            check_key=(scenario,module.layer_idx)
            if check_key not in checked:
                kernel=q@k.transpose(-1,-2);kernel=kernel.tril()
                reference=(kernel/kernel.sum(-1,keepdim=True))@v
                error=float((y-reference).norm()/reference.norm())
                assert error<1e-10
                checks.append(dict(scenario=scenario,layer=module.layer_idx,relative_l2=error));checked.add(check_key)
            out=out.clone();out[0,:,heads]=y.transpose(0,1).to(out.dtype)
        return out,weights
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',replaced)
    model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['model_revision'],
        cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval()
    paths=[None]+[P/'fits'/f'{loss}_{method}_n4096_b1_s{seed}.pt'
                   for loss in ['logcosh','poisson'] for method in ['mlp','kan','mulkan'] for seed in [11,29,47]]
    results=[]
    for path in paths:
        scenario=path.stem if path else 'teacher';active={}
        if path:
            for layer in [14,27]:
                indices=[i for i,label in enumerate(manifest['head_labels']) if label[0]==layer]
                fmap,meta=load_model(path,device='cuda',dtype=torch.float64,indices=indices)
                active[layer]=(fmap,[manifest['head_labels'][i][1] for i in indices])
        rows=[]
        for ordinal,document in enumerate(ids):
            hidden=model.model(input_ids=document[None],use_cache=False).last_hidden_state[0,:-1]
            losses=[]
            for start in range(0,len(hidden),128):
                logits=model.lm_head(hidden[start:start+128]).float()
                losses.append(F.cross_entropy(logits,document[start+1:start+1+len(logits)],reduction='none'))
            loss=torch.cat(losses)
            assert torch.isfinite(loss).all()
            rows.append(dict(ordinal=ordinal,nll=float(loss.mean()),tokens=len(loss),last512_nll=float(loss[-512:].mean())))
        nll=sum(r['nll']*r['tokens'] for r in rows)/sum(r['tokens'] for r in rows)
        tail=sum(r['last512_nll'] for r in rows)/len(rows)
        result=dict(scenario=scenario,nll=nll,perplexity=math.exp(nll),last512_nll=tail,
                    last512_perplexity=math.exp(tail),documents=rows,status='complete')
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='documents'}),flush=True)
        (P/'replacement.json').write_text(json.dumps(dict(model=manifest['model'],test_documents=32,
            replaced_heads=[[14,0],[14,6],[27,0],[27,6]],results=results,scan_checks=checks,
            notes=['Internal held-out WikiText-103 TRAIN articles, not official benchmark perplexity.',
                   'All query positions and all causal keys; training only sampled last-half Q and first-half K.',
                   'Frozen full model, no finetuning; downstream hidden states and Q/K recomputed.',
                   'Float64 feature network and prefix accumulation; raw kernel scale cancels in denominator.',
                   'Quality ablation only: original SDPA still computed for unselected heads.',
                   'Per-feature reweighting uses whole-sequence maxima for arithmetic stability; exact result is causal.']),indent=2))
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)


if __name__=='__main__':main()
