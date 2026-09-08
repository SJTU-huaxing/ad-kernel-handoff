"""Surgical head replacement in the complete frozen LLM; actual causal linear scan."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2
from fit_features import FeaturePair


def load_maps(path, layers, dtype=torch.float32):
    saved=torch.load(path,weights_only=True,map_location='cpu');meta=saved['metadata'];state=saved['state_dict']
    labels=meta['head_labels'];maps={}
    for layer in layers:
        idx=[i for i,label in enumerate(labels) if label[0]==layer]
        if not idx:continue
        norm={key:state[key][idx].cuda() for key in ['q_mean','q_std','k_mean','k_std']}
        module=FeaturePair(len(idx),128,meta['m'],meta['method'],norm,
                           state['amplitude'][idx,0,0].cuda(),grid=meta['grid'],width=16).cuda().to(dtype)
        substate={key:(value if 'knots' in key else value[idx]).cuda() for key,value in state.items()}
        module.load_state_dict(substate);module.eval()
        maps[layer]=(module,[labels[i][1] for i in idx])
    return maps


@torch.inference_mode()
def linear_scan(q,k,v):
    state=(k[:,:,:,None]*v[:,:,None,:]).cumsum(1)
    numerator=(q[:,:,:,None]*state).sum(2)
    denominator=(q*k.cumsum(1)).sum(-1,keepdim=True)
    if (denominator<=0).any():
        raise RuntimeError('Nonpositive denominator in positive feature scan; '
            f'zero_query_feature_rows={int((q.abs().sum(-1)==0).sum())}; '
            f'zero_key_feature_rows={int((k.abs().sum(-1)==0).sum())}')
    return numerator/denominator


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--fits',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--documents',type=int,default=12)
    p.add_argument('--objectives',nargs='+',choices=['row','raw'],default=['row','raw'])
    p.add_argument('--feature-dtype',choices=['float32','float64'],default='float32')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);manifest=json.loads((args.data/'manifest.json').read_text())
    docs=[x for x in manifest['documents'] if x['split']=='test'][:args.documents]
    active={};original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];check=[]
    def replaced(module,query,key,value,attention_mask,**kwargs):
        out,weights=original(module,query,key,value,attention_mask,**kwargs)
        if module.layer_idx in active:
            fmap,heads=active[module.layer_idx];kv=[h//module.num_key_value_groups for h in heads]
            dtype=next(fmap.parameters()).dtype
            q=fmap.feature(query[0,heads].to(dtype),'q');k=fmap.feature(key[0,kv].to(dtype),'k')
            v=value[0,kv].to(dtype);y=linear_scan(q,k,v)
            if not check:
                dense=q@k.transpose(-1,-2);mask=torch.ones_like(dense,dtype=torch.bool).tril()
                dense=dense.masked_fill(~mask,0);reference=(dense/dense.sum(-1,keepdim=True))@v
                error=float((y-reference).norm()/reference.norm())
                check.append(dict(relative_l2=error,description='causal cumsum scan versus dense masked feature kernel'))
                if error>1e-5:raise RuntimeError(f'Scan check failed: {error}')
            out=out.clone();out[0,:,heads]=y.transpose(0,1).to(out.dtype)
        return out,weights
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',replaced)
    model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],
        cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval()
    scenarios=[('teacher',None,[])]
    for seed in [11,29,47]:
        for objective in args.objectives:
            for method in ['mlp_positive','kan_positive']:
                path=args.fits/f'{objective}_{method}_m64_s{seed}.pt'
                for scope,layers in ([('middle',[14]),('late',[27]),('four_heads',[14,27]),('six_heads',[0,14,27])]
                                     if objective=='row' else [('four_heads',[14,27])]):
                    scenarios.append((f'{objective}_{method}_s{seed}_{scope}',path,layers))
    results=[]
    for name,path,layers in scenarios:
        active=load_maps(path,layers,getattr(torch,args.feature_dtype)) if path else {}
        rows=[];failure=None
        for doc in docs:
            ids=torch.load(args.data/doc['file'],weights_only=True)['input_ids'][None].cuda()
            try:
                hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1]
            except RuntimeError as error:
                if 'Nonpositive denominator in positive feature scan' not in str(error):raise
                failure=dict(document=doc['file'],reason=str(error));break
            losses=[]
            for start in range(0,len(hidden),128):
                logits=model.lm_head(hidden[start:start+128]).float()
                target=ids[0,start+1:start+1+len(logits)]
                losses.append(F.cross_entropy(logits,target,reduction='none'))
            loss=torch.cat(losses)
            rows.append(dict(document=doc['file'],nll=float(loss.mean()),tokens=len(loss),
                             last512_nll=float(loss[-512:].mean())))
        if failure:
            record=dict(scenario=name,status='failed',failure=failure,
                        replaced_heads=sum(len(v[1]) for v in active.values()),documents=rows)
        else:
            nll=sum(r['nll']*r['tokens'] for r in rows)/sum(r['tokens'] for r in rows)
            tail=sum(r['last512_nll'] for r in rows)/len(rows)
            record=dict(scenario=name,status='complete',replaced_heads=sum(len(v[1]) for v in active.values()),
                        nll=nll,perplexity=math.exp(nll),last512_nll=tail,last512_perplexity=math.exp(tail),documents=rows)
        results.append(record);print(json.dumps({k:v for k,v in record.items() if k!='documents'}),flush=True)
        (args.out/'replacement.json').write_text(json.dumps(dict(model=manifest['model'],results=results,feature_dtype=args.feature_dtype,
            scan_validation=check,notes=['All unselected heads remain original; no model finetuning.',
                'Q/K/V and hidden states are recomputed after every replacement.',
                'Original SDPA also computed for unselected heads: quality ablation, not a speed benchmark.',
                'All query positions replaced, including positions before the training query interval.']),indent=2))


if __name__=='__main__':main()
