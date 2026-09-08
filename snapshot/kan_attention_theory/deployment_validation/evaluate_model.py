"""Frozen full-Qwen PPL with only four selected query heads replaced."""
import argparse,json,math,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2
from runtime import P,AttentionReplacement

def documents():
    data=P.parent/'single_pass_mulkan/data';manifest=json.loads((data/'manifest.json').read_text())
    internal=[]
    for r in manifest['shards']:
        if r['split']=='test':internal.extend(torch.load(data/r['file'],weights_only=True)['input_ids'])
    old=P.parent/'real_llm_pilot/data_qwen25_1p5b';om=json.loads((old/'manifest.json').read_text())
    official=[torch.load(old/r['file'],weights_only=True)['input_ids'] for r in om['documents'] if r['split']=='test']
    return manifest,dict(internal=internal,official=official)

def load_model(manifest):
    return AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['model_revision'],
        cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--internal',type=int,default=256)
    parser.add_argument('--fp64-check',action='store_true');parser.add_argument('--output')
    parser.add_argument('--methods',nargs='*');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest,sets=documents();sets['internal']=sets['internal'][:args.internal]
    original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];model=load_model(manifest)
    scenarios=[('teacher',0),('exact_split',0),('partition',0),('galerkin',0)]+[('favor_plus',s) for s in [1009,1046,1083,1120,1157]]+[('favor_plus_640',0)]
    if args.methods:scenarios=[r for r in scenarios if r[0] in args.methods]
    dtype=torch.float64 if args.fp64_check else torch.float32
    outpath=P/'results'/(args.output or ('ppl_fp64.json' if args.fp64_check else 'ppl.json'))
    results=json.loads(outpath.read_text())['results'] if outpath.exists() else []
    for method,seed in scenarios:
        runtime=AttentionReplacement(original,method,seed,dtype,optimized=not args.fp64_check)
        runtime.collect_diagnostics=True
        modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',runtime)
        for ds,docs in sets.items():
            if args.fp64_check:docs=docs[:4]
            if any(r['method']==method and r['seed']==seed and r['dataset']==ds and r['documents_count']==len(docs) for r in results):continue
            rows=[];started=time.perf_counter()
            for ordinal,document in enumerate(docs):
                runtime.reset();document=document.cuda()
                hidden=model.model(input_ids=document[None],use_cache=False).last_hidden_state[0,:-1]
                losses=[]
                for start in range(0,len(hidden),128):
                    logits=model.lm_head(hidden[start:start+128]).float()
                    losses.append(F.cross_entropy(logits,document[start+1:start+1+len(logits)],reduction='none'))
                loss=torch.cat(losses)
                valid=bool(torch.isfinite(loss).all())
                counts=torch.stack(runtime.diagnostics).sum(0).tolist() if runtime.diagnostics else [0,0,0]
                rows.append(dict(ordinal=ordinal,nll=float(loss.mean()) if valid else None,tokens=len(loss),
                    last512_nll=float(loss[-512:].mean()) if valid else None,
                    nonpositive_denominators=counts[0],nonfinite_attention_rows=counts[1],attention_rows=counts[2]))
                if ordinal%32==31:print(json.dumps(dict(event='ppl_progress',method=method,seed=seed,dataset=ds,documents=ordinal+1)),flush=True)
            valid=all(r['nll'] is not None for r in rows)
            nll=sum(r['nll']*r['tokens'] for r in rows)/sum(r['tokens'] for r in rows) if valid else None
            row=dict(method=method,seed=seed,dataset=ds,documents_count=len(docs),nll=nll,
                perplexity=math.exp(nll) if nll is not None and nll<700 else None,
                last512_perplexity=math.exp(sum(r['last512_nll'] for r in rows)/len(rows)) if valid and sum(r['last512_nll'] for r in rows)/len(rows)<700 else None,
                invalid_documents=sum(r['nll'] is None for r in rows),documents=rows,seconds=time.perf_counter()-started)
            row.update(nonpositive_denominators=sum(r['nonpositive_denominators'] for r in rows),
                       nonfinite_attention_rows=sum(r['nonfinite_attention_rows'] for r in rows))
            results.append(row)
            outpath.parent.mkdir(exist_ok=True)
            outpath.write_text(json.dumps(dict(results=results,model=manifest['model'],revision=manifest['model_revision'],
                replaced_heads=[[14,0],[14,6],[27,0],[27,6]],total_query_heads=336,m=64,budget_control_m=640,feature_dtype=str(dtype),
                protocol='Frozen full-model forward, all causal positions; only 4/336 Q heads replaced; no finetuning or test fitting. Exact attention is computed only for remaining heads. BF16 model, FP32 feature/state arithmetic except explicit FP64 check.',
                datasets='Internal: 256 document-disjoint WikiText103 source-train article prefixes. Official: fixed 20 WikiText2 test article prefixes. Each 1024 tokens; these are subset PPLs, not standard concatenated WikiText2 benchmark PPL.',
                seeds='All five pre-existing FAVOR+ seeds, no selection.'),indent=2,allow_nan=False))
            print(json.dumps({k:v for k,v in row.items() if k!='documents'}),flush=True)
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':main()
