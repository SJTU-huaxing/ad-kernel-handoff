"""Full frozen model, local replacement; exact prior dataset/protocol."""
import argparse,time
import torch
import torch.nn.functional as F
from common import *
from deploy import make_runtime
from evaluate_model import documents,load_model
from transformers.models.qwen2 import modeling_qwen2

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='*');parser.add_argument('--fp64-check',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    methods=args.methods or json.loads((P/'results/selection.json').read_text())['ppl_methods']
    manifest,sets=documents();model=load_model(manifest);original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa']
    if args.fp64_check:sets={k:v[:8] for k,v in sets.items()}
    path=P/'results'/('ppl_fp64.json' if args.fp64_check else 'ppl.json');results=json.loads(path.read_text())['results'] if path.exists() else []
    for name in methods:
        replacement=make_runtime(original,name,dtype=torch.float64 if args.fp64_check else torch.float32);replacement.collect_diagnostics=True
        modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',replacement)
        for ds,docs in sets.items():
            if any(r['method']==name and r['dataset']==ds for r in results):continue
            rows=[];started=time.perf_counter()
            for i,doc in enumerate(docs):
                replacement.reset();ids=doc.cuda()
                hidden=model.model(input_ids=ids[None],use_cache=False).last_hidden_state[0,:-1]
                losses=[]
                for start in range(0,len(hidden),128):
                    logits=model.lm_head(hidden[start:start+128]).float()
                    losses.append(F.cross_entropy(logits,ids[start+1:start+1+len(logits)],reduction='none'))
                losses=torch.cat(losses);valid=bool(torch.isfinite(losses).all())
                counts=torch.stack(replacement.diagnostics).sum(0).tolist()
                rows.append(dict(ordinal=i,nll=float(losses.mean()) if valid else None,tokens=len(losses),
                    nonpositive_denominators=counts[0],nonfinite_attention_rows=counts[1]))
                if i%32==31:print(json.dumps(dict(event='ppl',method=name,dataset=ds,documents=i+1,seconds=time.perf_counter()-started)),flush=True)
            valid=all(r['nll'] is not None for r in rows);nll=sum(r['nll']*r['tokens'] for r in rows)/sum(r['tokens'] for r in rows) if valid else None
            row=dict(method=name,dataset=ds,documents_count=len(rows),documents=rows,nll=nll,
                perplexity=math.exp(nll) if nll is not None and nll<700 else None,
                nonpositive_denominators=sum(r['nonpositive_denominators'] for r in rows),
                nonfinite_attention_rows=sum(r['nonfinite_attention_rows'] for r in rows),seconds=time.perf_counter()-started)
            results.append(row);save(path,dict(results=results,model=manifest['model'],revision=manifest['model_revision'],
                replaced_heads=HEADS,total_query_heads=336,feature_dtype='FP64' if args.fp64_check else 'FP32',model_dtype='BF16',
                scope=('Frozen model, only 4/336 query heads replaced at all causal positions; '
                    +('first 8 internal and first 8 official documents for arithmetic diagnostics. ' if args.fp64_check else 'same 256 internal and 20 official 1024-token document prefixes as prior evaluation. ')
                    +'No fine-tuning. Official subset is not standard concatenated WikiText2 PPL.'),
                selection=json.loads((P/'results/selection.json').read_text())))
            print(json.dumps({k:v for k,v in row.items() if k!='documents'}),flush=True)
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':main()
