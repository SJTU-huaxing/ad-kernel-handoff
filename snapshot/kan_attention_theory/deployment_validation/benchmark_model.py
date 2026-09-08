"""Actual full-model cached prefill/decode, batch 1, 4/336-head ablation."""
import argparse,json,time,statistics,gc
import torch
from transformers.models.qwen2 import modeling_qwen2
from evaluate_model import documents,load_model
from runtime import P,AttentionReplacement

def events(fn,repeats=7):
    values=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();values.append(a.elapsed_time(b))
    return dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),samples_ms=values)

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='model_benchmark_optimized.json')
    parser.add_argument('--methods',nargs='*');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest,sets=documents();model=load_model(manifest)
    tokens=torch.cat(sets['internal'][:9]).cuda();original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa']
    outputpath=P/'results'/args.output
    previous=json.loads(outputpath.read_text()) if args.methods and outputpath.exists() else {}
    rows=[r for r in previous.get('results',[]) if r['method'] not in args.methods];checks=[r for r in previous.get('checks',[]) if r['method'] not in args.methods]
    for method in args.methods or ['teacher','exact_split','partition','favor_plus','galerkin']:
        runtime=AttentionReplacement(original,method);modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',runtime)
        # Real model prefix-vs-cache consistency on the first 1024-token document.
        runtime.reset();full=model(input_ids=tokens[None,:1024],use_cache=False,logits_to_keep=1).logits.float()
        runtime.reset();prefix=model(input_ids=tokens[None,:1000],use_cache=True,logits_to_keep=1)
        cache=prefix.past_key_values
        for t in range(1000,1024):
            last=model(input_ids=tokens[None,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
        err=float((last.logits.float()-full).norm()/full.norm())
        checks.append(dict(method=method,logits_prefill_vs_cached_relative_l2=err,
            top1_equal=bool(last.logits.argmax(-1).eq(full.argmax(-1)).all()),
            note='BF16 model arithmetic differs between GEMM prefill and GEMV decode; kernel recurrence checked independently in Float64.'))
        del prefix,last,cache,full
        for n in [1024,4096,8192]:
            ids=tokens[None,:n]
            def prefill():
                runtime.reset();return model(input_ids=ids,use_cache=True,logits_to_keep=1)
            for _ in range(3):prefill()
            torch.cuda.synchronize();timing=events(prefill)
            runtime.reset();gc.collect();torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            result=prefill();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-base
            cache=result.past_key_values
            kvbytes=sum(layer.keys.numel()*layer.keys.element_size()+layer.values.numel()*layer.values.element_size() for layer in cache.layers)
            statebytes=sum(s[key].numel()*s[key].element_size() for s in runtime.states.values() for key in ['s','z'])
            del result,cache
            # Teacher-forced fixed next tokens avoid comparing different generated work.
            decode=[];wall=[]
            for repeat in range(6):
                result=prefill();cache=result.past_key_values
                torch.cuda.synchronize();t0=time.perf_counter()
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                for t in range(n,n+32):
                    result=model(input_ids=tokens[None,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
                b.record();b.synchronize()
                if repeat:
                    decode.append(a.elapsed_time(b)/32);wall.append((time.perf_counter()-t0)*1000/32)
                del result,cache
            row=dict(method=method,prompt_tokens=n,batch_size=1,decode_tokens=32,prefill=timing,
                decode_ms_per_token=statistics.median(decode),decode_tokens_per_second=1000/statistics.median(decode),
                decode_samples_ms=decode,wall_ms_per_token=statistics.median(wall),kv_cache_bytes=kvbytes,
                linear_state_bytes=statebytes,prefill_incremental_peak_bytes=peak)
            rows.append(row);print(json.dumps(row),flush=True)
            (P/'results'/args.output).write_text(json.dumps(dict(results=rows,checks=checks,
                hardware=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
                protocol='Full frozen Qwen model, BF16; feature/state FP32, TF32 disabled. Batch=1. 3 prefill warmups, 7 repeats; decode 1 warmup + 5 repeats of 32 fixed next tokens. CUDA events; no concurrent GPU workload. HF eager model, not a production serving engine.',
                scope='Only 4/336 Q heads replaced. Remaining Q heads share ALL KV heads, so original KV cache is retained. Adds recurrent state. Prompts concatenate real heldout documents for timing; no long-context quality claim.',
                exact_split='Identity control with the same remaining/selected Q-head split and exact SDPA for both groups.'),indent=2,allow_nan=False))
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':main()
