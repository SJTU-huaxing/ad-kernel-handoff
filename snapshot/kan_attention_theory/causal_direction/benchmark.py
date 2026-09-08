"""Full frozen-model batch1 inference, actual cache storage and no GPU co-runs."""
import gc,statistics
from common import *
from runtime import Replacement,cache_for
from hybrid import Hybrid,hybrid_storage
from swa import Sliding
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2
from operators import LayerFeatures,prefill,step

@torch.inference_mode()
def microbenchmark():
 ds=torch.load(P/'data/validation_000.pt',weights_only=True);q=ds['q'][0,:12,:1].cuda();k=ds['k'][0,:2,:1].cuda();v=ds['v'][0,:2,:1].cuda();del ds
 rows=[]
 for name in ['favor_s11','exp_kl_s11','split_01_s11','softplus_kl_s11']:
  f=LayerFeatures(name,14);lq,lk=f.logs(q,k);_,_,state=prefill(lq,lk,v.float().repeat_interleave(6,0))
  def combined():
   a,b=f.logs(q,k);return step(a,b,v,state,True)
  for what,fn in [('features',lambda:f.logs(q,k)),('state',lambda:step(lq,lk,v,state,True)),('features_and_state',combined)]:
   for _ in range(10):out=fn()
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    with torch.cuda.graph(g,stream=stream):
     for _ in range(100):out=fn()
   torch.cuda.current_stream().wait_stream(stream);g.replay();torch.cuda.synchronize();vals=[]
   for _ in range(7):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000/100)
   rows.append(dict(name=name,component=what,microseconds_per_layer=statistics.median(vals),samples_us=vals));del g
 save(P/'results/microbenchmark.json',dict(results=rows,protocol='Real heldout Q/K/V,12 heads of one layer, batch1 decode. Same fused projection and state implementations used in full-model benchmark. CUDA graph containing100 operations replayed7times after warmup, removes Python per-op dispatch. Repeated hot-cache microbenchmark, not full-model speed.',arithmetic=dict(mlp_feature_flops_per_head=147456,favor_feature_flops_per_head=32768,state_dominant_flops_per_head=32768,combined_mlp_to_favor_flop_ratio=2.75,note='FMA counted as2, excludes scalar activations/normalizations; fixed m64,d128,hidden192.')))

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 manifest=json.loads((P/'data/manifest.json').read_text());docs=[r['input_ids'] for r in manifest['records'] if r['split']=='confirm_long'];tokens=torch.tensor([docs[0]+docs[1][:32]],device='cuda');original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 cases=[('pure','teacher'),('pure','favor_s11'),('pure','exp_kl_s11'),('pure','split_01_s11'),('hybrid','favor_s11'),('hybrid','calibrated_s11'),('hybrid','gate_s11'),('window','swa448_sink4')]
 rows=[]
 for n in [1024,4096,8192]:
  for mode,name in cases:
   rt=Sliding(original) if mode=='window' else (Hybrid if mode=='hybrid' else Replacement)(original,name);rt.collect=False;modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',rt)
   def pre():
    rt.reset();cache=cache_for(model,name!='teacher');out=model(input_ids=tokens[:,:n],past_key_values=cache,use_cache=True,logits_to_keep=1);return out,cache
   for _ in range(3):out,cache=pre()
   del out,cache;torch.cuda.synchronize();pf=[];decode=[];wall=[]
   for repeat in range(6):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();out,cache=pre();end.record();end.synchronize();pf.append(start.elapsed_time(end));sz=hybrid_storage(cache,rt)
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();t0=time.perf_counter();a.record()
    for t in range(n,n+32):out=model(input_ids=tokens[:,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
    b.record();b.synchronize()
    if repeat:decode.append(a.elapsed_time(b)/32);wall.append((time.perf_counter()-t0)*1000/32)
    after=hybrid_storage(cache,rt)
    if name!='teacher':
     assert cache.layers[14].keys.untyped_storage().nbytes()==0 and cache.layers[27].keys.untyped_storage().nbytes()==0
     assert after['state_bytes']==sz['state_bytes'] and after['window_bytes']==sz['window_bytes']
    del out,cache
   rt.reset();gc.collect();torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();out,cache=pre();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-base
   row=dict(mode=mode,name=name,prompt_tokens=n,prefill_ms=statistics.median(pf[1:]),prefill_samples_ms=pf[1:],decode_ms_per_token=statistics.median(decode),decode_samples_ms=decode,wall_ms_per_token=statistics.median(wall),prefill_incremental_peak_bytes=peak,**sz);rows.append(row);print(json.dumps(row),flush=True)
   save(P/'results/benchmark.json',dict(hardware=torch.cuda.get_device_name(),torch_version=str(torch.__version__),results=rows,protocol='No concurrent GPU workload. BF16 frozen model, FP32 learned features/state, TF32 disabled. Batch1, unpadded. Three prefill warmups; six paired prefill+32 decode trials, discard first for both. CUDA events plus wall clock. Fixed teacher-forced continuation for timing. Only24/336 heads, two complete GQA layers. HF eager execution; neither compiled serving engine nor full linearized model.',precision_note='MLP and FAVOR both use Triton fused log-feature projection plus the same fused state update. Observed timings characterize these implementations, not intrinsic architecture optimality. Hybrid reference local path not fused.'))
   del out,cache,rt;gc.collect();torch.cuda.empty_cache()
 modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
 microbenchmark()

if __name__=='__main__':main()
