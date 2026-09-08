"""CPU arithmetic and sensitivity estimates only; no LLM/kernel experiment."""
import json,pathlib,math
P=pathlib.Path(__file__).resolve().parent
cfgpath=pathlib.Path('/root/autodl-tmp/hf-cache/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323/config.json')
c=json.loads(cfgpath.read_text());L=c['num_hidden_layers'];H=c['num_attention_heads'];HK=c['num_key_value_heads'];D=c['hidden_size'];d=D//H;ff=c['intermediate_size'];V=c['vocab_size'];m=64;hidden=192
base_no_head=2*L*(D*(H*d)+2*D*(HK*d)+(H*d)*D+3*D*ff)
head=2*D*V
features=4*(d*hidden+hidden*m)*L*H
aggregation=4*m*d*L*H
chunk_extra=2*64*(m+d)*L*H
state=L*H*m*(d+1)*4
new_parameters=L*H*2*(d*hidden+hidden*m+m)
new_weight_bytes=new_parameters*4
base_weight_read_bytes=base_no_head+head # BF16 weights, 2 FLOP/multiply-add.
kv_per_token=2*L*HK*d*2
rows=[]
for n in [1024,4096,8192,16384,32768,65536]:
 old_decode=base_no_head+head+4*L*H*n*d
 new_decode=base_no_head+head+features+aggregation
 old_prefill=base_no_head*n+head+2*L*H*d*n*(n+1)
 new_prefill=base_no_head*n+head+(features+aggregation)*n
 row=dict(context=n,original_decode_gflop=old_decode/1e9,linear_decode_gflop=new_decode/1e9,decode_flop_change_pct=100*(new_decode/old_decode-1),original_prefill_tflop=old_prefill/1e12,linear_prefill_tflop=new_prefill/1e12,current_chunk_prefill_tflop=(new_prefill+chunk_extra*n)/1e12,prefill_flop_change_pct=100*(new_prefill/old_prefill-1),original_kv_mib=kv_per_token*n/2**20,linear_state_mib=state/2**20,cache_reduction_factor=kv_per_token*n/state,
  simplified_byte_ratio=(base_weight_read_bytes+new_weight_bytes+2*state)/(base_weight_read_bytes+kv_per_token*n))
 rows.append(row)
f=json.loads((P.parent/'candidate_validation/results/feature_benchmark.json').read_text())['results']
proxy=lambda n,graph:next(r['median_ms'] for r in f if r['method']=='nn_mlp_11' and r['n']==n and r['cuda_graph']==graph)
graph_min=L*proxy(1,True);graph_max=3*graph_min
out=dict(scope='Pure estimate; all 336 Q heads hypothetically use independent two-layer 128->192->64 Q/K maps. Model weights unchanged. No full-head experiment or quality validation.',config_path=str(cfgpath),assumptions=['BF16 original dense weights; FP32 new feature weights and S,z state; batch1.', 'Multiply-add counted as 2 FLOPs; excludes norm, activation, exp/softmax, RoPE, denominator/scalar operations and memory overhead.', 'Original GQA projections retained: only two original K/V projections per layer. Independent nonlinear K feature maps for all twelve Q heads.', 'Decode includes vocabulary projection once per generated token. Prefill computes logits only for final prompt token.', 'Ideal linear prefix aggregation; separate current block64 matrix overhead reported. Parallel scan I/O not modeled.', 'Byte ratio assumes every dense weight and each original KV element read once; linear state read and written once. It is an optimistic traffic model, NOT a timing prediction; backend, cache reuse, KV copying and launch costs differ.'],
 base_dense_decode_gflop=(base_no_head+head)/1e9,new_attention_gflop=(features+aggregation)/1e9,new_parameters=new_parameters,new_weights_fp32_mib=new_weight_bytes/2**20,linear_state_mib=state/2**20,rows=rows,
 timing=dict(proxy='Previously measured ordinary softplus MLP with same affine dimensions, not the new amplitude-direction map. Only reuses existing measurements.',batched_feature_only_graph_ms=[graph_min,graph_max],batched_feature_only_eager_ms=[L*proxy(1,False),3*L*proxy(1,False)],scaling_assumption='Each 12-head layer takes 1-3 times the measured four-head feature time; an engineering scenario, not a proven bound.',decode_sensitivity=dict(original_ms=22.,removed_attention_cache_ms=[0,3],linear_state_and_glue_ms=[.2,1],calculated_total_ms=[22-3+graph_min+.2,22+graph_max+1],caveat='Assumed removed path and state/glue costs, not separately profiled. Preserves original model implementation and only captures/batches added features. Rounded 20-30 ms is a planning envelope, not measured or statistically identified.'),prefill_8192_feature_proxy_ms=84*proxy(8192,True),prefill_caveat='~98ms new features at linear head scaling. Original quadratic component is ~50ms from a three-point curve fit, not a profile. Linear scan cost and new-map difference remain unknown.'))
(P/'results/all_heads_mlp_estimate.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ['assumptions','config_path']},indent=2))
