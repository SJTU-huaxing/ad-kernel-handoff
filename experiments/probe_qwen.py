"""Exercise the fixed teacher and post-RoPE capture on the installed HF/NPU stack."""
import json
from pathlib import Path
import time

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
device="npu:1"
location=ROOT/"work/models/Qwen2.5-1.5B"
tokenizer=AutoTokenizer.from_pretrained(location,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(location,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").to(device).eval()
original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS["sdpa"]
captured={}


def capture(module,q,k,v,mask,**kwargs):
    if module.layer_idx in [14,27]:
        captured[module.layer_idx]=(q,k,v)
    return original(module,q,k,v,mask,**kwargs)


modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register("sdpa",capture)
text=("A scientific experiment compares predictions with measurements. "
      "Researchers record the method, data, and uncertainty so that others can reproduce the result. ")*100
ids=tokenizer(text,add_special_tokens=False,return_tensors="pt").input_ids[:,:1024].to(device)
start=time.perf_counter()
with torch.inference_mode():
    hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state
    torch.npu.synchronize()
    forward_seconds=time.perf_counter()-start
    logits=model.lm_head(hidden[:,:128]).float()
    nll=torch.nn.functional.cross_entropy(logits[:,:-1].flatten(0,1),ids[:,1:128].flatten()).item()
    assert torch.isfinite(hidden).all().item()
    shapes={str(layer):[list(x.shape) for x in qkv] for layer,qkv in captured.items()}
    for q,k,v in captured.values():
        assert tuple(q.shape)==(1,12,1024,128)
        assert tuple(k.shape)==tuple(v.shape)==(1,2,1024,128)
    result={"passed":True,"teacher":"Qwen/Qwen2.5-1.5B",
            "revision":"8faed761d45a263340a0528343f099c05c9a4323",
            "device":device,"length":1024,"captured_post_rope_shapes":shapes,
            "cold_forward_seconds":forward_seconds,"synthetic_text_nll":nll,
            "scope":"Teacher/capture compatibility smoke test; synthetic repeated passage, not a quality evaluation"}
    (ROOT/"work/reproduction/qwen_capture_smoke.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
