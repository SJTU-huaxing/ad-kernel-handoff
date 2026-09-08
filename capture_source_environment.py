"""Whitelisted source environment and external model fingerprints, no secrets."""
import importlib.metadata as metadata
import json,platform,sys,subprocess
from pathlib import Path
from build_snapshot import sha,write,P

def main():
    import torch
    info={'python':sys.version,'machine':platform.machine(),'torch':torch.__version__,
          'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'packages':{}}
    for name in ['torch','transformers','triton','numpy','safetensors','datasets','huggingface-hub','einops','flash-linear-attention']:
        try:info['packages'][name]=metadata.version(name)
        except metadata.PackageNotFoundError:info['packages'][name]=None
    source=P.parent/'kan_attention_theory'
    info['fla_commit']=subprocess.check_output(['git','-C',str(source/'flash-linear-attention'),'rev-parse','HEAD'],text=True).strip()
    write(P/'SOURCE_ENVIRONMENT.json',info)
    revision='8faed761d45a263340a0528343f099c05c9a4323'
    snapshot=P.parent/'hf-cache/models--Qwen--Qwen2.5-1.5B/snapshots'/revision
    rows=[]
    for f in sorted(snapshot.iterdir()):
        if f.is_file():rows.append({'path':f.name,'bytes':f.stat().st_size,'sha256':sha(f),'is_symlink':f.is_symlink()})
    config=json.loads((snapshot/'config.json').read_text())
    write(P/'EXTERNAL_MODEL_MANIFEST.json',{'model':'Qwen/Qwen2.5-1.5B','revision':revision,
        'original_snapshot':str(snapshot),'files':rows,'bytes':sum(r['bytes'] for r in rows),
        'config':{k:config.get(k) for k in ['model_type','hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','rms_norm_eps','rope_theta','max_position_embeddings','vocab_size','torch_dtype','dtype']},
        'scope':'External base model, not bundled. Copy complete resolved snapshot or the matching HF cache blobs; never copy account/token files.'})
    print(json.dumps({'source_environment_saved':True,'model_files':len(rows),'model_bytes':sum(r['bytes'] for r in rows)}),flush=True)

if __name__=='__main__':main()
