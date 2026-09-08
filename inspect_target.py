"""Read-only target inventory. No installs, model loads, training, or env dump."""
import importlib.metadata as metadata
import json,platform,subprocess,sys

def main():
    result={'python':sys.version,'machine':platform.machine(),'platform':platform.platform(),'packages':{}}
    for name in ['torch','torch-npu','transformers','triton','triton-ascend','safetensors','datasets','numpy']:
        try:result['packages'][name]=metadata.version(name)
        except metadata.PackageNotFoundError:result['packages'][name]=None
    try:
        import torch
        result['torch_version']=torch.__version__
        try:
            import torch_npu
            result['torch_npu_import']='ok'
        except Exception as e:result['torch_npu_import']=f'{type(e).__name__}: {e}'
        if hasattr(torch,'npu'):
            result['npu_available']=torch.npu.is_available()
            if result['npu_available']:
                result['npu_count']=torch.npu.device_count()
                result['npu_names']=[torch.npu.get_device_name(i) for i in range(result['npu_count'])]
        else:result['npu_available']=False
    except Exception as e:result['torch_import']=f'{type(e).__name__}: {e}'
    try:
        p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15)
        result['npu_smi']={'returncode':p.returncode,'stdout':p.stdout[:12000],'stderr':p.stderr[:2000]}
    except (FileNotFoundError,subprocess.TimeoutExpired) as e:result['npu_smi']=type(e).__name__
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
