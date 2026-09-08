"""Check raw target semantics and evaluate training risk for final raw checkpoints."""
import json
from pathlib import Path
import torch
from fit_features import FeaturePair,read_data,evaluate,pooled_kernel_nmse,target_for


@torch.inference_mode()
def main():
    p=Path(__file__).resolve().parent;root=p/'data_qwen25_1p5b'
    torch.set_num_threads(4)
    manifest=json.loads((root/'manifest.json').read_text());labels=[[14,0],[14,6],[27,0],[27,6]]
    idx=[manifest['head_labels'].index(label) for label in labels]
    data=read_data(root,manifest,idx);output=[]
    for folder in ['raw_longer','raw_importance']:
        for path in sorted((p/folder).glob('*.pt')):
            saved=torch.load(path,weights_only=True);meta=saved['metadata'];state=saved['state_dict']
            norm={key:state[key].cuda() for key in ['q_mean','q_std','k_mean','k_std']}
            model=FeaturePair(4,128,meta['m'],meta['method'],norm,state['amplitude'][:,0,0].cuda(),
                              grid=meta['grid'],width=16).cuda().double();model.load_state_dict(state);model.eval()
            scale=torch.tensor(meta['log_scale'],device='cuda',dtype=torch.float64)
            q=data['test'][0]['q'][:,-16:].float();k=data['test'][0]['k'].float()
            positions=torch.full((16,),1023,device='cuda',dtype=torch.int64)
            a,_,_=target_for(q,k[:,:32],positions,'raw',scale)
            b,_,_=target_for(q,k[:,:64],positions,'raw',scale)
            torch.testing.assert_close(a,b[:,:,:32],rtol=2e-5,atol=1e-8)
            original=(q.double()@k[:,:32].double().transpose(-1,-2)/(128**.5)).exp()
            recovered=a.double()*scale.double().exp()[:,None,None]
            relative=float((recovered-original).norm()/original.norm())
            if relative>2e-5:raise RuntimeError(f'Raw target check failed: {relative}')
            rows=evaluate(model,data['train'],'raw',scale,exact=True)
            row=dict(folder=folder,name=meta['name'],training_pooled_kernel_nmse=pooled_kernel_nmse(rows).tolist(),
                     target_recovery_relative_l2=relative,unchanged_when_extra_keys_added=True,
                     training_evaluations=rows)
            output.append(row);print(json.dumps({k:v for k,v in row.items() if k!='training_evaluations'}),flush=True)
    (p/'raw_audit.json').write_text(json.dumps(output,indent=2))


if __name__=='__main__':main()
