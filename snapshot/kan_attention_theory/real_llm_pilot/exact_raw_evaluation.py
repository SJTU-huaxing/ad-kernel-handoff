"""Final raw evaluation in Float64, with the exact LA denominator and no epsilon."""
import json
from pathlib import Path
import torch
from fit_features import FeaturePair,read_data,evaluate,pooled_kernel_nmse


@torch.inference_mode()
def main():
    p=Path(__file__).resolve().parent;torch.set_num_threads(4)
    manifest=json.loads((p/'data_qwen25_1p5b/manifest.json').read_text())
    labels=[[14,0],[14,6],[27,0],[27,6]];idx=[manifest['head_labels'].index(label) for label in labels]
    data=read_data(p/'data_qwen25_1p5b',manifest,idx)
    backup=p/'legacy_float32_evaluation_backup';backup.mkdir(exist_ok=True)
    for folder in ['raw_main','raw_dimensions','raw_longer','raw_importance']:
        for path in sorted((p/folder).glob('*.pt')):
            result_path=path.with_suffix('.json');result=json.loads(result_path.read_text())
            if result.get('evaluation_protocol')=='float64_exact_denominator':continue
            saved=torch.load(path,weights_only=True);meta=saved['metadata'];state=saved['state_dict']
            norm={key:state[key].cuda().double() for key in ['q_mean','q_std','k_mean','k_std']}
            model=FeaturePair(4,128,meta['m'],meta['method'],norm,state['amplitude'][:,0,0].cuda(),
                              grid=meta['grid'],width=16).cuda().double()
            model.load_state_dict(state);model.eval();scale=torch.tensor(meta['log_scale'],device='cuda',dtype=torch.float64)
            evaluations={split:evaluate(model,data[split],'raw',scale,exact=True) for split in ['validation','test','test_long','ood']}
            evaluations['test_block']=evaluate(model,data['test'],'raw',scale,block=True,exact=True)
            previous=backup/f'{folder}__{path.stem}.json'
            if not previous.exists():previous.write_text(result_path.read_text())
            result['evaluations']=evaluations;result['evaluation_protocol']='float64_exact_denominator'
            result['evaluation_note']='Parameters unchanged; raw Q/K unchanged; Float64 feature and kernel evaluation, exact division by row sum, no epsilon or clipping. Checkpoints were selected during original Float32 training.'
            result_path.write_text(json.dumps(result,indent=2))
            print(json.dumps(dict(folder=folder,name=meta['name'],pooled_raw_nmse=pooled_kernel_nmse(evaluations['test']).tolist())),flush=True)


if __name__=='__main__':main()
