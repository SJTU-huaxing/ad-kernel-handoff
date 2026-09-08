"""Exploratory beta-divergence ablation, reusing fixed original holdout matrices."""
import math,time
import torch
from experiment import P,Features,save,json
from evaluate import load_eval,evaluate_matrix

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    cal={k:v.cuda() for k,v in torch.load(P/'results/calibration.pt',weights_only=True).items()}
    pools,_=load_eval();bank=[];start=time.perf_counter()
    for kind in ['shared','untied']:
        for seed in [11,29,47]:
            f=Features(cal,64,seed,learn=kind)
            f.load_state_dict(torch.load(P/f'beta15_fits/{kind}_m64_s{seed}.pt',weights_only=True)['state']);f.double()
            bank.append((dict(method='beta15_'+kind,m=64,seed=seed),f))
    checks=violations=0
    for dataset,pool in pools.items():
        qp=pool['q'][:,:,512:].flatten(1,2);kp=pool['k'][:,:,:512].flatten(1,2)
        for source in ['product','same_context']:
            for rep in range(4):
                original=json.loads((P/f'results/{dataset}_{source}_{rep}.json').read_text());sample=original['sampling']
                if source=='product':q=qp[:,sample['q_indices']];k=kp[:,sample['k_indices']];v=None
                else:q=pool['q'][:,rep,512:];k=pool['k'][:,rep,:512];v=pool['v'][:,rep,:512]
                rows=[]
                for meta,f in bank:
                    row=dict(**meta,**evaluate_matrix(f,q,k,v));rows.append(row)
                    for e,floor in zip(row['log_nmse'],original['svd']['64']):checks+=1;violations+=int(math.exp(e)+1e-9<floor)
                save(P/f'results/beta15_{dataset}_{source}_{rep}.json',dict(dataset=dataset,source=source,rep=rep,methods=rows))
    assert violations==0
    save(P/'results/beta15_evaluation_completion.json',dict(seconds=time.perf_counter()-start,rank_checks=checks,rank_violations=violations,
        note='Adaptive loss ablation; no model/checkpoint selection from these test scores.'))

if __name__=='__main__':main()
