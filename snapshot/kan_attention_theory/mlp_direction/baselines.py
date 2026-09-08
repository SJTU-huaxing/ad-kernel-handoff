"""Existing positive random-feature controls on the final untouched documents."""
from core import *
sys.path.insert(0,str(ROOT/'kernel_comparison'))
from rf import RandomFeatures,SEEDS

class RF:
    def __init__(self,method,seed,scale):
        bank=torch.load(ROOT/'kernel_comparison/results/rf_models.pt',weights_only=True)
        self.base=RandomFeatures(next(r for r in bank if r['method']==method and r['seed']==seed),64)
        self.scale=torch.as_tensor(scale,device='cuda',dtype=torch.float64)
    def log_feature(self,x,side):return self.base.log_features(x,side)-self.scale[:,None,None]/2
    def log_matrix(self,q,k):
        a,b=self.log_feature(q,'q'),self.log_feature(k,'k')
        aq=a.amax(-1,keepdim=True);bk=b.amax(-1,keepdim=True)
        return ((a-aq).exp()@(b-bk).exp().transpose(-1,-2)).clamp_min(1e-300).log()+aq+bk.transpose(-1,-2)

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    meta=json.loads((P/'fits/raw_m64_s11_n4096.json').read_text())['metadata']
    scale=torch.tensor(meta['log_scale'],device='cuda');rows=[]
    for ds in ['fresh_wiki3','swde3']:
        data=torch.load(P/'results'/f'data_{ds}.pt',weights_only=True)
        for method in ['favor_plus','centered_favor_plus','sderf','aderf']:
            for seed in SEEDS:
                metrics=evaluate(RF(method,seed,scale),data,scale)
                rows.append(dict(name=f'{method}_m64_s{seed}',kind=method,seed=seed,dataset=ds,**metrics))
                save(P/'results/kernel_rf.json',dict(results=rows,scope='Previously fixed m64 random-feature banks, five seeds; no new training or selection.'))
        print(json.dumps(dict(event='rf_complete',dataset=ds)),flush=True)

if __name__=='__main__':main()
