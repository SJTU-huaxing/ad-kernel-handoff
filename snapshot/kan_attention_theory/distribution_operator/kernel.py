"""Frozen nonnegative feature maps from training spectral partitions and means."""
import json
import torch
from run import P,profiles,assign

class ConditionalMeanKernel:
    def __init__(self,m=64,epsilon=0.):
        assert 0<=epsilon<1
        self.m=m;self.epsilon=epsilon
        self.basis={k:v.cuda() if torch.is_tensor(v) else v for k,v in torch.load(P/'results/train_basis.pt',weights_only=True).items()}
        self.trees=json.loads((P/'results/partitions.json').read_text())
        params=json.loads((P/'results/positive_kernel_coefficients.json').read_text())
        self.coefficients=torch.tensor([h[str(m)] for h in params['coefficients']],device='cuda',dtype=torch.float64)
        self.scale=self.basis['scale'].exp()
        self.maps={side:torch.tensor([t['maps'][str(m)] for t in self.trees[side]],device='cuda') for side in ['q','k']}

    @torch.no_grad()
    def cells(self,x,side):
        fine=assign(profiles(x,side,self.basis),self.trees[side])
        return self.maps[side].gather(1,fine)

    @torch.no_grad()
    def features(self,x,side):
        cells=self.cells(x,side)
        if side=='q':
            f=torch.nn.functional.one_hot(cells,self.m).double()
            f=(1-self.epsilon)*f+self.epsilon/self.m
        else:
            f=self.coefficients.transpose(-1,-2).gather(1,cells[:,:,None].expand(-1,-1,self.m))
        return f*self.scale.sqrt()[:,None,None]

    @torch.no_grad()
    def paired(self,q,k):return (self.features(q,'q')*self.features(k,'k')).sum(-1)

if __name__=='__main__':
    from run import DATA,save
    torch.set_num_threads(2)
    manifest=json.loads((DATA/'manifest.json').read_text())
    row=next(r for r in manifest['shards'] if r['split']=='test');b=torch.load(DATA/row['file'],weights_only=True)
    q=b['q'][:2,:,512:].permute(1,0,2,3).flatten(1,2).cuda();k=b['k'][:2,:,:512].permute(1,0,2,3).flatten(1,2).cuda()
    checks=[]
    for m in [16,32,64,128]:
        f=ConditionalMeanKernel(m);cq=f.cells(q,'q');ck=f.cells(k,'k')
        direct=torch.stack([f.coefficients[h,cq[h],ck[h]]*f.scale[h] for h in range(4)])
        pred=f.paired(q,k);error=((pred-direct).abs()/direct).max().item()
        assert error<1e-12 and (pred>0).all()
        checks.append(dict(m=m,feature_product_relative_error=error))
    f=ConditionalMeanKernel(64,epsilon=1e-8)
    assert (f.features(q,'q')>0).all() and (f.features(k,'k')>0).all()
    save(P/'checks/deployed_features.json',dict(checks=checks,optional_strictly_positive_features_verified=True,
         default='epsilon=0 matches reported piecewise-constant kernel; epsilon>0 is an unevaluated smoothing option.'))
    print(checks)
