"""Construct training-only quadratures, continuous bases and spectral envelope."""
import time
import torch
from common import *

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter();data=load_data()
    raw=torch.load(OP/'results/train_basis.pt',weights_only=True)
    u,v,s=[raw[x].cuda().double() for x in ['u','v','s']]
    # Simultaneous SVD sign choice preserves the anchor decomposition.
    sign=v[:,:,0].sum(1).sign();u[:,:,0]*=sign[:,None];v[:,:,0]*=sign[:,None]
    assert (u[:,:,0]>0).all() and (v[:,:,0]>0).all()
    envq=(v[:,:,1:32]/v[:,:,:1]).abs().amax(1)*(s[:,:1]/s[:,1:32]).sqrt()
    envk=(u[:,:,1:32]/u[:,:,:1]).abs().amax(1)*(s[:,:1]/s[:,1:32]).sqrt()
    envelope_sum=(envq*envk).sum(-1);delta=(envelope_sum-1).clamp_min(0);c0=(1-envelope_sum).clamp_min(0)
    # First token-position stratum spans ALL 2048 moment-fit documents.
    package=dict(scale=raw['scale'],anchor_q=raw['q'],anchor_k=raw['k'],
        positive_q=raw['q'][:,:64],moment_k=data['moment_train']['k'][:,:2048].double(),
        nys_q=v[:,:,:64]/s[:,:64].sqrt()[:,None],nys_k=u[:,:,:64]/s[:,:64].sqrt()[:,None],
        envelope_q=envq,envelope_k=envk,pair_delta=delta,pair_c0=c0,bases={})
    # VQ baseline: ordinary Euclidean key codebook, calibration-only k-means.
    x=data['calibration']['k'].double();centers=x[:,:64].clone()
    for it in range(30):
        total=torch.zeros_like(centers);count=torch.zeros(4,64,device='cuda',dtype=torch.float64)
        for start in range(0,x.shape[1],4096):
            xx=x[:,start:start+4096]
            dist=xx.square().sum(-1,keepdim=True)+centers.square().sum(-1)[:,None]-2*xx@centers.transpose(-1,-2)
            labels=dist.argmin(-1)
            total.scatter_add_(1,labels[:,:,None].expand(-1,-1,128),xx)
            count.scatter_add_(1,labels,torch.ones_like(labels,dtype=torch.float64))
        centers=torch.where((count>0)[:,:,None],total/count.clamp_min(1)[:,:,None],centers)
    package['vq_centers']=centers
    mk=package['moment_k'];condition=[]
    for kind in ['anchor','mlp','kan','mulkan']:
        if kind=='anchor':b=(mk@package['positive_q'].cuda().transpose(-1,-2)/math.sqrt(128)).softmax(-1)
        else:
            model,_=learned(kind,11);b=model.log_feature(mk,'k').softmax(-1);del model
        gram=b.transpose(-1,-2)@b/b.shape[1];p=b.mean(1);diag=gram.diagonal(dim1=-2,dim2=-1).sqrt()
        correlation=gram/diag[:,:,None]/diag[:,None]
        eig=torch.linalg.eigvalsh(correlation)
        assert p.min()>0 and eig.min()>-1e-10,(kind,p.min(),eig.min())
        package['bases'][kind]=dict(weights=b/b.shape[1],p=p,gram=gram,diag=diag,
            correlation=correlation,lipschitz=eig[:,-1])
        condition.append(dict(basis=kind,condition_number=[float(z[-1]/z[0]) if z[0]>0 else None for z in eig],
            minimum_eigenvalue=eig[:,0].tolist(),effective_rank_1e10=(eig>eig[:,-1:]*1e-10).sum(-1).tolist(),
            min_mass=p.amin(-1).tolist(),quadrature_nodes=2048))
    torch.save(cpu(package),P/'results/construction.pt')
    save(P/'results/protocol.json',dict(model='Qwen/Qwen2.5-1.5B',heads=HEADS,m=64,
        training_documents=4096,calibration_documents=2048,moment_documents=2048,
        quadrature='2048 independent-of-test moment keys, one key from each moment-fit document; empirical quadrature is NOT the true population integral.',
        neural_baselines='Frozen existing one-pass Poisson/KL fits, 4096 documents, 2097152 unique within-document pairs/head, 73856 active parameters per QK pair; all 3 seeds.',
        basis_note='Anchor basis uses 64 calibration query anchors. Neural bases reuse the frozen k-net (seed 11), soft-normalized over features. This feature normalization does NOT normalize kernel targets.',
        nystrom='Positive anchor kernel SVD, Nyström extension, 64 modes; no test fit.',
        spectral_pair='32 modes, 63 nonzero-or-zero candidate feature slots padded to 64. Global envelopes follow positive weighted averages of anchor singular-vector ratios; no sample-maximum assumption.',
        spectral_warning='Exact E_r*+delta^2 pertains to true orthonormal population singular functions, not these empirical Nystrom functions. Actual heldout risk is measured.',
        cone='128 deterministic accelerated projected-gradient iterations of the coefficient quadratic projection; numerical cone candidate, no assertion of exact NNLS or population optimality. This is not a neural MSE training objective.',
        vq='64 Euclidean calibration key centroids, 30 Lloyd iterations. Kernel-level VQ baseline, not full end-to-end-trained Transformer-VQ architecture.',
        target='exp(q^T k/sqrt(128)); all methods represented after a fixed headwise scale which is restored for raw metrics.',
        feature_seeds='No test-based seed selection; m=64 primary, fixed FAVOR+ m640 compute-budget supplement.',
        pair_delta=delta.tolist(),envelope_sum=envelope_sum.tolist(),bases=condition,seconds=time.perf_counter()-started))
    print(json.dumps(dict(event='prepared',seconds=time.perf_counter()-started,delta=delta.tolist(),bases=condition)),flush=True)

if __name__=='__main__':main()
