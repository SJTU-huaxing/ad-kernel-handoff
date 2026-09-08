"""Distribution-level coarse spectral witness on fixed empirical marginals."""
from common import *

def toy():
 n=256;m=64;eps=.1;C=torch.ones(n,n,dtype=torch.float64)*(1-eps)/n**2+torch.eye(n,dtype=torch.float64)*eps/n;s=torch.linalg.svdvals(C);bound=.5*s[m:].sum().square();mi=(C*(C*n*n).log()).sum();expected=.5*(eps*(n-m)/n)**2
 assert abs(float(bound)-expected)<1e-12
 torch.manual_seed(313);f=torch.rand(300,32,dtype=torch.float64);g=torch.rand(400,32,dtype=torch.float64);predict=f@g.T;predict/=predict.sum();qi=torch.arange(300)%80;ki=torch.arange(400)%90;coarse=torch.zeros(80,90,dtype=torch.float64);coarse.index_put_((qi[:,None].expand(300,400).reshape(-1),ki[None].expand(300,400).reshape(-1)),predict.reshape(-1),accumulate=True);s2=torch.linalg.svdvals(coarse);assert float(s2[32:].sum())<1e-12
 save(P/'checks/theory.json',dict(passed=True,toy=dict(n=n,m=m,epsilon=eps,nuclear_tail_bound=float(bound),analytic_bound=expected,mutual_information=float(mi),old_mi_bound=max(0,float(mi)-math.log(m))),coarsened_rank32_tail=float(s2[32:].sum()),note='Numerical checks support implementation of algebra; theorem proof is separate, not empirically proved.'))

@torch.inference_mode()
def centers(x,n=256):
 gen=torch.Generator(device=x.device).manual_seed(8711);c=x[torch.randperm(len(x),generator=gen,device=x.device)[:n]].clone()
 for _ in range(15):
  ids=(x.square().sum(-1,keepdim=True)+c.square().sum(-1)[None]-2*x@c.T).argmin(-1);s=torch.zeros_like(c).index_add_(0,ids,x);count=torch.bincount(ids,minlength=n);c=torch.where(count[:,None]>0,s/count.clamp_min(1)[:,None],c)
 return c

def assign(x,c):return (x.square().sum(-1,keepdim=True)+c.square().sum(-1)[None]-2*x@c.T).argmin(-1)

@torch.inference_mode()
def empirical():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 # Four fixed keys and Q per first 1024 training documents. All subsequent
 # computations integrate the WHOLE product of their selected marginals.
 tr=data('train');bank=tr['k'][:1024,:,::256].permute(1,0,2,3).flatten(1,2).cuda().double()
 tq=tr['q'][:1024,:,::16].permute(1,0,2,3).flatten(1,2).cuda().float();del tr
 nets={name:load(name) for name in ['split_01_s11','exp_kl_s11','global_bal_s11','calibrated_s11','gate_s11','favor_s11']};base,meta=nets['split_01_s11'];norm={k:base.state_dict()[k] for k in ['q_mean','q_std','k_mean','k_std']}
 cq=[];ck=[]
 for h in range(H):
  cq.append(centers((tq[h]-norm['q_mean'][h].float())/norm['q_std'][h].float()))
  ck.append(centers((bank[h//6].float()-norm['k_mean'][h].float())/norm['k_std'][h].float()))
 torch.save(dict(q=torch.stack(cq).cpu(),k=torch.stack(ck).cpu()),P/'results/witness_partitions.pt');del tq
 for split in ['validation','confirm_wiki','confirm_long','confirm_prose']:
  ds=data(split);q=ds['q'][:,:,::4].permute(1,0,2,3).flatten(1,2).cuda().double();n_docs=len(ds['q']);del ds
  # Test queries fixed before outcomes. Keys never taken from confirmation.
  fqs={n:net.log_feature(q,'q') for n,(net,_) in nets.items()};fks={n:net.log_feature(bank[KI.cuda()],'k') for n,(net,_) in nets.items()};rows=[]
  for h in range(H):
   qr=assign(((q[h]-norm['q_mean'][h])/norm['q_std'][h]).float(),cq[h]);kr=assign(((bank[h//6]-norm['k_mean'][h])/norm['k_std'][h]).float(),ck[h]);kt=F.one_hot(kr,256).double();C=torch.zeros(256,256,device='cuda',dtype=torch.float64);raw=C.clone();energies=[];risk={n:dict(kl=0.,raw_i=0.,energy=0.,raw_sse=0.) for n in nets};total=0.
   for begin in range(0,q.shape[1],256):
    end=min(begin+256,q.shape[1]);lt=q[h,begin:end]@bank[h//6].T/math.sqrt(D)-meta['log_scale'][h];a=lt.softmax(-1);truth=lt.exp();C.index_add_(0,qr[begin:end],a@kt/q.shape[1]);raw.index_add_(0,qr[begin:end],truth@kt);total+=float(truth.sum())
    for name in nets:
     lq=fqs[name][h,begin:end];lk=fks[name][h];qa=lq.amax(-1,keepdim=True);kb=lk.amax(-1,keepdim=True);lp=((lq-qa).exp()@(lk-kb).exp().T).clamp_min(1e-300).log()+qa+kb.T
     risk[name]['kl']+=float((a*(lt-lt.logsumexp(-1,keepdim=True)-lp+lp.logsumexp(-1,keepdim=True))).sum())/q.shape[1]
     pred=lp.exp();risk[name]['raw_i']+=float((pred-truth+truth*(lt-lp)).sum());risk[name]['raw_sse']+=float((pred-truth).square().sum());risk[name]['energy']+=float(truth.square().sum())
   raw/=total;sv=torch.linalg.svdvals(C);srv=torch.linalg.svdvals(raw);marg=C.sum(1)[:,None]*C.sum(0)[None];mi=float(torch.where(C>0,C*(C.clamp_min(1e-300)/marg.clamp_min(1e-300)).log(),0).sum());err=16*(1+math.sqrt(math.log(20)))/math.sqrt(n_docs)
   for values in risk.values():values['raw_i']/=total;values['raw_nmse']=values.pop('raw_sse')/values.pop('energy')
   curves=[dict(m=m,balanced_empirical_lb=.5*float(sv[m:].sum())**2,raw_empirical_lb=.5*float(srv[m:].sum())**2,balanced_iid_document_95_lower_bound=.5*max(0,float(sv[m:].sum())-err)**2,old_mi_lb=max(0,mi-math.log(m))) for m in [16,32,64,128]]
   assert risk['split_01_s11']['kl']+1e-10>=curves[2]['balanced_empirical_lb']
   rows.append(dict(head=HEADS[h],mutual_information=mi,curves=curves,risk=risk,confidence_nuclear_error=err,occupied_q=int(qr.unique().numel()),occupied_k=int(kr.unique().numel())))
  save(P/'results'/f'witness_{split}.json',dict(split=split,rows=rows,query_documents=n_docs,query_count=q.shape[1],key_count=bank.shape[1],key_reference='Fixed train-only bank: four keys of each of first1024 training documents',scope='Complete selected empirical P_Q x fixed-P_K product. Not one-context matrix optimization, not a certificate for unknown full P_K. Confidence correction assumes iid documents; source dependencies can violate that assumption.',partition='256 kmeans cells per side, 15 fixed iterations, learned only on training vectors; network itself has continuous features.'));print(json.dumps(dict(event='witness',split=split)),flush=True)

if __name__=='__main__':toy();empirical()
