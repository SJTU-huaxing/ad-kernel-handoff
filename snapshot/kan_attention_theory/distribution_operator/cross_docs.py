"""Disjoint Q-document / K-document marginal pools, frozen trained kernel."""
import json,time
import torch
from run import P,load_data,profiles,assign,scan,summarize_package,save

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=load_data()['internal']
    basis={k:v.cuda() if torch.is_tensor(v) else v for k,v in torch.load(P/'results/train_basis.pt',weights_only=True).items()}
    trees=json.loads((P/'results/partitions.json').read_text());means=json.loads((P/'results/positive_kernel_coefficients.json').read_text())['coefficients']
    gen=torch.Generator().manual_seed(20260912);order=torch.randperm(256,generator=gen);a,b=order[:128],order[128:]
    save(P/'results/disjoint_protocol.json',dict(A_documents=a.tolist(),B_documents=b.tolist(),
        directions=['Q_A x K_B','Q_B x K_A'],tokens_per_document_per_marginal=512,
        samples_per_marginal_per_direction=65536,all_pairs_per_head_per_direction=65536**2,
        kernel='Unchanged positive kernel from original independent training/moment-fit documents.',
        interpretation='Each direction is a separate empirical product measure with disjoint document pools; their spectral intervals are not a population confidence interval.'))
    for suffix,qd,kd in [('ab',a,b),('ba',b,a)]:
        name='disjoint_'+suffix;path=P/f'results/{name}_moments.pt'
        if path.exists():package=torch.load(path,weights_only=True)
        else:
            qi=(torch.arange(512)[:,None]*256+qd[None]).flatten().cuda()
            ki=(torch.arange(512)[:,None]*256+kd[None]).flatten().cuda()
            q=data['q'][:,qi];k=data['k'][:,ki]
            fq=profiles(q,'q',basis);fk=profiles(k,'k',basis)
            lq=assign(fq,trees['q']);lk=assign(fk,trees['k'])
            package=scan(q,k,fq,fk,lq,lk,basis,[16384,65536],path)
            del fq,fk,lq,lk,q,k
        save(P/f'results/{name}_summary.json',summarize_package(package,basis,trees,means))
        print(name,'complete',flush=True)

if __name__=='__main__':main()
