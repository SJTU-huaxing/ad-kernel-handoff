"""Remove the optional partition-of-unity restriction in the cone construction."""
from common import *

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    package=torch.load(P/'results/construction.pt',weights_only=True);mk=package['moment_k'].cuda()
    rows=[]
    for method in ['mlp','kan','mulkan']:
        model,_=learned(method,11);b=model.log_feature(mk,'k').exp();del model
        gram=b.transpose(-1,-2)@b/b.shape[1];p=b.mean(1);diag=gram.diagonal(dim1=-2,dim2=-1).sqrt()
        correlation=gram/diag[:,:,None]/diag[:,None];eig=torch.linalg.eigvalsh(correlation)
        assert p.min()>0 and eig.min()>-1e-10
        package['bases']['raw_'+method]=cpu(dict(weights=b/b.shape[1],p=p,gram=gram,diag=diag,
            correlation=correlation,lipschitz=eig[:,-1],initial_scale=gram.sum(-1)))
        rows.append(dict(method='cone_raw_'+method,condition_number=[float(z[-1]/z[0]) if z[0]>0 else None for z in eig],
            effective_rank_1e10=(eig>eig[:,-1:]*1e-10).sum(-1).tolist()))
    torch.save(package,P/'results/construction.pt')
    save(P/'results/raw_basis_protocol.json',dict(reason='The general positive cone needs no partition-of-unity constraint. This ablation retains the pretrained positive key-feature amplitude.',
        methods=rows,selection='All three architectures evaluated, fixed seed 11, same nodes and 128 projection steps. No selection based on test performance.',
        distinction='Removing feature normalization changes the basis; neither version normalizes raw kernel targets. This is an additional exploratory ablation after the first candidate screen.'))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
