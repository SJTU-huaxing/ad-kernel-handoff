from core_attribution import *

def main():
    torch.set_num_threads(4)
    box=torch.load(OLD/'fits'/f'{old_name(11)}.pt',weights_only=True)
    sd=box['state_dict'];norm={k:sd[k] for k in ['q_mean','q_std','k_mean','k_std']};scale=sd['numerical_scale']
    torch.manual_seed(8264)
    q=(norm['q_mean'][:,None]+norm['q_std'][:,None]*torch.randn(H,9,D)).double()
    k=(norm['k_mean'][:,None]+norm['k_std'][:,None]*torch.randn(H,17,D)).double()
    mask=torch.arange(17)[None]<=torch.arange(8,17)[:,None]
    rows=[]
    for init in ['standard','matched_zero']:
        nets=[]
        for kind in ['ad','exp']:
            torch.manual_seed(11);nets.append(Features(norm,kind,init,scale).double())
        assert all(torch.equal(x,y) for x,y in zip(nets[0].parameters(),nets[1].parameters()))
        pa=[n.log_matrix(q,k).masked_fill(~mask[None],-torch.inf).softmax(-1) for n in nets]
        difference=float((pa[0]-pa[1]).abs().max().detach())
        if init=='matched_zero':
            assert difference<1e-12
            uniform=mask.double()/mask.sum(-1,keepdim=True)
            assert float((pa[0]-uniform[None]).abs().max())<1e-12
        for kind,n in zip(['ad','exp'],nets):
            assert sum(p.numel() for p in n.parameters())//H==73536
            optimizer=torch.optim.AdamW(n.parameters(),lr=.002,weight_decay=1e-4)
            maxnorm={name:0. for name,_ in n.named_parameters()}
            for _ in range(3):
                logp=n.log_matrix(q,k)
                logt=q@k.transpose(-1,-2)/math.sqrt(D)-scale[:,None,None].double()
                loss=source.rows_loss(logp,logt,mask[None],0.)[0].mean()
                optimizer.zero_grad();loss.backward()
                for name,p in n.named_parameters():maxnorm[name]=max(maxnorm[name],float(p.grad.norm()))
                optimizer.step()
            assert all(v>0 for v in maxnorm.values())
            rows.append(dict(initialization=init,kind=kind,params_per_head=73536,
                paired_initial_attention_max_difference=difference,gradient_norms_over_first3steps=maxnorm))
    save(P/'checks/structure.json',dict(passed=True,rows=rows,
        scope='Synthetic structure/gradient checks; matched_zero has identical initial attention, standard merely identical weights.'))
    print(json.dumps(dict(passed=True,rows=rows)),flush=True)

if __name__=='__main__':main()
