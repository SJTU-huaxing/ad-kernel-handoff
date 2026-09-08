"""Convex query-amplitude calibration in the existing MLP parameter budget.

Directions come from the matched exponential-feature KL control. Calibration
uses the next 64 Q in each per-document permutation, disjoint from the 64 Q used
to train directions. Each calibration query/key pair is visited once by SGD.
No model selection uses round-3 confirmation data.
"""
from core import *

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest,data,norm,scale=load_train();val=load_full('validation')
    kg=torch.Generator().manual_seed(20260919)
    bank_indices=torch.stack([torch.randperm(512,generator=kg)[:4] for _ in range(4096)]).cuda()
    bank=data['k'].gather(2,bank_indices[:,None,:,None].expand(-1,4,-1,128)).permute(1,0,2,3).flatten(1,2).float().contiguous()
    for seed in [11,29,47]:
        name=f'gauge_calibrated_m64_s{seed}_n4096'
        if (P/'fits'/f'{name}.json').exists():continue
        base,bmeta=load_fit(f'exp_control_m64_s{seed}_n4096',torch.float32)
        generator=torch.Generator().manual_seed(20260908+seed)
        permutations=torch.stack([torch.randperm(512,generator=generator) for _ in range(4096)])
        queries=permutations[:,64:128].cuda();initial=permutations[:,:64]
        assert hashlib.sha256(initial.numpy().tobytes()).hexdigest()==bmeta['query_indices_sha256']
        assert all(not set(x.tolist())&set(y.tolist()) for x,y in zip(initial,queries.cpu()))
        hs=[];ls=[];oldmass=[];targetmass=[];start=time.perf_counter()
        with torch.no_grad():
            log_mu=base.log_feature(bank,'k').double().logsumexp(1)-math.log(bank.shape[1])
            for i in range(4096):
                q=data['q'][i,:,queries[i]].float();k=data['k'][i].float()
                normalized=(q-base.q_mean[:,None])/base.q_std[:,None]
                hidden=F.silu(normalized@base.qnet.w1.transpose(-1,-2))
                z=hidden@base.qnet.w2.transpose(-1,-2)+base.qnet.bias[:,None]
                lp=(z.double()-.5*math.log(64)+log_mu[:,None]).logsumexp(-1)
                lt=(q@bank.transpose(-1,-2)/math.sqrt(128)-scale[:,None,None]).logsumexp(-1)-math.log(bank.shape[1])
                # Unit-mean K features make the predicted distributional raw
                # kernel mass exp(readout); the target is the empirical MGF.
                hs.append(hidden.cpu());ls.append((-lt).cpu())
                oldmass.append(lp.cpu());targetmass.append(lt.cpu())
        h=torch.stack(hs,1).cuda();ell=torch.stack(ls,1).cuda()
        hm=h.double().mean((1,2)).float();sd=h.double().std((1,2)).clamp_min(.03).float()
        h=(h-hm[:,None,None])/sd[:,None,None]
        initial_bias=-(ell.double().flatten(1).logsumexp(-1)-math.log(4096*64)).float()
        u=nn.Parameter(torch.zeros(4,192,device='cuda'));bias=nn.Parameter(initial_bias.clone())
        opt=torch.optim.Adam([u,bias],lr=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4096,eta_min=.001)
        order=torch.randperm(4096,generator=torch.Generator().manual_seed(20260917+seed)).tolist();history=[]
        for step,i in enumerate(order,1):
            residual=ell[:,i]+(h[:,i]*u[:,None]).sum(-1)+bias[:,None]
            perhead=(torch.expm1(residual)-residual).mean(-1)+1e-4*u.square().mean(-1)
            assert torch.isfinite(perhead).all()
            opt.zero_grad(set_to_none=True);perhead.mean().backward()
            factor=(10/(u.grad.square().sum(-1)+bias.grad.square()).sqrt().clamp_min(1e-20)).clamp_max(1)
            u.grad.mul_(factor[:,None]);bias.grad.mul_(factor);opt.step();scheduler.step()
            if step%1024==0:
                row=dict(step=step,loss=perhead.detach().tolist(),seconds=time.perf_counter()-start);history.append(row)
                print(json.dumps(dict(event='gauge',seed=seed,**row)),flush=True)
        with torch.no_grad():
            weight=u.double()/sd.double();intercept=bias.double()-(weight*hm.double()).sum(-1)
            state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
            w=state['qnet.w2'].double();b=state['qnet.bias'].double()
            # 63 relative logits plus 1 calibrated amplitude: an invertible
            # reallocation of the original 64 output-coordinate budget.
            state['qnet.w2']=torch.cat([w[:,:-1]-w[:,-1:],weight.cpu()[:,None]],1)
            shifted=b+log_mu.cpu()
            state['qnet.bias']=torch.cat([shifted[:,:-1]-shifted[:,-1:],(intercept.cpu()-.5*math.log(64))[:,None]],1)
            state['knet.bias']=state['knet.bias'].double()-log_mu.cpu()
            meta=dict(bmeta,name=name,kind='gauge_calibrated',variant='gauge_calibrated',
                target='Original raw exponential kernel; direction-pretraining KL followed by raw I-divergence amplitude calibration.',
                training_seconds=bmeta['training_seconds']+time.perf_counter()-start,
                calibration_seconds=time.perf_counter()-start,calibration_epochs=1,calibration_steps=4096,
                calibration_parameters_optimized=193,parameters_per_head=73856,
                calibration_pairs_per_head=4096*64*bank.shape[1],key_bank_size=bank.shape[1],
                calibration_distribution='Empirical P_Q x P_K product: 4 fixed keys per each training document, paired with every new calibration Q. No same-document-only calibration.',
                calibration_key_indices_sha256=hashlib.sha256(bank_indices.cpu().numpy().tobytes()).hexdigest(),
                log_key_feature_means=log_mu.tolist(),
                calibration_query_indices_sha256=hashlib.sha256(queries.cpu().numpy().tobytes()).hexdigest(),
                calibration_queries_disjoint_from_direction_training=True,
                calibration_note='Same training documents, disjoint query positions and Q/K pairs; one-pass readout updates. Additional distributional kernel evaluations and compute beyond one-stage models. Scalar log-MGF readout uses one-pass Adam, not claimed exact convex optimum.',
                target_layer_precision='Relative-logit differences folded in FP64 for algebraic checks; deployed features/weights FP32, same parameter count and state budget.')
            torch.save(dict(state_dict=state,metadata=meta),P/'fits'/f'{name}.pt')
            net,_=load_fit(name);metrics=evaluate(net,val,scale)
            save(P/'fits'/f'{name}.json',dict(metadata=meta,history=history,validation=metrics))
            om=torch.stack(oldmass,1).double().flatten(1);tm=torch.stack(targetmass,1).double().flatten(1)
            shifts=dict(global_bal=-(om-tm).logsumexp(-1)+math.log(om.shape[-1]),global_raw=tm.logsumexp(-1)-om.logsumexp(-1))
            for mode,shift in shifts.items():
                cname=f'{mode}_m64_s{seed}_n4096';cs={k:v.cpu().clone() for k,v in base.state_dict().items()}
                cs['qnet.bias']=cs['qnet.bias'].double()+shift[:,None]
                cm=dict(bmeta,name=cname,kind=mode,variant='exp',target='Raw-kernel global-scale calibration of the normalized-KL control.',calibration_log_scale=shift.tolist(),
                    calibration_queries_disjoint_from_direction_training=True,calibration_pairs_per_head=4096*64*bank.shape[1],
                    parameters_per_head=73856,calibration_steps=0,calibration_note='Closed-form scalar per head; no additional trainable parameters. Both predeclared scale criteria reported, no test selection.')
                torch.save(dict(state_dict=cs,metadata=cm),P/'fits'/f'{cname}.pt')
                cn,_=load_fit(cname);vv=evaluate(cn,val,scale);save(P/'fits'/f'{cname}.json',dict(metadata=cm,validation=vv))
            print(json.dumps(dict(event='calibrated',name=name,validation=metrics['summary'])),flush=True)
        del h,ell,hs,ls,base,net
    save(P/'results/gauge_plan.json',dict(cases={f'{kind}_m64_s{s}':[f'{kind}_m64_s{s}_n4096']*4 for kind in ['gauge_calibrated','global_bal','global_raw','exp_control','factorized_both','raw','balanced','kl_control'] for s in [11,29,47]},
        confirmation='Round 3, selected independently before inspecting gauge-calibrated confirmation scores. Main calibration settings fixed: lr .01->.001, ridge 1e-4, one pass, 64 disjoint queries per training document.',
        training_budget='Same inference parameter/state budget; calibration adds 4096 document passes on NEW query/key pairs using a 16384-key product-distribution bank. No equal-total-training-compute claim.'))

if __name__=='__main__':main()
