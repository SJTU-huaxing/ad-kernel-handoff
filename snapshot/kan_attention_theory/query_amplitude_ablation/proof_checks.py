from ablation import *


def main():
    torch.set_num_threads(2)
    norm = {s+'_'+k:torch.zeros(1,D) if k=='mean' else torch.ones(1,D)
            for s in ['q','k'] for k in ['mean','std']}
    torch.manual_seed(7002)
    q, k = torch.randn(1,11,D,dtype=torch.float64), torch.randn(1,17,D,dtype=torch.float64)
    rows = []
    for regime in REGIMES:
        for variant in VARIANTS:
            torch.manual_seed(11)
            net = Reduced(norm,variant,regime,torch.tensor([1.27])).double()
            lp = net.log_matrix(q,k)
            lq,lk = net.raw_log_feature(q,'q'),net.raw_log_feature(k,'k')
            raw = lq.exp() @ lk.exp().transpose(-1,-2)
            restore = (lp+net.numerical_scale[:,None,None]).exp()
            err = float(((raw-restore).abs().max()/raw.abs().max()).detach())
            assert err < 1e-12
            truth = q@k.transpose(-1,-2)/math.sqrt(D)-net.numerical_scale[:,None,None]
            if regime == 'causal_kl':
                loss = (truth.softmax(-1)*(truth.log_softmax(-1)-lp.log_softmax(-1))).mean()
            else:
                loss = (lp.exp()-truth.exp()+truth.exp()*(truth-lp)).mean()
            loss.backward()
            grads = {n:float(p.grad.norm()) for n,p in net.named_parameters()}
            assert all(v>0 and math.isfinite(v) for v in grads.values())
            rows.append(dict(regime=regime,variant=variant,parameters=sum(p.numel() for p in net.parameters()),
                             scale_restore_relative_error=err,parameter_gradient_norms=grads))
    torch.manual_seed(11)
    original = parent.Matched(norm,'ad_raw').double()
    with torch.no_grad(): original.log_scale.fill_(1.3)
    view = inference_cancelled(original)
    original_lp, view_lp = original.log_matrix(q,k),view.log_matrix(q,k)
    error = float((original_lp.softmax(-1)-view_lp.softmax(-1)).abs().max().detach())
    assert error < 1e-12
    # In pure KL, deleting query-only scales preserves gradients on shared weights.
    reference = (q@k.transpose(-1,-2)/math.sqrt(D)).softmax(-1)
    original.zero_grad(set_to_none=True);view.zero_grad(set_to_none=True)
    (-reference*original_lp.log_softmax(-1)).mean().backward()
    (-reference*view_lp.log_softmax(-1)).mean().backward()
    grad_errors={}
    for key in ['qnet.w1','qnet.w2','knet.w1','knet.w2']:
        a=dict(original.named_parameters())[key].grad
        b=dict(view.named_parameters())[key].grad
        if key=='qnet.w2':a=a[:,:63]
        relative=float((a-b).norm()/a.norm())
        assert relative<1e-10,(key,relative)
        grad_errors[key]=relative
    save(P/'checks'/'structure.json', dict(passed=True,rows=rows,
         original_inference_cancellation_attention_max_error=error,
         original_vs_pure_deletion_KL_shared_gradient_relative_errors=grad_errors,
         scope='Synthetic FP64 structure and gradient checks only; not LLM performance or population validation.'))
    print(json.dumps(dict(passed=True,cancellation_error=error,rows=rows)),flush=True)


if __name__ == '__main__':main()
