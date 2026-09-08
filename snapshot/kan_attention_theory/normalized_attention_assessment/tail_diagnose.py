"""Exploratory follow-up: locate ratio-sensitive KL errors in normalized attention."""
from common_eval import *

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ds=source.data('confirm_long')
    nets={model_name(g,s):load_model(model_name(g,s))[0] for g in GROUPS for s in SEEDS}
    output={name:[] for name in nets}
    for i in range(len(ds['q'])):
        q=ds['q'][i].cuda().double();k=ds['k'][i].cuda().double()[KI.cuda()]
        pos=ds['query_positions'][i].cuda()
        mask=(torch.arange(k.shape[1],device='cuda')[None]<=pos[:,None])[None]
        lt=(q@k.transpose(-1,-2)/math.sqrt(D)).masked_fill(~mask,-torch.inf).log_softmax(-1)
        a=lt.exp()
        for name,net in nets.items():
            lp=net.log_matrix(q,k).masked_fill(~mask,-torch.inf).log_softmax(-1)
            b=lp.exp();ratio=(lt-lp).masked_fill(~mask,0)
            rows=[]
            for cutoff in [0.,math.log(100),math.log(10000),20.]:
                selected=(ratio>cutoff)&mask
                rows.append(dict(log_ratio_threshold=cutoff,
                    teacher_mass=(a*selected).sum(-1).mean(-1).tolist(),
                    prediction_mass=(b*selected).sum(-1).mean(-1).tolist(),
                    positive_kl=(a*ratio*selected).sum(-1).mean(-1).tolist(),
                    keys=selected.sum(-1).double().mean(-1).tolist()))
            output[name].append(dict(ordinal=i,thresholds=rows,
                positive_kl=(a*ratio.clamp_min(0)).sum(-1).mean(-1).tolist(),
                negative_kl=(a*ratio.clamp_max(0)).sum(-1).mean(-1).tolist()))
        print('tail',i+1,len(ds['q']),flush=True)
    save(P/'results/tail_diagnostics.json',dict(models=output,
        scope='Post-hoc normalized probability-ratio diagnostic on existing8k cases; no fitting, no model selection. Positive KL contributions do not sum to full KL without negative contributions.'))

if __name__=='__main__':main()
