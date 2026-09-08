from aggregate import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    s=read(P/'results/summary.json');tr={r['group']:r for r in s['training']}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for ax,init,title in zip(axes,['standard','matched_zero'],['Same random weight tensors','Same initial normalized attention']):
        for kind,color in [('ad','#1565c0'),('exp','#d95f02')]:
            r=tr[f'{init}_{kind}'];a=np.asarray(r['seed_mean_head_curves'])
            ax.plot(r['steps'],a.mean(0),label=kind.upper(),color=color)
            ax.fill_between(r['steps'],a.min(0),a.max(0),color=color,alpha=.2)
        ax.set(title=title,xlabel='Unique training documents (one pass)',ylabel='Causal attention KL (last 512 docs)')
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(P/'figures/training_curves.png',dpi=180);fig.savefig(P/'figures/training_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,metric,title in zip(axes,['kl','nll'],['8k attention KL: EXP - AD','8k model NLL/token: EXP - AD']):
        rows=[next(r for r in s['comparisons'] if r['split']=='confirm_long' and r['initialization']==init and r['metric']==metric) for init in ['standard','matched_zero']]
        for i,r in enumerate(rows):
            lo,hi=r['conditional_document_ci95'];mean=r['exp_minus_ad']
            ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color='#1565c0',capsize=5)
            ax.scatter([i-.12,i,i+.12],r['seed_differences'],s=15,color='#d95f02',alpha=.8)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks([0,1],['Random weights','Matched attention'])
        ax.set_title(title);ax.set_ylabel('Positive favors AD');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Bars: conditional document 95% CI; orange: three fixed seed effects',fontsize=10)
    fig.tight_layout();fig.savefig(P/'figures/attribution_effects.png',dpi=180);fig.savefig(P/'figures/attribution_effects.pdf');plt.close(fig)

if __name__=='__main__':main()
