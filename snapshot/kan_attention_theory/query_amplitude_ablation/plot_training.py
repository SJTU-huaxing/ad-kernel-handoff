import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
OLD=P.parent/'hedgehog_matched'
plan=json.loads((P/'results/plan.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
labels={'ad':'Original AD','reduced_matched':'Remove Q amplitude + C, matched','reduced_plain':'Remove Q amplitude + C, fewer params','hh':'Hedgehog-exp'}
colors={'ad':'#2667a8','reduced_matched':'#c34d36','reduced_plain':'#d69b33','hh':'#458450'}
for ax,regime,title in zip(axes,['product_i','causal_i','causal_kl'],['Product: raw I-divergence','Causal: balanced I-divergence','Causal: attention KL']):
    for group in labels:
        values=[]
        for seed in [11,29,47]:
            if group in ['ad','hh']:
                prefix='product_' if regime=='product_i' else ''
                objective='kl' if regime=='causal_kl' else 'raw'
                path=OLD/'fits'/f'{prefix}{group}_{objective}_s{seed}_lr0.002.json'
            else:
                lr=plan['learning_rates'][f'{regime}__{group}']
                path=P/'fits'/f'{regime}_{group}_s{seed}_lr{lr:g}.json'
            history=json.loads(path.read_text())['history']
            x=np.array([r['step'] for r in history])
            values.append([np.mean(r['per_head_loss']) for r in history])
        a=np.asarray(values)
        ax.plot(x,a.mean(0),color=colors[group],label=labels[group],linewidth=2,linestyle='--' if group=='reduced_plain' else '-')
        ax.fill_between(x,a.min(0),a.max(0),color=colors[group],alpha=.12)
    ax.set_title(title);ax.set_xlabel('Unique query documents processed');ax.set_yscale('log');ax.grid(alpha=.2)
axes[0].set_ylabel('Training objective, mean over each 512-doc block')
handles,legend=axes[0].get_legend_handles_labels()
fig.legend(handles,legend,loc='outside lower center',ncol=2,frameon=False)
fig.suptitle('One-pass training on frozen Qwen2.5-1.5B Q/K; 24 heads, 3 seeds (shaded range)',fontsize=12)
fig.savefig(P/'figures/training_curves.png',dpi=170)
fig.savefig(P/'figures/training_curves.pdf')
print('saved training curves')
