"""Predeclared validation selection; no internal/official test selection."""
from common import *

def main():
    rows=json.loads((P/'results/kernel_validation_8192.json').read_text())['results']
    eligible=[name for name in names() if name.startswith(('avg_','cone_'))]
    scores=[]
    for name in eligible:
        rr=[r for r in rows if r['method']==name]
        vals=[r['sample_normalized_idiv'] for r in rr]
        if len(vals)==4 and all(x is not None for x in vals):scores.append(dict(method=name,score=sum(vals)/4))
    scores.sort(key=lambda r:r['score']);selected=[r['method'] for r in scores[:2]]
    save(P/'results/selection.json',dict(rule='Top two continuous positive candidates by mean across four heads of validation raw generalized-KL / target mass. Fixed Monte Carlo pairs; validation documents separate from both training and final evaluation. No selection on internal or official tests.',
        candidates=scores,selected=selected,ppl_methods=selected+['vq','nn_mlp_11','nn_kan_11','nn_mulkan_11'],
        full_product_methods=selected+['vq','nn_mlp_11','nn_kan_11','nn_mulkan_11','nystrom']))
    print(json.dumps(dict(selected=selected,scores=scores)),flush=True)

if __name__=='__main__':main()
