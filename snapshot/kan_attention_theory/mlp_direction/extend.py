"""Sequential GPU stages; selection is validation-only and recorded before confirmation."""
import subprocess
from core import *

def run(script,*args):
    command=[sys.executable,str(P/script),*map(str,args)]
    print(json.dumps(dict(event='stage',command=command)),flush=True)
    subprocess.run(command,check=True)

rows={kind:json.loads((P/'fits'/f'{kind}_m64_s11_n4096.json').read_text()) for kind in ['raw','balanced','half','logcosh']}
scores={kind:sum(r['validation']['summary']['output_nmse'])/4 for kind,r in rows.items()}
winner=min(scores,key=scores.get)
save(P/'results/selection.json',dict(winner=winner,criterion='Mean across four heads and 128 validation documents of output NMSE; no test evaluation.',scores=scores,
    initial_methods=list(rows),primary_seeds=[11,29,47],ranks=[16,32,64,96,128],fixed_total_rank=256,
    primary_confirmations=['raw','balanced',winner],confirmation_scope='128 fresh Wikipedia documents and 96 distinct FDA files; checkpoints frozen before confirmation.',
    novelty_update='Degrees of Freedom for Linear Attention (NeurIPS 2025) already studies distribution-aware dimension allocation. Allocation alone is not claimed novel.'))
run('train.py','--kinds',*sorted(set(['raw','balanced',winner])),'--seeds',29,47)
run('train.py','--kinds',winner,'--ranks',16,32,96,128)
run('spectrum.py')
