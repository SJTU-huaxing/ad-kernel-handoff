"""Identical old end-to-end protocol, new candidate adapters."""
from common import *
from deploy import make_runtime
import benchmark_model

if __name__=='__main__':
    selected=json.loads((P/'results/selection.json').read_text())['ppl_methods']
    benchmark_model.P=P;benchmark_model.AttentionReplacement=make_runtime
    if len(sys.argv)==1:sys.argv=['benchmark_model','--output','model_benchmark.json','--methods','teacher','partition','favor_plus',*selected]
    benchmark_model.main()
