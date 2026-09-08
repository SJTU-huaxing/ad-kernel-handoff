"""Same existing full-model timing protocol, matched local replacement."""
from core import *
from runtime_new import Replacement
import benchmark_model as bench
from runtime import AttentionReplacement

cases=json.loads((P/'results/scenarios.json').read_text())['cases']
def build(original,method):
    if method in ['teacher','favor_plus']:return AttentionReplacement(original,method)
    runtime=Replacement(original,cases[method]);runtime.collect_diagnostics=False
    return runtime

bench.P=P;bench.AttentionReplacement=build
sys.argv=['benchmark.py','--output','benchmark.json','--methods','teacher','favor_plus','raw_m64_s11','balanced_m64_s11','half_m64_s11',
          'kl_control_m64_s11','factorized_m64_s11','factorized_both_m64_s11','exp_control_m64_s11','allocate_validation','allocate_spectrum_row_mass']
bench.main()
