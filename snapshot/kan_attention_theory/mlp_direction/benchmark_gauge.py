from core import *
from runtime_new import Replacement
import benchmark_model as bench
from runtime import AttentionReplacement

cases=json.loads((P/'results/gauge_plan.json').read_text())['cases']
def build(original,method):
    if method in ['teacher','favor_plus']:return AttentionReplacement(original,method)
    runtime=Replacement(original,cases[method]);runtime.collect_diagnostics=False
    return runtime
bench.P=P;bench.AttentionReplacement=build
sys.argv=['benchmark_gauge.py','--output','benchmark_gauge.json','--methods','teacher','favor_plus','factorized_both_m64_s11','exp_control_m64_s11','gauge_calibrated_m64_s11']
bench.main()
