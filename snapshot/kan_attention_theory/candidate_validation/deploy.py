"""Reuse verified true recurrent 4/336-head replacement, with new features."""
from common import *
sys.path.insert(0,str(DEP))
from runtime import AttentionReplacement

def make_runtime(original,method='teacher',seed=1009,dtype=torch.float32,optimized=True):
    if method in ['teacher','exact_split','partition','galerkin','favor_plus','favor_plus_640']:
        return AttentionReplacement(original,method,seed,dtype,optimized)
    result=AttentionReplacement(original,'teacher',seed,dtype,optimized)
    result.method=method;result.maps={14:Candidate(method,dtype,[0,1]),27:Candidate(method,dtype,[2,3])}
    return result
