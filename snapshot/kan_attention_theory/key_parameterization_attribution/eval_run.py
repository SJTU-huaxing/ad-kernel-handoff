import argparse
from core_attribution import *

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['all','verify','precision','kernels','ppl'])
    action=parser.parse_args().action
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    module=base_assess()
    for name in ['verify','precision','kernels','ppl'] if action=='all' else [action]:
        print(json.dumps(dict(event='begin_evaluation',action=name)),flush=True)
        getattr(module,name)()
        print(json.dumps(dict(event='end_evaluation',action=name)),flush=True)
