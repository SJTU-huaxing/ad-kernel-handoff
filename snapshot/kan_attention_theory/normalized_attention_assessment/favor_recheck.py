import argparse
from common_eval import *

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['verify','precision','kernels','ppl'])
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    getattr(base_assess(),args.action)()
