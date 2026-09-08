import argparse
import importlib.util
from ablation import *
spec = importlib.util.spec_from_file_location('matched_assess_parent', OLD / 'assess.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

BaseReplacement = base.Replacement


class Replacement(BaseReplacement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for net in self.maps.values():
            net.runtime_raw = True


base.P = P
base.load_fit = load_fit
base.plan = plan
base.Replacement = Replacement


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify','precision','kernels','products','ppl'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    getattr(base, args.action)()
