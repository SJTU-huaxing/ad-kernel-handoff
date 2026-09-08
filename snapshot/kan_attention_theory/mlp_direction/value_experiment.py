"""Exploratory output-sensitive raw-kernel divergence, fixed before new holdouts."""
import argparse
from core import *

@torch.inference_mode()
def extract():
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen2 import modeling_qwen2
    manifest=json.loads((DATA/'manifest.json').read_text())
    original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];captured={}
    def capture(module,q,k,v,mask,**kwargs):
        if module.layer_idx in [14,27]:captured[module.layer_idx]=[q[:,[0,6]].cpu(),k.cpu(),v.cpu()]
        return original(module,q,k,v,mask,**kwargs)
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
    model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['model_revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    folder=P/'values';folder.mkdir(exist_ok=True);checks=[];start=time.perf_counter()
    for r in manifest['shards']:
        if r['split']!='train':continue
        path=folder/r['file']
        if path.exists():continue
        b=torch.load(DATA/r['file'],weights_only=True);vs=[]
        for at in range(0,len(b['q']),4):
            ids=b['input_ids'][at:at+4].cuda();model.model(input_ids=ids,use_cache=False)
            q,k,v=[torch.cat([captured[l][j] for l in [14,27]],1) for j in range(3)]
            ix=b['key_positions'][at:at+4].long()[:,None,:,None].expand(-1,4,-1,128)
            v=v.gather(2,ix);vs.append(v)
            if at==0:
                qo=b['q'][at:at+4].float();ko=b['k'][at:at+4].float()
                q=q[:,:,512:].float();k=k.gather(2,ix).float()
                checks.append(dict(shard=r['file'],q_relative_l2=float((q-qo).norm()/qo.norm()),k_relative_l2=float((k-ko).norm()/ko.norm())))
        torch.save(dict(v=torch.cat(vs)),path)
        if len(checks)%8==0:print(json.dumps(dict(event='values',shards=len(checks),seconds=time.perf_counter()-start)),flush=True)
    save(P/'checks/value_extraction.json',dict(checks=checks,seconds=time.perf_counter()-start,alignment='Existing stored key_positions applied to separately captured V from identical frozen model and input IDs.'))
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

def value_weights(logt,v):
    p=logt.softmax(-1);y=p@v
    norm2=(v.square().sum(-1)[:,None]+y.square().sum(-1)[:,:,None]-2*y@v.transpose(-1,-2)).clamp_min(0)
    w=norm2.sqrt();mean=(p*w).sum(-1,keepdim=True)
    # Strictly positive floor keeps the raw target identified even for v_j=y.
    return w+.05*mean+1e-6

def fit():
    from train import fit as ordinary_fit
    manifest,data,norm,scale=load_train();val=load_full('validation')
    values=torch.cat([torch.load(P/'values'/r['file'],weights_only=True)['v'] for r in manifest['shards'] if r['split']=='train']).cuda()
    for seed in [11,29,47]:
        kind='value';m=64;n=4096;name=f'{kind}_m{m}_s{seed}_n{n}';path=P/'fits'/f'{name}.json'
        if path.exists():continue
        torch.manual_seed(seed);model=Pair(norm,m).cuda()
        order=torch.randperm(n,generator=torch.Generator().manual_seed(20260907+seed)).tolist()
        qgen=torch.Generator().manual_seed(20260908+seed)
        queries=torch.stack([torch.randperm(512,generator=qgen)[:64] for _ in range(n)]).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,n,eta_min=.0002)
        start=time.perf_counter();history=[];running=torch.zeros(4,device='cuda')
        for step,i in enumerate(order,1):
            q=data['q'][i,:,queries[i]].float();k=data['k'][i].float();v=values[i].float()
            target=q@k.transpose(-1,-2)/math.sqrt(128)-scale[:,None,None]
            with torch.no_grad():lw=value_weights(target,v).log()
            pred=model.log_matrix(q,k)
            perhead=loss_rows(pred+lw,target+lw,'balanced').mean(-1)
            assert torch.isfinite(perhead).all()
            opt.zero_grad(set_to_none=True);perhead.mean().backward()
            sums=torch.zeros(4,device='cuda')
            for p in model.parameters():sums+=p.grad.flatten(1).square().sum(-1)
            factor=(10/sums.sqrt().clamp_min(1e-20)).clamp_max(1)
            for p in model.parameters():p.grad.mul_(factor.reshape(4,*([1]*(p.ndim-1))))
            opt.step();sched.step();running+=perhead.detach()
            if step%512==0:
                row=dict(step=step,mean_loss=(running/512).tolist(),seconds=time.perf_counter()-start);running.zero_();history.append(row)
                print(json.dumps(dict(event='train',name=name,**row)),flush=True)
        seconds=time.perf_counter()-start
        meta=dict(name=name,kind=kind,m=m,seed=seed,training_documents=n,epochs=1,hidden_width=192,parameters_per_head=sum(p.numel() for p in model.parameters())//4,
            training_pairs_per_head=n*64*512,unique_query_vectors_per_head=n*64,
            document_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),query_indices_sha256=hashlib.sha256(queries.cpu().numpy().tobytes()).hexdigest(),
            log_scale=scale.tolist(),lr=.002,weight_decay=1e-4,training_seconds=seconds,
            target='Original raw exponential kernel, with strictly positive output-sensitive divergence weights; V used only to construct training weights.',
            weights='w_j=||v_j-y_teacher||_2 + .05 E_teacher||v-y_teacher||_2 + 1e-6; divide weighted I-divergence by sum_j w_j*kappa_j',
            selection='Exploratory additional construction after four-loss validation screen, before any fresh/FDA evaluation; final single-pass checkpoint.')
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},metadata=meta),P/'fits'/f'{name}.pt')
        model.double().eval();metrics=evaluate(model,val,scale)
        save(path,dict(metadata=meta,history=history,validation=metrics))
        print(json.dumps(dict(event='complete',name=name,validation=metrics['summary'])),flush=True)
    # Matched architecture/data/steps control using the published attention-weight
    # distillation objective. This is explicitly NOT a raw-kernel primary method.
    del values
    for seed in [11,29,47]:ordinary_fit(data,norm,scale,val,'kl_control',64,seed,4096)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['extract','fit']);a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    (extract if a.action=='extract' else fit)()
