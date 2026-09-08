"""Real frozen-LLM Q/K, unique documents and one-to-one legal Q/K training pairs."""
import hashlib
import json
import random
import time
from pathlib import Path
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2

P=Path(__file__).resolve().parent;OLD=P.parent/'real_llm_pilot';OUT=P/'data';OUT.mkdir(exist_ok=True)
CACHE=Path('/root/autodl-tmp/hf-cache');SEED=20260906
sha=lambda x:hashlib.sha256(x.encode()).hexdigest()


def main():
    torch.set_num_threads(6);random.seed(SEED);torch.manual_seed(SEED)
    old=json.loads((OLD/'data_qwen25_1p5b/manifest.json').read_text())
    labels=[[14,0],[14,6],[27,0],[27,6]];indices=[old['head_labels'].index(x) for x in labels]
    token=AutoTokenizer.from_pretrained(old['model'],cache_dir=CACHE,local_files_only=True)
    excluded={r['text_sha256'] for r in old['documents']}
    usedtokens={r['token_sha256'] for r in old['documents']}
    root=CACHE/'datasets/EleutherAI___wikitext_document_level'
    for split in ['validation','test']:
        table=Dataset.from_file(str(next(root.rglob(f'*-{split}.arrow'))))
        excluded.update(sha(row['page']) for row in table)
    parquet=CACHE/'datasets--EleutherAI--wikitext_document_level/snapshots/647234772b9554e208af6c826f23b99e3cac88c8/wikitext-103-raw-v1/wikitext-103-raw-v1-train.parquet'
    pages=pq.read_table(parquet,columns=['page']).column('page').to_pylist()
    order=list(range(len(pages)));random.Random(SEED).shuffle(order)
    selected=[]
    for start in range(0,len(order),64):
        candidates=[i for i in order[start:start+64] if sha(pages[i]) not in excluded]
        encoded=token([pages[i] for i in candidates],add_special_tokens=False,truncation=True,max_length=1024)['input_ids']
        for index,ids in zip(candidates,encoded):
            digest=sha(pages[index]);tsha=sha(json.dumps(ids))
            if len(ids)!=1024 or digest in excluded or tsha in usedtokens:continue
            excluded.add(digest);usedtokens.add(tsha)
            n=len(selected);split='validation' if n<128 else 'test' if n<384 else 'train'
            selected.append(dict(split=split,source='EleutherAI/wikitext_document_level',config='wikitext-103-raw-v1',
                source_split='train',document_index=index,text_sha256=digest,token_sha256=tsha,input_ids=ids))
            if len(selected)==4416:break # 128 validation + 256 test + 4032 added training
        if len(selected)==4416:break
    if len(selected)!=4416:raise RuntimeError(f'Not enough eligible documents: {len(selected)}')
    del pages
    manifest=dict(model=old['model'],model_revision=old['revision'],data_revision='647234772b9554e208af6c826f23b99e3cac88c8',
        seed=SEED,head_labels=labels,head_dimension=128,documents=[],shards=[],
        protocol='4096 train documents, one prefix each, 512 unique one-to-one (last-half Q, permuted first-half K) pairs per document; one epoch per run.',
        old_train_prefix=64,validation=128,test=256,source_note='New validation/test are internally held out WikiText-103 TRAIN articles; official WikiText validation/test excluded from all new subsets.')
    buffers={s:[] for s in ['train','validation','test']};records={s:[] for s in buffers};shardno={s:0 for s in buffers}
    counts={s:0 for s in buffers}
    def add_record(doc,values):
        split=doc['split'];ordinal=counts[split];counts[split]+=1
        row={k:v for k,v in doc.items() if k!='input_ids'};row['ordinal']=ordinal
        tensors=dict(input_ids=torch.tensor(doc['input_ids']),q=values[0],k=values[1])
        if split=='train':
            gen=torch.Generator().manual_seed(SEED+ordinal)
            perm=torch.randperm(512,generator=gen)
            tensors.update(q=values[0][:,512:],k=values[1][:,perm],key_positions=perm.to(torch.int16))
            row['pairs']=512;row['pairing_sha256']=sha(json.dumps(perm.tolist()))
        else:tensors['v']=values[2]
        buffers[split].append(tensors);records[split].append(row)
        if len(buffers[split])==64:flush(split)
    def flush(split):
        if not buffers[split]:return
        name=f'{split}_{shardno[split]:03d}.pt';shardno[split]+=1
        batch={key:torch.stack([x[key] for x in buffers[split]]) for key in buffers[split][0]}
        torch.save(batch,OUT/name)
        for offset,row in enumerate(records[split]):
            row.update(shard=name,offset=offset);manifest['documents'].append(row)
        manifest['shards'].append(dict(file=name,split=split,documents=len(buffers[split])))
        buffers[split].clear();records[split].clear()
        print(json.dumps(dict(event='shard',file=name,counts=counts)),flush=True)
    for doc in [r for r in old['documents'] if r['split']=='train']:
        data=torch.load(OLD/'data_qwen25_1p5b'/doc['file'],weights_only=True)
        doc=dict(doc,input_ids=data['input_ids'].tolist(),config='wikitext-2-raw-v1')
        add_record(doc,[data[k][indices] for k in ['q','k','v']])
    captured={};checks=[];original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa']
    def capture(module,query,key,value,attention_mask,**kwargs):
        output=original(module,query,key,value,attention_mask,**kwargs)
        if module.layer_idx in [14,27]:
            captured[module.layer_idx]=(query[:,[0,6]].detach().cpu(),key[:,[0,1]].detach().cpu(),value[:,[0,1]].detach().cpu())
            if len(checks)<2:
                q=query[:,[0,6],-16:].float();k=key[:,[0,1]].float();v=value[:,[0,1]].float()
                mask=torch.arange(1024,device='cuda')[None,:]<=torch.arange(1008,1024,device='cuda')[:,None]
                manual=(q@k.transpose(-1,-2)*kwargs['scaling']).masked_fill(~mask,-torch.inf).softmax(-1)@v
                actual=output[0][:,-16:,[0,6]].transpose(1,2).float()
                checks.append(dict(layer=module.layer_idx,relative_l2=float((manual-actual).norm()/actual.norm())))
        return output
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
    print(json.dumps(dict(event='model_loading',new_documents=len(selected))),flush=True)
    model=AutoModelForCausalLM.from_pretrained(old['model'],revision=old['revision'],cache_dir=CACHE,
        local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    start=time.perf_counter()
    with torch.inference_mode():
        for begin in range(0,len(selected),4):
            docs=selected[begin:begin+4];captured.clear()
            ids=torch.tensor([d['input_ids'] for d in docs],device='cuda')
            model.model(input_ids=ids,use_cache=False)
            allvalues=[torch.cat([captured[l][j] for l in [14,27]],1) for j in range(3)]
            for b,doc in enumerate(docs):add_record(doc,[x[b] for x in allvalues])
    for split in buffers:flush(split)
    manifest.update(counts=counts,numerical_checks=checks,extraction_seconds=time.perf_counter()-start)
    assert counts==dict(train=4096,validation=128,test=256)
    assert len({r['text_sha256'] for r in manifest['documents']})==4480
    assert len({r['token_sha256'] for r in manifest['documents']})==4480
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(event='complete',counts=counts,checks=checks,seconds=manifest['extraction_seconds'])),flush=True)


if __name__=='__main__':main()
