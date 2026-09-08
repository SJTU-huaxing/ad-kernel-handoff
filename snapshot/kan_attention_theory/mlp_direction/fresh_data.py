"""New untouched confirmation documents, selected without any method scores."""
import argparse,random
from core import *

def select(round2=False,round3=False):
    from datasets import Dataset
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    manifest=json.loads((DATA/'manifest.json').read_text())
    old=json.loads((ROOT/'real_llm_pilot/data_qwen25_1p5b/manifest.json').read_text())
    used_text={r['text_sha256'] for r in manifest['documents']+old['documents']}
    used_tokens={r['token_sha256'] for r in manifest['documents']+old['documents']}
    previous=[]
    if round2 or round3:
        previous=json.loads((P/'results/fresh_manifest.json').read_text())['records']
        if round3:previous+=json.loads((P/'results/fresh_manifest2.json').read_text())['records']
        used_text.update(r['text_sha256'] for r in previous);used_tokens.update(r['token_sha256'] for r in previous)
    tok=AutoTokenizer.from_pretrained(manifest['model'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True)
    cache=Path('/root/autodl-tmp/hf-cache')
    parquet=next((cache/'datasets--EleutherAI--wikitext_document_level').rglob('*train.parquet'))
    # Exact 103 source; the 2/103 cache can both exist.
    candidates=list((cache/'datasets--EleutherAI--wikitext_document_level').rglob('*train.parquet'))
    parquet=next(p for p in candidates if 'wikitext-103-raw-v1' in str(p))
    pages=pq.read_table(parquet,columns=['page']).column('page').to_pylist()
    sources={('fresh_wiki3' if round3 else 'fresh_wiki2' if round2 else 'fresh_wiki'):[dict(text=x,index=i,group=str(i)) for i,x in enumerate(pages)]}
    domain='hazyresearch___based-swde-v2' if round2 or round3 else 'hazyresearch___based-fda'
    other=Dataset.from_file(str(next((cache/'datasets'/domain).rglob('*validation.arrow'))))
    sources['swde3' if round3 else 'swde' if round2 else 'fda']=[dict(text=r['text'],index=i,group=r['file_name']) for i,r in enumerate(other)]
    records=[]
    for ds,rows in sources.items():
        order=list(range(len(rows)));random.Random((75091 if round3 else 74091 if round2 else 73091)+(0 if ds.startswith('fresh_wiki') else 1)).shuffle(order)
        chosen=[];groups={r['group'] for r in previous if ds.startswith('swde') and r['dataset'].startswith('swde')}
        target_count=128 if ds.startswith('fresh_wiki') else 40 if round3 else 96
        length=512 if ds.startswith('swde') else 1024
        for at in order:
            row=rows[at];digest=hashlib.sha256(row['text'].encode()).hexdigest()
            if digest in used_text or row['group'] in groups:continue
            ids=tok(row['text'],add_special_tokens=False,truncation=True,max_length=length)['input_ids']
            if len(ids)!=length:continue
            tsha=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if tsha in used_tokens:continue
            used_text.add(digest);used_tokens.add(tsha);groups.add(row['group'])
            chosen.append(dict(dataset=ds,source_index=row['index'],group=row['group'],text_sha256=digest,token_sha256=tsha,input_ids=ids))
            if len(chosen)==target_count:break
        assert len(chosen)==target_count,(ds,len(chosen))
        records.extend(chosen)
    save(P/'results'/('fresh_manifest3.json' if round3 else 'fresh_manifest2.json' if round2 else 'fresh_manifest.json'),dict(records=records,model=manifest['model'],revision=manifest['model_revision'],
        protocol=('Round 3: 128 additional unseen 1024-token Wikipedia documents and 40 distinct 512-token SWDE web source files; all disjoint from both prior confirmation rounds. Only 45 additional SWDE files meet source-file disjointness and length; target reduced before any evaluation. No score-based choice.' if round3 else 'Round 2: 128 additional unseen 1024-token Wikipedia documents and 96 distinct 512-token SWDE web documents; disjoint from all round-1 documents; no method-score selection.' if round2 else '128 unseen source-train Wikipedia documents + 96 distinct FDA source files, selected by fixed shuffled order and length/hash only; no evaluation-based choice. FDA has only 100 distinct eligible files of at least 1024 tokens; count reduced before any method evaluation.'),
        source_paths=dict(wiki=str(parquet),other=str(next((cache/'datasets'/domain).rglob('*validation.arrow'))))))
    print(json.dumps(dict(event='fresh_selected',counts={ds:sum(r['dataset']==ds for r in records) for ds in sources})),flush=True)

@torch.inference_mode()
def extract(round2=False,round3=False):
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen2 import modeling_qwen2
    manifest=json.loads((P/'results'/('fresh_manifest3.json' if round3 else 'fresh_manifest2.json' if round2 else 'fresh_manifest.json')).read_text())
    # Selection/checkpoints are fixed before fresh activations or scores exist.
    assert (P/'results/selection.json').exists()
    if round3:assert (P/'results/gauge_plan.json').exists()
    original=modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa'];captured={}
    def capture(module,q,k,v,mask,**kwargs):
        if module.layer_idx in [14,27]:captured[module.layer_idx]=[q[:,[0,6]].cpu(),k.cpu(),v.cpu()]
        return original(module,q,k,v,mask,**kwargs)
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
    model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],cache_dir='/root/autodl-tmp/hf-cache',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    for ds in (['fresh_wiki3','swde3'] if round3 else ['fresh_wiki2','swde'] if round2 else ['fresh_wiki','fda']):
        path=P/'results'/f'data_{ds}.pt'
        if path.exists():continue
        docs=[r for r in manifest['records'] if r['dataset']==ds]
        parts={s:[] for s in ['q','k','v','input_ids']}
        for begin in range(0,len(docs),4):
            ids=torch.tensor([r['input_ids'] for r in docs[begin:begin+4]],device='cuda')
            model.model(input_ids=ids,use_cache=False)
            for j,s in enumerate(['q','k','v']):parts[s].append(torch.cat([captured[l][j] for l in [14,27]],1))
            parts['input_ids'].append(ids.cpu())
        torch.save({k:torch.cat(v) for k,v in parts.items()},path)
        print(json.dumps(dict(event='fresh_extracted',dataset=ds,documents=len(docs))),flush=True)
    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['select','extract']);ap.add_argument('--round2',action='store_true');ap.add_argument('--round3',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4)
    (select if a.action=='select' else extract)(a.round2,a.round3)
