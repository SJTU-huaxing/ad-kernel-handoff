"""Build a private, read-only research handoff; never edit original experiments."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parent/'kan_attention_theory'
SNAP=P/'snapshot'/'kan_attention_theory'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False))

def main():
    P.mkdir(exist_ok=True);SNAP.mkdir(parents=True,exist_ok=True)
    chosen=[]
    for seed in [11,29,47]:
        chosen.append(f'query_amplitude_ablation/fits/causal_kl_reduced_plain_s{seed}_lr0.002.pt')
        for group in ['standard_exp','matched_zero_ad','matched_zero_exp']:
            chosen.append(f'key_parameterization_attribution/fits/{group}_s{seed}_lr0.002.pt')
        for name in [f'hh_kl_s{seed}_lr0.002',f'hh_softmax_kl_s{seed}_lr0.002']:
            chosen.append(f'hedgehog_matched/fits/{name}.pt')
        chosen.append(f'causal_direction/fits/favor_s{seed}.pt')
    chosen=set(chosen)
    ledger=[];docs=[];copied=[];large=[]
    for f in sorted(ROOT.rglob('*')):
        if not f.is_file():continue
        rel=f.relative_to(ROOT)
        if any(s in rel.parts for s in ['.git','__pycache__','flash-linear-attention']):continue
        size=f.stat().st_size
        external_paper=any(s in rel.parts for s in ['sources','papers'])
        is_source=f.suffix in ['.py','.md','.yaml','.yml','.toml','.txt'] and not external_paper
        is_figure=f.suffix in ['.png','.pdf','.svg'] and not external_paper
        is_small_metric=f.suffix in ['.json','.csv'] and size<=262144 and not external_paper
        # Full final summaries/checks and configuration JSONs are included even if larger.
        important=f.suffix=='.json' and (f.name in ['summary.json','plan.json','frozen_plan.json','selection.json'] or 'checks' in rel.parts) and not external_paper
        include=is_source or is_figure or is_small_metric or important
        row=dict(path=str(rel),bytes=size,in_document_bundle=include,in_checkpoint_bundle=str(rel) in chosen)
        if include:
            dst=SNAP/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
            row['sha256']=sha(f);copied.append(row)
            if f.suffix=='.md':docs.append(row)
        if str(rel) in chosen:
            dst=P/'checkpoints'/'kan_attention_theory'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
            row['sha256']=sha(f)
        if rel.parts[0]=='causal_direction' and len(rel.parts)>1 and rel.parts[1]=='data':
            row['sha256']=sha(f);large.append(row.copy())
        ledger.append(row)
    assert len([r for r in ledger if r['in_checkpoint_bundle']])==21
    write(P/'ARTIFACT_INVENTORY.json',dict(original_root=str(ROOT),files=ledger,
        exclusions='Third-party FLA tree, .git and __pycache__ omitted from inventory; paper fulltext not redistributed; large tensors and most per-document metric arrays external. All original project reports/theories/protocols are copied verbatim.'))
    write(P/'EXTERNAL_DATA_MANIFEST.json',dict(data_root='kan_attention_theory/causal_direction/data',files=large,
        bytes=sum(r['bytes'] for r in large),scope='Exact final-mainline cached activations plus input manifest; transfer separately to reproduce NVIDIA-extracted QKV. Regenerating on NPU is a separate extraction with rounding differences.'))
    index=['项目原文索引。每份原文保持原相对目录、原内容和SHA256。正文阶段概述不替代原始逐head/逐种子表。','',
        '| 原项目文件 | 字节 | SHA256 |','|---|---:|---|']
    alltext=['历史报告、协议与理论全文备份。以下按相对路径排序，不将文件顺序解释为实验时间。已被后续纠正的旧结论仍原样保存，当前有效结论以主交接文档为准。',
        '\n本合订本中的历史相对链接仍以各段标注的原文件目录为基准，直接点击可能无法定位；请从[SOURCE_INDEX.zh.md](SOURCE_INDEX.zh.md)打开保留原目录的独立文件。大型外部资产未全部附带，存在于原文的链接不代表对应资产已打包。独立快照的原文保持逐字节一致，新增解释只位于本合订本开头。']
    for r in docs:
        target='snapshot/kan_attention_theory/'+r['path']
        index.append(f'| [{r["path"]}]({target}) | {r["bytes"]} | `{r["sha256"]}` |')
        alltext.append('\n\n---\n\n原文件：'+r['path']+'\n\n'+(ROOT/r['path']).read_text())
    (P/'SOURCE_INDEX.zh.md').write_text('\n'.join(index)+'\n')
    (P/'HISTORICAL_REPORTS_FULL.zh.md').write_text('\n'.join(alltext))
    write(P/'SNAPSHOT_SUMMARY.json',dict(project_documents=len(docs),copied_files=len(copied),
        copied_bytes=sum(r['bytes'] for r in copied),checkpoint_count=21,
        checkpoint_bytes=sum(r['bytes'] for r in ledger if r['in_checkpoint_bundle']),
        external_final_data_bytes=sum(r['bytes'] for r in large)))
    print(json.dumps(json.loads((P/'SNAPSHOT_SUMMARY.json').read_text())),flush=True)

if __name__=='__main__':main()
