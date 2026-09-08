"""Distribution-level operator diagnostics and a positive conditional-mean kernel.

No test-matrix refitting for the deployed kernel. Exhaustive products of held-out
marginals are streamed, never materialized as a full matrix. All arithmetic Float64.
"""
import argparse, hashlib, heapq, json, math, time
from pathlib import Path
import torch

P=Path(__file__).resolve().parent
DATA=P.parent/'single_pass_mulkan/data'
HEADS=[[14,0],[14,6],[27,0],[27,6]]
RANKS=[16,32,64,128]

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def tensor_save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(obj,path)

@torch.no_grad()
def load_data():
    manifest=json.loads((DATA/'manifest.json').read_text());gen=torch.Generator().manual_seed(20260909)
    order=torch.randperm(4096,generator=gen);is_cal=torch.zeros(4096,dtype=torch.bool);is_cal[order[:2048]]=True
    result={s:{side:[] for side in ['q','k']} for s in ['calibration','moment_train','internal']}
    ordinal=0
    for row in manifest['shards']:
        if row['split'] not in ['train','test']:continue
        b=torch.load(DATA/row['file'],weights_only=True);n=len(b['q'])
        if row['split']=='train':
            ixq=torch.stack([torch.randperm(512,generator=gen)[:32] for _ in range(n)])
            ixk=torch.stack([torch.randperm(512,generator=gen)[:32] for _ in range(n)])
            for side,ix in [('q',ixq),('k',ixk)]:
                xx=b[side].gather(2,ix[:,None,:,None].expand(-1,4,-1,128))
                for split,mask in [('calibration',is_cal[ordinal:ordinal+n]),('moment_train',~is_cal[ordinal:ordinal+n])]:
                    result[split][side].append(xx[mask])
            ordinal+=n
        else:
            for side in ['q','k']:
                x=b[side][:,:,512:] if side=='q' else b[side][:,:,:512]
                result['internal'][side].append(x)
    for split in result:
        for side in ['q','k']:
            # Prefixes cover EVERY document, adding more token positions per document.
            x=torch.cat(result[split][side]);n=x.shape[0];tokens=x.shape[2]
            perm=torch.stack([torch.randperm(tokens,generator=gen) for _ in range(n)])
            x=x.gather(2,perm[:,None,:,None].expand(-1,4,-1,128))
            result[split][side]=x.permute(1,2,0,3).flatten(1,2).contiguous().cuda()
    old=P.parent/'real_llm_pilot/data_qwen25_1p5b';om=json.loads((old/'manifest.json').read_text())
    hi=[om['head_labels'].index(h) for h in HEADS];docs=[r for r in om['documents'] if r['split']=='test']
    boxes=[torch.load(old/r['file'],weights_only=True) for r in docs]
    result['official']={}
    for side in ['q','k']:
        x=torch.stack([b[side][hi,512:] if side=='q' else b[side][hi,:512] for b in boxes],1)
        result['official'][side]=x.permute(0,2,1,3).flatten(1,2).contiguous().cuda()
    hashes={s:{r['token_sha256'] for r in manifest['documents'] if r['split']==s} for s in ['train','validation','test']}
    assert not hashes['train']&hashes['test']
    assert not hashes['train']&{r['token_sha256'] for r in docs}
    save(P/'results/protocol.json',dict(model=manifest['model'],model_revision=manifest['model_revision'],
        head_labels=HEADS,ranks=RANKS,calibration_documents=2048,moment_fit_documents=2048,
        calibration_tokens_per_doc=32,moment_fit_tokens_per_doc=32,heldout_documents=256,
        heldout_tokens_per_marginal_per_doc=512,heldout_samples_per_marginal=131072,
        exhaustive_internal_pairs_per_head=131072**2,exhaustive_internal_pairs_all_heads=4*131072**2,
        calibration_doc_ordinals=order[:2048].tolist(),moment_fit_doc_ordinals=order[2048:].tolist(),
        sampling='Uniform document weights and sampled token positions. Nested internal prefixes include all 256 documents.',
        data_note='Internal evaluation is a document-disjoint partition of WikiText103 source train; official is the previous 20-document WikiText2 test subset.',
        distinctions=['Finite empirical product operator is integrated exhaustively; it is NOT the unknown true population operator.',
          'Anchor spectrum and partitions use calibration documents only; positive conditional means use separate moment-fit documents only.',
          'Held-out Galerkin coefficients are diagnostics and spectral brackets, never the deployed feature map.',
          'No stochastic neural training, no repeated epochs, no FAVOR Gaussian-feature representation.']))
    return result

@torch.no_grad()
def build_basis(cal,n=1024,r=256):
    gen=torch.Generator().manual_seed(20260910)
    qi=torch.randperm(cal['q'].shape[1],generator=gen)[:n];ki=torch.randperm(cal['k'].shape[1],generator=gen)[:n]
    q=cal['q'][:,qi].double();k=cal['k'][:,ki].double()
    logk=q@k.transpose(-1,-2)/math.sqrt(128);scale=logk.amax((1,2))
    kernel=(logk-scale[:,None,None]).exp()
    u,s,vh=torch.linalg.svd(kernel,full_matrices=False,driver='gesvd')
    basis=dict(q=q,k=k,scale=scale,u=u[:,:,:r],v=vh.transpose(-1,-2)[:,:,:r],s=s[:,:r],n=n,r=r)
    tensor_save(P/'results/train_basis.pt',{key:value.cpu() if torch.is_tensor(value) else value for key,value in basis.items()})
    save(P/'results/anchor_spectrum.json',dict(q_indices=qi.tolist(),k_indices=ki.tolist(),
        scaled_singular_values=s.tolist(),head_log_scale=scale.tolist(),note='Training quadrature spectrum, not a population spectral certificate.'))
    return basis

@torch.no_grad()
def profiles(x,side,basis,block=4096):
    other=basis['k' if side=='q' else 'q'];factor=basis['v' if side=='q' else 'u']
    out=torch.empty(4,x.shape[1],basis['r'],device='cuda',dtype=torch.float64)
    for i in range(0,x.shape[1],block):
        k=(x[:,i:i+block].double()@other.transpose(-1,-2)/math.sqrt(128)-basis['scale'][:,None,None]).exp()
        out[:,i:i+block]=(k@factor)/math.sqrt(basis['n'])
    return out

@torch.no_grad()
def make_tree(x,leaves=256,min_leaf=32):
    # Weighted singular-function coordinates: sigma_j u_j, NOT standardized PCs.
    # Greedy split minimizes within-cell spectral-coordinate variance analytically.
    x=x[:,:64];nodes=[];active={};heap=[]
    def add(indices,parent):
        node_id=len(nodes);z=x[indices];n=len(z);record=dict(parent=parent,count=n,left=-1,right=-1,axis=-1,threshold=0.)
        nodes.append(record);active[node_id]=indices
        if n<2*min_leaf:return
        center=z.mean(0);zc=z-center;axis=int(zc.square().sum(0).argmax())
        order=torch.argsort(z[:,axis]);sorted_x=z[order,axis];zs=zc[order]
        count=torch.arange(1,n,device='cuda',dtype=torch.float64)
        prefix=zs.cumsum(0)[:-1];total=zs.sum(0)
        delta=prefix/count[:,None]-(total-prefix)/(n-count)[:,None]
        gains=count*(n-count)/n*delta.square().sum(-1)
        valid=(count>=min_leaf)&(n-count>=min_leaf)&(sorted_x[:-1]<sorted_x[1:])
        gains.masked_fill_(~valid,-torch.inf);at=int(gains.argmax());gain=float(gains[at])
        if not math.isfinite(gain) or gain<=0:return
        threshold=float((sorted_x[at]+sorted_x[at+1])/2)
        record.update(axis=axis,threshold=threshold)
        heapq.heappush(heap,(-gain,node_id,indices[order[:at+1]],indices[order[at+1:]]))
    add(torch.arange(len(x),device='cuda'),-1);states={}
    while len(active)<leaves and heap:
        _,node,left,right=heapq.heappop(heap)
        if node not in active:continue
        del active[node];nodes[node]['left']=len(nodes);add(left,node)
        nodes[node]['right']=len(nodes);add(right,node)
        if len(active) in RANKS:states[str(len(active))]=sorted(active)
    final=sorted(active)
    assert len(final)==leaves,(len(final),leaves)
    maps={}
    for m,state in states.items():
        slots={node:i for i,node in enumerate(state)};mapping=[]
        for node in final:
            while node not in slots:node=nodes[node]['parent']
            mapping.append(slots[node])
        maps[m]=mapping
    return dict(nodes=nodes,leaves=final,maps=maps,embedding_dimensions=x.shape[1])

@torch.no_grad()
def assign(x,trees):
    labels=torch.empty(4,x.shape[1],device='cuda',dtype=torch.long)
    for h,tree in enumerate(trees):
        current=torch.zeros(x.shape[1],device='cuda',dtype=torch.long)
        for i,node in enumerate(tree['nodes']):
            if node['left']<0:continue
            mask=current==i
            current[mask]=torch.where(x[h,mask,node['axis']]<=node['threshold'],node['left'],node['right'])
        convert=torch.full((len(tree['nodes']),),-1,device='cuda',dtype=torch.long)
        convert[torch.tensor(tree['leaves'],device='cuda')]=torch.arange(len(tree['leaves']),device='cuda')
        labels[h]=convert[current];assert (labels[h]>=0).all()
    return labels

@torch.no_grad()
def orthonormal_profiles(x):
    # Column scaling leaves the span unchanged and helps numerical conditioning.
    scales=x.norm(dim=1).clamp_min(1e-300)
    q,r=torch.linalg.qr(x/scales[:,None],mode='reduced')
    r=r*scales[:,None]
    error=(q.transpose(-1,-2)@q-torch.eye(q.shape[-1],device='cuda')).abs().amax().item()
    assert error<1e-8,error
    return q,r,error

@torch.no_grad()
def scan(q,k,fq,fk,lq,lk,basis,sizes,out,block=4096,r=256):
    """Exhaustive product integration, normalized only by empirical probabilities."""
    n=q.shape[1];assert n==k.shape[1] and n==max(sizes)
    oq,rq,qe=orthonormal_profiles(fq);ok,rk,ke=orthonormal_profiles(fk)
    hq=torch.nn.functional.one_hot(lq,r).double();hk=torch.nn.functional.one_hot(lk,r).double()
    augk=torch.cat([ok,hk],-1)
    accum={nn:dict(energy=torch.zeros(4,device='cuda',dtype=torch.float64),mass=torch.zeros(4,device='cuda',dtype=torch.float64),
            logmoment=torch.zeros(4,device='cuda',dtype=torch.float64),galerkin=torch.zeros(4,r,r,device='cuda',dtype=torch.float64),
            cell_sum=torch.zeros(4,r,r,device='cuda',dtype=torch.float64)) for nn in sizes}
    doc={nn:dict(energy=torch.zeros(4,256,256,device='cuda',dtype=torch.float64),
                mass=torch.zeros(4,256,256,device='cuda',dtype=torch.float64)) for nn in sizes} if n==131072 else None
    start=time.perf_counter();tiles=0
    # Every diagonal boundary is a nested sample size, so no partial-tile ambiguity.
    edges=sorted(set(range(0,n,block))|set(sizes)|{n})
    for ia,(a,b) in enumerate(zip(edges,edges[1:])):
        for c,d in zip(edges,edges[1:]):
            logits=q[:,a:b].double()@k[:,c:d].double().transpose(-1,-2)/math.sqrt(128)-basis['scale'][:,None,None]
            kernel=logits.exp();assert torch.isfinite(kernel).all()
            energy=kernel.square().sum((1,2));mass=kernel.sum((1,2));lm=(kernel*logits).sum((1,2))
            applied=kernel@augk[:,c:d]
            core=oq[:,a:b].transpose(-1,-2)@applied[:,:,:r]
            cells=hq[:,a:b].transpose(-1,-2)@applied[:,:,r:]
            for nn in sizes:
                if b<=nn and d<=nn:
                    ac=accum[nn];ac['energy']+=energy;ac['mass']+=mass;ac['logmoment']+=lm;ac['galerkin']+=core;ac['cell_sum']+=cells
            if doc is not None:
                # Vector order is token-round, document: supports document-cluster diagnostics.
                assert a%256==c%256==b%256==d%256==0
                de=kernel.square().reshape(4,(b-a)//256,256,(d-c)//256,256).sum((1,3))
                dm=kernel.reshape(4,(b-a)//256,256,(d-c)//256,256).sum((1,3))
                for nn in sizes:
                    if b<=nn and d<=nn:doc[nn]['energy']+=de;doc[nn]['mass']+=dm
            tiles+=1
        print(json.dumps(dict(event='product_scan',dataset=out.name,rows_completed=b,rows=n,tiles=tiles,seconds=time.perf_counter()-start)),flush=True)
    package=dict(accumulators={str(nn):{key:value.cpu() for key,value in rec.items()} for nn,rec in accum.items()},
        q_r=rq.cpu(),k_r=rk.cpu(),sizes=sizes,orthonormal_error=[qe,ke],
        prefix_q_gram={str(nn):(oq[:,:nn].transpose(-1,-2)@oq[:,:nn]).cpu() for nn in sizes},
        prefix_k_gram={str(nn):(ok[:,:nn].transpose(-1,-2)@ok[:,:nn]).cpu() for nn in sizes},
        q_counts={str(nn):hq[:,:nn].sum(1).cpu() for nn in sizes},k_counts={str(nn):hk[:,:nn].sum(1).cpu() for nn in sizes},
        doc_moments={str(nn):{key:value.cpu() for key,value in rec.items()} for nn,rec in doc.items()} if doc else None,
        seconds=time.perf_counter()-start,tiles=tiles)
    tensor_save(out,package)
    return package

@torch.no_grad()
def summarize_package(package,basis,trees,train=None):
    results=[];r=basis['r'];headscale=basis['scale'].cpu();s=basis['s'].cpu();na=basis['n']
    rq,rk=package['q_r'],package['k_r']
    for nn in package['sizes']:
        ac=package['accumulators'][str(nn)];energy=ac['energy']/nn**2;mass=ac['mass']/nn**2
        gq=package['prefix_q_gram'][str(nn)];gk=package['prefix_k_gram'][str(nn)]
        rows=[]
        for h in range(4):
            whiten=[];retained=[]
            for gram in [gq[h],gk[h]]:
                ev,u=torch.linalg.eigh(gram);keep=ev>ev.max()*1e-10
                whiten.append(u[:,keep]/ev[keep].sqrt()[None]);retained.append(int(keep.sum()))
            compressed=whiten[0].T@ac['galerkin'][h]@whiten[1]/nn
            singular=torch.linalg.svdvals(compressed);captured=singular.square().sum()
            residual=float((energy[h]-captured)/energy[h])
            assert residual>-1e-7,(h,nn,residual)
            # Fixed train Nyström extension: F(q) diag(n_anchor/s_anchor) G(k)^T.
            cq=rq[h].T@gq[h]@rq[h]/nn;ck=rk[h].T@gk[h]@rk[h]/nn
            cross=rq[h].T@ac['galerkin'][h]@rk[h]/nn**2
            floors={};ny={};positive={};projection={}
            for m in RANKS:
                low=singular[m:].square().sum()/energy[h]
                upper=(energy[h]-singular[:m].square().sum())/energy[h]
                floors[str(m)]=dict(lower=float(low),upper=max(0,float(upper)))
                inv=torch.where(s[h,:m]>s[h,0]*1e-12,na/s[h,:m],0.)
                hat2=((cq[:m,:m]*ck[:m,:m])*inv[:,None]*inv[None]).sum()
                crossval=(cross.diag()[:m]*inv).sum()
                risk=energy[h]-2*crossval+hat2
                ny[str(m)]=dict(nmse=float(risk/energy[h]),active_rank=int((inv>0).sum()))
                maps=[]
                for side in ['q','k']:
                    mapping=torch.tensor(trees[side][h]['maps'][str(m)])
                    maps.append(torch.nn.functional.one_hot(mapping,m).double())
                countq=package['q_counts'][str(nn)][h]@maps[0];countk=package['k_counts'][str(nn)][h]@maps[1]
                cs=maps[0].T@ac['cell_sum'][h]@maps[1];paircounts=countq[:,None]*countk[None]
                popmean=cs/paircounts.clamp_min(1)
                conditional_energy=(cs.square()/paircounts.clamp_min(1)).sum()/nn**2
                projection[str(m)]=dict(nmse=float((energy[h]-conditional_energy)/energy[h]),
                    exact_partition_mean=popmean.tolist(),cell_mass=(paircounts/nn**2).tolist())
                if train is not None:
                    coefficient=torch.tensor(train[h][str(m)],dtype=torch.float64)
                    risk=energy[h]-2*(coefficient*cs).sum()/nn**2+(coefficient.square()*paircounts).sum()/nn**2
                    gap=((coefficient-popmean).square()*paircounts).sum()/nn**2
                    identity_error=abs(float(risk-(energy[h]-conditional_energy)-gap))/max(float(energy[h]),1e-300)
                    assert identity_error<1e-8,identity_error
                    assert float(risk/energy[h])+1e-7>=float(low)
                    idiv=(ac['logmoment'][h]-(cs*coefficient.clamp_min(1e-300).log()).sum()-ac['mass'][h]+(coefficient*paircounts).sum())/ac['mass'][h]
                    positive[str(m)]=dict(nmse=float(risk/energy[h]),projection_nmse=projection[str(m)]['nmse'],
                        conditional_mean_estimation_gap=float(gap/energy[h]),idiv_per_mass=float(idiv),pythagorean_identity_error=identity_error)
            rows.append(dict(head=HEADS[h],log_raw_kernel_second_moment=float(energy[h].log()+2*headscale[h]),
                log_raw_kernel_first_moment=float(mass[h].log()+headscale[h]),compression_dimension=retained,
                compression_residual_fraction=max(0,residual),compression_singular_values=singular.tolist(),
                signed_floor_bracket=floors,train_nystrom=ny,positive=positive,partition_projection=projection))
        results.append(dict(samples_per_marginal=nn,pairs_per_head=nn**2,heads=rows))
    return results

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--block',type=int,default=4096);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    data=load_data();print(json.dumps(dict(event='data_loaded',sizes={k:v['q'].shape[1] for k,v in data.items()})),flush=True)
    if (P/'results/train_basis.pt').exists():
        basis={k:v.cuda() if torch.is_tensor(v) else v for k,v in torch.load(P/'results/train_basis.pt',weights_only=True).items()}
    else:basis=build_basis(data['calibration'])
    if (P/'results/partitions.json').exists():trees=json.loads((P/'results/partitions.json').read_text())
    else:
        trees={}
        for side in ['q','k']:
            x=profiles(data['calibration'][side],side,basis);trees[side]=[]
            for h in range(4):
                trees[side].append(make_tree(x[h]));print(json.dumps(dict(event='tree_complete',side=side,head=HEADS[h])),flush=True)
        save(P/'results/partitions.json',trees)
    del data['calibration']
    means=None
    for dataset in ['moment_train','internal','official']:
        path=P/f'results/{dataset}_moments.pt'
        if path.exists():package=torch.load(path,weights_only=True)
        else:
            fq=profiles(data[dataset]['q'],'q',basis);fk=profiles(data[dataset]['k'],'k',basis)
            lq=assign(fq,trees['q']);lk=assign(fk,trees['k'])
            tensor_save(P/f'results/{dataset}_labels.pt',dict(q=lq.cpu(),k=lk.cpu()))
            sizes=[8192,32768,131072] if dataset=='internal' else [16384,65536] if dataset=='moment_train' else [2560,10240]
            package=scan(data[dataset]['q'],data[dataset]['k'],fq,fk,lq,lk,basis,sizes,path,args.block)
            del fq,fk,lq,lk
        summary=summarize_package(package,basis,trees,means)
        save(P/f'results/{dataset}_summary.json',summary)
        if dataset=='moment_train':
            means=[{str(m):row['partition_projection'][str(m)]['exact_partition_mean'] for m in RANKS} for row in summary[-1]['heads']]
            for h in range(4):
                # Unobserved training cells use the whole training mean, never test values.
                avg=math.exp(summary[-1]['heads'][h]['log_raw_kernel_first_moment']-float(basis['scale'][h]))
                for m in RANKS:
                    coef=torch.tensor(means[h][str(m)],dtype=torch.float64);zero=coef==0;coef[zero]=avg;means[h][str(m)]=coef.tolist()
            save(P/'results/positive_kernel_coefficients.json',dict(coefficients=means,head_log_scale=basis['scale'].tolist(),
                definition='exp(head_scale) * coefficient[query_cell(q), key_cell(k)]',
                no_test_fitting=True))
        print(json.dumps(dict(event='dataset_complete',dataset=dataset,seconds=time.perf_counter()-start)),flush=True)
    save(P/'results/completion.json',dict(seconds=time.perf_counter()-start,full_internal_pairs_all_heads=4*131072**2))

if __name__=='__main__':main()
