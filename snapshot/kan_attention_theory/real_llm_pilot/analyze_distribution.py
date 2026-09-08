"""Distribution diagnostics and empirical rank floors on legal attention blocks."""

import argparse
import json
import math
import time
from pathlib import Path

import torch


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def covariance(x):
    centered = x - x.mean(-2, keepdim=True)
    return centered.transpose(-1, -2) @ centered / (x.shape[-2] - 1)


def matrix_sqrt(c):
    values, vectors = torch.linalg.eigh(c)
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)


def spectrum(logits, ranks):
    raw = (logits - logits.amax((-2, -1), keepdim=True)).exp()
    normalized = logits.softmax(-1)
    result = []
    for name, matrix in [('raw', raw), ('row_normalized', normalized)]:
        try:
            singular = torch.linalg.svdvals(matrix.float(), driver='gesvdj').double()
        except RuntimeError as error:
            if 'converge' not in str(error):
                raise
            # Near rank-one rectangular kernels can defeat Jacobi iteration.
            singular = torch.linalg.svdvals(matrix.double(), driver='gesvd')
        energy = singular.square().sum(-1)
        rows = []
        for i in range(len(matrix)):
            rows.append(dict(kind=name,
                             floors={str(m):float(singular[i, m:].square().sum()/energy[i]) for m in ranks},
                             effective_rank=float(singular[i].square().sum().square()/singular[i].pow(4).sum()),
                             singular_values=(singular[i]/singular[i,0]).cpu().tolist()))
        result.append(rows)
    extra = dict(entropy=(-(normalized * normalized.clamp_min(1e-300).log()).sum(-1)).mean(-1),
                 top1=normalized.amax(-1).mean(-1),
                 first_key_mass=normalized[..., 0].mean(-1),
                 raw_max_energy_fraction=raw.square().amax((-2,-1))/raw.square().sum((-2,-1)))
    return result, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--spectral-documents', type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(8)
    torch.manual_seed(20260905)
    manifest = json.loads((args.data/'manifest.json').read_text())
    labels = manifest['head_labels']
    d = manifest['head_dimension']
    beta = d**-0.5
    train_docs = [x for x in manifest['documents'] if x['split']=='train']
    pools = {key:[] for key in ['q','k','q_norm','k_norm','paired_logit']}
    start=time.perf_counter()
    for doc in train_docs:
        data=load(args.data/doc['file'])
        length=data['q'].shape[1]
        qi=torch.randint(length//2,length,(64,))
        ki=torch.randint(0,length//2,(64,))
        ni=torch.randint(0,length,(64,))
        pools['q'].append(data['q'][:,qi].float())
        pools['k'].append(data['k'][:,ki].float())
        pools['q_norm'].append(data['q'][:,ni].float())
        pools['k_norm'].append(data['k'][:,ni].float())
        pools['paired_logit'].append((data['q'][:,qi].float()*data['k'][:,ki].float()).sum(-1)*beta)
    pools={key:torch.cat(value,1).cuda().double() for key,value in pools.items()}
    qc,kc=covariance(pools['q']),covariance(pools['k'])
    qr,kr=matrix_sqrt(qc),matrix_sqrt(kc)
    coupling=torch.linalg.svdvals(beta*qr@kr)
    statistics=[]
    for h,(layer,head) in enumerate(labels):
        row=dict(layer=layer,head=head,gaussian_max_coupling=float(coupling[h,0]),
                 gaussian_hilbert_schmidt_valid=bool(coupling[h,0]<0.5),
                 coupling_singular_values=coupling[h].cpu().tolist())
        for key,cov in [('q',qc),('k',kc)]:
            x=pools[key][h]
            centered=x-x.mean(0)
            eig=torch.linalg.eigvalsh(cov[h]).clamp_min(0).flip(0)
            variance=centered.square().mean(0)
            kurt=centered.pow(4).mean(0)/variance.clamp_min(1e-15).square()-3
            norm=x.norm(dim=-1)
            mean_direction=(x/norm[:,None].clamp_min(1e-15)).mean(0)
            row[key]=dict(mean_norm=float(x.mean(0).norm()),norm_mean=float(norm.mean()),
                          norm_p99=float(norm.quantile(.99)),
                          covariance_participation=float(eig.sum().square()/eig.square().sum()),
                          covariance_top8_fraction=float(eig[:8].sum()/eig.sum()),
                          mean_pair_cosine=float(mean_direction.square().sum()),
                          coordinate_excess_kurtosis_mean=float(kurt.mean()),
                          coordinate_excess_kurtosis_max=float(kurt.max()),
                          covariance_eigenvalues=eig.cpu().tolist())
        logits=pools['paired_logit'][h]
        row['causal_pair_logits']=dict(mean=float(logits.mean()),std=float(logits.std()),
                                      p01=float(logits.quantile(.01)),p99=float(logits.quantile(.99)),
                                      minimum=float(logits.min()),maximum=float(logits.max()))
        statistics.append(row)
    torch.save(dict(q_mean=pools['q_norm'].mean(1).cpu().float(),
                    q_std=pools['q_norm'].std(1).clamp_min(.03).cpu().float(),
                    k_mean=pools['k_norm'].mean(1).cpu().float(),
                    k_std=pools['k_norm'].std(1).clamp_min(.03).cpu().float(),
                    q_cov=qc.cpu(),k_cov=kc.cpu(),head_labels=labels),args.out/'normalization.pt')
    (args.out/'distribution.json').write_text(json.dumps(statistics,indent=2))
    print(json.dumps(dict(event='distribution',heads=len(labels),samples_per_head=pools['q'].shape[1],
                          gaussian_valid=sum(x['gaussian_hilbert_schmidt_valid'] for x in statistics))),flush=True)

    ranks=[16,32,64,128,256]
    records=[]
    for split in ['validation','test','test_long','ood']:
        docs=[x for x in manifest['documents'] if x['split']==split][:args.spectral_documents]
        for index,doc in enumerate(docs):
            data=load(args.data/doc['file'])
            length=data['q'].shape[1]
            qi=torch.arange(length-512,length)
            ki=torch.arange(512)
            q=data['q'][:,qi].cuda().double()
            k=data['k'][:,ki].cuda().double()
            logits=(q@k.transpose(-1,-2))*beta
            spectra,extra=spectrum(logits,ranks)
            for kind in spectra:
                for h,row in enumerate(kind):
                    row.update(split=split,document=doc['file'],layer=labels[h][0],head=labels[h][1],
                               source='same_context_legal_512x512_block',
                               **{key:float(value[h]) for key,value in extra.items()})
                    records.append(row)
            print(json.dumps(dict(event='spectrum',split=split,document=index+1,total=len(docs),
                                  seconds=round(time.perf_counter()-start,1))),flush=True)
    for rep in range(4):
        for mode in ['empirical_product','matched_gaussian']:
            if mode=='empirical_product':
                qi=torch.randperm(pools['q'].shape[1],device='cuda')[:512]
                ki=torch.randperm(pools['k'].shape[1],device='cuda')[:512]
                q,k=pools['q'][:,qi],pools['k'][:,ki]
            else:
                q=torch.randn(len(labels),512,d,device='cuda',dtype=torch.float64)@qr+pools['q'].mean(1)[:,None]
                k=torch.randn(len(labels),512,d,device='cuda',dtype=torch.float64)@kr+pools['k'].mean(1)[:,None]
            spectra,extra=spectrum(q@k.transpose(-1,-2)*beta,ranks)
            for kind in spectra:
                for h,row in enumerate(kind):
                    row.update(split='train_distribution_diagnostic',document=str(rep),layer=labels[h][0],
                               head=labels[h][1],source=mode,
                               **{key:float(value[h]) for key,value in extra.items()})
                    records.append(row)
    (args.out/'spectra.json').write_text(json.dumps(records,indent=2))
    summary=dict(model=manifest['model'],heads=labels,ranks=ranks,block_size=512,
                 entries=len(records),seconds=time.perf_counter()-start,
                 warnings=['Empirical matrix floors are not population certificates.',
                           'Only entirely legal rectangular blocks are decomposed, not causal-masked full matrices.',
                           'Row-normalized block kernels normalize over the first 512 keys only.',
                           'Gaussian surrogates are descriptive covariance matches, not asserted true distributions.'])
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(event='complete',**summary)),flush=True)


if __name__=='__main__':
    main()
