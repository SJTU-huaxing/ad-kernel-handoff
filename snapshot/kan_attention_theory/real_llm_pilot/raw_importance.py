"""Unbiased two-stage importance sampling for the ORIGINAL raw-kernel L2 risk."""
import math
import torch


class RawEnergySampler:
    @torch.no_grad()
    def __init__(self, records, query_start=128, mixture=.5):
        self.h,self.n,self.d=records[0]['q'].shape
        self.docs=len(records);self.query_start=query_start;self.mixture=mixture
        self.q=torch.stack([r['q'] for r in records],1)
        self.k=torch.stack([r['k'] for r in records],1)
        self.hidx=torch.arange(self.h,device='cuda')
        pos=torch.arange(query_start,self.n,device='cuda')
        mask=torch.arange(self.n,device='cuda')[None,:]<=pos[:,None]
        self.pairs_per_document=int(mask.sum())
        logrows=[];logmeans=[]
        for record in records:
            logits=record['q'][:,pos].double()@record['k'].double().transpose(-1,-2)/math.sqrt(self.d)
            logits=logits.masked_fill(~mask,-torch.inf)
            logrows.append((2*logits).logsumexp(-1))
            logmeans.append(logits.flatten(1).logsumexp(-1))
        logrows=torch.stack(logrows,1)  # H, documents, query positions
        docenergy=logrows.logsumexp(-1)
        totalenergy=docenergy.logsumexp(-1)
        self.log_scale=(.5*(totalenergy-math.log(self.docs*self.pairs_per_document))).float()
        logmean=torch.stack(logmeans,1).logsumexp(-1)-math.log(self.docs*self.pairs_per_document)
        self.amplitude=((logmean-self.log_scale.double())/2).exp().float()
        self.pdoc=((1-mixture)/self.docs+mixture*docenergy.softmax(-1)).float()
        self.pquery=((1-mixture)/(self.n-query_start)+mixture*logrows.softmax(-1)).float()
        self.row_target_energy=(logrows-2*self.log_scale.double()[:,None,None]).exp()

    def sample(self, generator, queries=32):
        doc=torch.multinomial(self.pdoc,1,replacement=True,generator=generator).squeeze(-1)
        conditional=self.pquery[self.hidx,doc]
        relpos=torch.multinomial(conditional,queries,replacement=True,generator=generator)
        positions=relpos+self.query_start
        q=self.q[self.hidx[:,None],doc[:,None],positions].float()
        k=self.k[self.hidx,doc].float()
        probability=self.pdoc[self.hidx,doc,None]*conditional.gather(-1,relpos)
        # E_proposal[sum_j error(i,j)^2 * weight] = uniform legal-pair L2 risk.
        weight=1/(self.docs*self.pairs_per_document*probability)
        return q,k,positions,weight

    @torch.no_grad()
    def verification(self):
        # Exact finite-sum check on the constant pairwise error integrand.
        counts=torch.arange(self.query_start+1,self.n+1,device='cuda').double()
        proposal=self.pdoc.double()[:,:,None]*self.pquery.double()
        weight=1/(self.docs*self.pairs_per_document*proposal)
        value=(proposal*counts[None,None,:]*weight).sum((-2,-1))
        generator=torch.Generator(device='cuda').manual_seed(721923)
        sampled=torch.multinomial(proposal.flatten(1),20000,replacement=True,generator=generator)
        integrand=(self.row_target_energy*weight).flatten(1).gather(-1,sampled)
        exact=self.row_target_energy.sum((-2,-1))/(self.docs*self.pairs_per_document)
        return dict(constant_integrand_expectation=value.tolist(),
                    max_absolute_error=float((value-1).abs().max()),
                    target_energy_exact_expectation=exact.tolist(),
                    target_energy_mc_expectation=integrand.mean(-1).tolist(),
                    target_energy_mc_standard_error=(integrand.std(-1)/math.sqrt(20000)).tolist(),
                    query_interval=[self.query_start,self.n-1],mixture=self.mixture,
                    pairs_per_document=self.pairs_per_document,
                    max_document_probability=self.pdoc.max(-1).values.tolist())
