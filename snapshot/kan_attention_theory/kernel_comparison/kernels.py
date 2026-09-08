"""Deployable m-dimensional features, with fixed training parameters.

Galerkin matrices are folded offline into landmark-to-feature projections.
Partition routing uses exact integer-valued path scores evaluated by GEMM,
avoiding a Python loop over hundreds of tree nodes for every input batch.
"""
import json, math
import torch
from rf import OP

class LandmarkFeatures:
    def __init__(self,dtype=torch.float64,device='cuda'):
        self.dtype=dtype;self.device=device
        b=torch.load(OP/'results/train_basis.pt',weights_only=True)
        self.q_anchor=b['q'].to(device=device,dtype=dtype)
        self.k_anchor=b['k'].to(device=device,dtype=dtype)
        self.scale=b['scale'].to(device=device,dtype=dtype)
        self.sqrt_scale=(self.scale/2).exp()
        self.a=b['n']
        self._training_basis=b

    def landmark_values(self,x,side):
        anchors=self.k_anchor if side=='q' else self.q_anchor
        return (x.to(self.dtype)@anchors.transpose(-1,-2)/math.sqrt(128)-self.scale[:,None,None]).exp()

class GalerkinFeatures(LandmarkFeatures):
    def __init__(self,m=64,dtype=torch.float64,device='cuda'):
        super().__init__(dtype,device);self.m=m
        models=torch.load(OP/'results/train_galerkin_kernel.pt',weights_only=True)
        for side,vec in [('q','v'),('k','u')]:
            folded=[]
            for h,model in enumerate(models):
                vectors=model['u' if side=='q' else 'v'][:,:m]*model['s'][:m].sqrt()[None]
                projection=self._training_basis[vec][h]@model['w'+side]@vectors/math.sqrt(self.a)
                folded.append(projection)
            setattr(self,'projection_'+side,torch.stack(folded).to(device=device,dtype=dtype))
        del self._training_basis

    def features(self,x,side):
        return (self.landmark_values(x,side)@getattr(self,'projection_'+side))*self.sqrt_scale[:,None,None]

class PartitionFeatures(LandmarkFeatures):
    def __init__(self,m=64,dtype=torch.float64,device='cuda'):
        super().__init__(dtype,device);self.m=m;self.embedding_dimension=64
        trees=json.loads((OP/'results/partitions.json').read_text())
        coefs=json.loads((OP/'results/positive_kernel_coefficients.json').read_text())
        self.coefficients=torch.tensor([row[str(m)] for row in coefs['coefficients']],device=device,dtype=dtype)
        for side,vec in [('q','v'),('k','u')]:
            setattr(self,'projection_'+side,(self._training_basis[vec][:,:,:64]/math.sqrt(self.a)).to(device=device,dtype=dtype))
            axes=[];thresholds=[];paths=[];rights=[];depths=[]
            for tree in trees[side]:
                ns=tree['nodes'][:2*m-1]
                internal=[i for i,n in enumerate(ns) if 0<=n['left']<len(ns)]
                leaves=[i for i in range(len(ns)) if i not in internal]
                assert len(internal)==m-1 and len(leaves)==m
                slots={node:j for j,node in enumerate(internal)}
                matrix=torch.zeros(m-1,m,dtype=torch.float32)
                right=torch.zeros(m,dtype=torch.float32)
                ds=[]
                for col,node in enumerate(leaves):
                    depth=0
                    while node:
                        parent=ns[node]['parent'];is_right=ns[parent]['right']==node
                        matrix[slots[parent],col]=1 if is_right else -1
                        right[col]+=int(is_right);depth+=1;node=parent
                    ds.append(depth)
                axes.append([ns[i]['axis'] for i in internal]);thresholds.append([ns[i]['threshold'] for i in internal])
                paths.append(matrix);rights.append(right);depths.append(max(ds))
            setattr(self,'axes_'+side,torch.tensor(axes,device=device,dtype=torch.long))
            setattr(self,'thresholds_'+side,torch.tensor(thresholds,device=device,dtype=dtype))
            setattr(self,'paths_'+side,torch.stack(paths).to(device))
            setattr(self,'rights_'+side,torch.stack(rights).to(device))
            setattr(self,'depths_'+side,depths)
        del self._training_basis

    def cells(self,x,side):
        coords=self.landmark_values(x,side)@getattr(self,'projection_'+side)
        axes=getattr(self,'axes_'+side)[:,None].expand(-1,x.shape[1],-1)
        decisions=(coords.gather(-1,axes)>getattr(self,'thresholds_'+side)[:,None]).float()
        scores=decisions@getattr(self,'paths_'+side)-getattr(self,'rights_'+side)[:,None]
        return scores.argmax(-1)

    def features(self,x,side):
        cells=self.cells(x,side)
        if side=='q':f=torch.nn.functional.one_hot(cells,self.m).to(self.dtype)
        else:f=self.coefficients.transpose(-1,-2).gather(1,cells[:,:,None].expand(-1,-1,self.m))
        return f*self.sqrt_scale[:,None,None]

def linear_attention(fq,fk,v):
    """Rectangular/all-visible attention; no epsilon or sign repair."""
    state=fk.transpose(-1,-2)@v
    denominator=(fq*fk.sum(1)[:,None]).sum(-1)
    return (fq@state)/denominator[:,:,None],denominator
