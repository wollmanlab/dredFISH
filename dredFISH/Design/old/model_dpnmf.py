
import json
import os
import pandas as pd
import numpy as np

from tqdm import tqdm
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.distributions as dist

from multiprocessing import Pool

from allen_data_iterators import DataIterCached


class InstNrm(nn.Module):
    def __init__(self, min_pos= 1e5, min_sgnl=5e4, max_sgnl= 2.5e5, scale= 1.5e4, noise= (1e4, 1e3)):
        super().__init__()
        self.scale= torch.tensor(scale).log()
        self.noise= noise
        self.median= torch.tensor(min_pos)
        self.min_sgnl= torch.tensor(min_sgnl)
        self.max_sgnl= torch.tensor(max_sgnl)

    def forward(self, X):
        if self.noise is None:
            X1= X.log()
        else:
            X1= (X + torch.poisson(self.noise[0]*torch.ones_like(X) + self.noise[1]*torch.randn_like(X))).log()
        o= X1.sort(1)[0]
        a= o[:,:o.shape[1]//2]
        b= o[:,o.shape[1]//2:]
        l= (a[:,-1:] + b[:,:1])/2
        lower= ((self.min_sgnl - X).clamp(0)**2).mean()
        upper= ((X - self.max_sgnl).clamp(0)**2).mean()
        median= ((self.median - b.exp()).clamp(0)**2).mean()
        return (X1-l)/self.scale, lower + upper + median


class LabPool(nn.Module):
    def __init__(self, labmap, reduction='max'):
        super().__init__()
        self.labmap= labmap.unsqueeze(1)
        self.maptot= (self.labmap!=0).sum(2).float()
        self.reduction= reduction

    def forward(self, X):
        prd= self.labmap*X
        if self.reduction=='max':
            return prd.max(2)[0].t()
        elif self.reduction=='mean':
            return (prd.sum(2)/self.maptot).t()

class CellTypeNet(nn.Module):
	def __init__(self, n_gns, n_cat, lab_map, embmat, reduction= 'max', 
			min_pos= 1e5, min_sgnl= 5e4, max_sgnl= 2.5e5,
			scale=1.5e4, noise= (1e4,1e3),
			n_act= 7, n_bit=14, cnst= 'half_nrml', mxpr= 9e4, 
			drprt= 0, lmd1= 1e-8, lmd2= 1e-2, lmd3= 1):
		super().__init__()
		self.name= '-'.join([str(i) for i in (reduction, cnst, mxpr, '%.2E'%min_pos, n_bit, drprt, '%.2E'%lmd1, lmd2, lmd3)])

		self.embmat= embmat
		self.lmd1= lmd1
		self.lmd2= lmd2
		self.lmd3= lmd3
		self.mxpr= mxpr
		self.rdcn= reduction

		self.n_cat= n_cat
		self.n_gns= n_gns

		self.dcd= nn.Embedding(n_bit, n_cat)
		self.drp= nn.Dropout(drprt)
		self.nrm= InstNrm(min_pos=  min_pos, 
			min_sgnl= min_sgnl, 
			max_sgnl= max_sgnl, 
			scale= scale, 
			noise= noise) 
		self.lab= LabPool(lab_map, reduction=reduction)

	def get_emb(self, X, embmat, rnd= False):
		prx= embmat
		prj= X.mm(self.drp(prx))
		q, mrgn= self.nrm(prj)
		return q.tanh(), mrgn

	def forward(self, X, embmat, rnd=False):
		emb, mrgn= self.get_emb(X, embmat, rnd)
		fine= emb.mm(self.dcd.weight)
		coarse= self.lab(fine)
		return fine, coarse, emb, mrgn

	def fit(self, data_iter, lr= 1e-1):
		learning_crvs= {}
		optimizer_gen= torch.optim.Adam(list(self.dcd.parameters()), lr= 1e-1)
		for i,data in tqdm(enumerate(data_iter)):
			tenx_ftrs= data['tenx_ftrs']
			tenx_embmat= data['tenx_embmat']
			tenx_fine= data['tenx_fine']
			tenx_crse= data['tenx_crse']
			smrt_ftrs= data['smrt_ftrs']
			smrt_embmat= data['smrt_embmat']
			smrt_fine= data['smrt_fine']
			smrt_crse= data['smrt_crse']

			optimizer_gen.zero_grad()
			tenx_lgts_fine, tenx_lgts_crse, tenx_emb, tenx_mrg_lss= self.forward(tenx_ftrs, tenx_embmat)
			tenx_ctg_lss1= nn.CrossEntropyLoss()(tenx_lgts_fine, tenx_fine)
			tenx_ctg_lss2= nn.CrossEntropyLoss()(tenx_lgts_crse, tenx_crse)
			
			smrt_lgts_fine, smrt_lgts_crse, smrt_emb, smrt_mrg_lss= self.forward(smrt_ftrs, smrt_embmat)
			smrt_ctg_lss1= nn.CrossEntropyLoss()(smrt_lgts_fine, smrt_fine)
			smrt_ctg_lss2= nn.CrossEntropyLoss()(smrt_lgts_crse, smrt_crse)

			ctg_lss1= tenx_ctg_lss1 + smrt_ctg_lss1
			ctg_lss2= tenx_ctg_lss2 + smrt_ctg_lss2

			loss_gen= ctg_lss1 + ctg_lss2
			loss_gen.backward()
			optimizer_gen.step()

			if not i%(data_iter.n_iter//10):
				self.eval()

				data= data_iter.validation()

				tenx_fine_acc, tenx_crse_acc, tenx_mrgn_lss= {}, {}, {}
				for l in data['tenx']:
					if len(data['tenx'][l]):
						tenx_ftrs= data['tenx'][l]['tenx_ftrs']
						tenx_embmat= data['tenx'][l]['tenx_embmat']
						tenx_fine= data['tenx'][l]['tenx_fine']
						tenx_crse= data['tenx'][l]['tenx_crse']
						tenx_lgts_fine, tenx_lgts_crse, tenx_emb, tenx_mrgn= self.forward(tenx_ftrs, tenx_embmat, rnd=True)
						tenx_prds_fine= tenx_lgts_fine.max(1)[1]
						tenx_prds_crse= tenx_lgts_crse.max(1)[1]
						tenx_fine_acc[l]= (tenx_prds_fine == tenx_fine).float().mean().item()
						tenx_crse_acc[l]= (tenx_prds_crse == tenx_crse).float().mean().item()
						tenx_mrgn_lss[l]= tenx_mrgn.item()

				smrt_fine_acc, smrt_crse_acc, smrt_mrgn_lss= {}, {}, {}
				for l in data['smrt']:
					if len(data['smrt'][l]):
						smrt_ftrs= data['smrt'][l]['smrt_ftrs']
						smrt_embmat= data['smrt'][l]['smrt_embmat']
						smrt_fine= data['smrt'][l]['smrt_fine']
						smrt_crse= data['smrt'][l]['smrt_crse']
						smrt_lgts_fine, smrt_lgts_crse, smrt_emb, smrt_mrgn= self.forward(smrt_ftrs, smrt_embmat, rnd=True)
						smrt_prds_fine= smrt_lgts_fine.max(1)[1]
						smrt_prds_crse= smrt_lgts_crse.max(1)[1]
						smrt_fine_acc[l]= (smrt_prds_fine == smrt_fine).float().mean().item()
						smrt_crse_acc[l]= (smrt_prds_crse == smrt_crse).float().mean().item()
						smrt_mrgn_lss[l]= smrt_mrgn.item()

				print('\n'+self.name)
				print('%d\t'%i,'|ttl (ctg1,ctg2,row): %.2E (%.2E,%.2E)  '%(loss_gen.item(), ctg_lss1.item(), ctg_lss2.item()) +
					'|10x acc: %.3f,%.3f  '%(sum(tenx_fine_acc.values())/len(tenx_fine_acc), 
					sum(tenx_crse_acc.values())/len(tenx_crse_acc)) +
					'|smrt acc: %.3f,%.3f  '%(sum(smrt_fine_acc.values())/len(smrt_fine_acc),
					sum(smrt_crse_acc.values())/len(smrt_crse_acc))
				)

				learning_crvs[i]= {  'smrt_crse_acc': smrt_crse_acc,
					'smrt_fine_acc': smrt_fine_acc,
					'tenx_crse_acc': tenx_crse_acc,
					'tenx_fine_acc': tenx_fine_acc,
					}
				self.train()
		return learning_crvs


