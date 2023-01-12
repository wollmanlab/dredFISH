"""
NN trained on gene expression data across sources to learn cell type while separate NN trained
so that data source outputs are indistinguishable 
"""
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
    """
    Performs normalization on projection with thresholding by value  
    
    Attributes
    ----------
        min_pos: minimum position  
        min_sgnl: minimum signal
        max_sgnl: maximum signal
        scale: scaling factors
        noise: range of noises for Poisson parameter
    """
    def __init__(self, min_pos= 1e5, min_sgnl=5e4, max_sgnl= 2.5e5, scale= 1.5e4, noise= (1e4, 1e3)):
        super().__init__()
        self.scale= torch.tensor(scale).log()
        self.noise= noise
        self.median= torch.tensor(min_pos) # median intensity -- as each cell has half +; it is the "minimum pos" as well 
        self.min_sgnl= torch.tensor(min_sgnl)
        self.max_sgnl= torch.tensor(max_sgnl)

    def forward(self, X):
        """
        Forward propagation with Poisson noise added
        
        Attributes
        ----------
        X: (projected) gene count matrix
        
        Returns
        -------
        (X1-l)/self.scale: standardized input
        lower + upper + median: quartile error 
        """
        
        # Poisson noise
        if self.noise is None:
            X1= X.log()
        else:
            X1= (X + torch.poisson(self.noise[0]*torch.ones_like(X) + self.noise[1]*torch.randn_like(X))).log()
        # each coarse level cell type will have a median expression value, which is the difference between the last low value
        # and the first high value 
        o= X1.sort(1)[0] # sort by bits (by col). [0] - val; [1] - indices
        a= o[:,:o.shape[1]//2] # smaller half
        b= o[:,o.shape[1]//2:] # bigger half
        l= (a[:,-1:] + b[:,:1])/2 # middle values for each cell
        
        # lower and upper are bounds on expression, we want counts within their threshold
        lower= ((self.min_sgnl - X).clamp(0)**2).mean() # X lower than min
        upper= ((X - self.max_sgnl).clamp(0)**2).mean() # X larger than max
        median= ((self.median - b.exp()).clamp(0)**2).mean() # larger half lower than median  
        return (X1-l)/self.scale, lower + upper + median

class CellTypeNet(nn.Module):
    """
    Neural network for learning bits for encoder probes 
    
    Attributes
    ----------
        n_gns: number of genes
        n_cat: number of categories
        lab_map: map from fine to coarse labels 
        reduction: max vs mean pooling
        min_pos: minimum position, difference from top gene counts 
        min_sgnl: minimum signal, difference to gene counts 
        max_sgnl: maximum signal, difference from gene counts 
        scale: normalizing factor
        noise: Poisson noise factors 
        n_act: number of nodes for discriminator
        n_bit: number of bits for probe set 
        cnst: not used, possibly the noise type 
        mxpr: max expression
        drprt: dropout proportion
        lmd1: penalty factor on margin loss (projection range: min, max, median) 
        lmd2: penalty factor on per-gene probe constraint
        lmd3: penalty factor on discriminator (10X vs SMART-seq)
    """
    def __init__(self, n_gns, n_cat, tenx_label, smrt_label, reduction= 'max', 
                 min_pos= 1e5, min_sgnl= 5e4, max_sgnl= 2.5e5,
                 # adjusted min and max signal thresholds to see how this would affect accuracy 
                 # min_pos= 1e5, min_sgnl= 5 * 5e4, max_sgnl= 5 * 2.5e5,
                 # min_pos= 1e5, min_sgnl= 10 * 5e4, max_sgnl= 10 * 2.5e5,
                 scale=1.5e4, noise= (1e4,1e3),
                 n_act= 7, n_bit=14, cnst= 'half_nrml', mxpr= 9e4, 
                 drprt= 0, lmd1= 1e-8, lmd2= 1e-2, lmd3= 1):
        super().__init__()
        
        # filename {pooling type} {noise type} {max expression} {min position} {number of bits} {dropout} {penalty factors}
        self.name= '-'.join([str(i) for i in (reduction, cnst, mxpr, '%.2E'%min_pos, n_bit, drprt, '%.2E'%lmd1, lmd2, lmd3)])
        self.tenx_label= tenx_label
        self.smrt_label= smrt_label

        self.lmd1= lmd1
        self.lmd2= lmd2
        self.lmd3= lmd3
        self.mxpr= mxpr
        self.rdcn= reduction

        self.n_cat= n_cat
        self.n_gns= n_gns

        # encoder
        self.enc= nn.Embedding(n_gns, n_bit)
        # decoder 
        self.dcd= nn.Embedding(n_bit, n_cat)
        # dropout
        self.drp= nn.Dropout(p=drprt)
        # transformation of data with objectives 
        self.nrm= InstNrm(min_pos=  min_pos, 
                          min_sgnl= min_sgnl, 
                          max_sgnl= max_sgnl, 
                          scale= scale, 
                          noise= noise) 

        # discriminator
        self.dsc= nn.Sequential(nn.Linear(n_bit, n_act), 
                                nn.Softplus(), # smooth version of ReLU 
                                nn.Linear(n_act, 1))

    def get_emb(self, X, rnd=False):
        """
        Finds tanh non-linear transform of data, essentially evaluting nodes at layer 1 and returns margin error
        
        Attributes
        ---------- 
            X: gene count matrix for some subset of cells 
            rnd: whether to round first layer gene counts 
        
        Returns
        -------
            q.tanh(): non-linear transform of normalization
            mrgn: margin error 
        -------
        """
        wts= self.enc.weight.exp()
        prx= wts / wts.sum() * self.mxpr
        if rnd: prx= prx.round()
        prj= X.mm(self.drp(prx))
        q, mrgn= self.nrm(prj)
        return q.tanh(), mrgn

    def forward(self, X, rnd=False):
        """
        Get labels at both levels and non-linear transform 
        
        Attributes
        ---------- 
            X: gene count matrix for some subset of cells 
            rnd: whether to round first layer gene counts 
        
        Returns
        -------
            fine: fine-grained cell type labels
            coarse: coarse-grained cell type labels
            emb: embedding of weights from first layer of network
            mrgn: margin error
        """
        emb, mrgn= self.get_emb(X, rnd)
        fine= emb.mm(self.dcd.weight)
        # coarse= self.lab(fine)
        return fine, emb, mrgn

    def fit(self, data_iter, device, lr= 1e-1):
        """
        Train NN on gene counts to predict cell type label at two levels for SM2 and 10X data
        where we do not want to learn features specific to one data type. 
        
        Attributes
        ---------- 
            data_iter: data iterable
            lr: learning rate
        
        Returns
        -------
            learning_crvs: performance over validation set
        """
        learning_crvs= {}
        optimizer_gen= torch.optim.Adam(list(self.enc.parameters()) + 
                                        list(self.dcd.parameters()) +
                                        list(self.nrm.parameters()), 
                                        lr= 1e-1)
        optimizer_dsc= torch.optim.Adam(self.dsc.parameters(), lr= lr)
        self.train() # get in the training mode
        for i,data in tqdm(enumerate(data_iter)):
            tenx_ftrs= data['tenx_ftrs'].to(device)
            tenx_fine= data[self.tenx_label].to(device)
            smrt_ftrs= data['smrt_ftrs'].to(device)
            smrt_fine= data[self.smrt_label].to(device)
            cnstrnts = data['cnstrnts'].to(device)

            # forward propagation and get predicted labels 
            optimizer_gen.zero_grad()
            tenx_lgts_fine, tenx_emb, tenx_mrg_lss= self.forward(tenx_ftrs)
            tenx_ctg_lss1= nn.CrossEntropyLoss()(tenx_lgts_fine, tenx_fine)
            tenx_gen_lss= 0 if not self.lmd3 else nn.BCEWithLogitsLoss()(self.dsc(tenx_emb), torch.ones(tenx_emb.shape[0], 1).to(device))
            
            smrt_lgts_fine, smrt_emb, smrt_mrg_lss= self.forward(smrt_ftrs)
            smrt_ctg_lss1= nn.CrossEntropyLoss()(smrt_lgts_fine, smrt_fine)
            smrt_gen_lss= 0 if not self.lmd3 else nn.BCEWithLogitsLoss()(self.dsc(smrt_emb), torch.zeros(smrt_emb.shape[0], 1).to(device))

            # fine loss 
            ctg_lss1= tenx_ctg_lss1 + smrt_ctg_lss1

            # forward loss
            mrg_lss= tenx_mrg_lss + smrt_mrg_lss
            # discriminator loss 
            gen_lss= tenx_gen_lss + smrt_gen_lss
            
            # subtract the 10X expected number of transcripts per gene as another loss value 
            # should not be expected number of transcripts(?), 
            # but the prx should not exceed the number of available probes per gene.
            if self.lmd2:
                wts= self.enc.weight.exp()
                prx= wts / wts.sum() * self.mxpr
                row_cnst= ((prx.sum(1) - cnstrnts).clamp(0)**2).mean() # number of probes per gene

            # overall loss
            loss_gen= (ctg_lss1) + mrg_lss*self.lmd1 + row_cnst*self.lmd2 + gen_lss*self.lmd3
            loss_gen.backward()
            optimizer_gen.step()

            # loss of discriminator 
            optimizer_dsc.zero_grad()
            loss_dsc= nn.BCEWithLogitsLoss()(torch.cat([self.dsc(smrt_emb.detach()),
                                                        self.dsc(tenx_emb.detach())]),
                                             torch.cat([torch.ones(smrt_emb.shape[0],1),
                                                        torch.zeros(tenx_emb.shape[0],1)]).to(device))
            # update weights with back propagation 
            loss_dsc.backward()
            optimizer_dsc.step()
            
            # add validation results to learning curve every 10%
            if not i%(np.clip(data_iter.n_iter//10, 1, None)):
                self.eval() # tell the model its now not training (for dropout and stuff)
                with torch.no_grad():
                    data= data_iter.validation()

                    tenx_fine_acc, tenx_mrgn_lss= {}, {}
                    for l in data['tenx']:
                        if len(data['tenx'][l]):
                            tenx_ftrs= data['tenx'][l]['tenx_ftrs'].to(device)
                            tenx_fine= data['tenx'][l][self.tenx_label].to(device)
                            tenx_lgts_fine, tenx_emb, tenx_mrgn= self.forward(tenx_ftrs, rnd=True)
                            tenx_prds_fine= tenx_lgts_fine.max(1)[1]
                            tenx_fine_acc[l]= (tenx_prds_fine == tenx_fine).float().mean().item()
                            tenx_mrgn_lss[l]= tenx_mrgn.item()

                    smrt_fine_acc, smrt_crse_acc, smrt_mrgn_lss= {}, {}, {}
                    for l in data['smrt']:
                        if len(data['smrt'][l]):
                            smrt_ftrs= data['smrt'][l]['smrt_ftrs'].to(device)
                            smrt_fine= data['smrt'][l][self.smrt_label].to(device)
                            smrt_lgts_fine, smrt_emb, smrt_mrgn= self.forward(smrt_ftrs, rnd=True)
                            smrt_prds_fine= smrt_lgts_fine.max(1)[1]
                            smrt_fine_acc[l]= (smrt_prds_fine == smrt_fine).float().mean().item()
                            smrt_mrgn_lss[l]= smrt_mrgn.item()

                    print('\n'+self.name)
                    print(f'{i}\t'
                        '|ttl (ctg1, row, mrg): '
                        f'{loss_gen.item():.2E} ({ctg_lss1.item():.2E}, {row_cnst.item():.2E},{mrg_lss.item():.2E}) '
                        f'|10x acc: {sum(tenx_fine_acc.values())/len(tenx_fine_acc):.3f}  '
                        f'|smrt acc: {sum(smrt_fine_acc.values())/len(smrt_fine_acc):.3f}  '
                        f'|dsc: {((loss_dsc / nn.BCELoss()(torch.ones(1)/2, torch.ones(1))).item()):.3f}'
                        )

                    learning_crvs[i]= {  
                                        'smrt_fine_acc': smrt_fine_acc,
                                        'smrt_mrgn_lss': smrt_mrgn_lss,
                                        'tenx_fine_acc': tenx_fine_acc,
                                        'tenx_mrgn_lss': tenx_mrgn_lss,
                                        'dsc_lss': (loss_dsc / nn.BCELoss()(torch.ones(1)/2, torch.ones(1))).item(),
                                        'row_cnst': row_cnst.item()
                                    }
                self.train() # set back to training mode
        return learning_crvs


def train_model(res_path, lmd1, min_pos, tenx_label, smrt_label, n_iter=2000, drprt=0):
    """
    Load some subset of data, train model, save model, performance, and encoder embedding to directory
    
    Attributes
    ----------
        res_path: results directory path
        lmd1: penalty factor on margin loss
        min_pos: lower bound on number of positions 
    
    Returns
    -------
        model.name: the model name with parameter specifications
    """
    # to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_iter= DataIterCached(n_iter=n_iter)
    model= CellTypeNet(n_gns=     data_iter.current['tenx_ftrs'].shape[1],                      # fixed
                       n_cat=     data_iter.labl_map.shape[1],                                  # fixed
                       tenx_label=tenx_label,
                       smrt_label=smrt_label,
                    #    lab_map=   data_iter.labl_map.to(device),                                           # reasonable val
                       cnst=      'half_nrml',                                                  # reasonable val
                       reduction= 'max',                                                        # reasonable val
                       n_bit=     24,                                                           # fixed
                       min_pos=   min_pos,                                                      # -- tune
                       lmd1=      lmd1,                                                         # -- tune
                       lmd2=      1e-2,                                                         # reasonable val
                       lmd3=      1e-0,                                                         # reasonable val
                       drprt=     drprt,                                                             # reasonable val (dropout applied at the gene level)
                      )
    model.to(device)
    
    # fit model; get results
    result= model.fit(data_iter, device)
    # get the encoding layer
    embmat= (model.enc.weight.exp() / model.enc.weight.exp().sum() * model.mxpr).round().detach().tolist()

    # save results
    # - model parameters
    # - result
    # - embmat
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    torch.save(model.state_dict(), os.path.join(res_path, 'model=%s.pt'%(model.name)))
    open(os.path.join(res_path, './result=%s.json'%model.name), 'w').write(json.dumps(result))
    open(os.path.join(res_path, './embmat=%s.json'%model.name), 'w').write(json.dumps(embmat))

    return model.name
