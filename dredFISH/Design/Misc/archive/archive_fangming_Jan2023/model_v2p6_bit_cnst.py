"""
NN trained on gene expression data across sources to learn cell type while separate NN trained
so that data source outputs are indistinguishable 
"""
import os
import json
from tqdm import tqdm
import logging
import numpy as np
import torch
import torch.nn as nn

from dredFISH.Utils import basicu

class InstNrmSimple(nn.Module):
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
    def __init__(self, 
        scale=1e4, #=1.5e4, 
        min_sgnl=None,
        max_sgnl=None,
        noise=None #(1e4, 1e3),
        ):
        super().__init__()
        self.logscale = torch.tensor(scale).log10()
        self.logmin = torch.tensor(min_sgnl).log10()
        self.logmax = torch.tensor(max_sgnl).log10()
        self.noise = noise

    def forward(self, Z):
        """
        Forward propagation with Poisson noise added
        
        Attributes
        ----------
        X: (projected) gene count matrix
        
        Returns
        -------
        (X1-l)/self.logscale: standardized input
        lower + upper + median: quartile error 
        """
        # Poisson noise
        if self.noise is not None:
            Z = Z + torch.poisson(self.noise[0]*torch.ones_like(Z) + self.noise[1]*torch.randn_like(Z))
        Zlog= Z.log10()

        # # penalty 1 (low or high overall -- pushed to oneside)
        # lo = (Zlog - self.logmin).clamp(0) # above low;  0 if below
        # hi = (self.logmax - Zlog).clamp(0) # below high; 0 if above
        # bit_cnst = torch.minimum(lo, hi).mean()
        
        # # penalty 2 (lower vs higher halves bits -- dimmer and brighter bits)
        # o = Zlog.sort(1)[0] # sort across cols/bits per row/cell. [0] - val; [1] - indices
        # a = o[:, :o.shape[1]//2] # smaller half
        # b = o[:, o.shape[1]//2:] # bigger half
        # lo = (a - self.logmin).clamp(0).mean() # above low;  0 if below
        # hi = (self.logmax - b).clamp(0).mean() # below high; 0 if above
        # bit_cnst = lo + hi 

        # penalty 3 (lower vs higher halves cells (in batch))
        o = Zlog.sort(0)[0] # sort across rows/cells per col/bit. [0] - val; [1] - indices
        hi_p = o.shape[0]//10 #4
        lo_p = o.shape[0]//4 #4
        a = o[     : lo_p, :] # smaller portion
        b = o[-hi_p:     , :] # bigger portion
        # l = o[ nint:-nint, :] # middle portion
        lo = (a - self.logmin).clamp(0).mean() # above low;  0 if below
        hi = (self.logmax - b).clamp(0).mean() # below high; 0 if above
        bit_cnst = lo + hi 

        # norm
        Zn = ((Zlog-self.logscale)/self.logscale).tanh()
        return Zn, bit_cnst 
    
class CellTypeNet(nn.Module):
    """
    Neural network for learning bits for encoder probes 
    
    Attributes
    ----------
        n_gns: number of genes
        n_cat: number of categories
        scale: normalizing factor
        n_bit: number of bits for probe set 
        mxpr: max expression
        drprt: dropout proportion
    """
    def __init__(self, n_gns, n_cat,  
                 gsubidx=None,
                 cnstrnts_idx=None,
                 cnstrnts=None,
                 n_rcn_layers=2,
                 n_bit=14, mxpr=9e4, 
                 drprt=0, 
                 lmd0=1e-10,
                 lmd1=0, # binarize
                 lmd2=1, # gene constraints
                 lmd3=1, # sparsity constraint
                 scale=0, ## 1.5e4, 
                 min_sgnl=None,
                 max_sgnl=None,
                 noise=None #(0,0), #(1e4, 1e3),
                 ):
        super().__init__()
        
        # filename {pooling type} {noise type} {max expression} {min position} {number of bits} {dropout} {penalty factors}
        self.name= '-'.join([str(i) for i in ('xxx', 'xxx', mxpr, 'xxx', n_bit, drprt, 'xxx', 'xxx', 'xxx')])
        self.scale = scale
        self.drprt = drprt
        self.n_rcn_layers=n_rcn_layers

        self.cnstrnts_idx = cnstrnts_idx 
        self.cnstrnts     = cnstrnts

        self.lmd0= lmd0
        self.lmd1= lmd1
        self.lmd2= lmd2
        self.lmd3= lmd3
        self.mxpr= mxpr

        self.n_cat= n_cat
        self.n_gns= n_gns

        self.gsubidx= gsubidx
        if self.gsubidx is not None and isinstance(self.gsubidx, torch.Tensor):
            n_gsub = len(self.gsubidx)
        else:
            n_gsub=0
        self.n_gsub = n_gsub

        # encoder
        self.enc= nn.Embedding(n_gns, n_bit)

        # decoder 
        self.dcd= nn.Embedding(n_bit, n_cat)

        # dropout
        self.drp= nn.Dropout(drprt)

        # transformation of data with objectives 
        self.nrm= InstNrmSimple(
            scale=scale, 
            min_sgnl=min_sgnl,
            max_sgnl=max_sgnl,
            noise=noise,
            )

        # decoder -- genes
        # self.rcn = nn.Embedding(n_bit, n_gns)
        if self.n_gsub == 0:
            self.n_rcn_layers = 0
        else:
            n_mid = max(self.n_gsub//2, n_bit)
            if self.n_rcn_layers == 0:
                pass
            elif self.n_rcn_layers == 1:
                self.rcn = nn.Sequential(
                    nn.Linear(n_bit, n_gsub), 
                    nn.ReLU(), 
                    )
            elif self.n_rcn_layers == 2:
                self.rcn = nn.Sequential(
                    nn.Linear(n_bit, n_mid), 
                    nn.ReLU(), 
                    nn.Linear(n_mid, n_gsub),
                    nn.ReLU(), 
                    )
            elif self.n_rcn_layers == 3:
                self.rcn = nn.Sequential(
                    nn.Linear(n_bit, n_mid), 
                    nn.ReLU(), 
                    nn.Linear(n_mid, n_mid), 
                    nn.ReLU(), 
                    nn.Linear(n_mid, n_gsub),
                    nn.ReLU(), 
                    )

    
    
    def get_encmat(self, rnd=False):
        """
        """
        wts= self.enc.weight.exp()
        prx= wts / wts.sum() * self.mxpr
        if rnd: prx= prx.round()
        return prx
    
    def get_prj(self, X, rnd=False):
        """
        X->Z (in-situ measured intensity)

        """
        prx = self.get_encmat(rnd=rnd)
        prj = X.mm(prx)

        # prj= X.mm(self.drp(prx)) # dropout applied to encoding probe set (gene by basis)
        if self.drprt > 0:
            prj = self.drp(prj)  # dropout applied to projected matrix   (cell by basis) 1/(1-p) rescaled 
            prj = prj + 1 # this will avoid causing downstream error when log normalizing it
            # prj = prj + 0.1*self.scale # this will avoid causing downstream error when log normalizing it
        return prj

    def get_emb(self, X, rnd=False):
        """
        X->Z->Z' (in-situ measured; then normed)

        Finds tanh non-linear transform of data, essentially evaluting nodes at layer 1 and returns margin error
        
        Attributes
        ---------- 
            X: gene count matrix for some subset of cells 
            rnd: whether to round first layer gene counts 
        
        Returns
        -------
            q.tanh(): non-linear transform of normalization
        -------
        """
        prj = self.get_prj(X, rnd=rnd)
        q, bit_cnst = self.nrm(prj)
        return q, bit_cnst

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
        """
        emb, bit_cnst = self.get_emb(X, rnd)
        if self.n_rcn_layers > 0: 
            Xrcn = self.rcn(emb)
        else:
            Xrcn = None
        fine = emb.mm(self.dcd.weight)
        return fine, Xrcn, emb, bit_cnst 
    
    def proc_batch(self, ftrs, clsts, device, libsize_norm=True):
        """
        Get ftrs and clsts from dataloader
        """
        # get data
        if libsize_norm:
            ftrs = basicu.libsize_norm(ftrs.float(), scale=1e6).to(device) # CPM
        else:
            ftrs = ftrs.float().to(device) # all features
        clsts = clsts.long().to(device)
        if self.n_rcn_layers > 0: # output features 
            ftrs_gsub = (ftrs[:,self.gsubidx]+1).log10() # log(x+1) norm
        else:
            ftrs_gsub = None
        ftrs = ftrs[:,self.cnstrnts_idx] # input features (do not use genes with unknown constraints)

        return ftrs, clsts, ftrs_gsub 

    def fit(self, dataloader, test_dataloader, device, lr=1e-1, n_iter=None, disable_tqdm=True, libsize_norm=True):
        """
        Train NN on gene counts to predict cell type label at two levels for SM2 and 10X data
        where we do not want to learn features specific to one data type. 
        
        Attributes
        ---------- 
            dataloader: pytorch dataloader instance
            lr: learning rate
        
        Returns
        -------
            learning_crvs: performance over validation set
        """
        learning_crvs= {}
        if self.n_rcn_layers > 0: 
            optimizer_gen= torch.optim.Adam(
                  list(self.enc.parameters()) 
                + list(self.dcd.parameters())
                + list(self.rcn.parameters())
                ,
                lr= lr)
        else:
            optimizer_gen= torch.optim.Adam(
                  list(self.enc.parameters()) 
                + list(self.dcd.parameters())
                ,
                lr= lr)

        self.train()
        for i,(ftrs, clsts) in tqdm(enumerate(dataloader), disable=disable_tqdm):
            if n_iter and i > n_iter:
                break

            # get data
            ftrs, clsts, ftrs_gsub = self.proc_batch(ftrs, clsts, device, libsize_norm=libsize_norm)

            # logistics 
            if i == 0:
                batch_size = len(ftrs)
                # report freq
                if n_iter:
                    en_iter = n_iter
                else:
                    en_iter = int(len(dataloader.dataset)/batch_size)
                report_freq = np.clip(en_iter//10, 1, 100) 

            # forward propagation and get predicted labels 
            optimizer_gen.zero_grad()
            
            plgt_fine, ftrs_rcn, emb, bit_cnst = self.forward(ftrs)
            ctg_lss = nn.CrossEntropyLoss()(plgt_fine, clsts)

            if self.n_rcn_layers > 0:
                rcn_lss = nn.MSELoss()(ftrs_rcn, ftrs_gsub)
            else:
                rcn_lss = torch.tensor(0) # 

            ## overall loss adds up
            # categorical
            loss_gen = ctg_lss 
            # recon (if any)
            if self.n_rcn_layers > 0:
                loss_gen = loss_gen + rcn_lss*self.lmd0 
            # brightness binarization 
            if self.lmd1 > 0:
                loss_gen = loss_gen + bit_cnst*self.lmd1 
            # gene constraints
            if self.lmd2:
                prx = self.get_encmat(rnd=False)
                row_cnst = ((prx.sum(1) - self.cnstrnts).clamp(0)**2).mean() # number of probes per gene
                loss_gen = loss_gen + row_cnst*self.lmd2
            # sparsity of encoding matrix
            if self.lmd3:
                if not self.lmd2:
                    prx = self.get_encmat(rnd=False)
                sparsity_cnst = prx.sqrt().mean() # ~ 1/2 norm
                loss_gen = loss_gen + sparsity_cnst*self.lmd3
            else:
                sparsity_cnst = torch.tensor(0)

            loss_gen.backward()
            optimizer_gen.step()
            
            # add validation results to learning curve every 10%
            if not i%report_freq:
                # training stats
                prds_fine = plgt_fine.max(1)[1]
                fine_acc = (prds_fine == clsts).float().mean()

                # eval mode
                self.eval()
                with torch.no_grad():
                    # validation dataset
                    ftrs, clsts = next(iter(test_dataloader))
                    # get data
                    ftrs, clsts, ftrs_gsub = self.proc_batch(ftrs, clsts, device, libsize_norm=libsize_norm)

                    # categorical loss/metrics
                    plgt_fine, ftrs_rcn, emb, bit_cnst = self.forward(ftrs, rnd=True)
                    prds_fine = plgt_fine.max(1)[1]
                    ctg_lss_eval = nn.CrossEntropyLoss()(plgt_fine, clsts)
                    fine_acc_eval = (prds_fine == clsts).float().mean()

                    # recon loss
                    if self.n_rcn_layers > 0:
                        rcn_lss_eval = nn.MSELoss()(ftrs_rcn, ftrs_gsub)
                    else:
                        rcn_lss_eval = torch.tensor(0) # 

                    if i == 0:
                        logging.info(f'|ttl, [cnstrnts:] bit, gene, sparsity, (trn vs tst) ctg, rcn, ctg_acc')
                    logging.info(
                        f' {i*batch_size:>5d}/{len(dataloader.dataset):>5d} | '
                        f'{loss_gen.item():.1E}, [' 
                        f'{self.lmd1*bit_cnst.item():.1E}, ' 
                        f'{self.lmd2*row_cnst.item():.1E}, ' 
                        f'{self.lmd3*sparsity_cnst.item():.1E}] ('

                        f'{ctg_lss.item():.1E}, ' 
                        f'{ctg_lss_eval.item():.1E}) ('

                        f'{rcn_lss.item():.1E}, ' 
                        f'{rcn_lss_eval.item():.1E}) ('

                        f'{fine_acc.item():.1E}, ' 
                        f'{fine_acc_eval.item():.1E})'
                        )
                learning_crvs[i]= {  
                                    'ttl': loss_gen.item(),
                                    'bit_cnst': self.lmd1*bit_cnst.item(),
                                    'row_cnst': self.lmd2*row_cnst.item(),
                                    'sparsity_cnst': self.lmd3*sparsity_cnst.item(),

                                    'rcn_lss': rcn_lss.item(),
                                    'ctg_lss': ctg_lss.item(),
                                    'fine_acc': fine_acc.item(),

                                    'rcn_lss_eval': rcn_lss_eval.item(),
                                    'ctg_lss_eval': ctg_lss_eval.item(),
                                    'fine_acc_eval': fine_acc_eval.item(),
                                }
                self.train()
        return learning_crvs

def init_weights(m):
    """Takes a nn.Module as input; initialize its weights depending on its type

    In our case, only two basic components: Linear and Embedding
    Embedding layers were initialized by N(0,1)
    Linear layers (combined with ReLU) will be initialized below by He initialization (He et al. 2015)
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    return

def train_model(
    trn_dataloader, tst_dataloader, 
    res_path,
    gsubidx, 
    cnstrnts_idx,
    cnstrnts,
    lmd0, lmd1, lmd2, lmd3,
    n_bit=24, n_rcn_layers=2, 
    drprt=0,
    scale=1e4, 
    min_sgnl=1e3,
    max_sgnl=1e5,
    noise=None, #(1e4, 1e3),
    lr=0.1, n_epochs=2, n_iter=None, 
    path_trained_model='',
    disable_tqdm=True,
    libsize_norm=True,
    device=None,
    ):
    """
    Load some subset of data, train model, save model, performance, and encoder embedding to directory
    
    Attributes
    ----------
        res_path: results directory path
    
    Returns
    -------
        model.name: the model name with parameter specifications
    """
    # set up 
    logging.basicConfig(format='%(asctime)s - %(message)s', 
                        datefmt='%m-%d %H:%M:%S', 
                        level=logging.INFO,
                        )

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU if exists
    logging.info(f'Start training on device: {device}')

    # accept numpy (but tensor is better)
    if gsubidx is not None and isinstance(gsubidx, np.ndarray): 
        gsubidx = torch.from_numpy(gsubidx) 
    gsubidx = gsubidx.to(device)

    if cnstrnts_idx is not None and isinstance(cnstrnts_idx, np.ndarray): 
        cnstrnts_idx = torch.from_numpy(cnstrnts_idx) 
    cnstrnts_idx = cnstrnts_idx.to(device)

    if cnstrnts is not None and isinstance(cnstrnts, np.ndarray): 
        cnstrnts = torch.from_numpy(cnstrnts) 
    cnstrnts = cnstrnts.to(device)

    if cnstrnts is not None:
        n_gns = len(cnstrnts)
    else:
        n_gns = trn_dataloader.dataset.X.shape[1] # number of genes

    n_cat = len(trn_dataloader.dataset.Ycat) # number of clusters

    # specify the model
    model= CellTypeNet(n_gns=           n_gns,                      
                       n_cat=           n_cat,                      
                       gsubidx=         gsubidx,
                       cnstrnts_idx=    cnstrnts_idx,
                       cnstrnts=        cnstrnts,
                       n_rcn_layers=    n_rcn_layers,
                       n_bit=           n_bit,                      
                       lmd0=            lmd0,
                       lmd1=            lmd1,
                       lmd2=            lmd2,
                       lmd3=            lmd3,
                       drprt=           drprt,              # reasonable val (dropout applied at the gene level)
                       scale=           scale,
                       min_sgnl=        min_sgnl,
                       max_sgnl=        max_sgnl,
                       noise=           noise,
                      )

    # load pretrained model if specified
    if len(path_trained_model) and os.path.isfile(path_trained_model):
        model.load_state_dict(torch.load(path_trained_model, map_location=device))
        logging.info(f"Loaded trained model from: {path_trained_model}")
    else: # otherwise init
        model.apply(init_weights) # apply `init_weights` to `model` **recursively**.
        # pass # auto init

    # get it ready
    model = model.float()
    model.to(device)
    
    # fit data 
    results = {}
    logging.info(f'\n {model.name}')
    for i in range(n_epochs):
        logging.info(f"epoch: {i}/{n_epochs} (count from 0)======================")
        result= model.fit(trn_dataloader, tst_dataloader, device, lr=lr, n_iter=n_iter, disable_tqdm=disable_tqdm, libsize_norm=libsize_norm)
        results[i] = result

    # save results
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    # - model parameters
    torch.save(model.state_dict(), os.path.join(res_path, f'model={model.name}.pt'))
    # - result
    open(os.path.join(res_path, f'./result={model.name}.json'), 'w').write(json.dumps(results))
    # - embmat: the encoding layer
    # embmat= (model.enc.weight.exp() / model.enc.weight.exp().sum() * model.mxpr).round()
    embmat = model.get_encmat(rnd=True).detach().tolist()
    open(os.path.join(res_path, f'./embmat={model.name}.json'), 'w').write(json.dumps(embmat))

    return model.name
