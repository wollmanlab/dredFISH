"""
NN trained on gene expression data across sources to learn cell type while separate NN trained
so that data source outputs are indistinguishable 
"""
from dis import dis
from faulthandler import disable
import os
import json
from tqdm import tqdm
from itertools import product
import logging
import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self, scale=1.5e4, noise=(1e4, 1e3)):
        super().__init__()
        self.logscale= torch.tensor(scale).log()
        self.noise= noise

    def forward(self, X):
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
        if self.noise is None:
            X1= X.log()
        else:
            X1= (X + torch.poisson(self.noise[0]*torch.ones_like(X) + self.noise[1]*torch.randn_like(X))).log()

        # # median
        # l = X1.median(1, keepdim=True)[0]
        # # norm
        # X1 = (X1-l)/self.logscale

        # norm
        X1 = (X1-self.logscale)/self.logscale
        X1 = X1.tanh()
        return X1 
    
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
                 n_rcn_layers=2,
                 n_bit=14, mxpr=9e4, 
                 drprt=0, lmd0=1e-10,
                 scale=1.5e4, noise=(1e4, 1e3),
                 ):
        super().__init__()
        
        # filename {pooling type} {noise type} {max expression} {min position} {number of bits} {dropout} {penalty factors}
        self.name= '-'.join([str(i) for i in ('xxx', 'xxx', mxpr, 'xxx', n_bit, drprt, 'xxx', 'xxx', 'xxx')])
        self.n_rcn_layers=n_rcn_layers

        self.lmd0= lmd0
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

        # dropout
        self.drp= nn.Dropout(drprt)
        # transformation of data with objectives 
        self.nrm= InstNrmSimple(scale=scale, noise=noise)
    
    def get_prj(self, X, rnd=False):
        """
        X->Z (in-situ measured intensity)

        """
        wts= self.enc.weight.exp()
        prx= wts / wts.sum() * self.mxpr
        if rnd: prx= prx.round()
        prj= X.mm(self.drp(prx))
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
        q = self.nrm(prj)
        return q 

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
        emb = self.get_emb(X, rnd)
        if self.n_rcn_layers > 0: 
            Xrcn = self.rcn(emb)
        else:
            Xrcn = None
        fine = emb.mm(self.dcd.weight)
        return fine, Xrcn, emb

    def fit(self, dataloader, test_dataloader, device, lr=1e-1, n_iter=None, disable_tqdm=True):
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

            # list(self.nrm.parameters()) + 

        self.train()
        for i,(ftrs, clsts) in tqdm(enumerate(dataloader), disable=disable_tqdm):
            if n_iter and i > n_iter:
                break

            # get data
            ftrs= ftrs.float().to(device)
            clsts= clsts.long().to(device)
            ftrs_gsub= (ftrs[:,self.gsubidx]+1).log() # log(x+1) norm

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
            
            plgt_fine, ftrs_rcn, emb = self.forward(ftrs)
            if self.n_rcn_layers > 0:
                rcn_lss = nn.MSELoss()(ftrs_rcn, ftrs_gsub)
            else:
                rcn_lss = torch.tensor(0) # 
            ctg_lss = nn.CrossEntropyLoss()(plgt_fine, clsts)

            # overall loss
            if self.n_rcn_layers > 0:
                loss_gen= ctg_lss + rcn_lss*self.lmd0 
            else:
                loss_gen= ctg_lss 

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
                    ftrs= ftrs.float().to(device)
                    clsts= clsts.long().to(device)
                    ftrs_gsub= (ftrs[:,self.gsubidx]+1).log() # log(x+1) norm

                    plgt_fine, ftrs_rcn, emb = self.forward(ftrs, rnd=True)
                    prds_fine = plgt_fine.max(1)[1]
                    if self.n_rcn_layers > 0:
                        rcn_lss_eval = nn.MSELoss()(ftrs_rcn, ftrs_gsub)
                    else:
                        rcn_lss_eval = torch.tensor(0) # 
                    ctg_lss_eval = nn.CrossEntropyLoss()(plgt_fine, clsts)
                    fine_acc_eval = (prds_fine == clsts).float().mean()

                    if i == 0:
                        logging.info(f'|ttl, trn vs tst: (ctg, rcn, ctg_acc): ')
                    logging.info(
                        f' {i*batch_size:>5d}/{len(dataloader.dataset):>5d} | '
                        f'{loss_gen.item():.2E} (' 

                        f'{ctg_lss.item():.2E}, ' 
                        f'{ctg_lss_eval.item():.2E}) ('

                        f'{rcn_lss.item():.2E}, ' 
                        f'{rcn_lss_eval.item():.2E}) ('

                        f'{fine_acc.item():.2E}, ' 
                        f'{fine_acc_eval.item():.2E})'
                        )
                learning_crvs[i]= {  
                                    'ttl': loss_gen.item(),

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


def train_model(trn_dataloader, tst_dataloader, gsubidx, res_path, lmd0, 
    n_bit=24, n_rcn_layers=2, 
    scale=1.5e4, noise=(1e4, 1e3),
    lr=0.1, n_epochs=2, n_iter=2000,
    path_trained_model='',
    disable_tqdm=True,
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU if exists
    logging.info(f'Start training on device: {device}')

    n_gns = trn_dataloader.dataset.X.shape[1] # number of genes
    n_cat = len(trn_dataloader.dataset.Ycat) # number of clusters
    # cnstrnts = torch.tensor(trn_dataloader.dataset.data['num_probe_limit']).to(device)
    if gsubidx is not None and isinstance(gsubidx, np.ndarray): 
        gsubidx = torch.from_numpy(gsubidx) 
        gsubidx = gsubidx.to(device)

    # specify the model
    model= CellTypeNet(n_gns=           n_gns,                      
                       n_cat=           n_cat,                      
                       gsubidx=         gsubidx,
                       n_rcn_layers=    n_rcn_layers,
                       n_bit=           n_bit,                      
                       lmd0=            lmd0,
                       drprt=           0,              # reasonable val (dropout applied at the gene level)
                       scale=scale, noise=noise,
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
        result= model.fit(trn_dataloader, tst_dataloader, device, lr=lr, n_iter=n_iter, disable_tqdm=disable_tqdm)
        results[i] = result

    # save results
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    # - model parameters
    torch.save(model.state_dict(), os.path.join(res_path, f'model={model.name}.pt'))
    # - result
    open(os.path.join(res_path, f'./result={model.name}.json'), 'w').write(json.dumps(results))
    # - embmat: the encoding layer
    embmat= (model.enc.weight.exp() / model.enc.weight.exp().sum() * model.mxpr).round().detach().tolist()
    open(os.path.join(res_path, f'./embmat={model.name}.json'), 'w').write(json.dumps(embmat))

    return model.name
