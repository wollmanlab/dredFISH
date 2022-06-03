# PyTorch data loader + Zarr persistent array
import os
import numpy as np
import zarr
import torch
from torch.utils.data import DataLoader

class scRNAseqDataset():
    """PyTorch interface of scRNA-seq data with Zarr storage format 
    """
    def __init__(self, zarr_file, keyX, keyY, keyYcat):
        assert os.path.isdir(zarr_file)
        assert zarr_file.endswith('.zarr')
        self.path = zarr_file
        self.data = zarr.open(self.path, mode='r')
        self.allkeys = list(self.data.keys())

        # these are zarr persistent array (file header; not actual data in memory)
        self.X = self.data[keyX]
        self.Y = self.data[keyY]
        self.Ycat = self.data[keyYcat]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # these are actual data retrieved (numpy array)
        x = self.X.oindex[idx,:] #.astype(float)) # expect to be 2d
        y = self.Y.oindex[idx] # expect to be 1d
        return x, y 

def load_Allen_data(keyX='counts', keyY='l3_code', keyYcat='l3_cat'):
    """
    """
    zarr_file = os.path.join('/bigstore/GeneralStorage/fangming/projects/dredfish/data/rna', 
                                'scrna_ss_ctxhippo_a_exon_count_matrix_v3.zarr')
    data = scRNAseqDataset(zarr_file, keyX, keyY, keyYcat) 

    # PyTorch data loader handles data streaming:
    # shuffling; batch size; turn to torch.tensor dtype
    dataloader = DataLoader(data, batch_size=64, shuffle=True) # dataloader.dataset gives back data
    return dataloader 

def load_Allen_data_train(keyX='counts', keyY='l3_code', keyYcat='l3_cat'):
    """
    """
    zarr_file = os.path.join('/bigstore/GeneralStorage/fangming/projects/dredfish/data/rna', 
                                'scrna_ss_ctxhippo_a_exon_count_matrix_v3_train.zarr')
    data = scRNAseqDataset(zarr_file, keyX, keyY, keyYcat) 

    # PyTorch data loader handles data streaming:
    # shuffling; batch size; turn to torch.tensor dtype
    dataloader = DataLoader(data, batch_size=64, shuffle=True) # dataloader.dataset gives back data
    return dataloader 

def load_Allen_data_test(keyX='counts', keyY='l3_code', keyYcat='l3_cat'):
    """
    """
    zarr_file = os.path.join('/bigstore/GeneralStorage/fangming/projects/dredfish/data/rna', 
                                'scrna_ss_ctxhippo_a_exon_count_matrix_v3_test.zarr')
    data = scRNAseqDataset(zarr_file, keyX, keyY, keyYcat) 

    # PyTorch data loader handles data streaming:
    # shuffling; batch size; turn to torch.tensor dtype
    dataloader = DataLoader(data, batch_size=500, shuffle=True) # dataloader.dataset gives back data
    return dataloader 