# PyTorch data loader + Zarr persistent array
import os
import numpy as np
import zarr
from torch.utils.data import DataLoader

from .data_files import DATABASE


class scRNAseqDataset():
    """PyTorch interface of scRNA-seq data with Zarr storage format 

    tinydata = 0 (default; do nothing); otherwise restrict the dataset access to the first few data points.
    """
    def __init__(self, zarr_file, keyX, keyY, keyYcat, tinydata=0):
        assert os.path.isdir(zarr_file)
        assert zarr_file.endswith('.zarr')
        self.path = zarr_file
        self.data = zarr.open(self.path, mode='r')
        self.allkeys = list(self.data.keys())
        self.tinydata = int(tinydata) # debugging only -- sometimes are needed; 0 means not used

        # these are zarr persistent array (file header; not actual data in memory)
        self.X = self.data[keyX]
        self.Y = self.data[keyY]
        self.Ycat = self.data[keyYcat]
    
    def __len__(self):
        if self.tinydata > 0:
            return min(self.tinydata, len(self.X)) # data appears small

        return len(self.X)

    def __getitem__(self, idx):
        if self.tinydata > 0:
            idx = idx % self.tinydata # only first few data points

        # these are actual data retrieved (numpy array)
        x = self.X.oindex[idx,:] #.astype(float)) # expect to be 2d
        y = self.Y.oindex[idx] # .astype(int) # expect to be 1d, has to be integers [0, C)
        return x, y 

def load_Allen_data(datasetkey='', path_zarr='', keyX='counts', keyY='l3_code', keyYcat='l3_cat', batch_size=64, tinydata=0, database=DATABASE):
    """
    """
    if len(path_zarr):
        pass
    elif len(datasetkey):
        path_zarr = database[datasetkey]
    else:
        raise ValueError("select a datasetkey or zarr_filepath")

    data = scRNAseqDataset(path_zarr, keyX, keyY, keyYcat, tinydata=tinydata) 

    # PyTorch data loader handles data streaming:
    # shuffling; batch size; turn to torch.tensor dtype
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True) # dataloader.dataset gives back data
    return dataloader 