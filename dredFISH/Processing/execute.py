#!/usr/bin/env python
import argparse
from datetime import datetime
import numpy as np
import anndata
from sklearn.cluster import KMeans
import pandas as pd
import importlib
from metadata import Metadata
import multiprocessing
import sys
from tqdm import tqdm
from cellpose import models
import cv2
import torch
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="Path to folder containing Raw Data /bigstore/Images20XX/User/Project/Dataset/")
    parser.add_argument("-c","--cword_config", type=str,dest="cword_config",default='dredfish_processing_config', action='store',help="Name of Config File for analysis ie. dredfish_processing_config")
    parser.add_argument("-b", "--batches", type=int, dest="batches", default=2000, action='store', help="Number of batches")
    parser.add_argument("-n", "--ncpu", type=int, dest="ncpu", default=10, action='store', help="Number of threads")
    parser.add_argument("-nr", "--nregions", type=int, dest="nregions", default=5, action='store', help="Number of Regions/Sections")
    parser.add_argument("-o", "--outpath", type=str, dest="outpath", default='/bigstore/GeneralStorage/Data/dredFISH/', action='store', help="Path to save data")
    parser.add_argument("-r", "--resolution", type=int, dest="resolution", default=10, action='store', help="esoltuino to round centroid before naming regions")
    args = parser.parse_args()
    
class dredFISH_Position_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 cword_config,
                 verbose=False):
        
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.cword_config = cword_config
        self.verbose = verbose
        self.proceed = True
        
        self.cword_config = cword_config
        self.config = importlib.import_module(self.cword_config)
        self.parameters = self.config.parameters
        self.bitmap = self.config.bitmap
        
        self.nucstain_acq = self.parameters['nucstain_acq']
        self.total_acq = self.parameters['total_acq']
        
    def run(self):
        self.check_data()
        if self.proceed:
            self.main()
            
    def update_user(self,message):
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]
        
    def main(self):
        if self.verbose:
            self.update_user('Processing Position '+self.posname)
        self.config_data()
        self.load_data()
        self.segmentation()
        if self.cell_metadata.shape[0]>0:
            self.pull_vectors()
            self.save_data()
        
    def check_data(self):
        if self.parameters['overwrite']:
            self.proceed = True
        else:
            if self.verbose:
                self.update_user('Checking Existing Data')
            """ Check If data already generated"""
            try:
                fname = os.path.join(self.metadata_path,
                                     'fishdata',
                                     self.dataset+'_'+self.posname+'_data.h5ad')
                data = anndata.read_h5ad(fname)
            except:
                data = None
            if isinstance(data,anndata._core.anndata.AnnData):
                self.proceed = False
                if self.verbose:
                    self.update_user(self.posname+' Data Found')
            else:
                self.proceed = True
        
    def config_data(self):
        if self.verbose:
            self.update_user('Loading Metadata')
        self.image_metadata = Metadata(self.metadata_path)
        
    def load_data(self):
        """ Load Nuclei Image """
        if self.verbose:
            self.update_user('Loading Nuclei Image')
        if self.nucstain_acq=='infer':
            self.nucstain_acq = np.array([i for i in self.image_metadata.acqnames if 'nucstain' in i])[-1]
        self.nucstain_image = self.image_metadata.stkread(
            Position=self.posname,
            Channel=self.parameters['nucstain_channel'],
            acq=self.nucstain_acq).mean(2).astype(float)
        
        """ Load Raw Data """
        if self.verbose:
            self.update_user('Loading Raw Data')
        self.raw_dict = {
            hybe.split('_')[0]:torch.tensor(
                self.image_metadata.stkread(
                    Position=self.posname,
                    Channel=self.parameters['total_channel'],
                    acq=hybe).mean(2).astype(float)) for hybe in self.image_metadata.acqnames if 'hybe' in hybe}
        
        """ Load or Calculate Background """
        self.background_dict = {
            'hybe'+strip.split('strip')[1].split('_')[0]:torch.tensor(
                self.image_metadata.stkread(
                    Position=self.posname,                                                                             
                    Channel=self.parameters['total_channel'],
                    acq=strip).mean(2).astype(float)) for strip in self.image_metadata.acqnames if 'strip' in strip}

        """ Subtract Background """
        if self.verbose:
            self.update_user('Subtracting Background')
        self.signal_dict = {hybe:self.raw_dict[hybe]-self.background_dict[hybe] for readout,hybe,channel in self.bitmap}
        
        """ Load Total Image """
        if self.verbose:
            self.update_user('Loading Total Image')
        if self.total_acq=='infer':
            self.total_acq = np.array([i for i in self.image_metadata.acqnames if 'nucstain' in i])[-1]
            self.total_image = self.image_metadata.stkread(Position=self.posname,
                                                           Channel=self.parameters['total_channel'],
                                                           acq=self.total_acq).mean(2).astype(float)
            self.background_acq = np.array([i for i in self.image_metadata.acqnames if 'background' in i])[-1]
            self.background_image = self.image_metadata.stkread(Position=self.posname,
                                                           Channel=self.parameters['total_channel'],
                                                           acq=self.background_acq).mean(2).astype(float)
            self.total_image = self.total_image-self.background_image
            
        elif self.total_acq=='none':
            self.total_image = np.dstack([self.signal_dict[hybe] for readout,hybe,channel in self.bitmap]).sum(2)
        else:
            self.total_image = self.image_metadata.stkread(Position=self.posname,
                                                           Channel=self.parameters['total_channel'],
                                                           acq=self.total_acq).mean(2).astype(float)
            
    def process_image(self,image):
        bkg = gaussian_filter(image,self.parameters['segment_diameter'])
        image = image-bkg
        image[image<0] = 0
        return image
    
    def nuclei_segmentation(self):      
        """ Segment Image """
        if self.verbose:
            self.update_user('Segmenting Nuclei')
        torch.cuda.empty_cache() 
        model = models.Cellpose(model_type='nuclei',gpu=self.parameters['segment_gpu'])
        raw_mask_image,flows,styles,diams = model.eval(self.nucstain_image,
                                            diameter=self.parameters['segment_diameter'],
                                            channels=[0,0],
                                            flow_threshold=1,
                                            cellprob_threshold=0,
                                            min_size=int(self.parameters['segment_diameter']*10))
        self.nuclei_mask = torch.tensor(raw_mask_image.astype(int))
        del model,raw_mask_image,flows,styles,diams
        torch.cuda.empty_cache() 
        
    def total_segmentation(self):
        stk = np.dstack([self.total_image.copy(),
                         self.nucstain_image.copy(),
                         np.zeros_like(self.total_image.copy())])
        """ Segment Image """
        if self.verbose:
            self.update_user('Segmenting Total')
        torch.cuda.empty_cache() 
        model = models.Cellpose(model_type='cyto',gpu=self.parameters['segment_gpu'])
        raw_mask_image,flows,styles,diams = model.eval(stk,
                                            diameter=self.parameters['segment_diameter'],
                                            channels=[1,2],
                                            flow_threshold=1,
                                            cellprob_threshold=0,
                                            min_size=int(self.parameters['segment_diameter']*10))
        self.total_mask = torch.tensor(raw_mask_image.astype(int))
        del model,raw_mask_image,flows,styles,diams
        torch.cuda.empty_cache() 

    def generate_cell_metadata(self):
        if self.verbose:
            self.update_user('Generating Cell Metadata')
        labels = np.unique(self.nuclei_mask.ravel()[self.nuclei_mask.ravel()>0])
        self.label_nuclei_coord_dict = {label:torch.where(self.nuclei_mask==label) for j,label in enumerate(labels)}
        self.label_cyto_coord_dict = {label:torch.where((self.cytoplasm_mask==label)) for j,label in enumerate(labels)}
        self.label_total_coord_dict = {label:torch.where(self.total_mask==label) for j,label in enumerate(labels)}
        self.total_image = torch.tensor(self.total_image)
        if labels.shape[0]==0:
            self.cell_metadata = pd.DataFrame(index=['label',
                                                     'pixel_x',
                                                     'pixel_y',
                                                     'nuclei_size',
                                                     'nuclei_signal',
                                                     'cytoplasm_size',
                                                     'cytoplasm_signal',
                                                     'total_size',
                                                     'total_signal']).T
        else:
            cell_metadata = []
            for j,label in enumerate(labels):
                nx,ny = self.label_nuclei_coord_dict[label]
                cx,cy = self.label_cyto_coord_dict[label]
                tx,ty = self.label_total_coord_dict[label]
                label = label
                data = [float(label),float(nx.float().mean()),float(ny.float().mean()),
                        float(nx.shape[0]),float(torch.median(self.total_image[nx,ny])),
                        float(cx.shape[0]),float(torch.median(self.total_image[cx,cy])),
                        float(tx.shape[0]),float(torch.median(self.total_image[tx,ty]))]
                cell_metadata.append(pd.DataFrame(data,index=['label',
                                                              'pixel_x',
                                                              'pixel_y',
                                                              'nuclei_size',
                                                              'nuclei_signal',
                                                              'cytoplasm_size',
                                                              'cytoplasm_signal',
                                                              'total_size',
                                                              'total_signal']).T)
            self.cell_metadata = pd.concat(cell_metadata,ignore_index=True)
            self.cell_metadata['posname'] = self.posname
            position_x,position_y = self.image_metadata.image_table[
                (self.image_metadata.image_table.acq==self.nucstain_acq)&
                (self.image_metadata.image_table.Position==self.posname)].XY.iloc[0]
            self.cell_metadata['posname_stage_x'] = position_x
            self.cell_metadata['posname_stage_y'] = position_y
            """ Update Cell name to be unique and use as index"""
            self.cell_metadata['cell_name'] = [
                self.dataset+'_'+self.posname+'_cell_'+str(label) for label in self.cell_metadata['label']]
            self.cell_metadata.index = np.array(self.cell_metadata['cell_name'])
            
    def segmentation(self):
        """ Segment """
        self.nuclei_segmentation()
        self.total_segmentation()
        
        """ Align Nuclei and Total Labels"""
        if self.verbose:
            self.update_user('Aligning Segmentation labels')
        from scipy import stats
        paired_nuclei_mask = torch.clone(self.nuclei_mask)
        labels = torch.unique(self.nuclei_mask.ravel()[self.nuclei_mask.ravel()>0])
        paired_total_mask = torch.zeros_like(self.total_mask)
        for label in labels:
            m = self.nuclei_mask==label
            cyto_label = stats.mode(self.total_mask[m]).mode[0]
            if cyto_label == 0:
                paired_nuclei_mask[m] = 0 # Eliminate Cell
                continue
            paired_total_mask[self.total_mask==cyto_label] = label
        self.nuclei_mask = paired_nuclei_mask
        self.total_mask = paired_total_mask
        self.cytoplasm_mask = torch.clone(paired_total_mask)
        self.cytoplasm_mask[self.nuclei_mask>0] = 0
        self.generate_cell_metadata()

    def pull_vectors(self):
        if self.verbose:
            self.update_user('Pulling Vectors')
        labels = torch.unique(self.nuclei_mask.ravel()[self.nuclei_mask.ravel()>0]).numpy()
        nuclei_vectors = torch.zeros([labels.shape[0],len(self.bitmap)])
        cyto_vectors = torch.zeros([labels.shape[0],len(self.bitmap)])
        total_vectors = torch.zeros([labels.shape[0],len(self.bitmap)])
        for i,(readout,hybe,channel) in enumerate(self.bitmap):
            image = self.signal_dict[hybe] # Alternatively Load Images Here maybe save on memory
            for j,label in enumerate(labels):
                nuclei_vectors[j,i] = torch.median(image[self.label_nuclei_coord_dict[label]])
                cyto_vectors[j,i] = torch.median(image[self.label_cyto_coord_dict[label]])
                total_vectors[j,i] = torch.median(image[self.label_total_coord_dict[label]])
        self.nuclei_vectors = nuclei_vectors
        self.cyto_vectors = cyto_vectors
        self.total_vectors = total_vectors
        
    def add_and_save_data(self,data,dtype='h5ad'):
        if not os.path.exists(os.path.join(self.metadata_path,'fishdata')):
            os.mkdir(os.path.join(self.metadata_path,'fishdata'))
        if dtype=='h5ad':
            fname = os.path.join(self.metadata_path,'fishdata',self.dataset+'_'+self.posname+'_data.h5ad')
            data.write(filename=fname)
        else:
            fname = os.path.join(self.metadata_path,'fishdata',self.dataset+'_'+self.posname+'_'+dtype+'.tif')
            cv2.imwrite(fname, data.astype('uint16'))
        
    def save_data(self):
        if self.verbose:
            self.update_user('Saving Data and Masks')
        data = anndata.AnnData(X=self.nuclei_vectors.numpy(),
                               var=pd.DataFrame(index=np.array([h for r,h,c in self.bitmap])),
                               obs=pd.DataFrame(index=self.cell_metadata.index.astype(str)))
        data.layers['nuclei_vectors'] = self.nuclei_vectors.numpy()
        data.layers['cytoplasm_vectors'] = self.cyto_vectors.numpy()
        data.layers['total_vectors'] = self.total_vectors.numpy()
        for column in self.cell_metadata.columns:
            data.obs[column] = np.array(self.cell_metadata[column])
        data.obs_names_make_unique()
        """ Remove Nans"""
        data = data[np.isnan(data.layers['total_vectors'].sum(1))==False]
        self.data = data
        self.add_and_save_data(self.data,dtype='h5ad')
        self.add_and_save_data(self.nuclei_mask.numpy(),dtype='nuclei_mask')
        self.add_and_save_data(self.cytoplasm_mask.numpy(),dtype='cytoplasm_mask')
        self.add_and_save_data(self.total_mask.numpy(),dtype='total_mask')

        
def wrapper(data):
    try:
        pos_class = dredFISH_Position_Class(data['metadata_path'],
                                            data['dataset'],
                                            data['posname'],
                                            data['cword_config'],
                                            verbose=False)
        pos_class.run()
    except Exception as e:
        print(posname)
        print(e)
    return data
    
if __name__ == '__main__':
    """ Unload Parameters"""
    metadata_path = args.metadata_path
    cword_config = args.cword_config
    config = importlib.import_module(cword_config)
    batches = args.batches
    ncpu=args.ncpu
    nregions = args.nregions
    outpath = args.outpath
    resolution = args.resolution
    print(args)
    
    """ Setup Batches"""
    dataset = [i for i in metadata_path.split('/') if not i==''][-1]
    image_metadata = Metadata(metadata_path)
    hybes = np.array([i for i in image_metadata.acqnames if 'hybe' in i])
    posnames = np.unique(image_metadata.image_table[image_metadata.image_table.acq==hybes[0]].Position)
    np.random.shuffle(posnames)
    Input = []
    for posname in tqdm(posnames):
        data = {
                'metadata_path':metadata_path,
                'dataset':dataset,
                'posname':posname,
                'cword_config':cword_config}
        Input.append(data)
    """ Process Batches """
    temp_input = []
    for i in Input:
        temp_input.append(i)
        if len(temp_input)>=batches:
            pool = multiprocessing.Pool(ncpu)
            sys.stdout.flush()
            results = pool.imap(wrapper, temp_input)
            iterable = tqdm(results,total=len(temp_input),desc=str(datetime.now().strftime("%H:%M:%S"))+' '+dataset,position=0)
            for i in iterable:
                pass
            pool.close()
            sys.stdout.flush()
            temp_input = []
    pool = multiprocessing.Pool(ncpu)
    sys.stdout.flush()
    results = pool.imap(wrapper, temp_input)
    iterable = tqdm(results,total=len(temp_input),desc=str(datetime.now().strftime("%H:%M:%S"))+' '+dataset,position=0)
    for i in iterable:
        pass
    pool.close()
    sys.stdout.flush()
    temp_input = []
        
    """ Merge Positions """
    data_list = []
    for posname in posnames:
        try:
            data = anndata.read_h5ad(os.path.join(metadata_path,'fishdata',dataset+'_'+posname+'_data.h5ad'))
        except:
            data = None
        if isinstance(data,anndata._core.anndata.AnnData):
            data_list.append(data)
    data = anndata.concat(data_list)
    
    """ Update Stage Coordiantes """
    pixel_size = config.parameters['pixel_size']
    camera_direction = config.parameters['camera_direction']
    stage_x = np.array(data.obs['posname_stage_x']) + camera_direction[0]*pixel_size*np.array(data.obs['pixel_x'])
    stage_y = np.array(data.obs['posname_stage_y']) + camera_direction[1]*pixel_size*np.array(data.obs['pixel_y'])
    data.obs['stage_x'] = stage_x
    data.obs['stage_y'] = stage_y
    xy = np.zeros([data.shape[0],2])
    xy[:,0] = data.obs['stage_x']
    xy[:,1] = data.obs['stage_y']
    data.obsm['stage'] = xy
    
    """ Remove Nans"""
    data = data[np.isnan(data.layers['total_vectors'].sum(1))==False]
    
    """ Assign Brains"""
    data.obs['dataset'] = dataset
    data.obs['brain_index'] = KMeans(n_clusters=nregions, random_state=0,n_init=50).fit(data.obsm['stage']).labels_
    
    """ Separate and Save Data """
    for region in tqdm(np.unique(data.obs['brain_index']),desc='Saving Brains'):
        mask = data.obs['brain_index']==region
        temp = data[mask]
        # Name Region by Centroid to resolution's of ums
        X = str(resolution*int(np.median(temp.obs['stage_x'])/resolution))
        Y = str(resolution*int(np.median(temp.obs['stage_x'])/resolution))
        brain_XY = 'Brain_'+X+'X_'+Y+'Y'
        # Give Cells Unique Name 
        temp.obs.index = [dataset+'_'+brain_XY+'_'+row.posname+'_Cell'+str(int(row.label)) for idx,row in temp.obs.iterrows()]
        matrix = pd.DataFrame(temp.X,
                              index=np.array(temp.obs.index),
                              columns=np.array(temp.var.index))
        fishdata.add_and_save_data(temp,
                                   dtype='h5ad',
                                   dataset=dataset+'_'+brain_XY)
        temp.write(filename=os.path.join(metadata_path,
                                         'fishdata',
                                         dataset+'_'+brain_XY+'_data.h5ad'))
        matrix.to_csv(os.path.join(metadata_path,
                                   'fishdata',
                                   dataset+'_'+brain_XY+'_matrix.csv'))
        temp.obs.to_csv(os.path.join(metadata_path,
                                     'fishdata',
                                     dataset+'_'+brain_XY+'_metadata.csv'))
        matrix.to_csv(os.path.join(outpath,
                                   dataset+'_'+brain_XY+'_matrix.csv'))
        temp.obs.to_csv(os.path.join(outpath,
                                     dataset+'_'+brain_XY+'_metadata.csv'))
        
