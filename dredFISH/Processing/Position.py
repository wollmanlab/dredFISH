from datetime import datetime
import numpy as np
import anndata
import pandas as pd
import importlib
from metadata import Metadata
from tqdm import tqdm
from cellpose import models
import cv2
import torch
import os
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy import stats
import time

class Position_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 cword_config,
                 verbose=False):
        """
        Position_Class : Segment and Pull Vector for each cell in a Position

        :param metadata_path: Path to Raw Data
        :type metadata_path: str
        :param dataset: Name of Dataset
        :type dataset: str
        :param posname: Name of Position
        :type posname: str
        :param cword_config: Name of Config Module
        :type cword_config: str
        :param verbose: Option to Print Loading Bars, defaults to False
        :type verbose: bool, optional
        """
        
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
        """
        run Main Executable
        """
        self.check_data()
        if self.proceed:
            self.main()
            
    def update_user(self,message):
        """
        update_user Communicate with User 

        :param message: message to be sent
        :type message: str
        """
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]
        
    def main(self):
        """
        main main functions to run for class
        """
        if self.verbose:
            self.update_user('Processing Position '+self.posname)
        self.config_data()
        self.load_data()
        self.segmentation()
        if self.cell_metadata.shape[0]>0:
            self.pull_vectors()
            self.save_data()
        
    def check_data(self):
        """
        check_data Check to see if data already exists
        """
        if self.parameters['overwrite']:
            self.proceed = True
        else:
            if self.verbose:
                self.update_user('Checking Existing Data')
            # Check If data already generated
            try:
                fname = os.path.join(self.metadata_path,
                                     self.config.parameters['fishdata'],
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
        """
        config_data load Metadata
        """
        if self.verbose:
            self.update_user('Loading Metadata')
        self.metadata = Metadata(self.metadata_path)
        
    def load_data(self):
        """
        load_data Load Raw Data 
        """
        # Load Nuclei Image 
        if self.verbose:
            self.update_user('Loading Nuclei Image')
        if self.nucstain_acq=='infer':
            self.nucstain_acq = np.array([i for i in self.metadata.acqnames if 'nucstain' in i])[-1]
        self.nucstain_image = self.metadata.stkread(
            Position=self.posname,
            Channel=self.parameters['nucstain_channel'],
            acq=self.nucstain_acq).mean(2).astype(float)
        
        # Load Raw Data 
        if self.verbose:
            self.update_user('Loading Raw Data')
        self.raw_dict = {
            hybe.split('_')[0]:torch.tensor(
                self.metadata.stkread(
                    Position=self.posname,
                    Channel=self.parameters['total_channel'],
                    acq=hybe).mean(2).astype(float)) for hybe in self.metadata.acqnames if 'hybe' in hybe}
        
        # Load or Calculate Background 
        self.background_dict = {
            'hybe'+strip.split('strip')[1].split('_')[0]:torch.tensor(
                self.metadata.stkread(
                    Position=self.posname,                                                                             
                    Channel=self.parameters['total_channel'],
                    acq=strip).mean(2).astype(float)) for strip in self.metadata.acqnames if 'strip' in strip}

        # Subtract Background 
        if self.verbose:
            self.update_user('Subtracting Background')
        self.signal_dict = {hybe:self.raw_dict[hybe]-self.background_dict[hybe] for readout,hybe,channel in self.bitmap}
        
        # Load Total Image 
        if self.verbose:
            self.update_user('Loading Total Image')
        if self.total_acq=='infer':
            self.total_acq = np.array([i for i in self.metadata.acqnames if 'nucstain' in i])[-1]
            self.total_image = self.metadata.stkread(Position=self.posname,
                                                           Channel=self.parameters['total_channel'],
                                                           acq=self.total_acq).mean(2).astype(float)
            self.background_acq = np.array([i for i in self.metadata.acqnames if 'background' in i])[-1]
            self.background_image = self.metadata.stkread(Position=self.posname,
                                                           Channel=self.parameters['total_channel'],
                                                           acq=self.background_acq).mean(2).astype(float)
            self.total_image = self.total_image-self.background_image
            
        elif self.total_acq=='none':
            self.total_image = np.dstack([self.signal_dict[hybe] for readout,hybe,channel in self.bitmap]).sum(2)
        else:
            self.total_image = self.metadata.stkread(Position=self.posname,
                                                           Channel=self.parameters['total_channel'],
                                                           acq=self.total_acq).mean(2).astype(float)
            
    def process_image(self,image):
        """
        process_image Background Subtract Image (unused)

        :param image: Image
        :type image: np.array
        :return: Processed Image
        :rtype: np.array
        """
        bkg = gaussian_filter(image,self.parameters['segment_diameter'])
        image = image-bkg
        image[image<0] = 0
        return image
    
    def nuclei_segmentation(self):      
        """
        nuclei_segmentation Wrapper for Cellpose Nuclei Segmentation
        """
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
        """
        total_segmentation Wrapper for Cellpose Total Cell Segmentation
        """
        stk = np.dstack([self.total_image.copy(),
                         self.nucstain_image.copy(),
                         np.zeros_like(self.total_image.copy())])
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
        """
        generate_cell_metadata Generate Metadata on Individual Cells 
        """
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
                # Which pixels are part of each cell
                nx,ny = self.label_nuclei_coord_dict[label]
                cx,cy = self.label_cyto_coord_dict[label]
                tx,ty = self.label_total_coord_dict[label]
                label = label
                if self.config.parameters['flipxy']:
                    x = float(nx.float().mean())
                    y = float(ny.float().mean())
                else:
                    y = float(nx.float().mean())
                    x = float(ny.float().mean())
                #  Pull Data from Images
                data = [int(label),x,y,
                        int(nx.shape[0]),float(torch.median(self.total_image[nx,ny])),
                        int(cx.shape[0]),float(torch.median(self.total_image[cx,cy])),
                        int(tx.shape[0]),float(torch.median(self.total_image[tx,ty]))]
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
            # Add Position Info
            self.cell_metadata['posname'] = self.posname
            self.cell_metadata['posname'] = self.cell_metadata['posname'].astype('category')
            position_x,position_y = self.metadata.image_table[
                (self.metadata.image_table.acq==self.nucstain_acq)&
                (self.metadata.image_table.Position==self.posname)].XY.iloc[0]
            self.cell_metadata['posname_stage_x'] = position_x
            self.cell_metadata['posname_stage_y'] = position_y
            # Make Cell names Unique
            self.cell_metadata['cell_name'] = [
                self.dataset+'_'+self.posname+'_cell_'+str(int(label)) for label in self.cell_metadata['label']]
            self.cell_metadata.index = np.array(self.cell_metadata['cell_name'])
            # Update Stage Coordinates
            pixel_size = self.config.parameters['pixel_size']
            camera_direction = self.config.parameters['camera_direction']
            self.cell_metadata['stage_x'] = np.array(self.cell_metadata['posname_stage_x']) + camera_direction[0]*pixel_size*np.array(self.cell_metadata['pixel_x'])
            self.cell_metadata['stage_y'] = np.array(self.cell_metadata['posname_stage_y']) + camera_direction[1]*pixel_size*np.array(self.cell_metadata['pixel_y'])
            
    def segmentation(self):
        """
        segmentation Segment Cells Nuclei and Total Before Pulling Vector
        """
        self.nuclei_segmentation()
        self.total_segmentation()
        if self.verbose:
            self.update_user('Aligning Segmentation labels')
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
        """
        pull_vectors Pull Median For Cell for each measurement
        """
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
        """
        add_and_save_data Save Data Objects

        :param data: Object to be saved
        :type data: various
        :param dtype: type of object to be saved defaults to 'h5ad'
        :type dtype: str, optional
        """
        if not os.path.exists(os.path.join(self.metadata_path,self.config.parameters['fishdata'])):
            os.mkdir(os.path.join(self.metadata_path,self.config.parameters['fishdata']))
        if dtype=='h5ad':
            fname = os.path.join(self.metadata_path,self.config.parameters['fishdata'],self.dataset+'_'+self.posname+'_data.h5ad')
            data.write(filename=fname)
        else:
            fname = os.path.join(self.metadata_path,self.config.parameters['fishdata'],self.dataset+'_'+self.posname+'_'+dtype+'.tif')
            cv2.imwrite(fname, data.astype('uint16'))
        
    def save_data(self):
        """
        save_data Save All Data Once Finished
        """
        if self.verbose:
            self.update_user('Saving Data and Masks')
        # data = anndata.AnnData(X=self.nuclei_vectors.numpy(),
        #                        var=pd.DataFrame(index=np.array([h for r,h,c in self.bitmap])),
        #                        obs=pd.DataFrame(index=self.cell_metadata.index.astype(str)))
        data = anndata.AnnData(X=self.nuclei_vectors.numpy(),
                               var=pd.DataFrame(index=np.array([h for r,h,c in self.bitmap])),
                               obs=self.cell_metadata)
        data.layers['nuclei_vectors'] = self.nuclei_vectors.numpy()
        data.layers['cytoplasm_vectors'] = self.cyto_vectors.numpy()
        data.layers['total_vectors'] = self.total_vectors.numpy()
        xy = np.zeros([data.shape[0],2])
        xy[:,0] = self.cell_metadata['stage_x']
        xy[:,1] = self.cell_metadata['stage_y']
        data.obsm['stage'] = xy
        # for column in self.cell_metadata.columns:
        #     data.obs[column] = np.array(self.cell_metadata[column])
        data.obs_names_make_unique()
        data = data[np.isnan(data.layers['total_vectors'].sum(1))==False]
        self.data = data
        
        self.add_and_save_data(self.data,dtype='h5ad')
        self.add_and_save_data(self.nuclei_mask.numpy(),dtype='nuclei_mask')
        self.add_and_save_data(self.cytoplasm_mask.numpy(),dtype='cytoplasm_mask')
        self.add_and_save_data(self.total_mask.numpy(),dtype='total_mask')

        