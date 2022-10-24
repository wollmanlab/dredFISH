from datetime import datetime
from dredFISH.Processing.Section import generate_FF
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
from scipy import interpolate
from scipy.ndimage import binary_dilation,binary_erosion
from skimage.morphology import remove_small_objects

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
        self.metadata = ''
        self.FF_dict = ''

        self.fishdata_path = os.path.join(self.metadata_path,self.config.parameters['fishdata'])
        if not os.path.exists(self.fishdata_path):
            os.mkdir(self.fishdata_path)
        
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
        if self.verbose:
            i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]
        
    def main(self):
        """
        main main functions to run for class
        """
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
        if self.parameters['vector_overwrite']:
            self.proceed = True
        else:
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
                self.update_user(self.posname+' Data Found')
            else:
                self.proceed = True
        
    def config_data(self):
        """
        config_data load Metadata
        """
        self.update_user('Loading Metadata')
        if isinstance(self.metadata,type(str)):
            self.metadata = Metadata(self.metadata_path)

    def load_flatfield(self):
        """
        load_flatfield 
        """
        if isinstance(self.FF_dict,str):
            self.update_user('Loading Flat Field Dictionary')
            self.FF_dict = {}
            for acq in self.acqs:
                if 'hybe' in acq:
                    hybe = acq.split('_')[0]
                elif 'strip' in acq:
                    hybe = 'hybe'+acq.split('_')[0].split('strip')[-1]
                channel = [c for r,h,c in self.config.bitmap if h==hybe][-1]
                fname = os.path.join(self.fishdata_path,self.dataset+'_'+acq+'_'+channel+'_FlatField.npy')
                if os.path.exists(fname):
                    self.update_user('Loading Flat Field '+acq)
                    self.FF_dict[acq] = np.load(fname)
                else:
                    self.update_user('Generating Flat Field '+acq)
                    try:
                        self.FF_dict[acq] = generate_FF(self.metadata,acq,channel).astype(np.float64)
                        np.save(fname,self.FF_dict[acq])
                    except:
                        self.update_user('Failed To Flat Field '+acq)
                        continue
        # nucstain
        acq = self.nucstain_acq
        channel = self.config.parameters['nucstain_channel']
        fname = os.path.join(self.fishdata_path,self.dataset+'_'+acq+'_'+channel+'_FlatField.npy')
        if os.path.exists(fname):
            self.FF_dict['nucstain'] = np.load(fname)
        else:
            self.update_user('Generating Flat Field '+'nucstain')
            self.FF_dict['nucstain'] = generate_FF(self.metadata,acq,channel).astype(np.float64)
            np.save(fname,self.FF_dict['nucstain'])

        # total
        acq = self.total_acq
        channel = self.config.parameters['total_channel']
        fname = os.path.join(self.fishdata_path,self.dataset+'_'+acq+'_'+channel+'_FlatField.npy')
        if os.path.exists(fname):
            self.FF_dict['total'] = np.load(fname)
        else:
            self.update_user('Generating Flat Field '+'total')
            self.FF_dict['total'] = generate_FF(self.metadata,acq,channel).astype(np.float64)
            np.save(fname,self.FF_dict['total'])

    def load_data(self):
        """
        load_data Load Raw Data 
        """
        self.hybes = [h for r,h,c in self.config.bitmap]
        self.acqs = self.metadata.image_table[self.metadata.image_table.Position==self.posname].acq.unique()
        np.random.shuffle(self.acqs)
        if not '_' in self.nucstain_acq:
            nucstain_acq = np.array([i for i in self.acqs if self.nucstain_acq+'_' in i])
            if len(nucstain_acq)>0:
                self.nucstain_acq = nucstain_acq[-1]

        if not self.total_acq == 'none':
            if not '_' in self.total_acq:
                total_acq = np.array([i for i in self.acqs if self.total_acq+'_' in i])
                if len(total_acq)>0:
                    self.total_acq = total_acq[-1]
                else:
                    # Raise Error
                    self.total_acq = 'none'

        acqs = []
        for acq in self.acqs:
            if 'hybe' in acq:
                hybe = acq.split('_')[0]
            elif 'strip' in acq:
                hybe = 'hybe'+acq.split('_')[0].split('strip')[-1]
            else:
                hybe = ''
            if hybe in self.hybes:
                acqs.append(acq)
        self.acqs = acqs
        self.load_flatfield() 
        # Load Nuclei Image 
        self.update_user('Loading Nuclei Image')
        self.nucstain_image = self.FF_dict['nucstain']*self.metadata.stkread(
            Position=self.posname,
            Channel=self.parameters['nucstain_channel'],
            acq=self.nucstain_acq).mean(2).astype(float)
        self.nuclei_segmentation()

        # Load Raw Data 
        self.update_user('Loading Raw Data')
        self.raw_dict = {
            hybe.split('_')[0]:torch.tensor(
                self.metadata.stkread(
                    Position=self.posname,
                    Channel=self.parameters['total_channel'],
                    acq=hybe).mean(2).astype(float)) for hybe in self.acqs if 'hybe' in hybe} #self.FF_dict[hybe]*
        
        # Load or Calculate Background 
        self.background_dict = {
            'hybe'+strip.split('strip')[1].split('_')[0]:torch.tensor(
                self.metadata.stkread(
                    Position=self.posname,                                                                             
                    Channel=self.parameters['total_channel'],
                    acq=strip).mean(2).astype(float)) for strip in self.acqs if 'strip' in strip} #self.FF_dict[strip]*
        self.process_data()
        # Load Total Image 
        self.update_user('Loading Total Image')
        if self.total_acq == 'none':
            self.total_image = np.dstack([self.signal_dict[hybe] for readout,hybe,channel in self.bitmap]).mean(2)
        else:
            hybe = self.config.parameters['total_acq']
            self.total_image = self.signal_dict[hybe].numpy()
        # self.offset_dict['total'] = np.percentile(temp.ravel(),self.config.parameters['non_cell_percentile'])
        # self.total_image = self.total_image-self.offset_dict['total']
        self.total_segmentation()

    def process_data(self):
         # Subtract Background 
        self.update_user('Subtracting Background')
        self.find_non_cell()
        self.signal_dict = {hybe:self.process_signal(basis) for basis,(readout,hybe,channel) in enumerate(self.config.bitmap)}

    def process_signal(self,basis):
        self.update_user('Processing Basis '+str(basis))
        hybe = self.raw_dict[self.config.bitmap[basis][1]]
        strip = self.background_dict[self.config.bitmap[basis][1]]
        # Assume non-cell percentile fo hybe and strip should be the same
        # hybe = hybe-np.percentile(hybe.ravel(),self.config.parameters['non_cell_percentile'])
        # strip = strip-np.percentile(strip.ravel(),self.config.parameters['non_cell_percentile'])
        # Subtract Strip from Hybe
        signal = hybe-strip 
        # # Assume area outside of cells but near cells should be 0 (2 std away from background is 0)
        # try: # see if self.non_cell exists
        #     test = self.non_cell*1
        # except:
        #     # Calculate non cell
        #     self.find_non_cell()
        # signal = signal-np.percentile(signal[self.non_cell],95)
        # to clip or not to clip
        
        # if self.config.bitmap[basis][1] in ['hybe23','hybe24','hybe25']:
        #     signal = np.array(signal)
        #     translation_x = -140
        #     translation_y = -15
        #     len_x = signal.shape[1]
        #     len_y = signal.shape[0]
        #     i2 = interpolate.interp2d(np.array(range(len_x))+translation_x,
        #                             np.array(range(len_y))+translation_y, signal,fill_value=0)
        #     signal = i2(range(len_x), range(len_y))
        #     signal = torch.tensor(signal.astype(float))
        return signal

    def find_non_cell(self):
        non_cell_completed = False
        if not self.config.parameters['non_cell_overwrite']:
            self.update_user('Checking Existing Non_cell')
            data = self.add_and_save_data('',dtype='non_cell_mask',load=True)
            if not isinstance(data,type(None)):
                self.non_cell = data.astype(float)>0
                non_cell_completed = True
            else:
                self.update_user('No Existing Non_cell '+str(type(data)))
        if not non_cell_completed:
            self.update_user('Calculating non cell')
            # Move from hard code
            nuclei = self.nuclei_mask.numpy()>0
            nuclei = remove_small_objects(nuclei,100)
            cell = binary_dilation(nuclei,iterations = 20) # window around nuclei
            non_cell = binary_dilation(cell,iterations = 100)
            non_cell = binary_erosion(non_cell,iterations = 100)
            non_cell[cell] = False #  window around nuclei
            if np.sum(non_cell)==0:
                # No Cells
                non_cell = (0*nuclei)==0
            self.non_cell = non_cell

            
    def process_image(self,image):
        """
        process_image Background Subtract Image (unused)

        :param image: Image
        :type image: np.array
        :return: Processed Image
        :rtype: np.array
        """
        img = image.copy()
        bkg = gaussian_filter(img,(self.parameters['segment_diameter']))
        img = img-bkg
        img[img<0] = 0
        return img
    
    def nuclei_segmentation(self):      
        """
        nuclei_segmentation Wrapper for Cellpose Nuclei Segmentation
        """
        nuclei_completed = False
        if not self.config.parameters['segment_overwrite']:
            self.update_user('Checking Existing Nuclei')
            data = self.add_and_save_data('',dtype='nuclei_mask',load=True)
            if not isinstance(data,type(None)):
                self.nuclei_mask = torch.tensor(data.astype(int))
                nuclei_completed = True
            else:
                self.update_user('No Existing Nuclei '+str(type(data)))
        if not nuclei_completed:
            self.update_user('Segmenting Nuclei')
            torch.cuda.empty_cache() 
            model = models.Cellpose(model_type='nuclei',gpu=self.parameters['segment_gpu'])
            raw_mask_image,flows,styles,diams = model.eval(self.process_image(self.nucstain_image),
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
        total_completed = False
        if not self.config.parameters['segment_overwrite']:
            self.update_user('Checking Existing Total')
            data = self.add_and_save_data('',dtype='total_mask',load=True)
            if not isinstance(data,type(None)):
                self.total_mask = torch.tensor(data.astype(int))
                total_completed = True
            else:
                self.update_user('No Existing Total '+str(type(data)))
        if not total_completed:
            stk = np.dstack([self.process_image(self.total_image.copy()),
                            self.process_image(self.nucstain_image.copy()),
                            np.zeros_like(self.total_image.copy())])
            self.update_user('Segmenting Total')
            torch.cuda.empty_cache() 
            model = models.Cellpose(model_type='cyto2',gpu=self.parameters['segment_gpu'])
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
        self.update_user('Generating Cell Metadata')
        labels = np.unique(self.nuclei_mask.ravel()[self.nuclei_mask.ravel()>0])
        self.label_nuclei_coord_dict = {label:torch.where(self.nuclei_mask==label) for j,label in enumerate(labels)}
        self.label_cyto_coord_dict = {label:torch.where((self.cytoplasm_mask==label)) for j,label in enumerate(labels)}
        self.label_total_coord_dict = {label:torch.where(self.total_mask==label) for j,label in enumerate(labels)}
        self.total_image = torch.tensor(self.total_image.astype(float))
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
                if nx.shape[0]<self.config.parameters['nuclei_size_threshold']:
                    continue
                cx,cy = self.label_cyto_coord_dict[label]
                tx,ty = self.label_total_coord_dict[label]
                label = label
                y = float(nx.float().mean())
                x = float(ny.float().mean())
                #  Pull Data from Images
                data = [int(label),x,y,
                        int(nx.shape[0]),float(torch.median(self.total_image[nx,ny])),float(np.median(self.nucstain_image[nx,ny])),
                        int(cx.shape[0]),float(torch.median(self.total_image[cx,cy])),
                        int(tx.shape[0]),float(torch.median(self.total_image[tx,ty]))]
                cell_metadata.append(pd.DataFrame(data,index=['label',
                                                              'pixel_x',
                                                              'pixel_y',
                                                              'nuclei_size',
                                                              'nuclei_signal',
                                                              'dapi_signal',
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
            if self.config.parameters['flipxy']:
                self.cell_metadata['stage_x'] = np.array(self.cell_metadata['posname_stage_x']) + camera_direction[0]*pixel_size*np.array(self.cell_metadata['pixel_y'])
                self.cell_metadata['stage_y'] = np.array(self.cell_metadata['posname_stage_y']) + camera_direction[1]*pixel_size*np.array(self.cell_metadata['pixel_x'])
            else:
                self.cell_metadata['stage_x'] = np.array(self.cell_metadata['posname_stage_x']) + camera_direction[0]*pixel_size*np.array(self.cell_metadata['pixel_x'])
                self.cell_metadata['stage_y'] = np.array(self.cell_metadata['posname_stage_y']) + camera_direction[1]*pixel_size*np.array(self.cell_metadata['pixel_y'])
            # # Add Offsets
            # for basis,(readout,hybe,channel) in enumerate(self.config.bitmap):
            #     self.cell_metadata[str(basis)+'_offset'] = self.offset_dict[hybe]

    def segmentation(self):
        """
        segmentation Segment Cells Nuclei and Total Before Pulling Vector
        """
        # self.nuclei_segmentation()
        # self.total_segmentation()
        self.update_user('Aligning Segmentation labels')
        paired_nuclei_mask = torch.clone(self.nuclei_mask)
        labels = torch.unique(self.nuclei_mask.ravel()[self.nuclei_mask.ravel()>0])
        paired_total_mask = torch.zeros_like(self.total_mask)
        for label in labels:
            m = self.nuclei_mask==label
            cyto_label = stats.mode(self.total_mask[m]).mode[0]
            if cyto_label == 0:
                # paired_nuclei_mask[m] = 0 # Eliminate Cell
                # Add Cell to total_mask
                cyto_label = self.total_mask.max()+1
                self.total_mask[m] = cyto_label
                # continue
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
        self.update_user('Pulling Vectors')
        labels = torch.unique(self.nuclei_mask.ravel()[self.nuclei_mask.ravel()>0]).numpy()
        labels = np.unique(self.cell_metadata['label'])
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
        
    def add_and_save_data(self,data,dtype='h5ad',load=False):
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
            if load:
                if os.path.exists(fname):
                    try:
                        data = anndata.read_h5ad(fname)
                    except:
                        data = None
                else:
                    data = None
                return data
            else:
                data.write(filename=fname)
        else:
            fname = os.path.join(self.metadata_path,self.config.parameters['fishdata'],self.dataset+'_'+self.posname+'_'+dtype+'.tif')
            if load:
                if os.path.exists(fname):
                    try:
                        data = cv2.imread(fname,-1)
                    except:
                        data = None
                else:
                    data = None
                return data
            else:
                cv2.imwrite(fname, data.astype('uint16'))
        
    def save_data(self):
        """
        save_data Save All Data Once Finished
        """
        self.update_user('Saving Data and Masks')
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
        data.obs_names_make_unique()
        data = data[np.isnan(data.layers['total_vectors'].sum(1))==False]
        self.data = data
        self.add_and_save_data(self.data,dtype='h5ad')
        self.add_and_save_data(self.nuclei_mask.numpy(),dtype='nuclei_mask')
        self.add_and_save_data(self.cytoplasm_mask.numpy(),dtype='cytoplasm_mask')
        self.add_and_save_data(self.total_mask.numpy(),dtype='total_mask')
        self.add_and_save_data(self.non_cell,dtype='non_cell_mask')

        