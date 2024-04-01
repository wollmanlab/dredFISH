import numpy as np
import torch
import os
import importlib
from tqdm import tqdm
from datetime import datetime
from metadata import Metadata
from functools import partial
from ashlar.utils import register
from PIL import Image
import multiprocessing
import sys
import cv2
from cellpose import models
from scipy.ndimage import gaussian_filter,percentile_filter,median_filter,minimum_filter
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import anndata
import pywt
import logging
from dredFISH.Utils import fileu
import shutil
from skimage.segmentation  import watershed
from skimage import morphology
import dredFISH.Processing as Processing
from skimage.feature import peak_local_max
import math
from dredFISH.Utils import basicu
import scanpy as sc
import matplotlib.colors as mcolors
from skimage import (
    data, restoration, util
)

""" TO DO LIST
 1. Add blocking so multiple computers could work on one dataset
 2. Check onfly processing (Rsync Blocking Permissions)
 3. 

"""

class Section_Class(object):
    """
    Section_Class Primary Class to Process dredFISH Sections
    """
    def __init__(self,
                 metadata_path,
                 section,
                 cword_config,
                 verbose=True):
        """
        __init__ Initialize Section_Class

        :param metadata_path: Path to Raw Data
        :type metadata_path: str
        :param section: Name of Section to Process. Should be the first part of position names before -Pos
        :type section: str
        :param cword_config: name of config file
        :type cword_config: str
        :param verbose: Deterimines if process is tracked via prints and logs, defaults to True
        :type verbose: bool, optional
        """
        self.metadata_path = metadata_path
        self.dataset = [i for i in self.metadata_path.split('/') if i!= ''][-1]
        self.section = str(section)
        self.cword_config = cword_config
        self.config = importlib.import_module(self.cword_config)
        self.image_metadata = ''
        self.reference_stitched=''
        self.FF=''
        self.nuc_FF=''
        self.data = None
        self.verbose=verbose
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger("Processing")

        if self.config.parameters['fishdata'] == 'fishdata':
            fishdata = self.config.parameters['fishdata']+str(datetime.today().strftime("_%Y%b%d"))
        else:
            fishdata = self.config.parameters['fishdata']
        self.path = os.path.join(self.metadata_path,fishdata)
        if not os.path.exists(self.path):
            self.update_user('Making fishdata Path',level=20)
            os.mkdir(self.path)
        self.path = os.path.join(self.path,'Processing')
        if not os.path.exists(self.path):
            self.update_user('Making Processing Path',level=20)
            os.mkdir(self.path)
        self.path = os.path.join(self.path,self.section)
        if not os.path.exists(self.path):
            self.update_user('Making Section Path',level=20)
            os.mkdir(self.path)

        logging.basicConfig(
                    filename=os.path.join(self.path,'processing_log.txt'),filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y %B %d %H:%M:%S',level=self.config.parameters['processing_log_level'], force=True)
        self.log = logging.getLogger("Processing")
        src = os.path.join(Processing.__file__.split('dredFISH/P')[0],self.cword_config+'.py')
        dst = os.path.join(self.path,self.cword_config+'.py')
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copyfile(src, dst)

        fileu_config = fileu.interact_config(self.path,return_config=True)
        if not 'version' in fileu_config.keys():
            fileu_config = fileu.interact_config(self.path,key='version',data=self.config.parameters['fileu_version'],return_config=True)

    def run(self):
        """
        run Primary Function to Process DredFISH Sections
        """
        self.load_metadata()
        if len(self.posnames)>0:
            completed = False
            for model_type in self.config.parameters['model_types']:
                self.model_type = model_type
                if (not self.config.parameters['vector_overwrite']):
                    self.update_user('Attempting To Load Data',level=20)
                    self.data = self.load(file_type='anndata',model_type=self.model_type)
                    if not isinstance(self.data,type(None)):
                        self.update_user('Existing Data Found')
                        completed = True
                        continue
                    self.update_user('No Data Found')
                    completed = False
            if not completed:
                self.stitch()
                for model_type in self.config.parameters['model_types']:
                    self.model_type = model_type
                    if (not self.config.parameters['vector_overwrite']):
                        self.update_user('Attempting To Load Data',level=20)
                        self.data = self.load(file_type='anndata',model_type=self.model_type)
                        if not isinstance(self.data,type(None)):
                            self.update_user('Existing Data Found')
                            continue
                        self.update_user('No Data Found')
                    self.segment()
                    if not self.any_incomplete_hybes:
                        self.pull_vectors()
            if not isinstance(self.data,type(None)):
                # self.remove_temporary_files()
                self.generate_report()
        else:
            self.update_user('No positions found for this section',level=50)
    
    def update_user(self,message,level=20):
        """
        update_user Wrapper to fileu.update_user

        :param message: Message to be logged
        :type message: str
        :param level: priority 0-50, defaults to 20
        :type level: int, optional
        """
        if self.verbose:
            print(message)
        fileu.update_user(message,level=level,logger=self.log)

    def check_existance(self,hybe='',channel='',file_type='',model_type=''):
        """
        check_existance wrapper to fileu.check_existance

        :param hybe: name of hybe ('hybe1','Hybe1','1')
        :type hybe: str, optional
        :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
        :type channel: str, optional
        :param type: Data Type to save ('stitched','mask','anndata',...)
        :type type: str, optional
        :param model_type: segmentation Type ('total','nuclei','cytoplasm')
        :type model_type: str, optional
        :return : True of file exists, False otherwise
        :rtype : bool
        """
        return fileu.check_existance(self.path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def generate_filename(self,hybe,channel,file_type,model_type=''):
        """
        fname Wrapper to fileu.generate_filename Generate Filename

        :param hybe: name of hybe ('hybe1','Hybe1','1')
        :type hybe: str
        :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
        :type channel: str
        :param type: Data Type to save ('stitched','mask','anndata',...)
        :type type: str
        :param model_type: segmentation Type ('total','nuclei','cytoplasm')
        :type model_type: str
        :return: File Path
        :rtype: str
        """
        if len(self.config.parameters['model_types'])==1:
            model_type = ''
        return fileu.generate_filename(self.path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def save(self,data,hybe='',channel='',file_type='',model_type=''):
        """
        save Wrapper to fileu.save Sae Files

        :param data: data object to be saved
        :type path: object
        :param hybe: name of hybe ('hybe1','Hybe1','1')
        :type hybe: str, optional
        :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
        :type channel: str, optional
        :param type: Data Type to save ('stitched','mask','anndata',...)
        :type type: str, optional
        :param model_type: segmentation Type ('total','nuclei','cytoplasm')
        :type model_type: str, optional
        """
        if len(self.config.parameters['model_types'])==1:
            model_type = ''
        fileu.save(data,path=self.path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def load(self,hybe='',channel='',file_type='anndata',model_type=''):
        """
        load Wrapper to fileu.load Load Files

        :param hybe: name of hybe ('hybe1','Hybe1','1')
        :type hybe: str, optional
        :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
        :type channel: str, optional
        :param type: Data Type to save ('stitched','mask','anndata',...)
        :type type: str, optional
        :param model_type: segmentation Type ('total','nuclei','cytoplasm')
        :type model_type: str, optional
        :return : Desired Data Object or None if not found
        :rtype : object
        """
        if len(self.config.parameters['model_types'])==1:
            model_type = ''
        return fileu.load(path=self.path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def generate_iterable(self,iterable,message,length=0):
        """
        generate_iterable Generate Iterable, Can be loading bar if self.verbose is True

        :param iterable: object to be iterated over 
        :type iterable: list
        :param message: short description of what is being iterated over
        :type message: str
        :param length: length of iterable if complex data type, defaults to ''
        :type length: int, optional
        :return: either initial iterable or iterable with loading bar
        :rtype: list,ProgressDialog?
        """
        if self.verbose:
            self.update_user(message,level=20)
        return iterable

    def filter_data(self):
        self.update_user('Filtering Out Non Cells')
        mask = self.data.obs['dapi']>self.config.parameters['dapi_thresh'] 
        bad_labels = self.data[mask==False].obs['label'].unique()
        idxes = np.nonzero(self.mask)
        l = self.mask[idxes[:,0],idxes[:,1]]
        l[torch.isin(l,torch.tensor(bad_labels))] = 0 # set bad labels to 0
        self.mask[idxes[:,0],idxes[:,1]] = l
        self.data = self.data[mask]
                                      
    def load_metadata(self):
        """
        load_metadata Load Raw Data Loading Class
        """
        if isinstance(self.image_metadata,str):
            self.update_user('Loading Metadata')
            self.image_metadata = Metadata(self.metadata_path)

        hybe = [i for i in self.image_metadata.acqnames if 'hybe' in i.lower()][0]
        posnames = np.unique(self.image_metadata.image_table[np.isin(self.image_metadata.image_table.acq,hybe)].Position)
        self.update_user(str(posnames.shape[0])+' Positions Found')
        sections = np.unique([i.split('-Pos')[0] for i in posnames if '-Pos' in i])

        if sections.shape[0] == 0:
            acqs = [i for i in self.image_metadata.acqnames if ('hybe' in i.lower())|('strip' in i.lower())]
            self.posnames = np.array(self.image_metadata.image_table[np.isin(self.image_metadata.image_table.acq,acqs)].Position.unique())
        else:
            self.posnames = np.array([i for i in self.image_metadata.posnames if i.split('-Pos')[0]==self.section])

        if len(self.posnames)>0:
            self.acqs = np.unique(self.image_metadata.image_table[np.isin(self.image_metadata.image_table.Position,self.posnames)].acq)
            if len(self.acqs)==0:
                print(self.image_metadata.acqnames)
            self.coordinates = {}
            for posname in self.posnames:
                self.coordinates[posname] = (self.image_metadata.image_table[(self.image_metadata.image_table.Position==posname)].XY.iloc[0]/self.config.parameters['pixel_size']).astype(int)

    def find_acq(self,hybe,protocol='hybe'):
        """
        find_acq Look through acqnames to find acquisition name for desired protocol and round

        :param hybe: Round of Imaging ("1","hybe1")
        :type hybe: str
        :param protocol: hybe or strip, defaults to 'hybe'
        :type protocol: str, optional
        :raises ValueError: Notify User if acq doesnt exist
        :return: name of acquisition within Metadata.acqnames
        :rtype: str
        """
        if 'hybe' in hybe.lower():
            hybe = hybe.lower().split('hybe')[-1]
        acqs = [i for i in self.acqs if protocol+hybe+'_' in i.lower()]
        if len(acqs)>1:
            new_acqs = []
            acq_times = []
            """ Remove Partially Imaged Hybes """
            for acq in acqs:
                acq_metadata = self.image_metadata[self.image_metadata.acq==acq]
                posnames = acq_metadata.Position.unique()
                if len(posnames)<self.posnames.shape[0]:
                    self.update_user('Ignoring Partially Imaged Acq '+acq,level=30)
                else:
                    avg_time = acq_metadata.TimestampImage.mean()
                    acq_times.append(avg_time)
            if len(new_acqs)>1:
                acqs = [np.array(new_acqs)[np.argmax(np.array(acq_times))]]
                self.update_user('Multiple Completed Acqs found choosing most recent '+str(acqs[0]),level=30)
            elif len(new_acqs)==0:
                self.update_user(f"No Complete Acqs found for {str(protocol)} {str(hybe)}",level=50)
            else:
                acqs = new_acqs
        if len(acqs)==0:
            self.update_user(f"No Acqs found for {str(protocol)} {str(hybe)}",level=50)
            raise ValueError(protocol+hybe+' not found in acqs')
        else:
            return acqs[0]

    def stitcher(self,hybe,channel,acq='',bkg_acq=''):
        """
        stitcher Function to stitch indivisual images into entire sections with rigid registration correction

        :param hybe: round of imaging ("1", "hybe1")
        :type hybe: str
        :param channel: name of channel to be stitched
        :type channel: str
        :param acq: name of acquisition if you dont want to infer from hybe, defaults to ''
        :type acq: str, optional
        :param bkg_acq: name of background acquisition if you dont want to infer from hybe, defaults to ''
        :type bkg_acq: str, optional
        :return:  stitched,nuclei,nuclei_down,signal,signal_down (down is downsampled for visualization)
        :rtype: torch.tensor,torch.tensor,torch.tensor,torch.tensor,torch.tensor
        """        
        if acq=='':
            acq = self.find_acq(hybe,protocol='hybe')
        if self.config.parameters['strip']&(bkg_acq=='')&(acq==''):
            bkg_acq = self.find_acq(hybe,protocol='strip')
        else:
            bkg_acq = ''
        """ Check if hybe is finished"""
        acq_posnames = np.unique(self.image_metadata.image_table[self.image_metadata.image_table.acq==acq].Position)
        bad_acq_positions = [pos for pos in self.posnames if not np.isin(pos,acq_posnames)]
        if bkg_acq!='':
            bkg_acq_posnames = np.unique(self.image_metadata.image_table[self.image_metadata.image_table.acq==bkg_acq].Position)
            bad_bkg_acq_positions = [pos for pos in self.posnames if not np.isin(pos,bkg_acq_posnames)]
        else:
            bad_bkg_acq_positions = []
        if len(bad_acq_positions)>0|len(bad_bkg_acq_positions)>0:
            self.update_user('Imaging isnt finished ',level=30)
            self.update_user(hybe+' '+channel+' '+acq+' '+bkg_acq,level=30)
            self.any_incomplete_hybes = True
            return None,None,None,None,None
        else:
            nuc_exists = self.check_existance(
                hybe=hybe,
                channel=self.config.parameters['nucstain_channel'],
                file_type='stitched'
            )
            signal_exists = self.check_existance(
                hybe=hybe,
                channel=channel,
                file_type='stitched'
            )
            if (not self.config.parameters['overwrite'])&(nuc_exists&signal_exists):
                self.update_user('Found Existing '+hybe+' Stitched')
                if (hybe==self.config.parameters['nucstain_acq'])&(self.reference_stitched==''):
                    nuc = self.load(
                        hybe=hybe,
                        channel=self.config.parameters['nucstain_channel'],
                        file_type='stitched'
                    )
                    signal = self.load(
                        hybe=hybe,
                        channel=channel,
                        file_type='stitched'
                    )
                    stitched = torch.dstack([nuc,signal])

                else:
                    stitched = 0
                return stitched
            else:
                if self.config.parameters['acq_FF']:
                    if (self.config.parameters['overwrite']==False)&self.check_existance(hybe=hybe,channel=self.config.parameters['total_channel'],file_type='constant')&self.check_existance(hybe=hybe,channel=self.config.parameters['total_channel'],file_type='FF'):
                        self.update_user(f"Found Existing Flat Field for {hybe} {self.config.parameters['total_channel']}")
                        self.FF = self.load(hybe=hybe,channel=self.config.parameters['total_channel'],file_type='FF')
                        self.constant = self.load(hybe=hybe,channel=self.config.parameters['total_channel'],file_type='constant')
                    else:
                        self.update_user(f"Calculating Flat Field for {self.find_acq(hybe,protocol='hybe')} {self.config.parameters['total_channel']}")
                        FF,constant = generate_FF(self.image_metadata,self.find_acq(hybe,protocol='hybe'),self.config.parameters['total_channel'],bkg_acq='',parameters=self.config.parameters)
                        self.FF = FF
                        self.save(FF,hybe=hybe,channel=self.config.parameters['total_channel'],file_type='FF')
                        self.save((FF*1000),hybe=hybe,channel=self.config.parameters['total_channel'],file_type='image_FF')
                        self.constant = constant
                        self.save(constant,hybe=hybe,channel=self.config.parameters['total_channel'],file_type='constant')
                        self.save(constant,hybe=hybe,channel=self.config.parameters['total_channel'],file_type='image_constant')
                    if (self.config.parameters['overwrite']==False)&self.check_existance(hybe=hybe,channel=self.config.parameters['nucstain_channel'],file_type='FF')&self.check_existance(hybe=hybe,channel=self.config.parameters['nucstain_channel'],file_type='constant'):
                        self.update_user(f"Found Existing Flat Field for {hybe} {self.config.parameters['nucstain_channel']}")
                        self.nuc_FF = self.load(hybe=hybe,channel=self.config.parameters['nucstain_channel'],file_type='FF')
                        self.nuc_constant = self.load(hybe=hybe,channel=self.config.parameters['nucstain_channel'],file_type='constant')
                    else:
                        self.update_user(f"Calculating Flat Field for {self.find_acq(hybe,protocol='hybe')} {self.config.parameters['nucstain_channel']}")
                        nuc_FF,nuc_constant = generate_FF(self.image_metadata,self.find_acq(hybe,protocol='hybe'),self.config.parameters['nucstain_channel'],bkg_acq='',parameters=self.config.parameters)
                        self.nuc_FF = nuc_FF
                        self.save(nuc_FF,hybe=hybe,channel=self.config.parameters['nucstain_channel'],file_type='FF')
                        self.save((nuc_FF*1000),hybe=hybe,channel=self.config.parameters['nucstain_channel'],file_type='image_FF')
                        self.nuc_constant = nuc_constant
                        self.save(nuc_constant,hybe=hybe,channel=channel,file_type='constant')
                        self.save(nuc_constant,hybe=hybe,channel=self.config.parameters['nucstain_channel'],file_type='image_constant')
                if isinstance(self.FF,str):
                    self.FF = 1
                if isinstance(self.nuc_FF,str):
                    self.nuc_FF = 1
                if isinstance(self.constant,str):
                    self.constant = 0
                if isinstance(self.nuc_constant,str):
                    self.nuc_constant = 0
                xy = np.stack([pxy for i,pxy in self.coordinates.items()])
                if (self.config.parameters['stitch_rotate'] % 2) == 0:
                    img_shape = np.flip(self.config.parameters['n_pixels'])
                else:
                    img_shape = self.config.parameters['n_pixels']
                y_min,x_min = xy.min(0)-self.config.parameters['border']
                y_max,x_max = xy.max(0)+img_shape+self.config.parameters['border']
                x_range = np.array(range(x_min,x_max+1))
                y_range = np.array(range(y_min,y_max+1))
                stitched = torch.zeros([len(x_range),len(y_range),2],dtype=torch.int32)
                if isinstance(self.reference_stitched,str):
                    pixel_coordinates_stitched = torch.zeros([len(x_range),len(y_range),2],dtype=torch.int32)
                Input = []
                for posname in self.posnames:
                    data = {}
                    data['acq'] = acq
                    data['bkg_acq'] = bkg_acq
                    data['posname'] = posname
                    data['image_metadata'] = self.image_metadata
                    data['channel'] = channel
                    data['parameters'] = self.config.parameters
                    # If not Reference Then pass destination through and do registration in parrallel
                    if not isinstance(self.reference_stitched,str):
                        position_y,position_x = self.coordinates[posname].astype(int)
                        img_x_min = position_x-x_min
                        img_x_max = (position_x-x_min)+img_shape[0]
                        img_y_min = position_y-y_min
                        img_y_max = (position_y-y_min)+img_shape[1]
                        destination = self.reference_stitched[img_x_min:img_x_max,img_y_min:img_y_max,0]
                        data['destination'] = destination
                    Input.append(data)
                pfunc = partial(preprocess_images,FF=self.FF,nuc_FF=self.nuc_FF,constant=self.constant,nuc_constant=self.nuc_constant)
                translation_y_list = []
                translation_x_list = []
                redo_posnames = []
                if (not self.config.parameters['register_stitch_reference'])&(isinstance(self.reference_stitched,str)):
                    redo_posnames = list(self.posnames)
                    # Input = []
                    translation_y_list = [0,0,0]
                    translation_x_list = [0,0,0]
                results = {}
                with multiprocessing.Pool(15) as p:
                    for posname,nuc,signal,translation_x,translation_y in self.generate_iterable(p.imap(pfunc,Input),'Processing '+acq+'_'+channel,length=len(Input)):
                        results[posname] = {}
                        results[posname]['nuc'] = nuc
                        results[posname]['signal'] = signal
                        results[posname]['translation_x'] = translation_x
                        results[posname]['translation_y'] = translation_y
                for posname in self.generate_iterable(results.keys(),'Stitching '+acq+'_'+channel,length=len(results.keys())):
                    nuc = results[posname]['nuc']
                    signal = results[posname]['signal']
                    translation_x = results[posname]['translation_x']
                    translation_y = results[posname]['translation_y']
                    if isinstance(translation_x,str):
                        redo_posnames.append(posname)
                        continue
                    position_y,position_x = self.coordinates[posname].astype(int)
                    img_x_min = position_x-x_min
                    img_x_max = (position_x-x_min)+img_shape[0] # maybe flipped 1,0
                    img_y_min = position_y-y_min
                    img_y_max = (position_y-y_min)+img_shape[1]# maybe flipped 1,0
                    if isinstance(self.reference_stitched,str):
                        translation_x = 0
                        translation_y = 0
                        destination = stitched[img_x_min:img_x_max,img_y_min:img_y_max,0]
                        mask = destination!=0
                        overlap =  mask.sum()/destination.ravel().shape[0]
                        if overlap>self.config.parameters['overlap']: 
                            mask_x = destination.max(1).values!=0
                            ref = destination[mask_x,:]
                            mask_y = ref.max(0).values!=0
                            ref = destination[mask_x,:]
                            ref = ref[:,mask_y]
                            non_ref = nuc[mask_x,:]
                            non_ref = non_ref[:,mask_y]
                            # Check if Beads work here
                            shift, error = register(ref.numpy(), non_ref.numpy(),10)
                            if (error!=np.inf)&(np.max(np.abs(shift))<=self.config.parameters['border']):
                                translation_y = int(shift[1])
                                translation_x = int(shift[0])
                                translation_y_list.append(translation_y)
                                translation_x_list.append(translation_x)
                            else: # Save for end and use median of good ones
                                redo_posnames.append(posname)
                                continue
                    else:
                        translation_y_list.append(translation_y)
                        translation_x_list.append(translation_x)
                    try:
                        stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),0] = nuc
                        stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),1] = signal
                        if isinstance(self.reference_stitched,str):
                            pixel_coordinates_stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),0] = torch.tensor(np.stack([np.array(range(nuc.shape[1])) for i in range(nuc.shape[0])])) #x
                            pixel_coordinates_stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),1] = torch.tensor(np.stack([np.array(range(nuc.shape[0])) for i in range(nuc.shape[1])]).T) #y
                            
                    except Exception as e:
                        print(posname,'Placing Image in Stitch with registration failed')
                        print(e)
                        translation_x = 0
                        translation_y = 0
                        stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),0] = nuc
                        stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),1] = signal
                        if isinstance(self.reference_stitched,str):
                            pixel_coordinates_stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),0] = torch.tensor(np.stack([np.array(range(nuc.shape[1])) for i in range(nuc.shape[0])])) #x
                            pixel_coordinates_stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),1] = torch.tensor(np.stack([np.array(range(nuc.shape[0])) for i in range(nuc.shape[1])]).T) #y
                if len(redo_posnames)>0:
                    for posname in self.generate_iterable(redo_posnames,'Redoing Failed Positions '+acq+'_'+channel,length=len(redo_posnames)):
                        nuc = results[posname]['nuc']
                        signal = results[posname]['signal']
                        translation_x = results[posname]['translation_x']
                        translation_y = results[posname]['translation_y']
                        position_y,position_x = self.coordinates[posname].astype(int)
                        img_x_min = position_x-x_min
                        img_x_max = (position_x-x_min)+img_shape[0] # maybe flipped 1,0
                        img_y_min = position_y-y_min
                        img_y_max = (position_y-y_min)+img_shape[1] # maybe flipped 1,0
                        translation_x = 0
                        translation_y = 0
                        if not isinstance(self.reference_stitched,str):
                            if len(translation_y_list)>0:
                                translation_y = int(np.median(translation_y_list))
                                translation_x = int(np.median(translation_x_list))

                        stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),0] = nuc
                        stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),1] = signal
                        if isinstance(self.reference_stitched,str):
                            pixel_coordinates_stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),0] = torch.tensor(np.stack([np.array(range(nuc.shape[1])) for i in range(nuc.shape[0])])) #x
                            pixel_coordinates_stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),1] = torch.tensor(np.stack([np.array(range(nuc.shape[0])) for i in range(nuc.shape[1])]).T) #y
                self.save(stitched[:,:,0],hybe=hybe,channel=self.config.parameters['nucstain_channel'],file_type='stitched')
                self.save(stitched[:,:,1],hybe=hybe,channel=channel,file_type='stitched')
                nuclei = stitched[self.config.parameters['border']:-self.config.parameters['border'],
                                self.config.parameters['border']:-self.config.parameters['border'],0].numpy()
                scale = (np.array(nuclei.shape)*self.config.parameters['ratio']).astype(int)
                nuclei_down = np.array(Image.fromarray(nuclei.astype(float)).resize((scale[1],scale[0]), Image.BICUBIC))
                self.save(nuclei_down,hybe=hybe,channel=self.config.parameters['nucstain_channel'],file_type='image')
                signal = stitched[self.config.parameters['border']:-self.config.parameters['border'],
                                self.config.parameters['border']:-self.config.parameters['border'],1].numpy()
                scale = (np.array(signal.shape)*self.config.parameters['ratio']).astype(int)
                signal_down = np.array(Image.fromarray(signal.astype(float)).resize((scale[1],scale[0]), Image.BICUBIC))
                self.save(signal_down,hybe=hybe,channel=channel,file_type='image')
                if isinstance(self.reference_stitched,str):
                    self.save(pixel_coordinates_stitched[:,:,0],hybe='image_x',channel='',file_type='stitched')
                    self.save(pixel_coordinates_stitched[:,:,1],hybe='image_y',channel='',file_type='stitched')
                return stitched

    def stitch(self):
        """
        stitch Wrapper to stitch all rounds of imaging
        """        
        self.update_user('Attempting To Stitch',level=20)
        acq = self.find_acq(self.config.parameters['nucstain_acq'],protocol='hybe')
        if self.config.parameters['use_FF']==False:
            self.FF = 1
            self.nuc_FF = 1
        elif isinstance(self.FF,str) | isinstance(self.nuc_FF,str):
            if (self.config.parameters['overwrite']==False)&self.check_existance(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['total_channel'],file_type='constant')&self.check_existance(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['total_channel'],file_type='FF'):
                self.update_user(f"Loading FF and Constant for {self.find_acq(self.config.parameters['nucstain_acq'],protocol='hybe')} {self.config.parameters['total_channel']}")
                self.FF = self.load(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['total_channel'],file_type='FF')
                self.constant = self.load(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['total_channel'],file_type='constant')
            else:
                self.update_user(f"Generating FF and Constant for {self.find_acq(self.config.parameters['nucstain_acq'],protocol='hybe')} {self.config.parameters['total_channel']}")
                FF,constant = generate_FF(self.image_metadata,self.find_acq(self.config.parameters['nucstain_acq'],protocol='hybe'),self.config.parameters['total_channel'],bkg_acq='',parameters=self.config.parameters)
                # FF,constant = generate_FF_constant(self.image_metadata,channel,posnames=self.posnames,bkg_acq='',parameters=self.config.parameters,verbose=self.verbose)
                self.FF = FF
                self.save(FF,hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['total_channel'],file_type='FF')
                self.save((FF*1000),hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['total_channel'],file_type='image_FF')
                self.constant = constant
                self.save(constant,hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['total_channel'],file_type='constant')
                self.save(constant,hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['total_channel'],file_type='image_constant')
            if (self.config.parameters['overwrite']==False)&self.check_existance(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='FF')&self.check_existance(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='constant'):
                self.update_user(f"Loading FF and Constant for {self.find_acq(self.config.parameters['nucstain_acq'],protocol='hybe')} {self.config.parameters['nucstain_channel']}")
                self.nuc_FF = self.load(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='FF')
                self.nuc_constant = self.load(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='constant')
            else:
                self.update_user(f"Generating FF and Constant for {self.find_acq(self.config.parameters['nucstain_acq'],protocol='hybe')} {self.config.parameters['nucstain_channel']}")
                nuc_FF,nuc_constant = generate_FF(self.image_metadata,self.find_acq(self.config.parameters['nucstain_acq'],protocol='hybe'),self.config.parameters['nucstain_channel'],bkg_acq='',parameters=self.config.parameters)
                # nuc_FF,nuc_constant = generate_FF_constant(self.image_metadata,self.config.parameters['nucstain_channel'],posnames=self.posnames,bkg_acq='',parameters=self.config.parameters,verbose=self.verbose)
                self.nuc_FF = nuc_FF
                self.save(nuc_FF,hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='FF')
                self.save((nuc_FF*1000),hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='image_FF')
                self.nuc_constant = nuc_constant
                self.save(nuc_constant,hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='constant')
                self.save((nuc_constant),hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='image_constant')
        if self.config.parameters['use_constant']==False:
            self.constant = 0
            self.nuc_constant = 0
        """ Generate Reference """
        self.any_incomplete_hybes = False
        self.reference_stitched = self.stitcher(self.config.parameters['nucstain_acq'],self.config.parameters['total_channel'])
        if isinstance(self.reference_stitched,type(None)):
            """ Reference Hasnt been Imaged"""
            self.any_incomplete_hybes = True
        else:
            for r,h,c in self.config.bitmap:
                try:
                    stitched = self.stitcher(h,self.config.parameters['total_channel'])
                except Exception as e:
                    self.update_user(e,level=40)
                    self.any_incomplete_hybes = True

    def segment(self):
        """
        segment Using stitched images segment cells 

        :param model_type: model for cellpose ('nuclei' 'total' 'cytoplasm'), defaults to 'nuclei'
        :type model_type: str, optional
        """      
        self.update_user(f"Attempting Segmentation {self.model_type}",level=20)
        if (not self.config.parameters['segment_overwrite'])&self.check_existance(file_type='mask',model_type=self.model_type):
            self.mask = self.load(file_type='mask',model_type=self.model_type)
            self.update_user(f"Existing Mask Found {self.model_type}",level=20)
        else:
            """ Cytoplasm"""
            if 'cytoplasm' in self.model_type:
                """ Check Total"""
                if self.check_existance(file_type='mask',model_type='total'):
                    """ Load """
                    total = self.load(file_type='mask',model_type='total')
                else:
                    backup_type = self.model_type
                    self.model_type = 'total'
                    self.segment()
                    self.model_type = backup_type
                    total = self.mask
                """ Check Nuclei"""
                if self.check_existance(file_type='mask',model_type='nuclei'):
                    """ Load """
                    nuclei = self.load(file_type='mask',model_type='nuclei')
                else:
                    backup_type = self.model_type
                    self.model_type = 'nuclei'
                    self.segment()
                    self.model_type = backup_type
                    nuclei = self.mask
                cytoplasm = total
                cytoplasm[nuclei>0] = 0
                self.mask = cytoplasm
                self.save(self.mask,file_type='mask',model_type=self.model_type)
            elif 'beads' in self.model_type:
                self.update_user('Bead Segmentation Not Implemented',level=50)
                """ Add Bead Segmentation Code Here """
            else:
                """ Total & Nuclei"""
                total = None
                nucstain = None
                model = None
                self.mask = None
                # nucstain
                if self.check_existance(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='stitched'):
                    nucstain = self.load(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='stitched')
                    dapi_mask = nucstain<self.config.parameters['dapi_thresh']
                    nucstain[dapi_mask] = 0
                else:
                    if self.config.parameters['nucstain_acq'] == 'all':
                        for r,h,c in self.generate_iterable(self.config.bitmap,'Loading Total by Averaging All Measurements'):
                            if self.check_existance(hybe=h,channel=self.config.parameters['nucstain_channel'],file_type='stitched'):
                                if isinstance(nucstain,type(None)):
                                    nucstain = self.load(hybe=h,channel=self.config.parameters['nucstain_channel'],file_type='stitched')/len(self.config.bitmap)
                                else:
                                    nucstain = nucstain+(self.load(hybe=h,channel=self.config.parameters['nucstain_channel'],file_type='stitched')/len(self.config.bitmap))
                            else:
                                self.update_user(f" Missing {h} for generating nucstain",value=50)
                                nucstain=None
                if not isinstance(nucstain,type(None)):
                    self.save(nucstain,hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='stitched')
                    nuclei = nucstain[self.config.parameters['border']:-self.config.parameters['border'],
                                    self.config.parameters['border']:-self.config.parameters['border']].numpy()
                    scale = (np.array(nuclei.shape)*self.config.parameters['ratio']).astype(int)
                    nuclei_down = np.array(Image.fromarray(nuclei.astype(float)).resize((scale[1],scale[0]), Image.BICUBIC))
                    self.save(nuclei_down,hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='image')
                # Total
                if 'total' in self.model_type:
                    if self.check_existance(hybe=self.config.parameters['total_acq'],channel=self.config.parameters['total_channel'],file_type='stitched'):
                            total = self.load(hybe=self.config.parameters['total_acq'],channel=self.config.parameters['total_channel'],file_type='stitched')
                            total[dapi_mask] = 0
                    else:
                        if self.config.parameters['total_acq'] == 'all':
                            for r,h,c in self.generate_iterable(self.config.bitmap,'Loading Total by Averaging All Measurements'):
                                if self.check_existance(hybe=h,channel=c,file_type='stitched'):
                                    if isinstance(total,type(None)):
                                        total = self.load(hybe=h,channel=c,file_type='stitched')/len(self.config.bitmap)
                                    else:
                                        total = total+(self.load(hybe=h,channel=c,file_type='stitched')/len(self.config.bitmap))
                                else:
                                    self.update_user(f" Missing {h} for generating total",value=50)
                                    total=None
                    if not isinstance(total,type(None)):
                        self.save(total,hybe=self.config.parameters['total_acq'],channel=self.config.parameters['total_channel'],file_type='stitched')
                        signal = total[self.config.parameters['border']:-self.config.parameters['border'],
                                        self.config.parameters['border']:-self.config.parameters['border']].numpy()
                        scale = (np.array(signal.shape)*self.config.parameters['ratio']).astype(int)
                        signal_down = np.array(Image.fromarray(signal.astype(float)).resize((scale[1],scale[0]), Image.BICUBIC))
                        self.save(signal_down,hybe=self.config.parameters['total_acq'],channel=self.config.parameters['total_channel'],file_type='image')
                        total[dapi_mask] = 0
                if not isinstance(nucstain,type(None)):
                    if 'total' in self.model_type:
                        model = models.Cellpose(model_type='cyto3',gpu=self.config.parameters['segment_gpu'])
                        self.mask = torch.zeros_like(total,dtype=torch.int32)
                        thresh = np.median(total[total!=0].numpy())+np.std(total[total!=0].numpy())
                    else:
                        model = models.Cellpose(model_type='nuclei',gpu=self.config.parameters['segment_gpu'])
                        self.mask = torch.zeros_like(nucstain,dtype=torch.int32)
                        thresh = np.median(nucstain[nucstain!=0].numpy())+np.std(nucstain[nucstain!=0].numpy())
                if not isinstance(model,type(None)):
                    window = 2000
                    x_step = round(nucstain.shape[0]/int(nucstain.shape[0]/window))
                    y_step = round(nucstain.shape[1]/int(nucstain.shape[1]/window))
                    n_x_steps = round(nucstain.shape[0]/x_step)
                    n_y_steps = round(nucstain.shape[1]/y_step)
                    Input = []
                    for x in range(n_x_steps):
                        for y in range(n_y_steps):
                            Input.append([x,y])
                    for (x,y) in self.generate_iterable(Input,'Segmenting Cells: '+self.model_type):
                        nuc = nucstain[(x*x_step):((x+1)*x_step),(y*y_step):((y+1)*y_step)].numpy()
                        if 'total' in self.model_type:
                            tot = total[(x*x_step):((x+1)*x_step),(y*y_step):((y+1)*y_step)].numpy()
                            stk = np.dstack([nuc,tot,np.zeros_like(nuc)])
                            diameter = int(self.config.parameters['segment_diameter']*1.5)
                            min_size = int(self.config.parameters['segment_min_size']*1.5)
                            channels = [1,2]
                        else:
                            stk = nuc
                            diameter = int(self.config.parameters['segment_diameter'])
                            min_size = int(self.config.parameters['segment_min_size'])
                            channels = [0,0]
                        if stk.max()==0:
                            continue
                        if np.min(stk.shape)==0:
                            continue
                        torch.cuda.empty_cache()
                        raw_mask_image,flows,styles,diams = model.eval(stk,
                                                            diameter=diameter,
                                                            channels=channels,
                                                            flow_threshold=1,
                                                            cellprob_threshold=0,
                                                            min_size=min_size,
                                                            batch_size=16)
                        mask = torch.tensor(raw_mask_image.astype(int),dtype=torch.int32)
                        updated_mask = mask.numpy().copy()
                        # Use Watershed to find missing cells 
                        if 'total' in self.model_type:
                            image = stk[:,:,1]
                        else:
                            image = stk
                        # Define Cell Borders
                        img = gaussian_filter(image.copy(),2)
                        cell_mask = img>thresh
                        cell_mask = morphology.remove_small_holes(cell_mask, 20)
                        cell_mask = morphology.remove_small_objects(cell_mask, int(self.config.parameters['segment_diameter']**1.5))
                        cell_mask = morphology.binary_dilation(cell_mask,footprint=create_circle_array(5, 5))
                        # Call Cell Centers
                        img = gaussian_filter(image.copy(),5)
                        img[~cell_mask]=0
                        if (np.sum(cell_mask)>1)&(mask.max().numpy()>5):
                            min_peak_height = np.percentile(img[cell_mask],5)
                            min_peak_distance = int(self.config.parameters['segment_diameter']/2)
                            peaks = peak_local_max(img, min_distance=min_peak_distance, threshold_abs=min_peak_height)
                            # Make Seeds for Watershed
                            seed = 0*np.ones_like(img)
                            seed[peaks[:,0],peaks[:,1]] = 1+np.array(range(peaks.shape[0]))
                            seed_max = morphology.binary_dilation(seed!=0,footprint=create_circle_array(int(1.5*self.config.parameters['segment_diameter']), int(1.5*self.config.parameters['segment_diameter'])))
                            for tx in range(-5,5):
                                for ty in range(-5,5):
                                    seed[peaks[:,0]+tx,peaks[:,1]+ty] = 1+np.array(range(peaks.shape[0]))
                            # Watershed
                            watershed_img = watershed(image=np.ones_like(img), markers=seed,mask=cell_mask&seed_max)
                            # Merge Cellpose and Watershed
                            tx,ty = np.where(watershed_img!=0)
                            if tx.shape[0]>0:
                                pixel_labels = watershed_img[tx,ty].copy()
                                cellpose_pixel_labels = mask.numpy()[tx,ty].copy()
                                current_label = cellpose_pixel_labels.max()
                                for i in np.unique(pixel_labels):
                                    m = pixel_labels==i
                                    if np.max(cellpose_pixel_labels[m])==0:
                                        current_label+=1
                                        cellpose_pixel_labels[m] = current_label
                                        # raise(ValueError('Finished'))
                                updated_mask[tx,ty] = cellpose_pixel_labels
                            # Remove Small Cells
                            tx,ty = np.where(updated_mask!=0)
                            if tx.shape[0]>0:
                                pixel_labels = updated_mask[tx,ty].copy()
                                for i in np.unique(pixel_labels):
                                    m = pixel_labels==i
                                    if np.sum(m)<self.config.parameters['segment_diameter']**1.5:
                                        pixel_labels[m] = 0
                                updated_mask[tx,ty] = pixel_labels

                            mask = torch.tensor(updated_mask.astype(int),dtype=torch.int32)

                        # Add tile to stitched
                        mask[mask>0] = mask[mask>0]+self.mask.max() # ensure unique labels
                        self.mask[(x*x_step):((x+1)*x_step),(y*y_step):((y+1)*y_step)] = mask
                        del raw_mask_image,flows,styles,diams
                        torch.cuda.empty_cache()
                    if 'nuclei' in self.model_type:
                        if self.check_existance(file_type='mask',model_type='total'):
                            total = self.load(file_type='mask',model_type='total')
                            total[self.mask==0] = 0 # Set non nuclear to 0
                            self.mask = total # replace mask with total&nuclear
                    self.save(self.mask,file_type='mask',model_type=self.model_type)

    def pull_vectors(self):
        """
        pull_vectors Using segmented cells pull pixels for each round for each cell and summarize into vector

        :param model_type: model for cellpose ('nuclei' 'total' 'cytoplasm'), defaults to 'nuclei'
        :type model_type: str, optional
        """      
        self.update_user(f"Attempting to Pull Vectors {self.model_type}",level=20)  
        if (not self.config.parameters['vector_overwrite'])&self.check_existance(file_type='anndata',model_type=self.model_type):
            self.data = self.load(file_type='anndata',model_type=self.model_type)
            self.update_user(f"Existing Data Found {self.model_type}",level=20)
        else:
            idxes = torch.where(self.mask!=0)
            labels = self.mask[idxes]

            """ Load Vector for each pixel """
            pixel_image_xy = torch.zeros([idxes[0].shape[0],2],dtype=torch.int32)
            pixel_image_xy[:,0] = self.load(hybe='image_x',channel='',file_type='stitched')[idxes]
            pixel_image_xy[:,1] = self.load(hybe='image_y',channel='',file_type='stitched')[idxes]

            pixel_vectors = torch.zeros([idxes[0].shape[0],len(self.config.bitmap)+1],dtype=torch.int32)
            nuc_pixel_vectors = torch.zeros([idxes[0].shape[0],len(self.config.bitmap)+1],dtype=torch.int32)
            for i,(r,h,c) in self.generate_iterable(enumerate(self.config.bitmap),'Generating Pixel Vectors',length=len(self.config.bitmap)):
                pixel_vectors[:,i] = self.load(hybe=h,channel=c,file_type='stitched')[idxes]
                nuc_pixel_vectors[:,i] = self.load(hybe=h,channel=self.config.parameters['nucstain_channel'],file_type='stitched')[idxes]
            pixel_vectors[:,-1] = self.load(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='stitched')[idxes]
            nuc_pixel_vectors[:,-1] = self.load(hybe=self.config.parameters['nucstain_acq'],channel=self.config.parameters['nucstain_channel'],file_type='stitched')[idxes]
            unique_labels = torch.unique(labels)
            self.vectors = torch.zeros([unique_labels.shape[0],pixel_vectors.shape[1]],dtype=torch.int32)
            self.nuc_vectors = torch.zeros([unique_labels.shape[0],nuc_pixel_vectors.shape[1]],dtype=torch.int32)
            pixel_xy = torch.zeros([idxes[0].shape[0],2])
            pixel_xy[:,0] = idxes[0]
            pixel_xy[:,1] = idxes[1]
            
            self.xy = torch.zeros([unique_labels.shape[0],2],dtype=torch.int32)
            self.image_xy = torch.zeros([unique_labels.shape[0],2],dtype=torch.int32)
            self.size = torch.zeros([unique_labels.shape[0],1],dtype=torch.int32)
            
            converter = {int(label):[] for label in unique_labels}
            for i in tqdm(range(labels.shape[0]),desc=str(datetime.now().strftime("%Y %B %d %H:%M:%S"))+' Generating Label Converter'):
                converter[int(labels[i])].append(i)
            for i,label in tqdm(enumerate(unique_labels),total=unique_labels.shape[0],desc=str(datetime.now().strftime("%H:%M:%S"))+' Generating Cell Vectors and Metadata'):
                m = converter[int(label)]
                self.vectors[i,:] = torch.median(pixel_vectors[m,:],axis=0).values
                self.nuc_vectors[i,:] = torch.median(nuc_pixel_vectors[m,:],axis=0).values
                pxy = pixel_xy[m,:]
                self.xy[i,:] = torch.median(pxy,axis=0).values
                pixy = pixel_image_xy[m,:]
                self.image_xy[i,:] = torch.median(pixy,axis=0).values
                self.size[i] = pxy.shape[0]
            cell_labels = [self.dataset+'_Section'+self.section+'_Cell'+str(i) for i in unique_labels.numpy()]
            self.cell_metadata = pd.DataFrame(index=cell_labels)
            self.cell_metadata['label'] = unique_labels.numpy()
            self.cell_metadata['stage_x'] = self.xy[:,0].numpy()
            self.cell_metadata['stage_y'] = self.xy[:,1].numpy()
            self.cell_metadata['image_x'] = self.image_xy[:,0].numpy()
            self.cell_metadata['image_y'] = self.image_xy[:,1].numpy()
            self.cell_metadata['size'] = self.size.numpy()
            self.cell_metadata['section_index'] = self.section
            self.cell_metadata['dapi'] = self.nuc_vectors[:,-1].numpy()
            self.vectors = self.vectors[:,0:-1]
            self.nuc_vectors = self.nuc_vectors[:,0:-1]
            self.data = anndata.AnnData(X=self.vectors.numpy(),
                                var=pd.DataFrame(index=np.array([r for r,h,c in self.config.bitmap])),
                                obs=self.cell_metadata)
            self.data.var['readout'] = [r for r,h,c in self.config.bitmap]
            self.data.var['hybe'] = [h for r,h,c in self.config.bitmap]
            self.data.var['channel'] = [c for r,h,c in self.config.bitmap]
            self.data.layers['processed_vectors'] = self.vectors.numpy()
            self.data.layers['nuc_processed_vectors'] = self.nuc_vectors.numpy()
            self.data.layers['raw'] = self.vectors.numpy()
            self.data.layers['nuc_raw'] = self.nuc_vectors.numpy()


            observations = ['PolyT','Housekeeping','Nonspecific_Encoding','Nonspecific_Readout'] # FIX upgrade to be everything but the rs in design 
            for obs in observations:
                if obs in self.data.var.index:
                    self.data.obs[obs.lower()] = self.data.layers['processed_vectors'][:,self.data.var.index==obs]

            self.data = self.data[:,self.data.var.index.isin(observations)==False]

            self.save(
                self.data.obs,
                file_type='metadata',
                model_type=self.model_type)
            self.save(
                self.data.var,
                file_type='metadata_bits',
                model_type=self.model_type)
            self.save(pd.DataFrame(
                self.data.layers['processed_vectors'],
                index=self.data.obs.index,
                columns=self.data.var.index),
                file_type='matrix',
                model_type=self.model_type)
            self.save(
                self.data,
                file_type='anndata',
                model_type=self.model_type)
    
    def generate_report(self):
        """ First Check if the report has been generated"""
        report_path = self.generate_filename(hybe='', channel='', file_type='Report', model_type=self.model_type)
        dpi = 200
        paths= []
        self.data.obs['sum'] = self.data.layers['processed_vectors'].sum(axis=1)
        additional_views = ['housekeeping','dapi','nonspecific_encoding','nonspecific_readout','sum','ieg']
        plt.close('all')
        if (os.path.exists(report_path)==False)|self.config.parameters['overwrite_report']:
            """ Vector """
            path = self.generate_filename(hybe='', channel='RawVectors', file_type='Figure', model_type=self.model_type)
            if (os.path.exists(path)==False)|self.config.parameters['overwrite_report']:
                self.update_user('Generating Raw Figure')
                n_bits = self.data.shape[1]
                n_figs = n_bits+len(additional_views)
                n_columns = 5
                n_rows = math.ceil(n_figs/n_columns)
                fig,axs = plt.subplots(n_columns,n_rows,figsize=[25,25])
                plt.suptitle('Stain Signal Log')
                axs = axs.ravel()
                for i in range(n_figs):
                    axs[i].axis('off')
                    axs[i].set_aspect('equal')
                x = self.data.obs['stage_x']
                y = self.data.obs['stage_y']
                X = basicu.normalize_fishdata_regress(self.data.layers['processed_vectors'].copy().astype(float),value='none',leave_log=True,log=True,bitwise=False)
                for i in range(n_bits):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i])
                    axs[i].set_title(np.array(self.data.var.index)[i])
                    axs[i].axis('off')
                shared_columns = [i for i in additional_views if i in self.data.obs.columns]
                X = basicu.normalize_fishdata_regress(np.array(self.data.obs[shared_columns]).astype(float),value='none',leave_log=True,log=True,bitwise=False)
                for i,obs in enumerate(shared_columns):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i+n_bits].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i+n_bits])
                    axs[i+n_bits].set_title(obs)
                    axs[i+n_bits].axis('off')
                paths.append(path)
                plt.savefig(path,dpi=dpi)
                plt.show()
            plt.close('all')
            """ Vector """
            path = self.generate_filename(hybe='', channel='NormVectors', file_type='Figure', model_type=self.model_type)
            if (os.path.exists(path)==False)|self.config.parameters['overwrite_report']:
                self.update_user('Generating SumNorm Figure')
                n_bits = self.data.shape[1]
                n_figs = n_bits+len(additional_views)
                n_columns = 5
                n_rows = math.ceil(n_figs/n_columns)
                fig,axs = plt.subplots(n_columns,n_rows,figsize=[25,25])
                plt.suptitle('Stain Signal Sum Normalized')
                axs = axs.ravel()
                for i in range(n_figs):
                    axs[i].axis('off')
                    axs[i].set_aspect('equal')
                x = self.data.obs['stage_x']
                y = self.data.obs['stage_y']
                X = basicu.normalize_fishdata_regress(self.data.layers['processed_vectors'].copy().astype(float),value='sum',leave_log=True,log=True,bitwise=False)
                for i in range(n_bits):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i])
                    axs[i].set_title(np.array(self.data.var.index)[i])
                    axs[i].axis('off')
                shared_columns = [i for i in additional_views if i in self.data.obs.columns]
                X = basicu.normalize_fishdata_regress(np.array(self.data.obs[shared_columns]).astype(float),value='sum',leave_log=True,log=True,bitwise=False)
                for i,obs in enumerate(shared_columns):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i+n_bits].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i+n_bits])
                    axs[i+n_bits].set_title(obs)
                    axs[i+n_bits].axis('off')
                paths.append(path)
                plt.savefig(path,dpi=dpi)
                plt.show()
            plt.close('all')
            """ Dapi  """
            path = self.generate_filename(hybe='', channel='Dapi', file_type='Figure', model_type=self.model_type)
            if (os.path.exists(path)==False)|self.config.parameters['overwrite_report']:
                self.update_user('Generating Dapi Figure')
                n_bits = self.data.shape[1]
                n_figs = n_bits+len(additional_views)
                n_columns = 5
                n_rows = math.ceil(n_figs/n_columns)
                fig,axs = plt.subplots(n_columns,n_rows,figsize=[25,25])
                plt.suptitle('Dapi Signal')
                axs = axs.ravel()
                for i in range(n_figs):
                    axs[i].axis('off')
                    axs[i].set_aspect('equal')
                x = self.data.obs['stage_x']
                y = self.data.obs['stage_y']
                X = basicu.normalize_fishdata_regress(self.data.layers['nuc_processed_vectors'].copy(),value='none',leave_log=True,log=True,bitwise=False)
                for i in range(n_bits):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i])
                    axs[i].set_title(np.array(self.data.var.index)[i])
                    axs[i].axis('off')
                shared_columns = [i for i in additional_views if i in self.data.obs.columns]
                X = basicu.normalize_fishdata_regress(np.array(self.data.obs[shared_columns]).astype(float),value='none',leave_log=True,log=True,bitwise=False)
                for i,obs in enumerate(shared_columns):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i+n_bits].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i+n_bits])
                    axs[i+n_bits].set_title(obs)
                    axs[i+n_bits].axis('off')
                paths.append(path)
                plt.savefig(path,dpi=dpi)
                plt.show()
            plt.close('all')
            """ Flat Field  """
            path = self.generate_filename(hybe='', channel='FlatFields', file_type='Figure', model_type=self.model_type)
            if (os.path.exists(path)==False)|self.config.parameters['overwrite_report']:
                self.update_user('Generating Flat Field Figure')
                if self.config.parameters['acq_FF']:
                    n_bits = len(self.config.bitmap)
                    n_figs = n_bits*2
                    n_columns = 10
                    n_rows = math.ceil(n_figs/n_columns)
                    fig,axs = plt.subplots(n_columns,n_rows,figsize=[25,25])
                    plt.suptitle('Image Processing Parameters')
                    axs = axs.ravel()
                    for i in range(n_figs):
                        axs[i].axis('off')
                        axs[i].set_aspect('equal')
                    ticker = 0
                    for readout,hybe,channel in self.config.bitmap:
                        img = self.load(hybe=hybe,channel='FarRed',file_type='image_FF')
                        vmin,vmax = np.percentile(img.ravel(),[5,95])
                        plot = axs[ticker].imshow(np.clip(img,vmin,vmax),cmap='jet')
                        axs[ticker].set_title(f"{readout} FlatField")
                        fig.colorbar(plot, ax=axs[ticker])
                        ticker+=1
                        img = self.load(hybe=hybe,channel='FarRed',file_type='image_constant')
                        vmin,vmax = np.percentile(img.ravel(),[5,95])
                        plot = axs[ticker].imshow(np.clip(img,vmin,vmax),cmap='jet')
                        axs[ticker].set_title(f"{readout} Constant")
                        fig.colorbar(plot, ax=axs[ticker])
                        ticker+=1
                else:
                    fig,axs  = plt.subplots(2,2,figsize=[25,25])
                    plt.suptitle('Image Processing Parameters')
                    axs = axs.ravel()

                    img = self.load(hybe='FF',channel='FarRed',file_type='image_FF')
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plot = axs[0].imshow(np.clip(img,vmin,vmax),cmap='jet')
                    axs[0].set_title('FarRed FF')
                    fig.colorbar(plot, ax=axs[0])

                    img = self.load(hybe='FF',channel='DeepBlue',file_type='image_FF')
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plot = axs[1].imshow(np.clip(img,vmin,vmax),cmap='jet')
                    axs[1].set_title('DeepBlue FF')
                    fig.colorbar(plot, ax=axs[1])

                    img = self.load(hybe='constant',channel='FarRed',file_type='image_constant')
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plot = axs[2].imshow(np.clip(img,vmin,vmax),cmap='jet')
                    axs[2].set_title('FarRed Constant')
                    fig.colorbar(plot, ax=axs[2])

                    img = self.load(hybe='constant',channel='DeepBlue',file_type='image_constant')
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plot = axs[3].imshow(np.clip(img,vmin,vmax),cmap='jet')
                    axs[3].set_title('DeepBlue Constant')
                    fig.colorbar(plot, ax=axs[3])

                paths.append(path)
                plt.savefig(path,dpi=dpi)
                plt.show()
            plt.close('all')
            """ Louvain """
            path = self.generate_filename(hybe='', channel='Louvain', file_type='Figure', model_type=self.model_type)
            if (os.path.exists(path)==False)|self.config.parameters['overwrite_report']:
                self.update_user('Generating Louvain Figure')
                if self.config.parameters['overwrite_louvain']| (not 'louvain' in self.data.obs.columns):
                    self.data.X = basicu.normalize_fishdata_regress(self.data.layers['processed_vectors'].copy(),value='sum',leave_log=True,log=True,bitwise=True)
                    sc.pp.neighbors(self.data, n_neighbors=15, use_rep='X')
                    sc.tl.umap(self.data, min_dist=0.1)
                    sc.tl.louvain(self.data,resolution=5,key_added='louvain')
                    self.data.X = self.data.layers['processed_vectors'].copy()
                cts = np.array(self.data.obs['louvain'].unique())
                louvain_counts = self.data.obs['louvain'].value_counts()
                unique_louvain_values_ordered = louvain_counts.sort_values(ascending=False).index.tolist()
                cts  = np.array(unique_louvain_values_ordered)
                colors = np.random.choice(np.array(list(mcolors.XKCD_COLORS.keys())),cts.shape[0],replace=False)
                # colors = np.array(list(mcolors.XKCD_COLORS.keys()))[0:cts.shape[0]]
                pallette = dict(zip(cts, colors))
                self.data.obs['louvain_colors'] = self.data.obs['louvain'].map(pallette)
                x = self.data.obs['stage_x']
                y = self.data.obs['stage_y']
                c = np.array(self.data.obs['louvain_colors'])
                fig,axs  = plt.subplots(1,3,figsize=[20,7])
                plt.suptitle('Unsupervised Classification')
                axs = axs.ravel()
                axs[0].scatter(self.data.obsm['X_umap'][:,0],self.data.obsm['X_umap'][:,1],c=c,s=0.1,marker='x')
                axs[0].set_title('UMAP Space')
                axs[0].axis('off')
                axs[1].scatter(x,y,s=0.1,c=c,marker='x')
                axs[1].set_title('Physical Space')
                axs[1].set_aspect('equal')
                axs[1].axis('off')
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pallette[key], markersize=10, label=key) for key in cts]
                axs[2].legend(handles=handles, loc='center',ncol=3, fontsize=8)
                axs[2].axis('off')
                paths.append(path)
                plt.savefig(path,dpi=dpi)

                fig,axs  = plt.subplots(8,8,figsize=[25,25])
                plt.suptitle('Unsupervised Classification')
                axs = axs.ravel()
                x = np.array(self.data.obs['stage_x'])
                y = np.array(self.data.obs['stage_y'])
                c = np.array(self.data.obs['louvain_colors'])
                for i in range(64):
                    if i>cts.shape[0]:
                        continue
                    ct = cts[i]
                    m = np.array(self.data.obs['louvain'])==ct
                    axs[i].scatter(x[m],y[m],c=c[m],s=0.1,marker='x')
                    axs[i].set_title(ct)
                    axs[i].axis('off')
                    axs[i].set_aspect('equal')
                path = self.generate_filename(hybe='Louvain_Split', channel='', file_type='Figure', model_type=self.model_type)
                paths.append(path)
                plt.savefig(path,dpi=dpi)
                plt.show()

            plt.close('all')

            self.save(
                self.data.obs,
                file_type='metadata',
                model_type=self.model_type)
            self.save(
                self.data.var,
                file_type='metadata_bits',
                model_type=self.model_type)
            self.save(pd.DataFrame(
                self.data.layers['processed_vectors'],
                index=self.data.obs.index,
                columns=self.data.var.index),
                file_type='matrix',
                model_type=self.model_type)
            self.save(
                self.data,
                file_type='anndata',
                model_type=self.model_type)
            
            # """ Generate Report """
            # from reportlab.pdfgen import canvas
            # def convert_pngs_to_pdf_reportlab(image_paths, output_filename):
            #     """Converts a list of PNG image paths to a single PDF with layout options.

            #     Args:
            #         image_paths: A list of strings containing file paths to PNG images.
            #         output_filename: The desired name for the output PDF file.
            #     """
            #     pdf = canvas.Canvas(output_filename)
            #     page_width, page_height = pdf.pagesize
            #     x, y = 0, page_height  # Starting coordinates for image placement

            #     for image_path in image_paths:
            #         img = canvas.drawImage(image_path, x, y)
            #         y -= img.height  # Adjust y position for next image

            #     pdf.save()
            # convert_pngs_to_pdf_reportlab(paths, report_path)

                

    def remove_temporary_files(self,data_types = ['stitched','stitched_raw','FF']):
        """
        remove_temporary_files Remove Temporary Processing Files To Save Disk Space

        :param data_types: Data Types to Remove, defaults to ['stitched','stitched_raw','FF']
        :type data_types: list, optional
        """
        for file_type in  self.generate_iterable(data_types,message='Removing Temporary Files'):
            fname = self.generate_filename(hybe='', channel='', file_type=file_type, model_type='')
            dirname = os.path.dirname(fname)
            shutil.rmtree(dirname)

def generate_constant_only(acq,image_metadata=None,channel=None,posnames=[],bkg_acq='',parameters={},verbose=False):
    if 'mask' in channel:
        return ''
    else:
        if len(posnames)==0:
            posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
        FF = []
        if verbose:
            iterable = tqdm(posnames,desc=str(datetime.now().strftime("%Y %B %d %H:%M:%S"))+' Generating FlatField '+acq+' '+channel)
        else:
            iterable = posnames
        for posname in iterable:
            try:
                img = image_metadata.stkread(Position=posname,Channel=channel,acq=acq).min(2).astype(float)
                img = median_filter(img,2)
                img = torch.tensor(img)
                if bkg_acq!='':
                    bkg = image_metadata.stkread(Position=posname,Channel=channel,acq=bkg_acq).mean(2).astype(float)
                    # bkg = median_filter(bkg,2)
                    bkg = torch.tensor(bkg)
                    img = img-bkg
                FF.append(img)
            except Exception as e:
                print(posname,acq,bkg_acq)
                print(e)
                continue
        # FF = torch.quantile(torch.dstack(FF),0.5,dim=2).numpy() # Assumption is that for each pixel half of the images wont have a cell there
        FF = torch.dstack(FF)
        constant = torch.min(FF,dim=2).values # There may be a more robust way 
        constant = gaussian_filter(constant,50,mode='nearest')  # causes issues with corners
        return constant


def generate_FF_only(acq,image_metadata=None,channel=None,constant=0,posnames=[],bkg_acq='',parameters={},verbose=False):
    """
    generate_FF Generate flat field to correct uneven illumination

    :param image_metadata: Data Loader Class
    :type image_metadata: Metadata Class
    :param acq: name of acquisition
    :type acq: str
    :param channel: name of channel
    :type channel: str
    :return: flat field image
    :rtype: np.array
    """
    if 'mask' in channel:
        return ''
    else:
        if len(posnames)==0:
            posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
        FF = []
        if verbose:
            iterable = tqdm(posnames,desc=str(datetime.now().strftime("%Y %B %d %H:%M:%S"))+' Generating FlatField '+acq+' '+channel)
        else:
            iterable = posnames
        for posname in iterable:
            try:
                img = image_metadata.stkread(Position=posname,Channel=channel,acq=acq).min(2).astype(float)
                img = median_filter(img,2)
                img = torch.tensor(img)
                if bkg_acq!='':
                    bkg = image_metadata.stkread(Position=posname,Channel=channel,acq=bkg_acq).mean(2).astype(float)
                    # bkg = median_filter(bkg,2)
                    bkg = torch.tensor(bkg)
                    img = img-bkg
                FF.append(img)
            except Exception as e:
                print(posname,acq,bkg_acq)
                print(e)
                continue
        # FF = torch.quantile(torch.dstack(FF),0.5,dim=2).numpy() # Assumption is that for each pixel half of the images wont have a cell there
        FF = torch.dstack(FF)
        FF = torch.mean(FF,dim=2).numpy() # There may be a more robust way in case of debris 
        if np.max(constant.ravel())>0:
            if isinstance(constant, torch.Tensor):
                constant = constant.numpy().copy()
            FF = FF-constant
        FF = gaussian_filter(FF,50,mode='nearest') # causes issues with corners
        vmin,vmid,vmax = np.percentile(FF[np.isnan(FF)==False],[0.1,50,99.9]) 
        # Maybe add median filter to FF 
        FF[FF<vmin] = vmin
        FF[FF>vmax] = vmax
        FF[FF==0] = vmid
        FF = vmid/FF
        return FF

def optional_tqdm(iterable,verbose=True,desc='',total=0):
    if verbose:
        return tqdm(iterable,desc=str(datetime.now().strftime("%Y %B %d %H:%M:%S"))+' '+desc,total=total)
    else:
        return iterable

def generate_FF_constant(image_metadata,channel,posnames=[],bkg_acq='',parameters={},verbose=False,ncpu=6):
    """
    generate_FF Generate flat field to correct uneven illumination

    :param image_metadata: Data Loader Class
    :type image_metadata: Metadata Class
    :param acq: name of acquisition
    :type acq: str
    :param channel: name of channel
    :type channel: str
    :return: flat field image
    :rtype: np.array
    """
    if 'mask' in channel:
        return ''
    if len(posnames)>0:
        image_metadata_table = image_metadata.image_table[image_metadata.image_table.Position.isin(posnames)]
    acqs = image_metadata_table.acq.unique()
    strip_acqs = [i for i in acqs if 'strip' in i.lower()]
    hybe_acqs = [i for i in acqs if 'hybe' in i.lower()]
    pfunc = partial(generate_constant_only,image_metadata=image_metadata,channel=channel,posnames=posnames,bkg_acq='',parameters=parameters,verbose=False)
    constants = []
    with multiprocessing.Pool(ncpu) as p:
        for constant in optional_tqdm(p.map(pfunc,strip_acqs),desc='Generating Image Constant',total=len(strip_acqs),verbose=verbose):
            constants.append(constant)
    constant = np.dstack(constants)
    constant = np.mean(constant,axis=2)
    pfunc = partial(generate_FF_only,constant=constant,image_metadata=image_metadata,channel=channel,posnames=posnames,bkg_acq='',parameters=parameters,verbose=False)
    FFs = []
    with multiprocessing.Pool(ncpu) as p:
        for FF in optional_tqdm(p.map(pfunc,hybe_acqs),desc='Generating Flat Field',total=len(hybe_acqs),verbose=verbose):
            FFs.append(FF)
    FF = np.dstack(FFs)
    FF = np.mean(FF,axis=2)
    return FF,constant

def generate_FF(image_metadata,acq,channel,posnames=[],bkg_acq='',parameters={},verbose=False):
    """
    generate_FF Generate flat field to correct uneven illumination

    :param image_metadata: Data Loader Class
    :type image_metadata: Metadata Class
    :param acq: name of acquisition
    :type acq: str
    :param channel: name of channel
    :type channel: str
    :return: flat field image
    :rtype: np.array
    """
    if 'mask' in channel:
        return ''
    else:
        if len(posnames)==0:
            posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
        FF = []
        if verbose:
            iterable = tqdm(posnames,desc=str(datetime.now().strftime("%Y %B %d %H:%M:%S"))+' Generating FlatField '+acq+' '+channel)
        else:
            iterable = posnames
        for posname in iterable:
            try:
                img = image_metadata.stkread(Position=posname,Channel=channel,acq=acq).min(2).astype(float)
                img = median_filter(img,2)
                img = torch.tensor(img)
                if bkg_acq!='':
                    bkg = image_metadata.stkread(Position=posname,Channel=channel,acq=bkg_acq).mean(2).astype(float)
                    # bkg = median_filter(bkg,2)
                    bkg = torch.tensor(bkg)
                    img = img-bkg
                FF.append(img)
            except Exception as e:
                print(posname,acq,bkg_acq)
                print(e)
                continue
        # FF = torch.quantile(torch.dstack(FF),0.5,dim=2).numpy() # Assumption is that for each pixel half of the images wont have a cell there
        FF = torch.dstack(FF)
        bkg = torch.min(FF,dim=2).values # There may be a more robust way 
        bkg = gaussian_filter(bkg,50,mode='nearest')  # causes issues with corners
        if parameters['constant_FF']:
            FF = FF-bkg[:,:,None]
            FF = torch.mean(FF,dim=2).numpy() # There may be a more robust way in case of debris 
            FF = gaussian_filter(FF,50,mode='nearest') # causes issues with corners
            FF = bkg + FF
        else:
            FF = FF-bkg[:,:,None]
            FF = torch.mean(FF,dim=2).numpy() # There may be a more robust way in case of debris 
            FF = gaussian_filter(FF,50,mode='nearest') # causes issues with corners
        vmin,vmid,vmax = np.percentile(FF[np.isnan(FF)==False],[0.1,50,99.9]) 
        # Maybe add median filter to FF 
        FF[FF<vmin] = vmin
        FF[FF>vmax] = vmax
        FF[FF==0] = vmid
        FF = vmid/FF
        return FF,bkg

def process_img(img,parameters,nuc=None,FF=1,constant=0):
    """
    process_img _summary_

    :param img: _description_
    :type img: _type_
    :param parameters: _description_
    :type parameters: _type_
    :param FF: _description_, defaults to 1
    :type FF: int, optional
    :param constant: _description_, defaults to 0
    :type constant: int, optional
    :return: _description_
    :rtype: _type_
    """
    # Remove Constant
    img = img-constant
    # FlatField 
    img = img*FF
    # Remove Dead Pixels
    img = median_filter(img,2) 
    # Smooth
    if parameters['highpass_smooth']>0:
        if parameters['highpass_smooth_function'] == 'gaussian':
            img = gaussian_filter(img,parameters['highpass_smooth']) 
        elif parameters['highpass_smooth_function'] == 'median':
            img = median_filter(img,parameters['highpass_smooth']) 
        elif parameters['highpass_smooth_function'] == 'minimum':
            img = minimum_filter(img,size=parameters['highpass_smooth']) 
        elif 'percentile' in parameters['highpass_smooth_function']:
            img = percentile_filter(img,int(parameters['highpass_function'].split('_')[-1]),size=parameters['highpass_smooth'])
        elif 'rolling_ball' in parameters['highpass_smooth_function']:
            img = gaussian_filter(restoration.rolling_ball(gaussian_filter(img,parameters['highpass_smooth']/5),radius=parameters['highpass_smooth'],num_threads=30),parameters['highpass_smooth'])
        
    # Background Subtract
    if parameters['highpass_sigma']>0:
        if parameters['highpass_function'] == 'gaussian':
            bkg = gaussian_filter(img,parameters['highpass_sigma']) 
        elif parameters['highpass_function'] == 'median':
            bkg = median_filter(img,parameters['highpass_sigma']) 
        elif parameters['highpass_function'] == 'minimum':
            bkg = minimum_filter(img,size=parameters['highpass_sigma']) 
        elif 'percentile' in parameters['highpass_function']:
            bkg = percentile_filter(img,int(parameters['highpass_function'].split('_')[-1]),size=parameters['highpass_sigma'])
        elif 'rolling_ball' in parameters['highpass_function']:
            bkg = gaussian_filter(restoration.rolling_ball(gaussian_filter(img,parameters['highpass_sigma']/5),radius=parameters['highpass_sigma'],num_threads=30),parameters['highpass_sigma'])
        else:
            bkg = 0
        img = img-bkg
    return img

def preprocess_images(data,FF=1,nuc_FF=1,constant=0,nuc_constant=0):
    """
    preprocess_images _summary_

    :param data: _description_
    :type data: _type_
    :param FF: _description_, defaults to 1
    :type FF: int, optional
    :param nuc_FF: _description_, defaults to 1
    :type nuc_FF: int, optional
    :param constant: _description_, defaults to 0
    :type constant: int, optional
    :param nuc_constant: _description_, defaults to 0
    :type nuc_constant: int, optional
    :return: _description_
    :rtype: _type_
    """    
    acq = data['acq']
    bkg_acq = data['bkg_acq']
    posname = data['posname']
    image_metadata = data['image_metadata']
    channel = data['channel']
    parameters = data['parameters']
    try:
        nuc = ((image_metadata.stkread(Position=posname,
                                        Channel=parameters['nucstain_channel'],
                                        acq=acq).max(axis=2).astype(float)))
        nuc = process_img(nuc,parameters,FF=nuc_FF,constant=nuc_constant)
        img = ((image_metadata.stkread(Position=posname,
                                        Channel=channel,
                                        acq=acq).max(axis=2).astype(float)))
        img = process_img(img,parameters,FF=FF,constant=constant)
        if not bkg_acq=='':
            bkg = ((image_metadata.stkread(Position=posname,
                                            Channel=channel,
                                            acq=bkg_acq).max(axis=2).astype(float)))
            bkg = process_img(bkg,parameters,FF=FF,constant=constant)
            bkg_nuc = ((image_metadata.stkread(Position=posname,
                                            Channel=parameters['nucstain_channel'],
                                            acq=bkg_acq).max(axis=2).astype(float)))
            bkg_nuc = process_img(bkg_nuc,parameters,FF=nuc_FF,constant=nuc_constant)
            # Check if beads work here
            shift, error = register(nuc, bkg_nuc,10)
            if error!=np.inf:
                translation_x = int(shift[1])
                translation_y = int(shift[0])
                x_correction = np.array(range(bkg.shape[1]))+translation_x
                y_correction = np.array(range(bkg.shape[0]))+translation_y
                i2 = interpolate.interp2d(x_correction,y_correction,bkg,fill_value=None)
                bkg = i2(range(bkg.shape[1]), range(bkg.shape[0]))
            img = img-bkg
        for iter in range(parameters['background_estimate_iters']):
            img = img-gaussian_filter(
                restoration.rolling_ball(
                gaussian_filter(img,
                                parameters['highpass_sigma']/5),
                                radius=parameters['highpass_sigma'],
                                num_threads=30),
                parameters['highpass_sigma'])
            nuc = nuc-gaussian_filter(
                restoration.rolling_ball(
                gaussian_filter(nuc,
                                parameters['highpass_sigma']/5),
                                radius=parameters['highpass_sigma'],
                                num_threads=30),
                parameters['highpass_sigma'])
        dtype = 'int32'
        nuc[nuc<np.iinfo(dtype).min] = np.iinfo(dtype).min
        img[img<np.iinfo(dtype).min] = np.iinfo(dtype).min
        nuc[nuc>np.iinfo(dtype).max] = np.iinfo(dtype).max
        img[img>np.iinfo(dtype).max] = np.iinfo(dtype).max
        
        img = torch.tensor(img,dtype=torch.int32)#+np.iinfo('int16').min
        nuc = torch.tensor(nuc,dtype=torch.int32)#+np.iinfo('int16').min
        
        if parameters['stitch_rotate']!=0:
            img = torch.rot90(img,parameters['stitch_rotate'])
            nuc = torch.rot90(nuc,parameters['stitch_rotate'])
        if parameters['stitch_flipud']:
            img = torch.flipud(img)
            nuc = torch.flipud(nuc)
        if parameters['stitch_fliplr']:
            img = torch.fliplr(img)
            nuc = torch.fliplr(nuc)
        if 'destination' in data.keys():
            destination = data['destination']
            """ Register to Correct Stage Error """
            mask = destination!=0
            overlap = mask.sum()/destination.ravel().shape[0]
            if overlap>parameters['overlap']:
                mask = destination!=0
                mask_x = destination.max(1).values!=0
                ref = destination[mask_x,:]
                mask_y = ref.max(0).values!=0
                ref = destination[mask_x,:]
                ref = ref[:,mask_y]
                non_ref = nuc[mask_x,:]
                non_ref = non_ref[:,mask_y]
                # Check if Beads work here
                shift, error = register(ref.numpy(), non_ref.numpy(),10)
                if (error!=np.inf)&(np.max(np.abs(shift))<=(parameters['border']/2)):
                    translation_y = int(shift[1])
                    translation_x = int(shift[0])
                else:
                    translation_x = ''
                    translation_y = ''
                    print(shift,error)
            else:
                translation_x = ''
                translation_y = ''
                print(posname,'Insufficient Overlap ',str(int(overlap*1000)/10),'%')
        else:
            translation_x = 0
            translation_y = 0
        return posname,nuc,img,translation_x,translation_y
    except Exception as e:
        print(e,posname)
        return posname,0,0,0,0,0,0

    
def visualize_merge(img1,img2,color1=np.array([1,0,1]),color2=np.array([0,1,1]),figsize=[20,20],pvmin=5,pvmax=95,title=''):
    """
    visualize_merge _summary_

    :param img1: _description_
    :type img1: _type_
    :param img2: _description_
    :type img2: _type_
    :param color1: _description_, defaults to np.array([1,0,1])
    :type color1: _type_, optional
    :param color2: _description_, defaults to np.array([0,1,1])
    :type color2: _type_, optional
    :param figsize: _description_, defaults to [20,20]
    :type figsize: list, optional
    :param pvmin: _description_, defaults to 5
    :type pvmin: int, optional
    :param pvmax: _description_, defaults to 95
    :type pvmax: int, optional
    :param title: _description_, defaults to ''
    :type title: str, optional
    """
    vmin,vmax = np.percentile(img1,[pvmin,pvmax])
    img1 = img1-vmin
    img1 = img1/vmax
    img1[img1<0] = 0
    img1[img1>1] = 1

    vmin,vmax = np.percentile(img2,[pvmin,pvmax])
    img2 = img2-vmin
    img2 = img2/vmax
    img2[img2<0] = 0
    img2[img2>1] = 1

    color_stk1 = color1*np.dstack([img1,img1,img1])
    color_stk2 = color2*np.dstack([img2,img2,img2])
    color_stk = color_stk1+color_stk2
    color_stk[color_stk<0] = 0
    color_stk[color_stk>1] = 1

    plt.figure(figsize=figsize)
    if title!='':
        plt.title(title)
    plt.imshow(color_stk)
    plt.show()
    
def create_circle_array(size, diameter):
    center = size // 2
    radius = diameter // 2
    array = np.zeros((size, size), dtype=int)

    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance <= radius:
                array[i, j] = 1

    return array

def colorize_segmented_image(img, color_type='rgb'):
    """
    Returns a randomly colorized segmented image for display purposes.
    :param img: Should be a numpy array of dtype np.int and 2D shape segmented
    :param color_type: 'rg' for red green gradient, 'rb' = red blue, 'bg' = blue green
    :return: Randomly colorized, segmented image (shape=(n,m,3))
    """
    # def split(a, n):
    #     k, m = divmod(len(a), n)
    #     return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    img = torch.tensor(img)
    # get empty rgb_img as a skeleton to add colors
    rgb_img = torch.zeros((img.shape[0], img.shape[1], 3),dtype=torch.int16)
    x,y = torch.where(img!=0)
    l = img[x,y]
    rgb = rgb_img[x,y,:]
    # make your colors
    cells = torch.unique(l).numpy()
    np.random.shuffle(cells)
    for cell in tqdm(cells,total=cells.shape[0]):
        rgb[l==cell,:] = torch.tensor(np.random.randint(0, 255, (3)),dtype=torch.int16).abs()
    # rgb[rgb<2] = 0
    rgb_img[x,y,:] = rgb
    return rgb_img.numpy().astype(np.int16)

def bin_pixels_pytorch(image, n):
    # Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.from_numpy(image)

    # Get the dimensions of the original image
    height, width, channels = image_tensor.shape

    # Calculate the new height and width after binning
    new_height = height // n
    new_width = width // n

    # Reshape the image to a new shape with n x n bins and 3 channels
    binned_image = image_tensor[:new_height * n, :new_width * n].view(new_height, n, new_width, n, channels)

    # Sum the pixels within each bin along the specified axes (axis 1 and axis 3)
    binned_image = binned_image.sum(dim=(1, 3))

    return binned_image.numpy()