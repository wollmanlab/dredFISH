import numpy as np
import torch
# from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import gaussian_blur
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
import inspect
from skimage import (
    data, restoration, util
)
from scipy.optimize import curve_fit

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
        self.well = self.section.split('-')[0]
        self.image_metadata = ''
        self.reference_stitched=''
        self.FF=''
        self.nuc_FF=''
        self.constant=''
        self.nuc_constant=''
        self.data = None
        self.verbose=verbose
        self.path = None
        self.cword_config = cword_config
        config = importlib.import_module(self.cword_config)
        self.parameters = config.parameters
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger("Processing")

        self.dtype_converter = {'float64':torch.float64,'float32':torch.float32,'float16':torch.float16,'int32':torch.int32}
        self.dtype_converter_numpy = {'float64':np.float64,'float32':np.float32,'float16':np.float16,'int32':np.int32}

    def run(self):
        try:
            self.main()
        except Exception as e:
            self.update_user('Catastrophic Error')
            self.update_user(str(e))
            self.update_user('Error occurred on line: {}'.format(sys.exc_info()[-1].tb_lineno))
            print(str(e))
            raise SystemError('Catastrophic Failure')

    def main(self):
        """
        run Primary Function to Process DredFISH Sections
        """
        self.setup_output()
        self.load_metadata()
        if len(self.posnames)>0:
            completed = False
            for model_type in self.parameters['model_types']:
                self.model_type = model_type
                if (not self.parameters['vector_overwrite']):
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
                for model_type in self.parameters['model_types']:
                    self.model_type = model_type
                    if (not self.parameters['vector_overwrite']):
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
                self.generate_report()
                self.remove_temporary_files()
                # self.copy_to_drive()
        else:
            self.update_user('No positions found for this section',level=50)

    def setup_output(self):
        config = importlib.import_module(self.cword_config)
        if isinstance(self.path,type(None)):
            self.path = config.parameters['fishdata']
        if self.dataset in self.path:
            """ Assume this is a full path """
            if not os.path.exists(self.path):
                self.update_user('User Defined Path Doesnt Exist')
                raise ValueError(self.path)
        else:
            if self.path == 'fishdata':
                self.path = self.path+str(datetime.today().strftime("_%Y%b%d"))

            """ Assume only fishdata name provided """
            self.scratch_path = os.path.join(self.parameters['scratch_path'])
            if not os.path.exists(self.scratch_path):
                self.update_user('Making Scratch Path',level=20)
                os.mkdir(self.scratch_path)
            self.scratch_path = os.path.join(self.scratch_path,self.dataset)
            if not os.path.exists(self.scratch_path):
                self.update_user('Making Scratch Path',level=20)
                os.mkdir(self.scratch_path)
            self.scratch_path = os.path.join(self.scratch_path,self.path)
            if not os.path.exists(self.scratch_path):
                self.update_user('Making Scratch Path',level=20)
                os.mkdir(self.scratch_path)
            self.scratch_path = os.path.join(self.scratch_path,self.section)
            if not os.path.exists(self.scratch_path):
                self.update_user('Making Scratch Path',level=20)
                os.mkdir(self.scratch_path)

            """ Assume only fishdata name provided """
            self.path = os.path.join(self.metadata_path,self.path)
            if not os.path.exists(self.path):
                self.update_user('Making fishdata Path',level=20)
                os.mkdir(self.path)
            self.path = os.path.join(self.path,'Processing')
            if not os.path.exists(self.path):
                self.update_user('Making Processing Path',level=20)
                os.mkdir(self.path)
            self.well_path = os.path.join(self.path,self.well)
            if not os.path.exists(self.well_path):
                self.update_user('Making Well Path',level=20)
                os.mkdir(self.well_path)
            self.path = os.path.join(self.path,self.section)
            if not os.path.exists(self.path):
                self.update_user('Making Section Path',level=20)
                os.mkdir(self.path)

        fileu_config = fileu.interact_config(self.path,return_config=True)
        if not 'version' in fileu_config.keys():
            fileu_config = fileu.interact_config(self.path,key='version',data=self.parameters['fileu_version'],return_config=True)

        fileu_config = fileu.interact_config(self.scratch_path,return_config=True)
        if not 'version' in fileu_config.keys():
            fileu_config = fileu.interact_config(self.scratch_path,key='version',data=self.parameters['fileu_version'],return_config=True)

        if config.parameters['config_overwrite']:
            self.parameters = config.parameters
        else:
            if self.check_existance(channel='parameters',file_type='Config'):
                self.parameters = self.load(channel='parameters',file_type='Config')
            else:
                self.parameters = config.parameters

        self.parameters['path'] = self.path
        self.parameters['well_path'] = self.well_path
        self.parameters['scratch_path'] = self.scratch_path
        self.parameters['section'] = self.dataset
        self.parameters['section'] = self.section
        self.parameters['well'] = self.well

        self.save(self.parameters,channel='parameters',file_type='Config')
        logging.basicConfig(
                    filename=os.path.join(self.path,'processing_log.txt'),filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y %B %d %H:%M:%S',level=logging.INFO, force=True)
        self.log = logging.getLogger("Processing")

    def update_user(self,message,level=20):
        """
        update_user Wrapper to fileu.update_user

        :param message: Message to be logged
        :type message: str
        :param level: priority 0-50, defaults to 20
        :type level: int, optional
        """
        if self.verbose:
            print(datetime.now().strftime("%Y %B %d %H:%M:%S") + ' ' + message)
        fileu.update_user(message,level=level,logger=self.log)

    def check_existance(self,hybe='',channel='',file_type='',model_type='',path=None):
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
        if isinstance(path,type(None)):
            path = self.path
        if file_type in ['FF','constant','image_FF','image_constant']:
            path = self.well_path
        elif file_type in ['stitched']:
            path = self.scratch_path
        return fileu.check_existance(path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def generate_filename(self,hybe,channel,file_type,model_type='',path=None):
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
        if len(self.parameters['model_types'])==1:
            model_type = ''
        if isinstance(path,type(None)):
            path = self.path
        if file_type in ['FF','constant','image_FF','image_constant']:
            path = self.well_path
        elif file_type in ['stitched']:
            path = self.scratch_path
        return fileu.generate_filename(path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def save(self,data,hybe='',channel='',file_type='',model_type='',path=None):
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
        if len(self.parameters['model_types'])==1:
            model_type = ''
        if isinstance(path,type(None)):
            path = self.path
        if file_type in ['FF','constant','image_FF','image_constant']:
            path = self.well_path
        elif file_type in ['stitched']:
            path = self.scratch_path
        fileu.save(data,path=path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

    def load(self,hybe='',channel='',file_type='anndata',model_type='',path=None):
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
        if len(self.parameters['model_types'])==1:
            model_type = ''
        if isinstance(path,type(None)):
            path = self.path
        if file_type in ['FF','constant','image_FF','image_constant']:
            path = self.well_path
        elif file_type in ['stitched']:
            path = self.scratch_path
        return fileu.load(path=path,hybe=hybe,channel=channel,file_type=file_type,model_type=model_type,logger=self.log)

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
        mask = self.data.obs['dapi']>self.parameters['dapi_thresh'] 
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
                self.coordinates[posname] = (self.image_metadata.image_table[(self.image_metadata.image_table.Position==posname)].XY.iloc[0]/self.parameters['pixel_size']).astype(int)

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
                acq_metadata = self.image_metadata.image_table[self.image_metadata.image_table.acq==acq]
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

    def generate_image_parameters(self,hybe,channel):
        if channel == self.parameters['nucstain_channel']:
            self_FF = self.nuc_FF
            self_constant = self.nuc_constant
        else:
            self_FF = self.FF
            self_constant = self.constant
        if isinstance(self_FF,str):
            save_FF = True
        else:
            save_FF = False
        if isinstance(self_constant,str):
            save_constant = True
        else:
            save_constant = False
        if self.parameters['acq_FF']:
            self_FF = ''
        if self.parameters['acq_constant']:
            self_constant = ''
        if not self.parameters['use_FF']:
            self_FF = torch.ones(self.parameters['n_pixels']).T
        if not self.parameters['use_constant']:
            self_constant = torch.zeros(self.parameters['n_pixels']).T
        if isinstance(self_FF,str)|isinstance(self_constant,str):
            if (self.parameters['overwrite']==False)&self.check_existance(hybe=hybe,channel=channel,file_type='FF'):
                self.update_user(f"Found Existing {'FF'} for {hybe} {channel}")
                self_FF = self.load(hybe=hybe,channel=channel,file_type='FF')
            if (self.parameters['overwrite']==False)&self.check_existance(hybe=hybe,channel=channel,file_type='constant'):
                self.update_user(f"Found Existing {'constant'} for {hybe} {channel}")
                self_constant = self.load(hybe=hybe,channel=channel,file_type='constant')
            if isinstance(self_FF,str)|isinstance(self_constant,str):
                self.update_user(f"Calculating {'FF&constant'} for {self.find_acq(hybe,protocol='hybe')} {channel}")
                if self.parameters['post_strip_FF']&(channel!=self.parameters['nucstain_channel']):
                    FF,constant = generate_FF_parallel(self.image_metadata,self.find_acq(hybe,protocol='hybe'),
                                            channel,
                                            bkg_acq=self.find_acq(hybe,protocol='strip'),
                                            parameters=self.parameters)
                else:
                    FF,constant = generate_FF_parallel(self.image_metadata,self.find_acq(hybe,protocol='hybe'),
                                            channel,
                                            bkg_acq='',
                                            parameters=self.parameters)
                if isinstance(self_FF,str):
                    self_FF = FF
                if isinstance(self_constant,str):
                    self_constant = constant
        
        if self.parameters['acq_FF']|save_FF:
            self.save(self_FF,hybe=hybe,channel=channel,file_type='FF')
            self.save((self_FF*1000),hybe=hybe,channel=channel,file_type='image_FF')
        if self.parameters['acq_constant']|save_constant:
            self.save(self_constant,hybe=hybe,channel=channel,file_type='constant')
            self.save((self_constant*1000),hybe=hybe,channel=channel,file_type='image_constant')

        if isinstance(self_FF,torch.Tensor):
            self_FF = self_FF.numpy()
        if isinstance(self_constant,torch.Tensor):
            self_constant = self_constant.numpy()

        if channel == self.parameters['nucstain_channel']:
            self.nuc_FF = self_FF
            self.nuc_constant = self_constant
        else:
            self.FF = self_FF
            self.constant = self_constant

    def image_parameters(self,hybe,channel):
        self.generate_image_parameters(hybe,channel)
        self.generate_image_parameters(hybe,self.parameters['nucstain_channel'])

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
        if self.parameters['strip']&(bkg_acq=='')&(acq==''):
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
                channel=self.parameters['nucstain_channel'],
                file_type='stitched'
            )
            signal_exists = self.check_existance(
                hybe=hybe,
                channel=channel,
                file_type='stitched'
            )
            if (not self.parameters['overwrite'])&(nuc_exists&signal_exists):
                self.update_user('Found Existing '+hybe+' Stitched')
                if (hybe==self.parameters['nucstain_acq'])&(self.reference_stitched==''):
                    nuc = self.load(
                        hybe=hybe,
                        channel=self.parameters['nucstain_channel'],
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
                self.image_parameters(hybe,channel)
                xy = np.stack([pxy for i,pxy in self.coordinates.items()])
                if (self.parameters['stitch_rotate'] % 2) == 0:
                    img_shape = np.flip(self.parameters['n_pixels'])
                else:
                    img_shape = self.parameters['n_pixels']
                y_min,x_min = xy.min(0)-self.parameters['border']
                y_max,x_max = xy.max(0)+img_shape+self.parameters['border']
                x_range = np.array(range(x_min,x_max+1))
                y_range = np.array(range(y_min,y_max+1))
                stitched = torch.zeros([len(x_range),len(y_range),2],dtype=self.dtype_converter[self.parameters['dtype']])
                if isinstance(self.reference_stitched,str):
                    pixel_coordinates_stitched = torch.zeros([len(x_range),len(y_range),2],dtype=self.dtype_converter[self.parameters['dtype']])
                Input = []
                for posname in self.posnames:
                    data = {}
                    data['acq'] = acq
                    data['bkg_acq'] = bkg_acq
                    data['posname'] = posname
                    data['image_metadata'] = self.image_metadata
                    data['channel'] = channel
                    data['parameters'] = self.parameters
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
                if (not self.parameters['register_stitch_reference'])&(isinstance(self.reference_stitched,str)):
                    redo_posnames = list(self.posnames)
                    # Input = []
                    translation_y_list = [0,0,0]
                    translation_x_list = [0,0,0]
                results = {}
                with multiprocessing.Pool(self.parameters['ncpu']) as p:
                    for posname,nuc,signal,translation_x,translation_y in self.generate_iterable(p.imap(pfunc,Input),'Processing '+acq+'_'+channel,length=len(Input)):
                        results[posname] = {}
                        results[posname]['nuc'] = nuc
                        results[posname]['signal'] = signal
                        results[posname]['translation_x'] = translation_x
                        results[posname]['translation_y'] = translation_y
                if self.parameters['post_processing_constant']:
                    loc = ''
                    n_pos = len(list(results.keys()))
                    out_img = ''
                    n_pixels = 5
                    for i,posname in enumerate(results.keys()):
                        result = results[posname]

                        signal_image = result['signal'].numpy()
                        nuc_image = result['nuc'].numpy()

                        signal_image = image_filter(signal_image,'median',3)
                        nuc_image = image_filter(nuc_image,'median',3)

                        signal_image = np.clip(signal_image,None,np.mean(signal_image))
                        nuc_image = np.clip(nuc_image,None,np.mean(nuc_image))

                        if isinstance(loc,str):
                            loc = {}
                            for r in range(n_pixels):
                                loc[r]= np.random.randint(np.ones([signal_image.shape[0],signal_image.shape[1]])*n_pos)
                            out_stk_signal = np.zeros([signal_image.shape[0],signal_image.shape[1],n_pixels],dtype= np.float32)
                            out_stk_nuc = np.zeros([signal_image.shape[0],signal_image.shape[1],n_pixels],dtype= np.float32)
                        for r in range(n_pixels):
                            x,y = np.where(loc[r]==i)
                            out_stk_signal[x,y,r] = signal_image[x,y]
                            out_stk_nuc[x,y,r] = nuc_image[x,y]
                    for r in range(n_pixels):
                        out_stk_signal[:,:,r] = image_filter(out_stk_signal[:,:,r],'median',10)
                        out_stk_nuc[:,:,r] = image_filter(out_stk_nuc[:,:,r],'median',10)
                    constant = np.min(out_stk_signal,axis=2)
                    nuc_constant = np.min(out_stk_nuc,axis=2)
                    constant = image_filter(constant,'median',10)
                    nuc_constant = image_filter(nuc_constant,'median',10)
                    constant = torch.tensor(constant,dtype=self.dtype_converter[self.parameters['dtype']])
                    nuc_constant = torch.tensor(nuc_constant,dtype=self.dtype_converter[self.parameters['dtype']])
                    constant = torch.clip(constant,0,None)
                    nuc_constant = torch.clip(nuc_constant,0,None)
                    self.save((constant).numpy(),hybe=hybe,channel=channel,file_type='image_post_constant')
                    self.save((nuc_constant).numpy(),hybe=hybe,channel=self.parameters['nucstain_channel'],file_type='image_post_constant')
                    # constant = torch.min(torch.dstack([result['signal'] for key,result in results.items()]),dim=2).values
                    # nuc_constant = torch.min(torch.dstack([result['nuc'] for key,result in results.items()]),dim=2).values
                    # constant = torch.tensor(median_filter(constant.numpy(),5),dtype=self.dtype_converter[self.parameters['dtype']])
                    # nuc_constant = torch.tensor(median_filter(nuc_constant.numpy(),5),dtype=self.dtype_converter[self.parameters['dtype']])

                    # constant = torch.min(torch.dstack([torch.tensor(gaussian_filter(result['signal'].numpy(),5)) for key,result in results.items()]),dim=2).values
                    # nuc_constant = torch.min(torch.dstack([torch.tensor(gaussian_filter(result['nuc'].numpy(),5)) for key,result in results.items()]),dim=2).values
                    # constant = torch.tensor(median_filter(constant.numpy(),5),dtype=self.dtype_converter[self.parameters['dtype']])
                    # nuc_constant = torch.tensor(median_filter(nuc_constant.numpy(),5),dtype=self.dtype_converter[self.parameters['dtype']])

                    # constant = torch.quantile(torch.dstack([torch.tensor(image_filter(result['signal'].numpy(),'median',5),dtype=self.dtype_converter[self.parameters['dtype']]) for key,result in results.items()]),0.05,axis=2)
                    # nuc_constant = torch.quantile(torch.dstack([torch.tensor(image_filter(result['nuc'].numpy(),'median',5),dtype=self.dtype_converter[self.parameters['dtype']]) for key,result in results.items()]),0.05,axis=2)
                    # constant = torch.tensor(image_filter(constant.numpy(),'polyfit_8',5),dtype=self.dtype_converter[self.parameters['dtype']])
                    # nuc_constant = torch.tensor(image_filter(nuc_constant.numpy(),'polyfit_8',5),dtype=self.dtype_converter[self.parameters['dtype']])
                else:
                    constant = 0
                    nuc_constant = 0
                for posname in self.generate_iterable(results.keys(),'Stitching '+acq+'_'+channel,length=len(results.keys())):
                    nuc = results[posname]['nuc']
                    signal = results[posname]['signal']
                    # Maybe here accound for differences in overall brightness of the image
                    nuc = nuc-nuc_constant
                    signal = signal-constant
                    nuc = torch.clip(nuc,0,None)
                    signal = torch.clip(signal,0,None)
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
                        if overlap>self.parameters['overlap']: 
                            mask_x = destination.max(1).values!=0
                            ref = destination[mask_x,:]
                            mask_y = ref.max(0).values!=0
                            ref = destination[mask_x,:]
                            ref = ref[:,mask_y]
                            non_ref = nuc[mask_x,:]
                            non_ref = non_ref[:,mask_y]
                            # Check if Beads work here
                            shift, error = register(ref.numpy(), non_ref.numpy(),10)
                            if (error!=np.inf)&(np.max(np.abs(shift))<=self.parameters['border']):
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
                        self.update_user('Error occurred on line: {}'.format(sys.exc_info()[-1].tb_lineno))
                        print(e)
                        translation_x = 0
                        translation_y = 0
                        stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),0] = nuc
                        stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),1] = signal
                        if isinstance(self.reference_stitched,str):
                            pixel_coordinates_stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),0] = torch.tensor(np.stack([np.array(range(nuc.shape[1])) for i in range(nuc.shape[0])])) #x
                            pixel_coordinates_stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),1] = torch.tensor(np.stack([np.array(range(nuc.shape[0])) for i in range(nuc.shape[1])]).T) #y
                if len(redo_posnames)>0:
                    for posname in self.generate_iterable(redo_posnames,f"Redoing {str(len(redo_posnames))} Failed Positions  {acq} {channel}",length=len(redo_posnames)):
                        nuc = results[posname]['nuc']-nuc_constant
                        signal = results[posname]['signal']-constant
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
                if self.parameters['set_min_zero']:
                    thresh = torch.min(stitched[:,:,0][stitched[:,:,0]>0]).values
                    stitched[:,:,0] = stitched[:,:,0]-thresh
                    thresh = torch.min(stitched[:,:,1][stitched[:,:,1]>0]).values
                    stitched[:,:,1] = stitched[:,:,1]-thresh
                
                self.save(stitched[:,:,0],hybe=hybe,channel=self.parameters['nucstain_channel'],file_type='stitched')
                self.save(stitched[:,:,1],hybe=hybe,channel=channel,file_type='stitched')
                nuclei = stitched[self.parameters['border']:-self.parameters['border'],
                                self.parameters['border']:-self.parameters['border'],0].numpy()
                scale = (np.array(nuclei.shape)*self.parameters['ratio']).astype(int)
                nuclei_down = np.array(Image.fromarray(nuclei.astype(np.float32)).resize((scale[1],scale[0]), Image.BICUBIC))
                self.save(nuclei_down,hybe=hybe,channel=self.parameters['nucstain_channel'],file_type='image')
                signal = stitched[self.parameters['border']:-self.parameters['border'],
                                self.parameters['border']:-self.parameters['border'],1].numpy()
                scale = (np.array(signal.shape)*self.parameters['ratio']).astype(int)
                signal_down = np.array(Image.fromarray(signal.astype(np.float32)).resize((scale[1],scale[0]), Image.BICUBIC))
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
        # if (self.parameters['acq_FF']==False)&(self.parameters['acq_constant']==False):
        #     self.image_parameters(self.parameters['nucstain_acq'],self.parameters['total_channel'])
        """ Generate Reference """
        self.any_incomplete_hybes = False
        self.reference_stitched = self.stitcher(self.parameters['nucstain_acq'],self.parameters['total_channel'])
        if isinstance(self.reference_stitched,type(None)):
            """ Reference Hasnt been Imaged"""
            self.any_incomplete_hybes = True
        else:
            for r,h,c in self.parameters['bitmap']:
                try:
                    stitched = self.stitcher(h,self.parameters['total_channel'])
                except Exception as e:
                    self.update_user(e,level=40)
                    self.update_user('Error occurred on line: {}'.format(sys.exc_info()[-1].tb_lineno))
                    self.any_incomplete_hybes = True

    def segment(self):
        """
        segment Using stitched images segment cells 

        :param model_type: model for cellpose ('nuclei' 'total' 'cytoplasm'), defaults to 'nuclei'
        :type model_type: str, optional
        """      
        self.update_user(f"Attempting Segmentation {self.model_type}",level=20)
        if (not self.parameters['segment_overwrite'])&self.check_existance(file_type='mask',model_type=self.model_type):
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
                if self.check_existance(hybe=self.parameters['nucstain_acq'],channel=self.parameters['nucstain_channel'],file_type='stitched'):
                    nucstain = self.load(hybe=self.parameters['nucstain_acq'],channel=self.parameters['nucstain_channel'],file_type='stitched')
                    dapi_mask = nucstain<self.parameters['dapi_thresh']
                    nucstain[dapi_mask] = 0
                else:
                    if self.parameters['nucstain_acq'] == 'all':
                        for r,h,c in self.generate_iterable(self.parameters['bitmap'],'Loading Total by Averaging All Measurements'):
                            if self.check_existance(hybe=h,channel=self.parameters['nucstain_channel'],file_type='stitched'):
                                if isinstance(nucstain,type(None)):
                                    nucstain = self.load(hybe=h,channel=self.parameters['nucstain_channel'],file_type='stitched')/len(self.parameters['bitmap'])
                                else:
                                    nucstain = nucstain+(self.load(hybe=h,channel=self.parameters['nucstain_channel'],file_type='stitched')/len(self.parameters['bitmap']))
                            else:
                                self.update_user(f" Missing {h} for generating nucstain",value=50)
                                nucstain=None
                if not isinstance(nucstain,type(None)):
                    self.save(nucstain,hybe=self.parameters['nucstain_acq'],channel=self.parameters['nucstain_channel'],file_type='stitched')
                    nuclei = nucstain[self.parameters['border']:-self.parameters['border'],
                                    self.parameters['border']:-self.parameters['border']].numpy()
                    scale = (np.array(nuclei.shape)*self.parameters['ratio']).astype(int)
                    nuclei_down = np.array(Image.fromarray(nuclei.astype(np.float32)).resize((scale[1],scale[0]), Image.BICUBIC))
                    self.save(nuclei_down,hybe=self.parameters['nucstain_acq'],channel=self.parameters['nucstain_channel'],file_type='image')
                # Total
                if 'total' in self.model_type:
                    if self.check_existance(hybe=self.parameters['total_acq'],channel=self.parameters['total_channel'],file_type='stitched'):
                            total = self.load(hybe=self.parameters['total_acq'],channel=self.parameters['total_channel'],file_type='stitched')
                            total[dapi_mask] = 0
                    else:
                        if self.parameters['total_acq'] == 'all':
                            for r,h,c in self.generate_iterable(self.parameters['bitmap'],'Loading Total by Averaging All Measurements'):
                                if self.check_existance(hybe=h,channel=c,file_type='stitched'):
                                    if isinstance(total,type(None)):
                                        total = self.load(hybe=h,channel=c,file_type='stitched')/len(self.parameters['bitmap'])
                                    else:
                                        total = total+(self.load(hybe=h,channel=c,file_type='stitched')/len(self.parameters['bitmap']))
                                else:
                                    self.update_user(f" Missing {h} for generating total",value=50)
                                    total=None
                    if not isinstance(total,type(None)):
                        self.save(total,hybe=self.parameters['total_acq'],channel=self.parameters['total_channel'],file_type='stitched')
                        signal = total[self.parameters['border']:-self.parameters['border'],
                                        self.parameters['border']:-self.parameters['border']].numpy()
                        scale = (np.array(signal.shape)*self.parameters['ratio']).astype(int)
                        signal_down = np.array(Image.fromarray(signal.astype(np.float32)).resize((scale[1],scale[0]), Image.BICUBIC))
                        self.save(signal_down,hybe=self.parameters['total_acq'],channel=self.parameters['total_channel'],file_type='image')
                        total[dapi_mask] = 0
                if not isinstance(nucstain,type(None)):
                    if 'total' in self.model_type:
                        model = models.Cellpose(model_type='cyto3',gpu=self.parameters['segment_gpu'])
                        self.mask = torch.zeros_like(total,dtype=self.dtype_converter[self.parameters['dtype']])
                        thresh = np.median(total[total!=0].numpy())+np.std(total[total!=0].numpy())
                    else:
                        model = models.Cellpose(model_type='nuclei',gpu=self.parameters['segment_gpu'])
                        self.mask = torch.zeros_like(nucstain,dtype=self.dtype_converter[self.parameters['dtype']])
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
                            diameter = int(self.parameters['segment_diameter']*1.5)
                            min_size = int(self.parameters['segment_min_size']*1.5)
                            channels = [1,2]
                        else:
                            stk = nuc
                            diameter = int(self.parameters['segment_diameter'])
                            min_size = int(self.parameters['segment_min_size'])
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
                        mask = torch.tensor(raw_mask_image.astype(int),dtype=self.dtype_converter[self.parameters['dtype']])
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
                        cell_mask = morphology.remove_small_objects(cell_mask, int(self.parameters['segment_diameter']**1.5))
                        cell_mask = morphology.binary_dilation(cell_mask,footprint=create_circle_array(5, 5))
                        # Call Cell Centers
                        img = gaussian_filter(image.copy(),5)
                        img[~cell_mask]=0
                        if (np.sum(cell_mask)>1)&(mask.max().numpy()>5):
                            min_peak_height = np.percentile(img[cell_mask],5)
                            min_peak_distance = int(self.parameters['segment_diameter']/2)
                            peaks = peak_local_max(img, min_distance=min_peak_distance, threshold_abs=min_peak_height)
                            # Make Seeds for Watershed
                            seed = 0*np.ones_like(img)
                            seed[peaks[:,0],peaks[:,1]] = 1+np.array(range(peaks.shape[0]))
                            seed_max = morphology.binary_dilation(seed!=0,footprint=create_circle_array(int(1.5*self.parameters['segment_diameter']), int(1.5*self.parameters['segment_diameter'])))
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
                                    if np.sum(m)<self.parameters['segment_diameter']**1.5:
                                        pixel_labels[m] = 0
                                updated_mask[tx,ty] = pixel_labels

                            mask = torch.tensor(updated_mask.astype(int),dtype=self.dtype_converter[self.parameters['dtype']])

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

    def metric(self,V):
        if self.parameters['metric'] =='median':
            return torch.median(V,axis=0).values
        elif self.parameters['metric'] =='mean':
            return torch.mean(V,axis=0)
        elif self.parameters['metric'] =='sum':
            return torch.sum(V,axis=0)
        elif self.parameters['metric'] =='min':
            return torch.min(V,axis=0).values
        elif self.parameters['metric'] =='max':
            return torch.max(V,axis=0).values
        else:
            self.update_user(f"{self.parameters['metric']} not supporte metric",level=50)
            raise ValueError(f"{self.parameters['metric']} not supporte metric")

    def pull_vectors(self):
        """
        pull_vectors Using segmented cells pull pixels for each round for each cell and summarize into vector

        :param model_type: model for cellpose ('nuclei' 'total' 'cytoplasm'), defaults to 'nuclei'
        :type model_type: str, optional
        """      
        self.update_user(f"Attempting to Pull Vectors {self.model_type}",level=20)  
        if (not self.parameters['vector_overwrite'])&self.check_existance(file_type='anndata',model_type=self.model_type):
            self.data = self.load(file_type='anndata',model_type=self.model_type)
            self.update_user(f"Existing Data Found {self.model_type}",level=20)
        else:
            idxes = torch.where(self.mask!=0)
            labels = self.mask[idxes]

            """ Load Vector for each pixel """
            pixel_image_xy = torch.zeros([idxes[0].shape[0],2],dtype=self.dtype_converter[self.parameters['dtype']])
            pixel_image_xy[:,0] = self.load(hybe='image_x',channel='',file_type='stitched')[idxes]
            pixel_image_xy[:,1] = self.load(hybe='image_y',channel='',file_type='stitched')[idxes]

            pixel_vectors = torch.zeros([idxes[0].shape[0],len(self.parameters['bitmap'])],dtype=self.dtype_converter[self.parameters['dtype']])
            nuc_pixel_vectors = torch.zeros([idxes[0].shape[0],len(self.parameters['bitmap'])],dtype=self.dtype_converter[self.parameters['dtype']])
            for i,(r,h,c) in self.generate_iterable(enumerate(self.parameters['bitmap']),'Generating Pixel Vectors',length=len(self.parameters['bitmap'])):
                pixel_vectors[:,i] = self.load(hybe=h,channel=c,file_type='stitched')[idxes]
                nuc_pixel_vectors[:,i] = self.load(hybe=h,channel=self.parameters['nucstain_channel'],file_type='stitched')[idxes]
            pixel_vectors[:,-1] = self.load(hybe=self.parameters['nucstain_acq'],channel=self.parameters['nucstain_channel'],file_type='stitched')[idxes]
            nuc_pixel_vectors[:,-1] = self.load(hybe=self.parameters['nucstain_acq'],channel=self.parameters['nucstain_channel'],file_type='stitched')[idxes]
            unique_labels = torch.unique(labels)
            self.vectors = torch.zeros([unique_labels.shape[0],pixel_vectors.shape[1]],dtype=self.dtype_converter[self.parameters['dtype']])
            self.nuc_vectors = torch.zeros([unique_labels.shape[0],nuc_pixel_vectors.shape[1]],dtype=self.dtype_converter[self.parameters['dtype']])
            pixel_xy = torch.zeros([idxes[0].shape[0],2])
            pixel_xy[:,0] = idxes[0]
            pixel_xy[:,1] = idxes[1]
            
            self.xy = torch.zeros([unique_labels.shape[0],2],dtype=self.dtype_converter[self.parameters['dtype']])
            self.image_xy = torch.zeros([unique_labels.shape[0],2],dtype=self.dtype_converter[self.parameters['dtype']])
            self.size = torch.zeros([unique_labels.shape[0],1],dtype=self.dtype_converter[self.parameters['dtype']])
            
            converter = {int(label):[] for label in unique_labels}
            for i in tqdm(range(labels.shape[0]),desc=str(datetime.now().strftime("%Y %B %d %H:%M:%S"))+' Generating Label Converter'):
                converter[int(labels[i])].append(i)

            
            for i,label in tqdm(enumerate(unique_labels),total=unique_labels.shape[0],desc=str(datetime.now().strftime("%H:%M:%S"))+' Generating Cell Vectors and Metadata'):
                m = converter[int(label)]
                self.vectors[i,:] = self.metric(pixel_vectors[m,:])
                self.nuc_vectors[i,:] = self.metric(nuc_pixel_vectors[m,:])
                pxy = pixel_xy[m,:]
                self.xy[i,:] = self.metric(pxy)
                pixy = pixel_image_xy[m,:]
                self.image_xy[i,:] =self.metric(pixy)
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
            # self.vectors = self.vectors[:,0:-1]
            # self.nuc_vectors = self.nuc_vectors[:,0:-1]
            self.data = anndata.AnnData(X=self.vectors.numpy(),
                                var=pd.DataFrame(index=np.array([r for r,h,c in self.parameters['bitmap']])),
                                obs=self.cell_metadata,dtype=self.dtype_converter_numpy[self.parameters['anndata_dtype']])
            self.data.var['readout'] = [r for r,h,c in self.parameters['bitmap']]
            self.data.var['hybe'] = [h for r,h,c in self.parameters['bitmap']]
            self.data.var['channel'] = [c for r,h,c in self.parameters['bitmap']]
            self.data.layers['processed_vectors'] = self.vectors.numpy()
            self.data.layers['nuc_processed_vectors'] = self.nuc_vectors.numpy()
            self.data.layers['raw'] = self.vectors.numpy()
            self.data.layers['nuc_raw'] = self.nuc_vectors.numpy()


            # observations = ['PolyT','Housekeeping','Nonspecific_Encoding','Nonspecific_Readout'] # FIX upgrade to be everything but the rs in design 
            # for obs in observations:
            #     if obs in self.data.var.index:
            #         self.data.obs[obs.lower()] = self.data.layers['processed_vectors'][:,self.data.var.index==obs]

            # self.data = self.data[:,self.data.var.index.isin(observations)==False]

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
        # additional_views = ['housekeeping','dapi','nonspecific_encoding','nonspecific_readout','sum','ieg']
        additional_views = ['sum','dapi']
        plt.close('all')
        if (os.path.exists(report_path)==False)|self.parameters['overwrite_report']:
            """ Vector """
            path = self.generate_filename(hybe='', channel='RawVectors', file_type='Figure', model_type=self.model_type)
            if (os.path.exists(path)==False)|self.parameters['overwrite_report']:
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
                X = self.data.layers['processed_vectors'].copy()
                # X = basicu.normalize_fishdata_robust_regression(X)
                # X = basicu.normalize_fishdata_regress(X,value='sum',leave_log=True,log=True,bitwise=False)
                X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                for i in range(n_bits):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i])
                    axs[i].set_title(np.array(self.data.var.index)[i])
                    axs[i].axis('off')
                shared_columns = [i for i in additional_views if i in self.data.obs.columns]
                X = np.array(self.data.obs[shared_columns]).astype(np.float32)
                # X = basicu.normalize_fishdata_robust_regression(X)
                # X = basicu.normalize_fishdata_regress(X,value='sum',leave_log=True,log=True,bitwise=False)
                X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                for i,obs in enumerate(shared_columns):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i+n_bits].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i+n_bits])
                    axs[i+n_bits].set_title(obs)
                    axs[i+n_bits].axis('off')
                paths.append(path)
                plt.savefig(path,dpi=dpi)
                plt.show(block=False)
            plt.close('all')
            """ Vector """
            path = self.generate_filename(hybe='', channel='NormVectors', file_type='Figure', model_type=self.model_type)
            if (os.path.exists(path)==False)|self.parameters['overwrite_report']:
                self.update_user('Generating Norm Figure')
                n_bits = self.data.shape[1]
                n_figs = n_bits+len(additional_views)
                n_columns = 5
                n_rows = math.ceil(n_figs/n_columns)
                fig,axs = plt.subplots(n_columns,n_rows,figsize=[25,25])
                plt.suptitle('Stain Signal Normalized')
                axs = axs.ravel()
                for i in range(n_figs):
                    axs[i].axis('off')
                    axs[i].set_aspect('equal')
                x = self.data.obs['stage_x']
                y = self.data.obs['stage_y']
                X = self.data.layers['processed_vectors'].copy()
                X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                X = basicu.normalize_fishdata_robust_regression(X)
                # X = basicu.normalize_fishdata_regress(X,value='sum',leave_log=True,log=True,bitwise=False)
                # X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                for i in range(n_bits):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i])
                    axs[i].set_title(np.array(self.data.var.index)[i])
                    axs[i].axis('off')
                shared_columns = [i for i in additional_views if i in self.data.obs.columns]
                X = np.array(self.data.obs[shared_columns]).astype(np.float32).copy()
                X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                X = basicu.normalize_fishdata_robust_regression(X)
                # X = basicu.normalize_fishdata_regress(X,value='sum',leave_log=True,log=True,bitwise=False)
                # X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                for i,obs in enumerate(shared_columns):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i+n_bits].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i+n_bits])
                    axs[i+n_bits].set_title(obs)
                    axs[i+n_bits].axis('off')
                paths.append(path)
                plt.savefig(path,dpi=dpi)
                plt.show(block=False)
            plt.close('all')
            """ Dapi  """
            path = self.generate_filename(hybe='', channel='Dapi', file_type='Figure', model_type=self.model_type)
            if (os.path.exists(path)==False)|self.parameters['overwrite_report']:
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
                X = self.data.layers['nuc_processed_vectors'].copy()
                # X = basicu.normalize_fishdata_robust_regression(X)
                # X = basicu.normalize_fishdata_regress(X,value='sum',leave_log=True,log=True,bitwise=False)
                X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                for i in range(n_bits):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i])
                    axs[i].set_title(np.array(self.data.var.index)[i])
                    axs[i].axis('off')
                shared_columns = [i for i in additional_views if i in self.data.obs.columns]
                X = np.array(self.data.obs[shared_columns]).astype(np.float32).copy()
                
                # X = basicu.normalize_fishdata_robust_regression(X)
                # X = basicu.normalize_fishdata_regress(X,value='sum',leave_log=True,log=True,bitwise=False)
                X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                for i,obs in enumerate(shared_columns):
                    c = X[:,i].copy()
                    vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                    scatter = axs[i+n_bits].scatter(x,y,c=np.clip(c,vmin,vmax),s=0.01,cmap='jet',marker='x')
                    fig.colorbar(scatter, ax=axs[i+n_bits])
                    axs[i+n_bits].set_title(obs)
                    axs[i+n_bits].axis('off')
                paths.append(path)
                plt.savefig(path,dpi=dpi)
                plt.show(block=False)
            plt.close('all')
            """ Louvain """
            path = self.generate_filename(hybe='', channel='Louvain', file_type='Figure', model_type=self.model_type)
            if (os.path.exists(path)==False)|self.parameters['overwrite_report']:
                self.update_user('Generating Louvain Figure')
                if self.parameters['overwrite_louvain']| (not 'louvain' in self.data.obs.columns):
                    #  self.data.X = basicu.normalize_fishdata_regress(self.data.layers['processed_vectors'].copy(),value='sum',leave_log=True,log=True,bitwise=True)
                    X = self.data.layers['processed_vectors'].copy()
                    X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                    X = basicu.normalize_fishdata_robust_regression(X)
                    # X = basicu.normalize_fishdata_regress(X,value='none',leave_log=True,log=True,bitwise=False)
                    self.data.X = X.copy()
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
                plt.show(block=False)

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

                

    def remove_temporary_files(self,data_types = ['stitched','stitched_raw']):
        """
        remove_temporary_files Remove Temporary Processing Files To Save Disk Space

        :param data_types: Data Types to Remove, defaults to ['stitched','stitched_raw','FF']
        :type data_types: list, optional
        """
        for file_type in  self.generate_iterable(data_types,message='Removing Temporary Files'):
            fname = self.generate_filename(hybe='', channel='', file_type=file_type, model_type='')
            dirname = os.path.dirname(fname)
            shutil.rmtree(dirname)
        shutil.rmtree(self.scratch_path)

# def generate_constant_only(acq,image_metadata=None,channel=None,posnames=[],bkg_acq='',parameters={},verbose=False):
#     if 'mask' in channel:
#         return ''
#     else:
#         if len(posnames)==0:
#             posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
#         FF = []
#         if verbose:
#             iterable = tqdm(posnames,desc=str(datetime.now().strftime("%Y %B %d %H:%M:%S"))+' Generating FlatField '+acq+' '+channel)
#         else:
#             iterable = posnames
#         for posname in iterable:
#             try:
#                 img = image_metadata.stkread(Position=posname,Channel=channel,acq=acq).min(2).astype(np.float32)
#                 img = median_filter(img,2)
#                 img = torch.tensor(img)
#                 if bkg_acq!='':
#                     bkg = image_metadata.stkread(Position=posname,Channel=channel,acq=bkg_acq).mean(2).astype(np.float32)
#                     # bkg = median_filter(bkg,2)
#                     bkg = torch.tensor(bkg)
#                     img = img-bkg
#                 FF.append(img)
#             except Exception as e:
#                 print(posname,acq,bkg_acq)
#                 print(e)
#                 continue
#         # FF = torch.quantile(torch.dstack(FF),0.5,dim=2).numpy() # Assumption is that for each pixel half of the images wont have a cell there
#         FF = torch.dstack(FF)
#         constant = torch.min(FF,dim=2).values # There may be a more robust way 
#         constant = gaussian_filter(constant,50,mode='nearest')  # causes issues with corners
#         return constant


# def generate_FF_only(acq,image_metadata=None,channel=None,constant=0,posnames=[],bkg_acq='',parameters={},verbose=False):
#     """
#     generate_FF Generate flat field to correct uneven illumination

#     :param image_metadata: Data Loader Class
#     :type image_metadata: Metadata Class
#     :param acq: name of acquisition
#     :type acq: str
#     :param channel: name of channel
#     :type channel: str
#     :return: flat field image
#     :rtype: np.array
#     """
#     if 'mask' in channel:
#         return ''
#     else:
#         if len(posnames)==0:
#             posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
#         FF = []
#         if verbose:
#             iterable = tqdm(posnames,desc=str(datetime.now().strftime("%Y %B %d %H:%M:%S"))+' Generating FlatField '+acq+' '+channel)
#         else:
#             iterable = posnames
#         for posname in iterable:
#             try:
#                 img = image_metadata.stkread(Position=posname,Channel=channel,acq=acq).min(2).astype(np.float32)
#                 img = median_filter(img,2)
#                 img = torch.tensor(img)
#                 if bkg_acq!='':
#                     bkg = image_metadata.stkread(Position=posname,Channel=channel,acq=bkg_acq).mean(2).astype(np.float32)
#                     # bkg = median_filter(bkg,2)
#                     bkg = torch.tensor(bkg)
#                     img = img-bkg
#                 FF.append(img)
#             except Exception as e:
#                 print(posname,acq,bkg_acq)
#                 print(e)
#                 continue
#         # FF = torch.quantile(torch.dstack(FF),0.5,dim=2).numpy() # Assumption is that for each pixel half of the images wont have a cell there
#         FF = torch.dstack(FF)
#         FF = torch.mean(FF,dim=2).numpy() # There may be a more robust way in case of debris 
#         if np.max(constant.ravel())>0:
#             if isinstance(constant, torch.Tensor):
#                 constant = constant.numpy().copy()
#             FF = FF-constant
#         FF = gaussian_filter(FF,50,mode='nearest') # causes issues with corners
#         vmin,vmid,vmax = np.percentile(FF[np.isnan(FF)==False],[0.1,50,99.9]) 
#         # Maybe add median filter to FF 
#         FF[FF<vmin] = vmin
#         FF[FF>vmax] = vmax
#         FF[FF==0] = vmid
#         FF = vmid/FF
#         return FF

def optional_tqdm(iterable,verbose=True,desc='',total=0):
    if verbose:
        return tqdm(iterable,desc=str(datetime.now().strftime("%Y %B %d %H:%M:%S"))+' '+desc,total=total)
    else:
        return iterable

# def generate_FF_constant(image_metadata,channel,posnames=[],bkg_acq='',parameters={},verbose=False,ncpu=6):
#     """
#     generate_FF Generate flat field to correct uneven illumination

#     :param image_metadata: Data Loader Class
#     :type image_metadata: Metadata Class
#     :param acq: name of acquisition
#     :type acq: str
#     :param channel: name of channel
#     :type channel: str
#     :return: flat field image
#     :rtype: np.array
#     """
#     if 'mask' in channel:
#         return ''
#     if len(posnames)>0:
#         image_metadata_table = image_metadata.image_table[image_metadata.image_table.Position.isin(posnames)]
#     acqs = image_metadata_table.acq.unique()
#     strip_acqs = [i for i in acqs if 'strip' in i.lower()]
#     hybe_acqs = [i for i in acqs if 'hybe' in i.lower()]
#     pfunc = partial(generate_constant_only,image_metadata=image_metadata,channel=channel,posnames=posnames,bkg_acq='',parameters=parameters,verbose=False)
#     constants = []
#     with multiprocessing.Pool(ncpu) as p:
#         for constant in optional_tqdm(p.map(pfunc,strip_acqs),desc='Generating Image Constant',total=len(strip_acqs),verbose=verbose):
#             constants.append(constant)
#     constant = np.dstack(constants)
#     constant = np.mean(constant,axis=2)
#     pfunc = partial(generate_FF_only,constant=constant,image_metadata=image_metadata,channel=channel,posnames=posnames,bkg_acq='',parameters=parameters,verbose=False)
#     FFs = []
#     with multiprocessing.Pool(ncpu) as p:
#         for FF in optional_tqdm(p.map(pfunc,hybe_acqs),desc='Generating Flat Field',total=len(hybe_acqs),verbose=verbose):
#             FFs.append(FF)
#     FF = np.dstack(FFs)
#     FF = np.mean(FF,axis=2)
#     return FF,constant

def generate_FF_parallel(image_metadata,acq,channel,posnames=[],bkg_acq='',parameters={},verbose=False):
    if 'mask' in channel:
        return ''
    else:
        if len(posnames)==0:
            posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
        posnames = np.array(posnames)
        np.random.shuffle(posnames)
        posnames = list(posnames)
        n_cpu = parameters['FF_n_cpu']
        if n_cpu == 1:
            FF,constant = generate_FF(image_metadata,acq,channel,posnames=posnames,bkg_acq=bkg_acq,parameters=parameters,verbose=verbose)
        else:
            Input = [posnames[i:i + len(posnames)//n_cpu] for i in range(0, len(posnames), len(posnames)//n_cpu)]
            FF = []
            Constant = []
            pfunc = partial(generate_FF_wrapper,
                            image_metadata=image_metadata,
                            acq=acq,
                            channel=channel,
                            bkg_acq=bkg_acq,
                            parameters=parameters,
                            verbose=False)
            with multiprocessing.Pool(n_cpu) as p:
                for ff,constant in p.imap(pfunc,Input):
                    FF.append(ff)
                    Constant.append(constant)
            FF = torch.quantile(torch.dstack(FF),0.5,axis=2)
            constant = torch.quantile(torch.dstack(Constant),0.5,axis=2)
        if isinstance(FF,np.ndarray):
            FF = torch.tensor(FF,dtype=torch.float32)
        if isinstance(constant,np.ndarray):
            constant = torch.tensor(constant,dtype=torch.float32)
        return FF,constant
        
def generate_FF_wrapper(posnames,image_metadata=None,acq=None,channel=None,bkg_acq='',parameters={},verbose=False):
    return generate_FF(image_metadata,acq,channel,posnames=posnames,bkg_acq=bkg_acq,parameters=parameters,verbose=verbose)

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
        loc = ''
        n_pos = len(posnames)
        out_img = ''
        n_pixels = 5
        for i,posname in enumerate(iterable):
            # try:
            img = image_metadata.stkread(Position=posname,Channel=channel,acq=acq).min(2).astype(np.float32)
            img = median_filter(img,2)
            img = torch.tensor(img,dtype=torch.float32)
            if bkg_acq!='':
                bkg = image_metadata.stkread(Position=posname,Channel=channel,acq=bkg_acq).min(2).astype(np.float32)
                bkg = median_filter(bkg,2)
                bkg = torch.tensor(bkg,dtype=torch.float32)
                img = img-bkg
                img = torch.clip(img,0,None)
            if parameters['process_img_before_FF']:
                temp_parameters = parameters.copy()
                temp_parameters['highpass_function'] = 'none'
                temp_parameters['highpass_sigma'] = 0
                temp_parameters['highpass_smooth_function'] = 'median'
                temp_parameters['highpass_smooth'] = 2
                img = torch.tensor(process_img(img.numpy(),temp_parameters,nuc=None,FF=1,constant=0))
            if isinstance(loc,str):
                loc = {}
                for r in range(n_pixels):
                    loc[r]= torch.tensor(np.random.randint(np.ones([img.shape[0],img.shape[1]])*n_pos))
                out_stk = torch.zeros([img.shape[0],img.shape[1],n_pixels],dtype= torch.float32)
            for r in range(n_pixels):
                x,y = torch.where(loc[r]==i)
                out_stk[x,y,r] = img[x,y]
                # FF.append(img)
            # except Exception as e:
            #     print(posname,acq,bkg_acq)
            #     print('Error occurred at line number:', inspect.currentframe().f_back.f_lineno)
            #     print(e)
            #     continue
        
        FF = out_stk
        constant = torch.zeros_like(FF[:,:,0]) # There may be a more robust way 
        if parameters['use_constant']:
            constant = torch.quantile(FF,0.05,axis=2)
            if parameters['debug']:
                plt.figure()
                title='Constant raw'
                img = constant.numpy()
                vmin,vmax = np.percentile(img.ravel(),[5,95])
                plt.imshow(np.clip(img,vmin,vmax),cmap='jet')
                plt.colorbar()
                plt.title(title)
                plt.savefig(fileu.generate_filename(parameters['scratch_path'],hybe=acq.split('_')[0],channel=channel+'_'+title,file_type='Figure',model_type='',logger='Debugging'))
                plt.close('all')

            if parameters['fit_constant']:
                scale = np.percentile(constant.ravel(),50)
                constant = median_filter(constant,5)
                x = gaussian_filter(np.percentile(constant,50,axis=0),1)
                y = gaussian_filter(np.percentile(constant,50,axis=1),1)
                x = np.poly1d(np.polyfit(range(x.shape[0]), x, parameters['constant_poly_degrees']))(range(x.shape[0]))
                y = np.poly1d(np.polyfit(range(y.shape[0]), y, parameters['constant_poly_degrees']))(range(y.shape[0]))
                constant = ((np.ones_like(constant)*x)*(np.ones_like(constant).T*y).T)
                constant = constant/np.median(constant.ravel())
                constant = scale*constant
                constant = torch.tensor(constant,dtype=FF.dtype)
                if parameters['debug']:
                    plt.figure()
                    title='Constant fit'
                    img = constant.numpy()
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plt.imshow(np.clip(img,vmin,vmax),cmap='jet')
                    plt.colorbar()
                    plt.title(title)
                    plt.savefig(fileu.generate_filename(parameters['scratch_path'],hybe=acq.split('_')[0],channel=channel+'_'+title,file_type='Figure',model_type='',logger='Debugging'))
                    plt.close('all')
            if parameters['smooth_constant']:
                constant = image_filter(constant,'downsample_quantile_0.1',150)
                if parameters['debug']:
                    plt.figure()
                    title='Constant processed'
                    img = constant.numpy()
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plt.imshow(np.clip(img,vmin,vmax),cmap='jet')
                    plt.colorbar()
                    plt.title(title)
                    plt.savefig(fileu.generate_filename(parameters['scratch_path'],hybe=acq.split('_')[0],channel=channel+'_'+title,file_type='Figure',model_type='',logger='Debugging'))
                    plt.close('all')
            if parameters['clip_constant']:
                vmin,vmid,vmax = np.percentile(constant[np.isnan(constant)==False],[0.1,50,99.9]) 
                constant[constant<vmin] = vmin
                constant[constant>vmax] = vmax
                if parameters['debug']:
                    plt.figure()
                    title='Constant clipped'
                    img = constant.numpy()
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plt.imshow(np.clip(img,vmin,vmax),cmap='jet')
                    plt.colorbar()
                    plt.title(title)
                    plt.savefig(fileu.generate_filename(parameters['scratch_path'],hybe=acq.split('_')[0],channel=channel+'_'+title,file_type='Figure',model_type='',logger='Debugging'))
                    plt.close('all')
            constant = torch.clip(constant,0,None)
            FF = FF-constant[:,:,None]

        if parameters['use_FF']:
            FF = torch.quantile(FF,0.5,axis=2)
            if parameters['debug']:
                plt.figure()
                title='FF raw'
                img = FF.numpy()
                vmin,vmax = np.percentile(img.ravel(),[5,95])
                plt.imshow(np.clip(img,vmin,vmax),cmap='jet')
                plt.colorbar()
                plt.title(title)
                plt.savefig(fileu.generate_filename(parameters['scratch_path'],hybe=acq.split('_')[0],channel=channel+'_'+title,file_type='Figure',model_type='',logger='Debugging'))
                plt.close('all')
            if parameters['fit_FF']:
                FF = median_filter(FF.numpy(),5)
                x = gaussian_filter(np.percentile(FF,50,axis=0),1)
                y = gaussian_filter(np.percentile(FF,50,axis=1),1)
                x = np.poly1d(np.polyfit(range(x.shape[0]), x, parameters['FF_poly_degrees']))(range(x.shape[0]))
                y = np.poly1d(np.polyfit(range(y.shape[0]), y, parameters['FF_poly_degrees']))(range(y.shape[0]))
                # from scipy.optimize import curve_fit
                # def gaussian(x, amplitude, mean, stddev):
                #     return amplitude * np.exp(-((x - mean) / 2 / stddev)**2)
                # x_data = np.arange(x.shape[0])
                # popt, _ = curve_fit(gaussian, x_data, x, p0=[1, np.mean(x_data), np.std(x_data)])
                # x = gaussian(x_data, *popt)
                # x_data = np.arange(y.shape[0])
                # popt, _ = curve_fit(gaussian, x_data, y, p0=[1, np.mean(x_data), np.std(x_data)])
                # y = gaussian(x_data, *popt)
                # FF = ((np.ones_like(FF)*x)+(np.ones_like(FF).T*y).T)/2
                FF = ((np.ones_like(FF)*x)*(np.ones_like(FF).T*y).T)
                FF = FF/np.median(FF)
                FF = torch.tensor(FF,dtype=constant.dtype)
                if parameters['debug']:
                    plt.figure()
                    title='FF fit'
                    img = FF.numpy()
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plt.imshow(np.clip(img,vmin,vmax),cmap='jet')
                    plt.colorbar()
                    plt.title(title)
                    plt.savefig(fileu.generate_filename(parameters['scratch_path'],hybe=acq.split('_')[0],channel=channel+'_'+title,file_type='Figure',model_type='',logger='Debugging'))
                    plt.close('all')
            if parameters['smooth_FF']:
                # FF = gaussian_filter(FF,50,mode='nearest') # Edge Issues
                # FF = image_filter(FF.numpy(),'downsample_quantile_0.5',150)
                FF = image_filter(FF.numpy(),'gaussian',50)
                FF = torch.tensor(FF,dtype=constant.dtype)
                if parameters['debug']:
                    plt.figure()
                    title='FF processed'
                    img = FF.numpy()
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plt.imshow(np.clip(img,vmin,vmax),cmap='jet')
                    plt.colorbar()
                    plt.title(title)
                    plt.savefig(fileu.generate_filename(parameters['scratch_path'],hybe=acq.split('_')[0],channel=channel+'_'+title,file_type='Figure',model_type='',logger='Debugging'))
                    plt.close('all')
            vmin,vmid,vmax = np.percentile(FF[np.isnan(FF)==False],[0.1,50,99.9]) 
            if parameters['clip_FF']:
                FF[FF<vmin] = vmin
                FF[FF>vmax] = vmax
                FF[FF==0] = vmid
                if parameters['debug']:
                    plt.figure()
                    title='FF clipped'
                    img = FF.numpy()
                    vmin,vmax = np.percentile(img.ravel(),[5,95])
                    plt.imshow(np.clip(img,vmin,vmax),cmap='jet')
                    plt.colorbar()
                    plt.title(title)
                    plt.savefig(fileu.generate_filename(parameters['scratch_path'],hybe=acq.split('_')[0],channel=channel+'_'+title,file_type='Figure',model_type='',logger='Debugging'))
                    plt.close('all')
            FF = torch.clip(FF,1,None)
            FF = vmid/FF
        else:
            FF = torch.ones_like(FF[:,:,0])
        plt.close('all')
        return FF,constant

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
        img = image_filter(img,parameters['highpass_smooth_function'],parameters['highpass_smooth'])
    # Background Subtract
    if parameters['highpass_sigma']>0:
        bkg = image_filter(img,parameters['highpass_function'],parameters['highpass_sigma'])
        img = img-bkg
    return img

def image_filter(img,function,value):
    if function == 'gaussian':
        img = gaussian_filter(img,value) 
    elif function == 'median':
        img = median_filter(img,value) 
    elif function == 'minimum':
        img = minimum_filter(img,size=value) 
    elif 'percentile' in function:
        img = percentile_filter(img,int(function.split('_')[-1]),size=value)
    elif 'rolling_ball' in function:
        img = gaussian_filter(restoration.rolling_ball(gaussian_filter(img,value/5),radius=value,num_threads=30),value)
    elif 'downsample' in function:
        binsize = value
        scale = np.array(img.shape)
        original_width, original_height = img.shape
        new_width = int(original_width/binsize)
        new_height = int(original_height/binsize)
        original_width = new_width*binsize
        original_height = new_height*binsize
        # Resize to be a multiple of binsize
        img = np.array(Image.fromarray(img.astype(np.float32)).resize((original_height,original_width), Image.BICUBIC))
        img = torch.tensor(img)
        img_down = torch.zeros([new_width,new_height])
        for x in range(new_width):
            for y in range(new_height):
                if 'quantile' in function:
                    quantile = float(function.split('_')[-1])
                    img_down[x,y] = torch.quantile(img[x*binsize:(x+1)*binsize,y*binsize:(y+1)*binsize],quantile)
                else:
                    img_down[x,y] = torch.quantile(img[x*binsize:(x+1)*binsize,y*binsize:(y+1)*binsize],0.5)
        img = median_filter(img_down,2)
        img = np.array(Image.fromarray(img.astype(np.float32)).resize((scale[1],scale[0]), Image.BICUBIC))
    elif 'polyfit' in function:
        function,degrees = function.split('_')
        degrees = int(degrees)
        x = gaussian_filter(np.percentile(img,value,axis=0),1)
        y = gaussian_filter(np.percentile(img,value,axis=1),1)
        x = np.poly1d(np.polyfit(range(x.shape[0]), x, degrees))(range(x.shape[0]))
        y = np.poly1d(np.polyfit(range(y.shape[0]), y, degrees))(range(y.shape[0]))
        img = ((np.ones_like(img)*x)+(np.ones_like(img).T*y).T)/2
    else:
        img = 0
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
    dtype_converter = {'float64':torch.float64,'float32':torch.float32,'float16':torch.float16,'int32':torch.int32}
    try:
        nuc = ((image_metadata.stkread(Position=posname,
                                        Channel=parameters['nucstain_channel'],
                                        acq=acq).max(axis=2).astype(np.float32)))
        nuc = process_img(nuc,parameters,FF=nuc_FF,constant=nuc_constant)
        img = ((image_metadata.stkread(Position=posname,
                                        Channel=channel,
                                        acq=acq).max(axis=2).astype(np.float32)))
        if not bkg_acq=='':
            bkg = ((image_metadata.stkread(Position=posname,
                                            Channel=channel,
                                            acq=bkg_acq).max(axis=2).astype(np.float32)))
            bkg_nuc = ((image_metadata.stkread(Position=posname,
                                            Channel=parameters['nucstain_channel'],
                                            acq=bkg_acq).max(axis=2).astype(np.float32)))
            bkg_nuc = process_img(bkg_nuc,parameters,FF=nuc_FF,constant=nuc_constant)
            # Check if beads work here
            shift, error = register(nuc, bkg_nuc,10)
            if error!=np.inf:
                translation_x = int(shift[1])
                translation_y = int(shift[0])
            else:
                translation_x = 0
                translation_y = 0
            x_correction = np.array(range(bkg.shape[1]))+translation_x
            y_correction = np.array(range(bkg.shape[0]))+translation_y
            if not parameters['post_strip_FF']:
                bkg = process_img(bkg,parameters,FF=FF,constant=constant)
                img = process_img(img,parameters,FF=FF,constant=constant)
            i2 = interpolate.interp2d(x_correction,y_correction,bkg,fill_value=None)
            bkg = i2(range(bkg.shape[1]), range(bkg.shape[0]))
            img = img-bkg
            if parameters['post_strip_FF']:
                img = process_img(img,parameters,FF=FF,constant=constant)
        else:
          img = process_img(img,parameters,FF=FF,constant=constant)  
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
        dtype = parameters['dtype']
        if 'float' in dtype:
            nuc[nuc<np.finfo(dtype).min] = np.finfo(dtype).min
            img[img<np.finfo(dtype).min] = np.finfo(dtype).min
            nuc[nuc>np.finfo(dtype).max] = np.finfo(dtype).max
            img[img>np.finfo(dtype).max] = np.finfo(dtype).max
        else:
            nuc[nuc<np.iinfo(dtype).min] = np.iinfo(dtype).min
            img[img<np.iinfo(dtype).min] = np.iinfo(dtype).min
            nuc[nuc>np.iinfo(dtype).max] = np.iinfo(dtype).max
            img[img>np.iinfo(dtype).max] = np.iinfo(dtype).max
        img = torch.tensor(img,dtype=dtype_converter[dtype])
        nuc = torch.tensor(nuc,dtype=dtype_converter[dtype])
        
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
        print(nuc.dtype,nuc_FF.dtype,nuc_constant.dtype)
        print(img.dtype,FF.dtype,constant.dtype)
        print(e,posname)
        print('Error occurred on line: {}'.format(sys.exc_info()[-1].tb_lineno))
        return posname,0,0,0,0

    
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
    plt.show(block=False)
    
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