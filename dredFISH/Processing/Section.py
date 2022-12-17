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

from skimage import (
    data, restoration, util
)

class Section_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 section,
                 cword_config,
                 verbose=False):
        """
        Section_Class: Class to Generate QC Files for a Section

        :param metadata_path: Path to Raw Data
        :type metadata_path: str
        :param dataset: Name of Dataset
        :type dataset: str
        :param section: Name of Section
        :type section: str
        :param cword_config: Name of Config Module
        :type cword_config: str
        :param verbose: Option to Print Loading Bars, defaults to False
        :type verbose: bool, optional
        """
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.section = str(section)
        self.cword_config = cword_config
        self.config = importlib.import_module(self.cword_config)
        self.image_metadata = ''
        self.reference_stitched=''
        self.FF=''
        self.nuc_FF=''
        self.data = ''
        self.verbose=verbose
        # self.out_path = os.path.join(self.metadata_path,self.config.parameters['fishdata'])
        # if not os.path.exists(self.out_path):
        #     os.mkdir(self.out_path)
    
    def run(self):
        """
        run Main Executable
        """
        self.load_metadata()
        if len(self.posnames)>0:
            self.stitch()
            for model_type in self.config.parameters['model_types']:
                self.segment(model_type=model_type)
                self.pull_vectors(model_type=model_type)
                self.save_data(model_type=model_type)
        else:
            self.update_user('No positions found for this section')
    
    def update_user(self,message):
        """
        update_user Send Message to User

        :param message: Message
        :type message: str
        """
        if self.verbose:
            i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]

    def generate_iterable(self,iterable,message,length=''):
        if self.verbose:
            if isinstance(length,str):
                return tqdm(iterable,desc=str(datetime.now().strftime("%H:%M:%S"))+' '+message)
            else:
                return tqdm(iterable,total=length,desc=str(datetime.now().strftime("%H:%M:%S"))+' '+message)
        else:
            return iterable


    def save_data(self,model_type='nuclei'):
        proceed = True
        if not self.config.parameters['vector_overwrite']:
            fname = self.generate_fname('',model_type,'data',dtype='h5ad')
            if os.path.exists(fname):
                self.update_user('Loading Existing Data')
                self.data = anndata.read_h5ad(fname)
                proceed = False
        if proceed:
            self.update_user('Saving Data')
            matrix = pd.DataFrame(self.data.X,
                                index=np.array(self.data.obs.index),
                                columns=np.array(self.data.var.index))
            self.data.write(filename=self.generate_fname('',model_type,'data',dtype='h5ad'))
            matrix.to_csv(self.generate_fname('',model_type,'matrix',dtype='csv'))
            self.data.obs.to_csv(self.generate_fname('',model_type,'metadata',dtype='csv'))
                                      
    def load_metadata(self):
        """
        load_data Load Previously Processed Data
        """
        if isinstance(self.image_metadata,str):
            self.update_user('Loading Metadata')
            self.image_metadata = Metadata(self.metadata_path)
        
        self.posnames = np.array([i for i in self.image_metadata.posnames if i.split('-')[0]==self.section])
        # if len(self.posnames)==0:
        #     hybe1 = [i for i in self.image_metadata.acqnames if 'hybe1_' in i][-1]
        #     self.posnames = np.array(self.image_metadata.image_table[self.image_metadata.image_table.acq==hybe1].Position.unique())
        if len(self.posnames)>0:
            self.acqs = np.unique(self.image_metadata.image_table[np.isin(self.image_metadata.image_table.Position,self.posnames)].acq)
            self.coordinates = {}
            for posname in self.posnames:
                self.coordinates[posname] = (self.image_metadata.image_table[(self.image_metadata.image_table.Position==posname)].XY.iloc[0]/self.config.parameters['pixel_size']).astype(int)
        
    def find_acq(self,hybe,protocol='hybe'):
        return [i for i in self.acqs if protocol+hybe+'_' in i][-1]

    def generate_fname(self,hybe,channel,name,dtype = 'pt'):
        if hybe!='':
            fname = self.dataset+'_Section'+self.section+'_Hybe'+hybe+'_'+channel+'_'+name+'.'+dtype
        else:
            if channel!='':
                fname = self.dataset+'_Section'+self.section+'_'+channel+'_'+name+'.'+dtype
            else:
                fname = self.dataset+'_Section'+self.section+'_'+name+'.'+dtype
        self.out_path = os.path.join(self.metadata_path,self.config.parameters['fishdata'])
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        fname = os.path.join(self.out_path,fname)
        return fname

    
    def stitcher(self,hybe,channel,acq='',bkg_acq=''):
        if hybe!='':
            acq = self.find_acq(hybe,protocol='hybe')
            if self.config.parameters['strip']:
                bkg_acq = self.find_acq(hybe,protocol='strip')
        else:
            hybe = acq.split('_')[0]
        nuc_fname = self.generate_fname(hybe,self.config.parameters['nucstain_channel'],'stitched')
        signal_fname = self.generate_fname(hybe,channel,'stitched')
        if (os.path.exists(nuc_fname))&(os.path.exists(signal_fname))&(self.config.parameters['overwrite']==False):
            self.update_user('Found Existing Hybe'+hybe+' Stitched')
            if (hybe==self.config.parameters['nucstain_acq'].split('hybe')[-1])&(self.reference_stitched==''):
                stitched = torch.dstack([self.load_tensor(nuc_fname),self.load_tensor(signal_fname)])
            else:
                stitched = 0
            return stitched,0,0,0,0# nuclei,nuclei_down,signal,signal_down
        else:
            if isinstance(self.FF,str):
                self.FF=1
                # self.FF = generate_FF(self.image_metadata,acq,channel,bkg_acq=bkg_acq,verbose=self.verbose)
            if isinstance(self.nuc_FF,str):
                self.nuc_FF = 1
                # self.nuc_FF = generate_FF(self.image_metadata,acq,self.config.parameters['nucstain_channel'],bkg_acq='',verbose=self.verbose)

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
            pfunc = partial(preprocess_images,FF=self.FF,nuc_FF=self.nuc_FF)
            translation_y_list = []
            translation_x_list = []
            redo_posnames = []
            if (not self.config.parameters['register_stitch_reference'])&(isinstance(self.reference_stitched,str)):
                redo_posnames = list(self.posnames)
                Input = []
                translation_y_list = [0,0,0]
                translation_x_list = [0,0,0]
            results = {}
            with multiprocessing.Pool(60) as p:
                for posname,nuc,signal,translation_x,translation_y in self.generate_iterable(p.imap(pfunc,Input),'Processing '+acq+'_'+channel,length=len(Input)):
                    results[posname] = {}
                    results[posname]['nuc'] = nuc
                    results[posname]['signal'] = signal
                    results[posname]['translation_x'] = translation_x
                    results[posname]['translation_y'] = translation_y
            # nuc_FF = torch.stack([results[posname]['nuc'].float() for posname in results.keys()]).mean(0)
            # nuc_FF = nuc_FF.mean()/nuc_FF

            # FF = torch.stack([results[posname]['signal'].float() for posname in results.keys()]).mean(0)
            # FF = FF.mean()/FF
            
            for posname in self.generate_iterable(results.keys(),'Stitching '+acq+'_'+channel,length=len(results.keys())):
                nuc = results[posname]['nuc']#*nuc_FF
                signal = results[posname]['signal']#*FF
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
                    if overlap>0.02: # atleast 2% of overlap???
                        mask_x = destination.max(1).values!=0
                        ref = destination[mask_x,:]
                        mask_y = ref.max(0).values!=0
                        ref = destination[mask_x,:]
                        ref = ref[:,mask_y]
                        non_ref = nuc[mask_x,:]
                        non_ref = non_ref[:,mask_y]
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
                except Exception as e:
                    print(posname,'Placing Image in Stitch with registration failed')
                    print(e)
                    translation_x = 0
                    translation_y = 0
                    stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),0] = nuc
                    stitched[(img_x_min+translation_x):(img_x_max+translation_x),(img_y_min+translation_y):(img_y_max+translation_y),1] = signal

            # Input = []
            # for posname in redo_posnames:
            #     data = {}
            #     data['acq'] = acq
            #     data['bkg_acq'] = bkg_acq
            #     data['posname'] = posname
            #     data['image_metadata'] = self.image_metadata
            #     data['channel'] = channel
            #     data['parameters'] = self.config.parameters
            #     Input.append(data)
            # with multiprocessing.Pool(60) as p:
            #     for posname,nuc,signal,translation_x,translation_y in self.generate_iterable(p.imap(pfunc,Input),'Redoing Failed Positions '+acq+'_'+channel,length=len(Input)):

            for posname in self.generate_iterable(redo_posnames,'Redoing Failed Positions '+acq+'_'+channel,length=len(redo_posnames)):
                nuc = results[posname]['nuc']#*nuc_FF
                signal = results[posname]['signal']#*FF
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
        
            self.save_tensor(stitched[:,:,0],nuc_fname)
            self.save_tensor(stitched[:,:,1],signal_fname)
            
            nuclei = stitched[self.config.parameters['border']:-self.config.parameters['border'],
                            self.config.parameters['border']:-self.config.parameters['border'],0].numpy()
            scale = (np.array(nuclei.shape)*self.config.parameters['ratio']).astype(int)
            nuclei_down = np.array(Image.fromarray(nuclei.astype(float)).resize((scale[1],scale[0]), Image.BICUBIC))
            self.save_image(nuclei_down,self.generate_fname(hybe,self.config.parameters['nucstain_channel'],'stitched',dtype='tif'))

            signal = stitched[self.config.parameters['border']:-self.config.parameters['border'],
                            self.config.parameters['border']:-self.config.parameters['border'],1].numpy()
            scale = (np.array(nuclei.shape)*self.config.parameters['ratio']).astype(int)
            signal_down = np.array(Image.fromarray(signal.astype(float)).resize((scale[1],scale[0]), Image.BICUBIC))
            self.save_image(signal_down,self.generate_fname(hybe,channel,'stitched',dtype='tif'))

            return stitched,nuclei,nuclei_down,signal,signal_down

    def stitch(self):
        hybe = self.config.parameters['nucstain_acq'].split('hybe')[-1]
        channel = self.config.parameters['total_channel']
        if hybe!='':
            acq = self.find_acq(hybe,protocol='hybe')
            if self.config.parameters['strip']:
                bkg_acq = self.find_acq(hybe,protocol='strip')
            else:
                bkg_acq = ''
        if isinstance(self.FF,str) | isinstance(self.nuc_FF,str):
            FF_fname = self.generate_fname('',channel,'FF')
            nuc_FF_fname = self.generate_fname('',self.config.parameters['nucstain_channel'],'FF')
            # if os.path.exists(FF_fname)&os.path.exists(nuc_FF_fname):
            #     try:
            #         self.FF = self.load_tensor(FF_fname)
            #         self.nuc_FF = self.load_tensor(nuc_FF_fname)
            #     except:
            #         pass
            if isinstance(self.FF,str) | isinstance(self.nuc_FF,str):
                # nuc_FF,FF = generate_FFs(self.image_metadata,acq,channel,bkg_acq=bkg_acq,posnames=self.posnames,parameters=self.config.parameters,verbose=self.verbose)
                if isinstance(self.FF,str):
                    FF = generate_FF(self.image_metadata,acq,channel,bkg_acq='',posnames=self.posnames,parameters=self.config.parameters,verbose=self.verbose)
                    self.FF = FF
                if isinstance(self.nuc_FF,str):
                    nuc_FF = generate_FF(self.image_metadata,acq,self.config.parameters['nucstain_channel'],bkg_acq='',posnames=self.posnames,parameters=self.config.parameters,verbose=self.verbose)
                    self.nuc_FF = nuc_FF
                self.save_tensor(self.nuc_FF,nuc_FF_fname)
                self.save_tensor(self.FF,FF_fname)

        # if self.FF=='':
        #     if not self.config.parameters['FF']:
        #         self.FF=1
        #     else:
        #         fname = self.generate_fname('',channel,'FF')
        #         if os.path.exists(fname):
        #             try:
        #                 self.FF = self.load_tensor(fname).numpy()
        #             except:
        #                 self.FF = generate_FF(self.image_metadata,acq,channel,bkg_acq=bkg_acq,verbose=self.verbose)
        #                 self.save_tensor(torch.tensor(self.FF),fname)
        #         else:
        #             self.FF = generate_FF(self.image_metadata,acq,channel,bkg_acq=bkg_acq,verbose=self.verbose)
        #             self.save_tensor(torch.tensor(self.FF),fname)
        # if self.nuc_FF == '':
        #     fname = self.generate_fname('',self.config.parameters['nucstain_channel'],'FF')
        #     if os.path.exists(fname):
        #         try:
        #             self.nuc_FF = self.load_tensor(fname).numpy()
        #         except:
        #             self.nuc_FF = generate_FF(self.image_metadata,acq,self.config.parameters['nucstain_channel'],bkg_acq='',verbose=self.verbose)
        #             self.save_tensor(torch.tensor(self.nuc_FF),fname)
        #     else:
        #         self.nuc_FF = generate_FF(self.image_metadata,acq,self.config.parameters['nucstain_channel'],bkg_acq='',verbose=self.verbose)
        #         self.save_tensor(torch.tensor(self.nuc_FF),fname)
        """ Generate Refernce """
        hybe = self.config.parameters['nucstain_acq'].split('hybe')[-1]
        self.reference_stitched,self.ref_nuclei,self.ref_nuclei_down,signal,signal_down = self.stitcher(hybe,self.config.parameters['total_channel'])
        for r,h,c in self.config.bitmap:
            hybe = h.split('hybe')[-1]
            stitched,nuclei,nuclei_down,signal,signal_down = self.stitcher(hybe,self.config.parameters['total_channel'])
            if self.config.parameters['visualize']:
                visualize_merge(self.ref_nuclei_down,signal_down,title='hybe'+hybe)
                xy = (np.array(self.ref_nuclei.shape)/3).astype(int)
                x_min,y_min = xy-self.config.parameters['border']
                x_max,y_max = xy+self.config.parameters['border']
                visualize_merge(self.ref_nuclei[x_min:x_max,y_min:y_max],signal[x_min:x_max,y_min:y_max],title='hybe'+hybe+' zoom')
                x,y = np.array(self.ref_nuclei.shape)-(np.array(self.ref_nuclei.shape)/3).astype(int)
                x_min,y_min = xy-self.config.parameters['border']
                x_max,y_max = xy+self.config.parameters['border']
                visualize_merge(self.ref_nuclei[x_min:x_max,y_min:y_max],signal[x_min:x_max,y_min:y_max],title='hybe'+hybe+' zoom')
            
    def save_image(self,img,fname):
        if not isinstance(img,np.ndarray):
            img = img.numpy()
        img = img.copy()
        img[img<np.iinfo('uint16').min] = np.iinfo('uint16').min
        img[img>np.iinfo('uint16').max] = np.iinfo('uint16').max
        if os.path.exists(fname):
            os.remove(fname)
        cv2.imwrite(fname, img.astype('uint16'))

    def save_tensor(self,tensor,fname):
        if os.path.exists(fname):
            os.remove(fname)
        torch.save(tensor,fname)

    def load_tensor(self,fname):
        return torch.load(fname)

    def segment(self,model_type='nuclei'):
        """ FIGURE OUT GPU FOR SPEED"""
        proceed = True
        if not self.config.parameters['segment_overwrite']:
            fname = self.generate_fname('',model_type+'_mask','stitched')
            if os.path.exists(fname):
                self.update_user('Loading Existing Segmentation')
                self.mask = self.load_tensor(fname)
                proceed = False
        if proceed:
            """ Cytoplasm"""
            if 'cytoplasm' in model_type:
                """ Check Total"""
                fname = self.generate_fname('','total_mask','stitched')
                if os.path.exists(fname):
                    """ Load """
                    total = self.load_tensor(fname)
                else:
                    self.segment(model_type='total')
                    total = self.mask
                """ Check Nuclei"""
                fname = self.generate_fname('','nuclei_mask','stitched')
                if os.path.exists(fname):
                    """ Load """
                    nuclei = self.load_tensor(fname)
                else:
                    self.segment(model_type='nuclei')
                    nuclei = self.mask
                cytoplasm = total
                cytoplasm[nuclei>0] = 0
                self.mask = cytoplasm
                self.save_tensor(self.mask,self.generate_fname('',model_type+'_mask','stitched'))
            else:
                """ Total & Nuclei"""
                hybe = self.config.parameters['nucstain_acq'].split('hybe')[-1]
                nucstain = self.load_tensor(self.generate_fname(hybe,self.config.parameters['nucstain_channel'],'stitched'))
                if 'total' in model_type:
                    if self.config.parameters['total_acq'] == 'all':
                        total = ''
                        for r,h,c in self.generate_iterable(self.config.bitmap,'Loading Total by Summing All Measurements'):
                            hybe = h.split('hybe')[-1]
                            temp = self.load_tensor(self.generate_fname(hybe,self.config.parameters['total_channel'],'stitched'))
                            if isinstance(total,str):
                                total = temp
                            else:
                                total = total+temp
                    else:
                        hybe = self.config.parameters['total_acq'].split('hybe')[-1]
                        total = self.load_tensor(self.generate_fname(hybe,self.config.parameters['total_channel'],'stitched'))
                if os.path.exists(self.generate_fname('',model_type+'_mask','stitched')):
                    self.update_user('Loading Existing Segmentation')
                    self.mask = self.load_tensor(self.generate_fname('',model_type+'_mask','stitched'))
                else:
                    if 'total' in model_type:
                        model = models.Cellpose(model_type='cyto2',gpu=self.config.parameters['segment_gpu'])
                        self.mask = torch.zeros_like(total)
                    else:
                        model = models.Cellpose(model_type='nuclei',gpu=self.config.parameters['segment_gpu'])
                        self.mask = torch.zeros_like(nucstain)
                    window = 2000
                    x_step = round(nucstain.shape[0]/int(nucstain.shape[0]/window))
                    y_step = round(nucstain.shape[1]/int(nucstain.shape[1]/window))
                    n_x_steps = round(nucstain.shape[0]/x_step)
                    n_y_steps = round(nucstain.shape[1]/y_step)
                    Input = []
                    for x in range(n_x_steps):
                        for y in range(n_y_steps):
                            Input.append([x,y])
                    for (x,y) in self.generate_iterable(Input,'Segmenting Cells: '+model_type):
                        nuc = nucstain[(x*x_step):((x+1)*x_step),(y*y_step):((y+1)*y_step)].numpy()
                        if 'cyto' in model_type:
                            tot = total[(x*x_step):((x+1)*x_step),(y*y_step):((y+1)*y_step)].numpy()
                            stk = np.dstack([nuc,tot,np.zeros_like(nuc)])
                            diameter = int(self.config.parameters['segment_diameter']*1.5)
                            min_size = int(self.config.parameters['segment_diameter']*10*1.5)
                            channels = [1,2]
                        else:
                            stk = nuc
                            diameter = int(self.config.parameters['segment_diameter'])
                            min_size = int(self.config.parameters['segment_diameter']*10)
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
                                                            min_size=min_size)
                        mask = torch.tensor(raw_mask_image.astype(int),dtype=torch.int32)
                        mask[mask>0] = mask[mask>0]+self.mask.max() # ensure unique labels
                        self.mask[(x*x_step):((x+1)*x_step),(y*y_step):((y+1)*y_step)] = mask
                        del raw_mask_image,flows,styles,diams
                        torch.cuda.empty_cache()
                    if 'nuclei' in model_type:
                        fname = self.generate_fname('','total_mask','stitched')
                        if os.path.exists(fname):
                            total = self.load_tensor(fname) # Load Total
                            total[self.mask==0] = 0 # Set non nuclear to 0
                            self.mask = total # replace mask with total&nuclear
                    self.save_tensor(self.mask,self.generate_fname('',model_type+'_mask','stitched'))

    def pull_vectors(self,model_type='nuclei'):
        proceed = True
        if not self.config.parameters['vector_overwrite']:
            fname = self.generate_fname('',model_type,'data',dtype='h5ad')
            if os.path.exists(fname):
                self.update_user('Loading Existing Data')
                self.data = anndata.read_h5ad(fname)
                proceed = False
        if proceed:
            idxes = torch.where(self.mask!=0)
            labels = self.mask[idxes]

            """ Load Vector for each pixel """
            pixel_vectors = torch.zeros([idxes[0].shape[0],len(self.config.bitmap)+1],dtype=torch.int32)
            for i,(r,h,c) in self.generate_iterable(enumerate(self.config.bitmap),'Generating Pixel Vectors',length=len(self.config.bitmap)):
                hybe = h.split('hybe')[-1]
                pixel_vectors[:,i] = self.load_tensor(self.generate_fname(hybe,c,'stitched'))[idxes]
            # Nucstain Signal
            hybe = self.config.parameters['nucstain_acq'].split('hybe')[-1]
            pixel_vectors[:,-1] = self.load_tensor(self.generate_fname(hybe,self.config.parameters['nucstain_channel'],'stitched'))[idxes]

            unique_labels = torch.unique(labels)
            self.vectors = torch.zeros([unique_labels.shape[0],pixel_vectors.shape[1]],dtype=torch.int32)
            pixel_xy = torch.zeros([idxes[0].shape[0],2])
            pixel_xy[:,0] = idxes[0]
            pixel_xy[:,1] = idxes[1]
            self.xy = torch.zeros([unique_labels.shape[0],2],dtype=torch.int32)
            self.size = torch.zeros([unique_labels.shape[0],1],dtype=torch.int32)
            converter = {int(label):[] for label in unique_labels}
            for i in tqdm(range(labels.shape[0]),desc=str(datetime.now().strftime("%H:%M:%S"))+' Generating Label Converter'):
                converter[int(labels[i])].append(i)
            for i,label in tqdm(enumerate(unique_labels),total=unique_labels.shape[0],desc=str(datetime.now().strftime("%H:%M:%S"))+' Generating Cell Vectors and Metadata'):
                m = converter[int(label)]#labels==label
                self.vectors[i,:] = torch.median(pixel_vectors[m,:],axis=0).values
                pxy = pixel_xy[m,:]
                self.xy[i,:] = torch.median(pxy,axis=0).values
                self.size[i] = pxy.shape[0]
            cell_labels = [self.dataset+'_Section'+self.section+'_Cell'+str(i) for i in unique_labels.numpy()]
            self.cell_metadata = pd.DataFrame(index=cell_labels)
            self.cell_metadata['label'] = unique_labels.numpy()
            self.cell_metadata['stage_x'] = self.xy[:,0].numpy()
            self.cell_metadata['stage_y'] = self.xy[:,1].numpy()
            self.cell_metadata['size'] = self.size.numpy()
            self.cell_metadata['section_index'] = self.section
            self.cell_metadata['dapi'] = self.vectors[:,-1].numpy()
            self.vectors = self.vectors[:,0:-1]
            self.data = anndata.AnnData(X=self.vectors.numpy(),
                                var=pd.DataFrame(index=np.array([h for r,h,c in self.config.bitmap])),
                                obs=self.cell_metadata)
            self.data.layers['raw_vectors'] = self.vectors.numpy()
            self.data.obs['polyt'] = self.data.X[:,-1]
            self.data = self.data[:,0:-1]

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
            iterable = tqdm(posnames,desc=str(datetime.now().strftime("%H:%M:%S"))+' Generating FlatField '+acq+' '+channel)
        else:
            iterable = posnames
        for posname in iterable:
            try:
                img = torch.tensor(image_metadata.stkread(Position=posname,Channel=channel,acq=acq).min(2).astype(float))
                if bkg_acq!='':
                    bkg = torch.tensor(image_metadata.stkread(Position=posname,Channel=channel,acq=bkg_acq).mean(2).astype(float))
                    img = img-bkg
                FF.append(img)
            except Exception as e:
                print(posname,acq,bkg_acq)
                print(e)
                continue
        FF = torch.quantile(torch.dstack(FF),0.5,dim=2).numpy()
        vmin,vmax = np.percentile(FF[np.isnan(FF)==False],[0.1,99.9])
        FF[FF<vmin] = vmin
        FF[FF>vmax] = vmax
        FF[FF==0] = np.median(FF)
        FF = np.median(FF)/FF
        return FF

def generate_FFs(image_metadata,acq,channel,posnames=[],bkg_acq='',parameters={},verbose=False):
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
    # if 'mask' in channel:
    #     return ''
    # else:
    #     posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
    #     random_posnames = posnames
    #     FF = []
    #     if verbose:
    #         iterable = tqdm(random_posnames,desc=str(datetime.now().strftime("%H:%M:%S"))+' Generating FlatField '+acq+' '+channel)
    #     else:
    #         iterable = random_posnames
    #     for posname in iterable:
    #         try:
    #             img = torch.tensor(image_metadata.stkread(Position=posname,Channel=channel,acq=acq).min(2).astype(float))
    #             if bkg_acq!='':
    #                 bkg = torch.tensor(image_metadata.stkread(Position=posname,Channel=channel,acq=bkg_acq).mean(2).astype(float))
    #                 img = img-bkg
    #             FF.append(img)
    #         except Exception as e:
    #             print(posname,acq,bkg_acq)
    #             print(e)
    #             continue
    #     FF = torch.quantile(torch.dstack(FF),0.5,dim=2).numpy()
    #     vmin,vmax = np.percentile(FF[np.isnan(FF)==False],[0.1,99.9])
    #     FF[FF<vmin] = vmin
    #     FF[FF>vmax] = vmax
    #     FF[FF==0] = np.median(FF)
    #     FF = np.median(FF)/FF
    if len(posnames)==0:
        posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
    Input = []
    for posname in posnames:
        data = {}
        data['acq'] = acq
        data['bkg_acq'] = bkg_acq
        data['posname'] = posname
        data['image_metadata'] = image_metadata
        data['channel'] = channel
        data['parameters'] = parameters
        Input.append(data)
    pfunc = partial(preprocess_images,FF=1,nuc_FF=1)
    results = {}
    with multiprocessing.Pool(60) as p:
        if verbose:
            iterable = tqdm(p.imap(pfunc,Input),total=len(Input),desc=str(datetime.now().strftime("%H:%M:%S"))+' '+'Calculating Flat Fields')
        else:
            iterable = p.imap(pfunc,Input)
        for posname,nuc,signal,translation_x,translation_y in iterable:
            results[posname] = {}
            results[posname]['nuc'] = nuc
            results[posname]['signal'] = signal

    nuc_FF = torch.stack([results[posname]['nuc'].float() for posname in results.keys()])
    nuc_FF = torch.quantile(nuc_FF,0.5,dim=0).numpy()
    vmin,vmax = np.percentile(nuc_FF[np.isnan(nuc_FF)==False],[0.1,99.9])
    nuc_FF[nuc_FF<vmin] = vmin
    nuc_FF[nuc_FF>vmax] = vmax
    nuc_FF[nuc_FF==0] = np.median(nuc_FF)
    nuc_FF = np.median(nuc_FF)/nuc_FF

    FF = torch.stack([results[posname]['signal'].float() for posname in results.keys()])
    FF = torch.quantile(FF,0.5,dim=0).numpy()
    vmin,vmax = np.percentile(FF[np.isnan(FF)==False],[0.1,99.9])
    FF[FF<vmin] = vmin
    FF[FF>vmax] = vmax
    FF[FF==0] = np.median(FF)
    FF = np.median(FF)/FF
    return nuc_FF,FF

def process_img(img,parameters,FF=1):
    # FlatField 
    img = img*FF
    # Smooth
    if parameters['highpass_smooth']>0:
        img = gaussian_filter(img,parameters['highpass_smooth'])
    # Background Subtract
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

def preprocess_images(data,FF=1,nuc_FF=1):
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
        nuc = process_img(nuc,parameters,FF=nuc_FF)
        # nuc = nuc*nuc_FF
        img = ((image_metadata.stkread(Position=posname,
                                        Channel=channel,
                                        acq=acq).max(axis=2).astype(float)))
        img = process_img(img,parameters,FF=FF)
        if not bkg_acq=='':
            bkg = ((image_metadata.stkread(Position=posname,
                                            Channel=channel,
                                            acq=bkg_acq).max(axis=2).astype(float)))
            bkg = process_img(bkg,parameters,FF=FF)
            bkg_nuc = ((image_metadata.stkread(Position=posname,
                                            Channel=parameters['nucstain_channel'],
                                            acq=bkg_acq).max(axis=2).astype(float)))
            bkg_nuc = process_img(bkg_nuc,parameters,FF=nuc_FF)
            # bkg_nuc = bkg_nuc*nuc_FF

            shift, error = register(nuc, bkg_nuc,10)
            if error!=np.inf:
                translation_x = int(shift[1])
                translation_y = int(shift[0])
                x_correction = np.array(range(bkg.shape[1]))+translation_x
                y_correction = np.array(range(bkg.shape[0]))+translation_y
                i2 = interpolate.interp2d(x_correction,y_correction,bkg,fill_value=None)
                bkg = i2(range(bkg.shape[1]), range(bkg.shape[0]))
            img = img-bkg
        # img = img*FF

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
            if overlap>0.05: # atleast 5% of overlap???
                mask = destination!=0
                mask_x = destination.max(1).values!=0
                ref = destination[mask_x,:]
                mask_y = ref.max(0).values!=0
                ref = destination[mask_x,:]
                ref = ref[:,mask_y]
                non_ref = nuc[mask_x,:]
                non_ref = non_ref[:,mask_y]
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
        return posname,0,0,0,0

    
def visualize_merge(img1,img2,color1=np.array([1,0,1]),color2=np.array([0,1,1]),figsize=[20,20],pvmin=5,pvmax=95,title=''):
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
    