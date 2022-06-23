from datetime import datetime
import numpy as np
import anndata
import pandas as pd
import importlib
from metadata import Metadata
from functools import partial
import multiprocessing
from tqdm import tqdm
import cv2
import torch
import os
from PIL import Image
from scipy.ndimage import gaussian_filter
import time
import torch

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
        self.section = section
        self.cword_config = cword_config
        self.config = importlib.import_module(self.cword_config)
        self.metadata = ''
        self.data = ''
        self.verbose=verbose
    
    def run(self):
        """
        run Main Executable
        """
        self.load_data()
        self.generate_QC()
        
        
    def update_user(self,message):
        """
        update_user Send Message to User

        :param message: Message
        :type message: str
        """
        if self.verbose:
            i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]

    def load_data(self):
        self.load_metadata()
        self.load_h5ad()
        
    def load_h5ad(self):
        """
        load_data Load Previously Processed Data
        """
        if isinstance(self.data,str):
            self.update_user('Loading Data')
            filename=os.path.join(self.metadata_path,
                                         self.config.parameters['fishdata'],
                                         self.dataset+'_'+self.section+'_data.h5ad')
            self.data = anndata.read_h5ad(filename)
        self.posnames = np.unique(self.data.obs['posname'])


    def load_metadata(self):
        """
        load_data Load Previously Processed Data
        """
        if isinstance(self.metadata,str):
            self.update_user('Loading Metadata')
            self.metadata = Metadata(self.metadata_path)
        
    def generate_stitched(self,acq,channel):
        """
        generate_stitched Stitch Images together for QC

        :param acq: name of acq
        :type acq: str
        :param channel: Name of Channel
        :type channel: str
        """

        self.update_user('Stitching Images')
        proceed = True
        self.out_path = os.path.join(self.metadata_path,self.config.parameters['results'])
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        self.out_path = os.path.join(self.out_path,self.section)
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        fname = os.path.join(self.out_path,self.dataset+'_'+self.section+'_'+acq+'_'+channel+'.tif')
        if not self.config.parameters['overwrite']:
            if os.path.exists(fname):
                self.update_user('Loading Stitched')
                stitched = cv2.imread(fname,-1)
                proceed = False
        if proceed:
            stitched = generate_stitched(self.metadata,
                                        acq,channel,
                                        pixel_size=self.config.parameters['pixel_size'],
                                        rotate = self.config.parameters['stitch_rotate'],
                                        flipud=self.config.parameters['stitch_flipud'],
                                        fliplr=self.config.parameters['stitch_fliplr'],
                                        posnames=self.posnames)
            self.update_user('Downsampling Stitched')
            ratio = self.config.parameters['pixel_size']/self.config.parameters['QC_pixel_size']
            scale = (np.array(stitched.shape)*ratio).astype(int)
            stitched = np.array(Image.fromarray(stitched.astype(float)).resize((scale[1],scale[0]), Image.BICUBIC))
            self.update_user('Saving Stitched')
            stitched[stitched<0] = 0
            cv2.imwrite(fname, stitched.astype('uint16'))
        return stitched
        
    def generate_QC(self):
        """
        generate_QC Wrapper to generate QC files for Section
        Generates Stitched Images for each measurement as well as segmentation QC
        """
        self.update_user('Generating QC')
        if self.config.parameters['nucstain_acq']=='infer':
            acq = [i for i in self.metadata.acqnames if 'nucstain_' in i][-1]
            stitched = self.generate_stitched(acq,self.config.parameters['nucstain_channel'])
        elif self.config.parameters['nucstain_acq']=='none':
            do = 'nothing'
        else:
            stitched = self.generate_stitched(self.config.parameters['nucstain_acq'],
                                   self.config.parameters['nucstain_channel'])
            
        if self.config.parameters['total_acq']=='infer':
            acq = [i for i in self.metadata.acqnames if 'nucstain_' in i][-1]
            stitched = self.generate_stitched(acq,self.config.parameters['total_channel'])
        elif self.config.parameters['total_acq']=='none':
            do = 'nothing'
        else:
            stitched = self.generate_stitched(self.config.parameters['total_acq'],
                                   self.config.parameters['total_channel'])
            
        for r,h,c in self.config.bitmap:
            acq = [i for i in self.metadata.acqnames if h+'_' in i][-1]
            stitched = self.generate_stitched(acq,c)

    def remove_outliers(self):
        self.update_user('Removing Outliers')
        XY = torch.tensor(self.data.obsm['stage'].copy())
        center = torch.median(XY,axis=0).values
        distances = torch.cdist(XY,center[:,None].T).numpy()
        thresh = np.percentile(distances,99)
        mask = distances<thresh
        self.data = self.data[mask]

    def load_allen(self):
        allen_template = regu.load_allen_template(allen_template_path)
        allen_tree, allen_maps = regu.load_allen_tree(allen_tree_path)
        allen_annot = regu.load_allen_annot(allen_annot_path) # takes about 30 seconds

    def register_preview(self):
        spatial_data = regu.check_run(self.data.obsm['stage'].copy(), 
                                    allen_template, 
                                    allen_annot, 
                                    allen_maps,
                                    idx_ccf, 
                                    flip=flip)

    def register(self):
        spatial_data = regu.real_run(self.data.obsm['stage'].copy(), 
                                    allen_template, 
                                    allen_annot, 
                                    allen_maps,
                                    idx_ccf, 
                                    flip=flip,
                    dataset="", # a name
                    outprefix=outprefix, 
                    force=force,
                    )
        

def generate_img(data,FF=''):
    """
    generate_img Load and Process Images for Stitching

    :param data: Example
                data['acq'] = acq
                data['posname'] = posname
                data['image_metadata'] = image_metadata
                data['channel'] = channel
                data['exposure'] = exposure
    :type data: dict
    :param FF: Flat Field Image array of same shape as image, defaults to ''
    :type FF: np.array, optional
    :return: processed Image
    :rtype: np.array
    """
    try:
        acq = data['acq']
        posname = data['posname']
        image_metadata = data['image_metadata']
        channel = data['channel']
        if 'hybe' in acq:
            strip = 'strip' + acq.split('hybe')[1].split('_')[0] + '_' +str(int(acq.split('_')[1])-1)
        if ('nucstain' in acq)&(channel!='DeepBlue'):
            strip = 'backround' + acq.split('nucstain')[1].split('_')[0] + '_' +str(int(acq.split('_')[1])-1)
        if not isinstance(FF,str):
            if data['exposure']!='':
                img = FF*image_metadata.stkread(Position=posname,Channel=channel,acq=acq,Exposure=data['exposure']).max(axis=2).astype(float)
            else:
                img = FF*image_metadata.stkread(Position=posname,Channel=channel,acq=acq).max(axis=2).astype(float)
        else:
            if data['exposure']!='':
                img = image_metadata.stkread(Position=posname,Channel=channel,acq=acq,Exposure=data['exposure']).max(axis=2).astype(float)
            else:
                img = image_metadata.stkread(Position=posname,Channel=channel,acq=acq).max(axis=2).astype(float)
            if 'hybe' in acq:
                if data['exposure']!='':
                    bkg = image_metadata.stkread(Position=posname,Channel=channel,acq=strip,Exposure=data['exposure']).max(axis=2).astype(float)
                else:
                    bkg = image_metadata.stkread(Position=posname,Channel=channel,acq=strip).max(axis=2).astype(float)
                img = img-bkg
        return posname,img
    except Exception as e:
        print(e)
        print(posname)
        img = np.zeros([2048, 2448]).astype(float)
        return posname,img
    
def generate_FF(image_metadata,acq,channel):
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
    posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
    random_posnames = posnames[0:20]
    FF = torch.dstack([torch.tensor(image_metadata.stkread(Position=posname,Channel=channel,acq=acq).min(2).astype(float)) for posname in random_posnames]).min(2).values
    FF = gaussian_filter(FF.numpy(),5)
    FF = FF.mean()/FF
    return FF

def generate_stitched(image_metadata,
                      acq,channel,
                      pixel_size=0.495,
                      n_pixels=np.array([2448, 2048]),
                      exposure='',
                      rotate=0,
                      flipud=False,
                      fliplr=False,
                      dtype = torch.int32,
                      verbose=True,
                      posnames=[]):
    """
    generate_stitched Generate Stitched View of all FOVs

    :param image_metadata: Data Loader
    :type image_metadata: Metadata Class
    :param acq: name of acquisition
    :type acq: str
    :param channel: name of channel
    :type channel: str
    :param pixel_size: size of pizel in um, defaults to 0.495
    :type pixel_size: float, optional
    :param n_pixels: shape of image in pixels, defaults to np.array([2448, 2048])
    :type n_pixels: np.array, optional
    :param exposure: exposure of image in ms, defaults to ''
    :type exposure: int, optional
    :param rotate: how many times to rotate 90 degrees, defaults to 0
    :type rotate: int, optional
    :param flipud: flip the image vertically, defaults to False
    :type flipud: bool, optional
    :param fliplr: flip the image horizantally, defaults to False
    :type fliplr: bool, optional
    :param dtype: datatype for final stitched image, defaults to torch.int32
    :type dtype: _type_, optional
    :param verbose: loading bars and print statements, defaults to True
    :type verbose: bool, optional
    :param posnames: list of position names if only a subset is desired, defaults to []
    :type posnames: list, optional
    :return: stitched image 
    :rtype: np.array
    """
    if len(posnames)==0:
        posnames = image_metadata.image_table[image_metadata.image_table.acq==acq].Position.unique()
    FF = generate_FF(image_metadata,acq,channel)
    coordinates = {}
    for posname in posnames:
        coordinates[posname] = (image_metadata.image_table[(image_metadata.image_table.acq==acq)&
                                                           (image_metadata.image_table.Position==posname)].XY.iloc[0]/pixel_size).astype(int)
    xy = np.stack([pxy for i,pxy in coordinates.items()])
    y_min,x_min = xy.min(0)-n_pixels
    y_max,x_max = xy.max(0)+(2*n_pixels)
    x_range = np.array(range(x_min,x_max+1))
    y_range = np.array(range(y_min,y_max+1))
    stitched = torch.tensor(np.zeros([len(x_range),len(y_range)]),dtype=dtype)
    img_dict = {}
    Input = []
    for posname in np.unique(posnames):
        data = {}
        data['acq'] = acq
        data['posname'] = posname
        data['image_metadata'] = image_metadata
        data['channel'] = channel
        data['exposure'] = exposure
        Input.append(data)
    pfunc = partial(generate_img,FF=FF)
    with multiprocessing.Pool(60) as p:
        if verbose:
            iterable = tqdm(p.imap(pfunc,Input),total=len(Input),desc='Generate Images '+acq+'_'+channel)
        else:
            iterable = p.imap(pfunc,Input)
        for posname,image in iterable:
            try:
                position_y,position_x = coordinates[posname].astype(int)
                if rotate!=0:
                    image = np.rot90(np.array(image),rotate) #3
                if flipud:
                    image = np.flipud(image)
                if fliplr:
                    image = np.fliplr(image)
                img = torch.tensor(np.array(image),dtype=dtype)
                img_x_min = position_x-x_min
                img_x_max = (position_x-x_min)+img.shape[0]
                img_y_min = position_y-y_min
                img_y_max = (position_y-y_min)+img.shape[1]
                img = torch.stack([img,stitched[img_x_min:img_x_max,img_y_min:img_y_max]]).max(0).values
                stitched[img_x_min:img_x_max,img_y_min:img_y_max] = img.type(dtype)
            except Exception as e:
                print(e)
                print(posname)
                # raise
                continue
    return stitched.numpy()

def colorize_segmented_image(img, color_type='rgb'):
    """
    colorize_segmented_image 
    Returns a randomly colorized segmented image for display purposes.
    :param img: Should be a numpy array of dtype np.int and 2D shape segmented
    :type img: np.array
    :param color_type: 'rg' for red green gradient, 'rb' = red blue, 'bg' = blue green, defaults to 'rgb'
    :type color_type: str, optional
    :return: Randomly colorized, segmented image (shape=(n,m,3))
    :rtype: np.array
    """
    # get empty rgb_img as a skeleton to add colors
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3))

    # make your colors
    num_cells = np.max(img)  # find the number of cells so you know how many colors to make
    colors = np.random.randint(0, 255, (num_cells, 3))
    if not 'r' in color_type:
        colors[:, 0] = 0  # remove red
    if not 'g' in color_type:
        colors[:, 1] = 0  # remove green
    if not 'b' in color_type:
        colors[:, 2] = 0  # remove blue

    regions = regionprops(img)
    for i in range(1, len(regions)):  # start at 1 because no need to replace background (0s already)
        rgb_img[tuple(regions[i].coords.T)] = colors[i]  # won't use the 1st color

    return rgb_img.astype(np.int)


