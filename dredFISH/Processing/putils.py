""" Max Projection"""
from PIL import Image
import torch
import multiprocessing
from functools import partial
from scipy.ndimage import gaussian_filter
import time
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

def generate_img(data,FF=''):
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
    Returns a randomly colorized segmented image for display purposes.
    :param img: Should be a numpy array of dtype np.int and 2D shape segmented
    :param color_type: 'rg' for red green gradient, 'rb' = red blue, 'bg' = blue green
    :return: Randomly colorized, segmented image (shape=(n,m,3))
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