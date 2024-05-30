"""
fileu module handles all IO that are part of TMG. 
Two key areas are all Processing related stuff and the save/load of TMG
"""
import os
import logging
from datetime import datetime
import importlib
import sys
import glob

import numpy as np
# import cv2 #opencv-python-headless
import tifffile
import pandas as pd
import torch

import anndata
import shapely
import shapely.wkt
import json
import dill as pickle
from skimage import io
# import pickle


def check_existance(path='',hybe='',channel='',file_type='',model_type='',dataset='',section='',fname='',logger='FileU'):
    """
    check_existance Check if File Existsin accordance with established file structure

    :param path: Path to Section /Storage/Images20**/User/Project/Dataset/fishdata/Dataset/Section/
    :type path: str, optional
    :param hybe: name of hybe ('hybe1','Hybe1','1')
    :type hybe: str, optional
    :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
    :type channel: str, optional
    :param file_type: Data Type to save ('stitched','mask','anndata',...)
    :type file_type: str, optional
    :param model_type: segmentation Type ('total','nuclei','cytoplasm')
    :type model_type: str, optional
    :param logger: Logger to send logs can be a name of logger, defaults to 'FileU'
    :type logger: str, logging.Logger, optional
    :return : True of file exists, False otherwise
    :rtype : bool
    """
    if fname=='':
        filename = generate_filename(path,hybe,channel,file_type,model_type,dataset,section,logger=logger)
    else:
        filename=fname
    return os.path.exists(filename)

def interact_config(path,key='',data=None,return_config=True):
    """
    
    """
    config_path = os.path.join(path,'fileu.json')
    if os.path.exists(config_path):
        with open(config_path, 'r',encoding="utf-8") as json_file:
            config = json.load(json_file)
    else:
        config = {}
    if key!='':
        config[key] = data
        with open(config_path, 'w',encoding="utf-8") as json_file:
            json.dump(config,json_file)
    if return_config:
        return config

def generate_filename(path='',hybe='',channel='',file_type='',model_type='',dataset='',section='',logger='FileU'):
    """
    fname Generate File Name 

    :param path: Path to Section /Storage/Images20**/User/Project/Dataset/fishdata/Dataset/Section/
    :type path: str
    :param hybe: name of hybe ('hybe1','Hybe1','1')
    :type hybe: str
    :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
    :type channel: str
    :param file_type: Data Type to save ('stitched','mask','anndata',...)
    :type file_type: str
    :param model_type: context dependents. For segmentation is Type ('total','nuclei','cytoplasm') for Layer it's layer type
    :type model_type: str
    :param logger: Logger to send logs can be a name of logger, defaults to 'FileU'
    :type logger: str, logging.Logger, optional
    :return: File Path
    :rtype: str
    """
    # normalize path (removes trailing / and makes two consecutive // into /)
    path = os.path.normpath(path)

    ## Check for config 
    config = interact_config(path,return_config=True)
    if not 'version' in config.keys():
        config = interact_config(path,key='version',data=1,return_config=True)
    
    # infer section and dataset from path if both are missing. 
    if section == '' and dataset == '':
        (prefix,section) = os.path.split(path)
        (prefix,dataset) = os.path.split(prefix)

    out_path = os.path.join(path,file_type)
    if not os.path.exists(out_path):
        update_user(f"Making Type Path {file_type}",logger=logger)
        try:
            os.mkdir(out_path)
        except:
            update_user(f"Issue Making Type Path {file_type}",logger=logger)

    backup_file_type = file_type
    file_type = file_type.split('_')[0]
    # if hybe != '':
    #     if 'hybe' in hybe:
    #         hybe = hybe.split('hybe')[-1]
    #     if not 'Hybe' in hybe:
    #         hybe = 'Hybe'+hybe
    ftype_converter = {'anndata':'.h5ad',
                       'matrix':'.csv',
                       'metadata':'.csv',
                       'mask':'.pt',
                       'image':'.tif',
                       'stitched':'.pt',
                       'FF':'.pt',
                       'constant':'.pt',
                       'Layer':'.h5ad',
                       'Taxonomy':'.csv',
                       'Geom':'.wkt',
                       'Figure':'.png',
                       'Report':'.pdf',
                       'Config':'.json',
                       'Model':'.pkl'}
    if not file_type in ftype_converter.keys():
        update_user('Unsupported File Type '+backup_file_type+'->'+file_type,level=40,logger=logger)
        raise ValueError('Unsupported File Type '+backup_file_type+'->'+file_type)
    fname = file_type+ftype_converter[file_type]
    if config['version'] == 1:
        optional_inputs = [model_type,channel,hybe,section,dataset]
    elif config['version'] == 2:
        optional_inputs = [model_type,channel,hybe]
    for optional_input in optional_inputs:
        if optional_input!='':
            fname = f"{str(optional_input)}_{fname}"
    fname = os.path.join(out_path,fname)
    return fname

def save(data,path='',hybe='',channel='',file_type='',model_type='',dataset='',section='',fname='',logger='FileU'):
    """
    save save File in accordance with established file structure

    :param data: data object to be saved
    :type path: object
    :param path: Path to Section /Storage/Images20**/User/Project/Dataset/fishdata/Dataset/Section/
    :type path: str
    :param hybe: name of hybe ('hybe1','Hybe1','1')
    :type hybe: str, optional
    :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
    :type channel: str, optional
    :param type: Data Type to save ('stitched','mask','anndata',...)
    :type type: str, optional
    :param model_type: segmentation Type ('total','nuclei','cytoplasm')
    :type model_type: str, optional
    :param logger: Logger to send logs can be a name of logger, defaults to 'FileU'
    :type logger: str, logging.Logger, optional
    """
    if fname=='':
        fname = generate_filename(path,hybe,channel,file_type,model_type,dataset,section,logger=logger)
    update_user(f"Saving {fname.split('/')[-1]}",level=20,logger=logger)
    file_type = file_type.split('_')[0]
    if file_type == 'anndata':
        data.write(fname)
    elif file_type =='matrix':
        data.to_csv(fname)
    elif file_type == 'metadata':
        data.to_csv(fname)
    elif file_type == 'mask':
        torch.save(data,fname)
    elif file_type == 'image':
        if not isinstance(data,np.ndarray):
            data = data.numpy()
        data = data.copy()
        data[data<np.iinfo('uint16').min] = np.iinfo('uint16').min
        data[data>np.iinfo('uint16').max] = np.iinfo('uint16').max
        # pylint: disable=no-member
        try:
            cv2.imwrite(fname, data.astype('uint16'))
        except:
            tifffile.imwrite(fname, data.astype('uint16'))
        # pylint: enable=no-member
    elif file_type == 'stitched':
        torch.save(data,fname)
    elif file_type == 'FF':
        torch.save(data,fname)
    elif file_type == 'constant':
        torch.save(data,fname)
    elif file_type == 'Layer':
        data.write(fname)
    elif file_type == 'Taxonomy':
        data.to_csv(fname)
    elif file_type == 'Geom':
        save_polygon_list(data,fname)
    elif file_type == 'Config':
        with open(fname, 'w',encoding="utf-8") as f:
            data = data.copy()
            data = {k:v for k,v in data.items() if isinstance(v, (str, int, float, bool, list, dict, tuple, set))}
            json.dump(data,f)
    elif file_type == 'Model':
        with open(fname, 'wb') as f:
            pickle.dump(data,f)
    else:
        update_user('Unsupported File Type '+file_type,level=30,logger=logger)

def load(path='',hybe='',channel='',file_type='anndata',model_type='',dataset='',section='',fname='',logger='FileU'):
    """
    load load File in accordance with established file structure

    :param path: Path to Section /Storage/Images20**/User/Project/Dataset/fishdata/Dataset/Section/
    :type path: str
    :param hybe: name of hybe ('hybe1','Hybe1','1')
    :type hybe: str, optional
    :param channel: Name of Channel ('DeepBlue', 'FarRed',...)
    :type channel: str, optional
    :param file_type: Data Type to save ('stitched','mask','anndata',...)
    :type file_type: str, optional
    :param model_type: segmentation Type ('total','nuclei','cytoplasm')
    :type model_type: str, optional
    :param logger: Logger to send logs can be a name of logger, defaults to 'FileU'
    :type logger: str, logging.Logger, optional
    :return : Desired Data Object or None if not found
    :rtype : object
    """
    if fname=='':
        fname = generate_filename(path,hybe,channel,file_type,model_type,dataset,section,logger=logger)
    file_type = file_type.split('_')[0]
    if os.path.exists(fname):
        try:
            update_user(f"Loading {fname.split('/')[-1]}",level=20,logger=logger)
            if file_type == 'anndata':
                data = anndata.read_h5ad(fname)
            elif file_type =='matrix':
                data = pd.read_csv(fname)
            elif file_type == 'metadata':
                data = pd.read_csv(fname)
            elif file_type == 'mask':
                data = torch.load(fname)
            elif file_type == 'image':
                try:
                    data = cv2.imread(fname,cv2.IMREAD_UNCHANGED)
                except:
                    data = tifffile.imread(fname)
                data = data.astype('uint16')
            elif file_type == 'tif':
                data = io.imread(fname)
            elif file_type == 'stitched':
                data = torch.load(fname)
            elif file_type == 'FF':
                data = torch.load(fname)
            elif file_type == 'constant':
                data = torch.load(fname)
            elif file_type == 'Layer':
                data = anndata.read_h5ad(fname)
            elif file_type == 'Taxonomy':
                data = pd.read_csv(fname)
            elif file_type == 'Geom':
                data = load_polygon_list(fname)
            elif file_type == 'Config':
                with open(fname, 'r',encoding="utf-8") as f:
                    data = json.load(f)
            elif file_type == 'Model':
                with open(fname, 'rb') as f:
                    data = pickle.load(f)
            else:
                update_user('Unsupported File Type '+file_type,level=30,logger=logger)
                data = None
        except Exception as e:
            print(fname)
            print(e)
            data = None
        return data
    else:
        update_user('File Does Not Exist '+fname,level=30,logger=logger)
        return None

def update_user(message,level=20,logger=None):
    """
    update_user Send string messages to logger with various levels OF IMPORTANCE

    :param message: _description_
    :type message: str
    :param level: _description_, defaults to 20
    :type level: int, optional
    :param logger: Logger to send logs can be a name of logger, defaults to 'FileU'
    :type logger: str, logging.Logger, optional
    """
    if isinstance(logger,logging.Logger):
        log = logger
    elif isinstance(logger,str):
        log = logging.getLogger(logger)
    elif isinstance(log,type(None)):
        log = logging.getLogger('Update_User')
    else:
        log = logging.getLogger('Unknown Logger')
    if level<=10:
        # pylint: disable=logging-not-lazy
        log.debug(str(message))
    elif level==20:
        # pylint: disable=logging-not-lazy
        log.info(str(message))
    elif level==30:
        # pylint: disable=logging-not-lazy
        log.warning(str(message))
    elif level==40:
        # pylint: disable=logging-not-lazy
        log.error(str(message))
    elif level>=50:
        # pylint: disable=logging-not-lazy
        log.critical(str(message))

def save_polygon_list(polys,fname):
    """
    Simple polygon saving utility 
    """
    with open(fname, "w", encoding='utf8') as f:
        for p in polys:
            wkt = shapely.wkt.dumps(p)
            f.write(wkt + "\n")

def load_polygon_list(fname):
    polys = list()
    with open(fname,encoding='utf8') as file:
        for line in file:
            wktstr = line.rstrip()
            p = shapely.wkt.loads(wktstr)
            polys.append(p)
    return polys


def load_config_module(inputpath):
    section_paths = [f.path for f in os.scandir(inputpath) if f.is_dir()]
    first_section_path = os.path.join(inputpath,section_paths[0])
    cword_config = glob.glob(os.path.join(first_section_path,'*.py'))[0]

    module_dir = os.path.dirname(cword_config)
    sys.path.append(module_dir)

    # Now import the module using its name (without .py)
    module_name = os.path.basename(cword_config).replace('.py', '')
    config = importlib.import_module(module_name)
    return config


def create_input_df(project_path, animal): 
    sections = {}
    for dataset in os.listdir(project_path):
        if not os.path.isdir(os.path.join(project_path,dataset)):
            continue
        if not animal in dataset:
            continue
        if animal==dataset.split('_')[0]:
            wells = [i.split('.')[-1] for i in dataset.split('_') if '.' in i]
        else:
            wells = [i.split('.')[-1] for i in dataset.split('_') if ('.' in i)&(animal in i)]
        if len(wells)==0:
            continue
        dataset_sections = []
        processing_paths = [i for i in os.listdir(os.path.join(project_path,dataset)) if 'Processing_' in i]
        processing_date = [os.path.getctime(os.path.join(project_path,dataset,processing)) for processing in processing_paths]
        sorted_processing_paths = [x for _, x in reversed(sorted(zip(processing_date, processing_paths)))]
        for processing in sorted_processing_paths:
            for section in [i for i in os.listdir(os.path.join(project_path,dataset,processing)) if i.split('-')[0].split('Well')[-1] in wells]:
                if section in sections.keys():
                    continue
                if not os.path.exists(os.path.join(project_path,dataset,processing,section)):
                    continue
                if check_existance(os.path.join(project_path,dataset,processing,section),file_type='anndata'):
                    sections[section] = {
                        'animal':animal,
                        'processing':processing,
                        'processing_path':os.path.join(project_path,dataset,processing),
                        'dataset':dataset,
                        'dataset_path':project_path}
                    dataset_sections.append(section)
        registration_paths = [i for i in os.listdir(os.path.join(project_path,dataset)) if 'Registration_' in i]
        registration_date = [os.path.getctime(os.path.join(project_path,dataset,registration)) for registration in registration_paths]
        sorted_registration_paths = [x for _, x in reversed(sorted(zip(registration_date, registration_paths)))]
        for registration in sorted_registration_paths:
            for section in dataset_sections:
                if 'registration_path' in sections[section].keys():
                    continue
                if not os.path.exists(os.path.join(project_path,dataset,registration,section)):
                    continue
                if check_existance(os.path.join(project_path,dataset,registration,section),channel='X',file_type='Model'):
                    sections[section]['registration_path'] = os.path.join(project_path,dataset,registration)
                    sections[section]['registration'] = registration
    incomplete_sections = []
    for section,items in sections.items():
        if not 'registration_path' in items.keys():
            incomplete_sections.append(section)
    for section in incomplete_sections:
        del sections[section]

    # Convert the sections dictionary to a DataFrame
    input_df = pd.DataFrame.from_dict(sections, orient='index')
    input_df['section_acq_name']  = input_df.index
    input_df.reset_index(drop=True, inplace=True)
    input_df = input_df[['animal', 'section_acq_name','processing','registration','dataset', 'registration_path', 'processing_path', 'dataset_path']]

    return input_df