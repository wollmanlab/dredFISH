from dredFISH.Processing.Position import *
from dredFISH.Processing.Section import *
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
import os
import time

class Dataset_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 cword_config,
                 verbose=False):
        """
        __init__ Class To process a dredFISH Dataset

        :param metadata_path: Path to Raw Data
        :type metadata_path: str
        :param dataset: name of dataset
        :type dataset: str
        :param cword_config: name of config module
        :type cword_config: str
        :param verbose: loading bars and print statments, defaults to False
        :type verbose: bool, optional
        """
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.cword_config = cword_config
        self.config = importlib.import_module(self.cword_config)
        self.metadata = ''
        self.data = ''
        self.verbose=verbose

    def run(self):
        """
        run Main Executable
        """
        self.main()

    def main(self):
        """
        process_positions Wrapper to Process individual Positions
        """
        self.setup_batches()
        self.process_batches()
        self.merge_positions()
        self.process_sections()

    def update_user(self,message):
        """
        update_user Communicate with User 

        :param message: message to be sent
        :type message: str
        """
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]

    def setup_batches(self):
        """
        setup_batches Create executables for multiprocessing wrapper
        """
        self.update_user('Setting up batches')
        self.metadata = Metadata(self.metadata_path)
        hybes = np.array([i for i in self.metadata.acqnames if 'hybe' in i])
        self.posnames = np.unique(self.metadata.image_table[self.metadata.image_table.acq==hybes[0]].Position)
        np.random.shuffle(self.posnames)
        Input = []
        for posname in tqdm(self.posnames):
            data = {
                    'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':posname,
                    'cword_config':self.cword_config}
            Input.append(data)

        self.Input = Input
    
    def process_batches(self):
        """
        process_batches Use multiprocessing wrapper to execute batches
        """
        self.update_user('Processing Batches')
        temp_input = []
        for i in self.Input:
            temp_input.append(i)
            if len(temp_input)>=self.config.parameters['batches']:
                pool = multiprocessing.Pool(self.config.parameters['ncpu'])
                sys.stdout.flush()
                results = pool.imap(wrapper, temp_input)
                iterable = tqdm(results,total=len(temp_input),desc=str(datetime.now().strftime("%H:%M:%S"))+' '+self.dataset,position=0)
                for i in iterable:
                    pass
                pool.close()
                sys.stdout.flush()
                temp_input = []
        pool = multiprocessing.Pool(self.config.parameters['ncpu'])
        sys.stdout.flush()
        results = pool.imap(wrapper, temp_input)
        iterable = tqdm(results,total=len(temp_input),desc=str(datetime.now().strftime("%H:%M:%S"))+' '+self.dataset,position=0)
        for i in iterable:
            pass
        pool.close()
        sys.stdout.flush()
        temp_input = []

    def merge_positions(self):
        """
        merge_positions load output from position class and merge
        """
        self.update_user('Merging Positions')
        data_list = []
        for posname in self.posnames:
            try:
                data = anndata.read_h5ad(os.path.join(self.metadata_path,
                                                    self.config.parameters['fishdata'],
                                                    self.dataset+'_'+posname+'_data.h5ad'))
            except:
                data = None
            if isinstance(data,anndata._core.anndata.AnnData):
                data_list.append(data)
        self.data = anndata.concat(data_list)
        self.data.write(filename=os.path.join(self.metadata_path,
                                         self.config.parameters['fishdata'],
                                         self.dataset+'_data.h5ad'))

    def assign_sections(self):
        """
        assign_sections use Kmeans to assign cells to sections
        """
        self.update_user('Assigning Sections')
        self.data.obs['dataset'] = self.dataset
        if self.config.parameters['nregions']>1:
            self.data.obs['section_index'] = KMeans(n_clusters=self.config.parameters['nregions'],
                                            random_state=0,
                                            n_init=50).fit(self.data.obsm['stage']).labels_
        else:
            self.data.obs['section_index'] = 0

    def process_section(self,region):
        """
        process_section create and execute section class for a specific section

        :param region: name of section
        :type region: str
        """
        self.update_user('Processing Section')
        mask = self.data.obs['section_index']==region
        temp = self.data[mask].copy()
        # Name Region by Centroid to resolution's of ums
        X = str(self.config.parameters['resolution']*int(np.median(temp.obs['stage_x'])/self.config.parameters['resolution']))
        Y = str(self.config.parameters['resolution']*int(np.median(temp.obs['stage_y'])/self.config.parameters['resolution']))
        section_XY = 'Section_'+X+'X_'+Y+'Y'
        # Give Cells Unique Name 
        temp.obs.index = [self.dataset+'_'+section_XY+'_'+row.posname+'_Cell'+str(int(row.label)) for idx,row in temp.obs.iterrows()]
        temp.obs['section'] = section_XY
        section = Section_Class(self.metadata_path,
                                         self.dataset,
                                         section_XY,
                                         self.cword_config,
                                         verbose=self.verbose)
        section.metadata = self.metadata
        section.data = temp
        section.run()

    def process_sections(self):
        """
        process_sections wrapper to process all sections
        """
        self.update_user('Processing Sections')
        self.assign_sections()
        for region in tqdm(np.unique(self.data.obs['section_index']),desc='Saving Sections'):
            self.process_section(region)

    
def wrapper(data):
    """
    wrapper Wrapper to Multiprocess Position Class

    :param data: Example
                    data = {
                    'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':posname,
                    'cword_config':self.cword_config}
    :type data: dict
    :return: Example
                    data = {
                    'metadata_path':self.metadata_path,
                    'dataset':self.dataset,
                    'posname':posname,
                    'cword_config':self.cword_config}
    :rtype: dict
    """
    try:
        pos_class = Position_Class(data['metadata_path'],
                                            data['dataset'],
                                            data['posname'],
                                            data['cword_config'],
                                            verbose=False)
        pos_class.run()
    except Exception as e:
        print(data['posname'])
        print(e)
    return data