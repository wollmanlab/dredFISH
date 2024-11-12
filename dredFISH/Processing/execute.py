#!/usr/bin/env python
import argparse
from dredFISH.Processing.Section import *
from dredFISH.Utils.imageu import *
import time
import gc
"""
conda activate dredfish_3.9; nice -n 10 nohup python -W ignore /home/zach/PythonRepos/dredFISH/dredFISH/Processing/execute.py /orangedata/Images2023/Gaby/dredFISH/Acrydite_77.5.A_DPNMF_97.5.B_2023Feb16/ -c dredfish_processing_config_v1 -w A; conda deactivate
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="Path to folder containing Raw Data /bigstore/Images20XX/User/Project/Dataset/")
    parser.add_argument("-c","--cword_config", type=str,dest="cword_config",default='dredfish_processing_config_tree', action='store',help="Name of Config File for analysis ie. dredfish_processing_config")
    parser.add_argument("-s","--section", type=str,dest="section",default='all', action='store',help="keyword in posnames to identify which section to process")
    parser.add_argument("-w","--well", type=str,dest="well",default='', action='store',help="keyword in well to identify which section to process")
    parser.add_argument("-f","--fishdata", type=str,dest="fishdata",default='', action='store',help="fishdata name for save directory")
    
    args = parser.parse_args()
    
if __name__ == '__main__':
    """
     Main Executable to process raw Images to dredFISH data
    """
    metadata_path = args.metadata_path
    cword_config = args.cword_config
    config = importlib.import_module(cword_config)

    print(args)
    if args.fishdata == '':
        fishdata = None
    else:
        fishdata = args.fishdata

    # if args.well=='X':
    # generate_image_parameters(metadata_path,overwrite=False,nthreads = 3)

    if args.section=='all':
        image_metadata = Metadata(metadata_path)
        hybe = [i for i in image_metadata.acqnames if config.parameters['nucstain_acq']+'_' in i.lower()]
        posnames = np.unique(image_metadata.image_table[np.isin(image_metadata.image_table.acq,hybe)].Position)
        sections = np.unique([i.split('-Pos')[0] for i in posnames if '-Pos' in i])
        del image_metadata
    else:
        sections = np.array([args.section])
    if sections.shape[0]==0:
        sections = np.array(['Section1'])
    if args.well!='':
        sections = np.array([i for i in sections if args.well in i])
    # np.random.shuffle(sections)
    print(sections)
    completion_array = np.array([False for i in sections])
    max_attempts = 5
    attempt = 1
    while np.sum(completion_array==False)>0:
        if attempt>max_attempts:
            raise(ValueError('Max Attempts Reached'))
        attempt+=1
        for idx,section in enumerate(sections):
            gc.collect()
            self = Section_Class(metadata_path,section,cword_config,verbose=True)
            if isinstance(fishdata,str):
                self.path = fishdata.copy()
            self.update_user(str(np.sum(completion_array==False))+ ' Unfinished Sections')
            self.update_user('Processing Section '+section)
            self.run()
            if isinstance(self.data,type(None)):
                continue
            completion_array[idx] = True
            del self
            gc.collect()
        time.sleep(60)
    print('Completed')

