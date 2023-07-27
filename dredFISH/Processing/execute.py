#!/usr/bin/env python
import argparse
import shutil
from datetime import datetime
import dredFISH.Processing as Processing
from dredFISH.Processing.Section import *
"""
conda activate dredfish_3.9; python -W ignore /home/zach/PythonRepos/dredFISH/dredFISH/Processing/execute.py /orangedata/Images2023/Gaby/dredFISH/Acrydite_77.5.A_DPNMF_97.5.B_2023Feb16/ -c dredfish_processing_config_v1; conda deactivate
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="Path to folder containing Raw Data /bigstore/Images20XX/User/Project/Dataset/")
    parser.add_argument("-c","--cword_config", type=str,dest="cword_config",default='dredfish_processing_config', action='store',help="Name of Config File for analysis ie. dredfish_processing_config")
    parser.add_argument("-s","--section", type=str,dest="section",default='all', action='store',help="keyword in posnames to identify which section to process")
    parser.add_argument("-w","--well", type=str,dest="well",default='', action='store',help="keyword in well to identify which section to process")
    # parser.add_argument("-f","--fishdata", type=str,dest="fishdata",default='fishdata', action='store',help="fishdata name for save directory")
    
    args = parser.parse_args()
    
if __name__ == '__main__':
    """
     Main Executable to process raw Images to dredFISH data
    """
    metadata_path = args.metadata_path
    cword_config = args.cword_config
    config = importlib.import_module(cword_config)
    config.parameters['nucstain_acq']
    # if args.fishdata == 'fishdata':
    #     fishdata = args.fishdata+str(datetime.today().strftime("_%Y%b%d"))
    # else:
    #     fishdata = args.fishdata
    print(args)
    if args.section=='all':
        image_metadata = Metadata(metadata_path)
        hybe = [i for i in image_metadata.acqnames if config.parameters['nucstain_acq']+'_' in i.lower()]
        posnames = np.unique(image_metadata.image_table[np.isin(image_metadata.image_table.acq,hybe)].Position)
        sections = np.unique([i.split('-Pos')[0] for i in posnames if '-Pos' in i])
    else:
        sections = [args.section]
    if sections.shape[0]==0:
        sections = ['Section1']

    if args.well!='':
        sections = [i for i in sections if args.well in i]
    np.random.shuffle(sections)
    for section in sections:
        # print('Processing Section ',section)
        self = Section_Class(metadata_path,section,cword_config,verbose=True)
        self.update_user('Processing Section '+section)
        # self.config.parameters['fishdata'] = fishdata
        # self.out_path = os.path.join(self.metadata_path,self.config.parameters['fishdata'])
        # if not os.path.exists(self.out_path):
        #     os.mkdir(self.out_path)
        src = os.path.join(Processing.__file__.split('dredFISH/P')[0],args.cword_config+'.py')
        dst = os.path.join(self.path,args.cword_config+'.py')
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copyfile(src, dst)
        self.run()
