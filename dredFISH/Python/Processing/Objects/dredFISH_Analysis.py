#!/usr/bin/env python
import argparse
from MERFISH_Objects.FISHData import *
from datetime import datetime
from fish_helpers import *
from cellpose import models
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="Path to folder containing Raw Data /bigstore/Images20XX/User/Project/Dataset/.")
    parser.add_argument("cword_config", type=str, help="Name of Config File for analysis ie. dredFISH_config")
    args = parser.parse_args()
    
class dredFISH_Position_Class(object):
    def __init__(self,
                 metadata_path,
                 dataset,
                 posname,
                 cword_config,
                 verbose=False):
        
        self.metadata_path = metadata_path
        self.dataset = dataset
        self.posname = posname
        self.cword_config = cword_config
        self.verbose = verbose
        self.proceed = True
        
        self.cword_config = cword_config
        self.merfish_config = importlib.import_module(self.cword_config)
        self.parameters = self.merfish_config.parameters
        self.fishdata = FISHData(os.path.join(self.metadata_path,self.parameters['fishdata']))
        self.bitmap = self.merfish_config.bitmap
        
        self.nucstain_acq = self.parameters['nucstain_acq']
        self.total_acq = self.parameters['total_acq']
        
    def run(self):
        self.check_data()
        if self.proceed:
            self.main()
            
    def update_user(self,message):
        i = [i for i in tqdm([],desc=str(datetime.now().strftime("%H:%M:%S"))+' '+str(message))]
        
    def main(self):
        if self.verbose:
            self.update_user('Processing Position '+self.posname)
        self.config_data()
        self.load_data()
        self.segmentation()
        if self.cell_metadata.shape[0]>0:
            self.pull_vectors()
            self.save_data()
        
    def check_data(self):
        if self.parameters['overwrite']==False:
            self.data = self.fishdata.load_data('h5ad',dataset=self.dataset,posname=self.posname)
            if isinstance(self.data,type(None))==False:
                self.proceed==False
            self.nuclei_mask = self.fishdata.load_data('nuclei_mask',dataset=self.dataset,posname=self.posname)
            self.cytoplasm_mask = self.fishdata.load_data('cytoplasm_mask',dataset=self.dataset,posname=self.posname)
            self.total_mask = self.fishdata.load_data('total_mask',dataset=self.dataset,posname=self.posname)
        
    def config_data(self):
        if self.verbose:
            self.update_user('Loading Metadata')
        self.image_metadata = Metadata(self.metadata_path)
        
    def load_data(self):
        """ Load Nuclei Image """
        if self.verbose:
            self.update_user('Loading Nuclei Image')
        if self.nucstain_acq=='infer':
            self.nucstain_acq = np.array([i for i in self.image_metadata.acqnames if 'nucstain' in i])[-1]
        self.nucstain_image = self.image_metadata.stkread(Position=posname,Channel=self.parameters['nucstain_channel'],acq=self.nucstain_acq).mean(2).astype(float)
        
        """ Load Raw Data """
        if self.verbose:
            self.update_user('Loading Raw Data')
        self.raw_dict = {hybe.split('_')[0]:torch.tensor(self.image_metadata.stkread(Position=self.posname,
                                                                        Channel=self.parameters['total_channel'],
                                                                        acq=hybe).mean(2).astype(float)) for hybe in self.image_metadata.acqnames if 'hybe' in hybe}
        self.background_dict = {'hybe'+strip.split('strip')[1].split('_')[0]:torch.tensor(self.image_metadata.stkread(Position=self.posname,
                                                                                                         Channel=self.parameters['total_channel'],
                                                                                                         acq=strip).mean(2).astype(float)) for strip in self.image_metadata.acqnames if 'strip' in strip}
        
        """ Subtract Background """
        if self.verbose:
            self.update_user('Subtracting Background')
        self.signal_dict = {hybe:self.raw_dict[hybe]-self.background_dict[hybe] for readout,hybe,channel in self.bitmap}
        
        """ Load Total Image """
        if self.verbose:
            self.update_user('Loading Total Image')
        if self.total_acq=='infer':
            self.total_acq = np.array([i for i in self.image_metadata.acqnames if 'nucstain' in i])[-1]
            self.total_image = self.image_metadata.stkread(Position=self.posname,
                                                           Channel=self.parameters['total_channel'],
                                                           acq=self.total_acq).mean(2).astype(float)
            self.background_acq = np.array([i for i in self.image_metadata.acqnames if 'background' in i])[-1]
            self.background_image = self.image_metadata.stkread(Position=self.posname,
                                                           Channel=self.parameters['total_channel'],
                                                           acq=self.background_acq).mean(2).astype(float)
            self.total_image = self.total_image-self.background_image
            
        elif self.total_acq=='none':
            self.total_image = np.dstack([self.signal_dict[hybe] for readout,hybe,channel in self.bitmap]).mean(2)
        else:
            self.total_image = self.image_metadata.stkread(Position=self.posname,
                                                           Channel=self.parameters['total_channel'],
                                                           acq=self.total_acq).mean(2).astype(float)
            
    def process_image(self,image):
        bkg = gaussian_filter(image,self.parameters['segment_diameter']/self.parameters['segment_pixel_size'])
        image = image-bkg
        image[image<0] = 0
        return image
    
    def nuclei_segmentation(self):      
        """ Segment Image """
        if self.verbose:
            self.update_user('Segmenting Nuclei')
        torch.cuda.empty_cache() 
        model = models.Cellpose(model_type='nuclei',gpu=self.parameters['segment_gpu'])
        raw_mask_image,flows,styles,diams = model.eval(self.process_image(self.nucstain_image.copy()),
                                          diameter=self.parameters['segment_diameter']/self.parameters['segment_pixel_size'],
                                          channels=[0,0],
                                          flow_threshold=1,
                                          cellprob_threshold=0)
        self.nuclei_mask = torch.tensor(raw_mask_image.astype(int))
        del model,raw_mask_image,flows,styles,diams
        torch.cuda.empty_cache() 
        
    def total_segmentation(self):
        stk = np.dstack([self.process_image(self.total_image.copy()),self.process_image(self.nucstain_image.copy()),np.zeros_like(self.total_image.copy())])
        """ Segment Image """
        if self.verbose:
            self.update_user('Segmenting Total')
        torch.cuda.empty_cache() 
        model = models.Cellpose(model_type='cyto',gpu=self.parameters['segment_gpu'])
        raw_mask_image,flows,styles,diams = model.eval(stk,
                                          diameter=self.parameters['segment_diameter']/self.parameters['segment_pixel_size'],
                                          channels=[1,2],
                                          flow_threshold=1,
                                          cellprob_threshold=0)
        self.total_mask = torch.tensor(raw_mask_image.astype(int))
        del model,raw_mask_image,flows,styles,diams
        torch.cuda.empty_cache() 

    def generate_cell_metadata(self):
        if self.verbose:
            self.update_user('Generating Cell Metadata')
        labels = np.unique(self.nuclei_mask.ravel()[self.nuclei_mask.ravel()>0])
        self.label_nuclei_coord_dict = {label:torch.where(self.nuclei_mask==label) for j,label in enumerate(labels)}
        self.label_cyto_coord_dict = {label:torch.where((self.cytoplasm_mask==label)) for j,label in enumerate(labels)}
        self.label_total_coord_dict = {label:torch.where(self.total_mask==label) for j,label in enumerate(labels)}
        self.total_image = torch.tensor(self.total_image)
        if labels.shape[0]==0:
            self.cell_metadata = pd.DataFrame(index=['label','pixel_x','pixel_y','nuclei_size','nuclei_signal','cytoplasm_size','cytoplasm_signal','total_size','total_signal']).T
        else:
            cell_metadata = []
            for j,label in enumerate(labels):
                nx,ny = self.label_nuclei_coord_dict[label]
                cx,cy = self.label_cyto_coord_dict[label]
                tx,ty = self.label_total_coord_dict[label]
                label = label
                data = [float(label),float(nx.float().mean()),float(ny.float().mean()),
                        float(nx.shape[0]),float(torch.median(self.total_image[nx,ny])),
                        float(cx.shape[0]),float(torch.median(self.total_image[cx,cy])),
                        float(tx.shape[0]),float(torch.median(self.total_image[tx,ty]))]
                cell_metadata.append(pd.DataFrame(data,index=['label','pixel_x','pixel_y','nuclei_size','nuclei_signal','cytoplasm_size','cytoplasm_signal','total_size','total_signal']).T)
            self.cell_metadata = pd.concat(cell_metadata,ignore_index=True)
            self.cell_metadata['posname'] = self.posname
            position_x,position_y = self.image_metadata.image_table[self.image_metadata.image_table.Position==self.posname].XY.iloc[0]
            self.cell_metadata['posname_stage_x'] = position_x
            self.cell_metadata['posname_stage_y'] = position_y
        
    def segmentation(self):
        """ Segment """
        self.nuclei_segmentation()
        self.total_segmentation()
        
        """ Align Nuclei and Total Labels"""
        if self.verbose:
            self.update_user('Aligning Segmentation labels')
        from scipy import stats
        paired_nuclei_mask = torch.clone(self.nuclei_mask)
        labels = torch.unique(self.nuclei_mask.ravel()[self.nuclei_mask.ravel()>0])
        paired_total_mask = torch.zeros_like(self.total_mask)
        for label in labels:
            m = self.nuclei_mask==label
            cyto_label = stats.mode(self.total_mask[m]).mode[0]
            if cyto_label == 0:
                paired_nuclei_mask[m] = 0 # Eliminate Cell
                continue
            paired_total_mask[self.total_mask==cyto_label] = label
        self.nuclei_mask = paired_nuclei_mask
        self.total_mask = paired_total_mask
        self.cytoplasm_mask = torch.clone(paired_total_mask)
        self.cytoplasm_mask[self.nuclei_mask>0] = 0
        self.generate_cell_metadata()

    def pull_vectors(self):
        if self.verbose:
            self.update_user('Pulling Vectors')
        labels = torch.unique(self.nuclei_mask.ravel()[self.nuclei_mask.ravel()>0]).numpy()
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
        
    def save_data(self):
        if self.verbose:
            self.update_user('Saving Data and Masks')
        data = anndata.AnnData(X=self.nuclei_vectors.numpy(),var=pd.DataFrame(index=np.array([h for r,h,c in self.bitmap])),obs=pd.DataFrame(index=self.cell_metadata.index.astype(str)))
        data.layers['nuclei_vectors'] = self.nuclei_vectors.numpy()
        data.layers['cytoplasm_vectors'] = self.cyto_vectors.numpy()
        data.layers['total_vectors'] = self.total_vectors.numpy()
        for column in self.cell_metadata.columns:
            data.obs[column] = np.array(self.cell_metadata[column])
        data.obs_names_make_unique()
        """ Remove Nans"""
        data = data[np.isnan(data.layers['total_vectors'].sum(1))==False]
        self.data = data
        self.fishdata.add_and_save_data(self.data,
                                        dtype='h5ad',
                                        dataset=self.dataset,
                                        posname=self.posname)
        self.fishdata.add_and_save_data(self.nuclei_mask.numpy(),
                                        dtype='nuclei_mask',
                                        dataset=self.dataset,
                                        posname=self.posname)
        self.fishdata.add_and_save_data(self.cytoplasm_mask.numpy(),
                                        dtype='cytoplasm_mask',
                                        dataset=self.dataset,
                                        posname=self.posname)
        self.fishdata.add_and_save_data(self.total_mask.numpy(),
                                        dtype='total_mask',
                                        dataset=self.dataset,
                                        posname=self.posname)
if __name__ == '__main__':
    metadata_path = args.metadata_path
    cword_config = args.cword_config
    
    if metadata_path[-1]=='/':
        dataset = metadata_path.split('/')[-2]
    else:
        dataset = metadata_path.split('/')[-1]
    image_metadata = Metadata(metadata_path)
    hybes = np.array([i for i in image_metadata.acqnames if 'hybe' in i])
    posnames = np.unique(image_metadata.image_table[image_metadata.image_table.acq==hybes[0]].Position)
    np.random.shuffle(posnames)
    for posname in tqdm(posnames):
        try:
            pos_class = dredFISH_Position_Class(metadata_path,dataset,posname,cword_config,verbose=False)
            pos_class.main()
        except Exception as e:
            print(posname)
            print(e)
    """ Merge Positions """
    data_list = []
    for posname in posnames:
        data = pos_class.fishdata.load_data('h5ad',dataset=dataset,posname=posname)
        if isinstance(data,anndata._core.anndata.AnnData):
            data_list.append(data)
    data = anndata.concat(data_list)
    """ Update Stage Coordiantes """
    pixel_size = pos_class.parameters['segment_pixel_size']
    camera_direction = pos_class.parameters['camera_direction']
    stage_x = np.array(data.obs['posname_stage_x']) + camera_direction[0]*pixel_size*np.array(data.obs['pixel_x'])
    stage_y = np.array(data.obs['posname_stage_y']) + camera_direction[1]*pixel_size*np.array(data.obs['pixel_y'])
    data.obs['stage_x'] = stage_x
    data.obs['stage_y'] = stage_y
    xy = np.zeros([data.shape[0],2])
    xy[:,0] = data.obs['stage_x']
    xy[:,1] = data.obs['stage_y']
    data.obsm['stage'] = xy
    """ Save Data """
    """ Remove Nans"""
    data = data[np.isnan(data.layers['total_vectors'].sum(1))==False]
    pos_class.fishdata.add_and_save_data(data,dtype='h5ad',dataset=pos_class.dataset)
    """ Manually Check Data and save to /bigstore/GeneralStorage/Data/dredFISH as dataset.h5ad"""
