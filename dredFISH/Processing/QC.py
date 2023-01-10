#!/usr/bin/env python
from dredFISH.Processing.execute import *
from dredFISH.Analysis.TissueGraph import *
import matplotlib.pyplot as plt
import seaborn as sns
from dredFISH.Utils import basicu
from dredFISH.Utils import pathu
import scanpy as sc
from dredFISH.Analysis import Classification
import argparse
import shutil
from datetime import datetime
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", type=str, help="Path to folder containing Raw Data /bigstore/Images20XX/User/Project/Dataset/")
    parser.add_argument("-c","--cword_config", type=str,dest="cword_config",default='dredfish_processing_config', action='store',help="Name of Config File for analysis ie. dredfish_processing_config")
    parser.add_argument("-s","--section", type=str,dest="section",default='all', action='store',help="keyword in posnames to identify which section to process")
    parser.add_argument("-f","--fishdata", type=str,dest="fishdata",default='fishdata', action='store',help="fishdata name for save directory")
    parser.add_argument("-m","--model_types", type=str,dest="model_types",default='', action='store',help="fishdata name for save directory")
    args = parser.parse_args()
    
if __name__ == '__main__':
    print(args)
    metadata_path = args.metadata_path# '/bigstore/Images2022/Gaby/dredFISH/DPNMF-FR_R1_4A_UC_R2_5C_2022Nov27/'
    cword_config = args.cword_config #'dredfish_processing_config'
    fishdata = args.fishdata #'fishdata_2022Dec08'
    section = args.section
    model_types = args.model_types


    config = importlib.import_module(cword_config)
    dataset = [i for i in metadata_path.split('/') if not i==''][-1]
    config.parameters['fishdata'] = fishdata


    if isinstance(model_types,str):
        if model_types=='':
            model_types = config.parameters['model_types']
        else:
            model_types = [model_types]

    if section=='all':
        image_metadata = Metadata(metadata_path)
        hybe1s = [i for i in image_metadata.acqnames if 'hybe1_' in i]
        posnames = np.unique(image_metadata.image_table[np.isin(image_metadata.image_table.acq,hybe1s)].Position)
        sections = np.unique([i.split('-')[0] for i in posnames])
        # np.random.shuffle(sections)
    else:
        sections = [section]

    for section in sections:
        for model_type in model_types:
            print('Section '+section+' '+model_type)
            try:
                figure_list = []
                """ Load Data """
                self = Section_Class(metadata_path,dataset,section,cword_config,verbose=True)
                self.config.parameters['fishdata'] = config.parameters['fishdata']
                fname = self.generate_fname('',model_type,'data',dtype='h5ad')
                self.data = anndata.read_h5ad(fname)
            except Exception as e:
                print(e)
                print(section)
                print(model_type)
            if isinstance(self.data,str):
                print('Not Finished')
                continue

            """ Filter Non Cell"""
            self.data = self.data[self.data.obs['dapi']>5000]

            """ DAPI """
            self.update_user('Visualize Dapi')
            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(25,12),layout="constrained")
            x = self.data.obs['stage_x']
            y = self.data.obs['stage_y']
            c = self.data.obs['dapi']
            vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
            im = axs[0].scatter(x,y,c=c,s=0.05,cmap='jet',vmin=vmin,vmax=vmax,marker='x')
            axs[0].axis('equal')
            axs[0].title.set_text('Cells')
            fig.colorbar(im,ax=axs[0], orientation="horizontal")

            hybe = self.config.parameters['nucstain_acq'].split('hybe')[-1]
            fname = self.generate_fname(hybe,self.config.parameters['nucstain_channel'],'stitched',dtype='tif')
            signal = np.rot90(np.array(cv2.imread(fname,-1)))
            vmin,vmax = np.percentile(signal[np.isnan(signal)==False],[5,95])
            im = axs[1].imshow(signal,cmap='jet',vmin=vmin,vmax=vmax)
            axs[1].title.set_text('Dapi')
            fig.colorbar(im,ax=axs[1], orientation="horizontal")
            fname = self.generate_fname('','dapi',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            plt.close()

            """ PolyT """
            self.update_user('Visualize PolyT')
            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(25,12),layout="constrained")
            x = self.data.obs['stage_x']
            y = self.data.obs['stage_y']
            c = self.data.obs['polyt']
            vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
            im = axs[0].scatter(x,y,c=c,s=0.05,cmap='jet',vmin=vmin,vmax=vmax,marker='x')
            axs[0].axis('equal')
            axs[0].title.set_text('Cells')
            fig.colorbar(im,ax=axs[0], orientation="horizontal")

            hybe = self.config.parameters['nucstain_acq'].split('hybe')[-1]
            fname = self.generate_fname(hybe,self.config.parameters['total_channel'],'stitched',dtype='tif')
            signal = np.rot90(np.array(cv2.imread(fname,-1)))
            vmin,vmax = np.percentile(signal[np.isnan(signal)==False],[5,95])
            im = axs[1].imshow(signal,cmap='jet',vmin=vmin,vmax=vmax)
            axs[1].title.set_text('polyT')
            fig.colorbar(im,ax=axs[1], orientation="horizontal")
            fname = self.generate_fname('','polyT',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            plt.close()

            """ Visualize Processing """
            self.update_user('Visualize Processing')
            nrows = 5
            ncols = 5
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(nrows*5,ncols*5),layout="constrained")
            fig.suptitle('Processed Images')
            for i,(r,h,channel) in enumerate(self.config.bitmap):
                # if h=='hybe25':
                #     continue
                hybe = h.split('hybe')[-1]
                fname = self.generate_fname(hybe,channel,'stitched',dtype='tif')
                signal = np.rot90(np.array(cv2.imread(fname,-1)))
                vmin,vmax = np.percentile(signal,[5,95])
                im = axs[int(i/ncols),i%ncols].imshow(signal,vmin=vmin,vmax=vmax,cmap='jet')
                axs[int(i/ncols),i%ncols].axis('off')
                axs[int(i/ncols),i%ncols].title.set_text(h)
                fig.colorbar(im,ax=axs[int(i/ncols),i%ncols], orientation="horizontal")

            fname = self.generate_fname('','processing',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            plt.close()

            """ Visualize Raw Vectors """
            self.update_user('Visualize Raw Vectors')
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(nrows*5,ncols*5),layout="constrained")
            fig.suptitle('Raw Vectors')
            x = self.data.obs['stage_x']
            y = self.data.obs['stage_y']
            for i,(r,h,channel) in enumerate(self.config.bitmap):
                if h == 'hybe25':
                    # continue
                    c = self.data.obs['polyt']
                else:
                    c = self.data.X[:,self.data.var.index==h]
                vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                im = axs[int(i/ncols),i%ncols].scatter(x,y,c=c,s=0.05,marker='x',vmin=vmin,vmax=vmax,cmap='jet')
                axs[int(i/ncols),i%ncols].axis('off')
                axs[int(i/ncols),i%ncols].title.set_text(h)
                fig.colorbar(im,ax=axs[int(i/ncols),i%ncols], orientation="horizontal")
            fname = self.generate_fname('','raw_vectors',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            plt.close()

            """ Calculate Correlation """

            """ Normalize """
            self.data.layers['raw'] = self.data.X.copy()
            self.data.X = basicu.normalize_fishdata(self.data.X, norm_cell=True, norm_basis=True)
            self.data.layers['normalized'] = self.data.X.copy()

            """ Visualize Normalized Vectors"""
            self.update_user('Visualize Normalized Vectors')
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(nrows*5,ncols*5),layout="constrained")
            fig.suptitle('Processed Vectors')
            x = self.data.obs['stage_x']
            y = self.data.obs['stage_y']
            for i,(r,h,channel) in enumerate(self.config.bitmap):
                if h == 'hybe25':
                    # continue
                    c = self.data.obs['polyt']
                    c = c-np.median(c)
                    c = c/np.std(c)
                else:
                    c = self.data.X[:,self.data.var.index==h]
                vmin,vmax = np.percentile(c[np.isnan(c)==False],[5,95])
                im = axs[int(i/ncols),i%ncols].scatter(x,y,c=c,s=0.05,marker='x',vmin=-1.5,vmax=1.5,cmap='coolwarm')
                axs[int(i/ncols),i%ncols].axis('off')
                axs[int(i/ncols),i%ncols].title.set_text(h)
                fig.colorbar(im,ax=axs[int(i/ncols),i%ncols], orientation="horizontal")
            fname = self.generate_fname('','normalized_vectors',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            plt.close()

            """ Calculate Correlation """

            """ Unsupervised Clustering """
            self.update_user('Generating Unsupervised Labels')
            adata = self.data.copy()
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=0)
            sc.tl.leiden(adata)
            self.data.obs['leiden'] = np.array(adata.obs['leiden'])
            # # sc.tl.paga(adata)
            # # sc.pl.paga(adata, plot=False)
            # # sc.tl.umap(adata, init_pos='paga')
            # # # sc.pl.umap(adata, color=['leiden', 'dapi', 'polyt'])
            # plt.figure(figsize=[10,7.5])
            # plt.title('Unsupervised')
            # x = adata.obs['stage_x']
            # y = adata.obs['stage_y']
            # for i in np.unique(adata.obs['leiden']).astype(int):
            #     m = adata.obs['leiden']==str(i)
            #     plt.scatter(x[m],y[m],s=0.05,marker='x',label=i)
            # plt.legend()
            # plt.axis('off')
            # fname = self.generate_fname('','Unsupervised_Clustering',model_type,dtype='png')
            # figure_list.append(fname)
            # plt.savefig(fname) #SAVE
            # plt.close()

            # plt.figure(figsize=[10,7.5])
            # plt.title('Unsupervised UMAP')
            # x = adata.obsm['X_umap'][:,0]
            # y = adata.obsm['X_umap'][:,1]
            # for i in np.unique(adata.obs['leiden']).astype(int):
            #     m = adata.obs['leiden']==str(i)
            #     plt.scatter(x[m],y[m],s=0.05,marker='x',label=i)
            # plt.legend()
            # plt.axis('off')
            # fname = self.generate_fname('','Unsupervised_Clustering_UMAP',model_type,dtype='png')
            # figure_list.append(fname)
            # fig.savefig(fname) #SAVE


            """ Supervised Clustering """
            self.update_user('Generating Supervised Labels')
            data = self.data.copy()
            FISHbasis = data.layers['raw'].copy()
            FISHbasis_norm = basicu.normalize_fishdata(FISHbasis, norm_cell=True, norm_basis=True)
            XYS = np.ones([data.shape[0],3])
            XYS[:,0:2] = np.array(data.obs[['stage_x','stage_y']])

            metric = 'cosine'
            TMG = TissueMultiGraph(basepath=os.path.join(metadata_path,config.parameters['fishdata']), 
                                               redo=True, # create an empty one
                                              )
            # creating first layer - cell tissue graph
            TG = TissueGraph(feature_mat=FISHbasis_norm,
                             feature_mat_raw=FISHbasis,
                             basepath=TMG.basepath,
                             layer_type="cell", 
                             redo=True)

            # add observations and init size to 1 for all cells
            TG.node_size = np.ones((FISHbasis_norm.shape[0],1))

            # add XY and slice information 
            TG.XY = XYS[:,0:2]
            TG.Slice = XYS[:,2]

            # build two key graphs
            logging.info('building spatial graphs')
            TG.build_spatial_graph(XYS)
            logging.info('building feature graphs')
            TG.build_feature_graph(FISHbasis_norm, metric=metric)

            TMG.Layers.append(TG)

            # create known cell type classifier and train and predict
            allen_classifier = Classification.KnownCellTypeClassifier(
                TMG.Layers[0], 
                tax_name='Allen_types',
                ref='allen_smrt_dpnmf',
                ref_levels=['class_label', 'neighborhood_label', 'subclass_label'], #, 'cluster_label'], 
                model='knn',
            )
            allen_classifier.train(verbose=True)
            type_mat = allen_classifier.classify()


            # register results in TMG and create Isozones
            TMG.add_type_information(0, 
                                     type_mat[:,-1], # only record the finest level of cell types
                                     allen_classifier.tax) # not exactly sure what .tax does
            TMG.create_isozone_layer()
            logging.info(f"TMG has {len(TMG.Layers)} Layers")

            """ Vector Averages """
            self.update_user('Generating Unsupervised Measured Vector Averages')
            labels = np.array(adata.obs['leiden'])
            cell_types = np.unique(labels)
            dredfish_mu = np.zeros([self.data.X.shape[1],cell_types.shape[0]])
            for i,ct in enumerate(cell_types):
                m = labels==ct
                dredfish_mu[:,i] = self.data.X[m,:].mean(0)
            m,r,c = basicu.diag_matrix_rows(dredfish_mu.T)
            fig, ax = plt.subplots(figsize=(10,10))  
            s = sns.heatmap(pd.DataFrame(m,index=cell_types[r]),cmap='bwr',center=0,vmin=-3,vmax=3,ax=ax)
            plt.title('Measured Unsupervised Vector Averages')
            fig = s.get_figure()
            fname = self.generate_fname('','Measured_Unsupervised_Vector_Averages',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            plt.close()

            """ Vector Averages """
            self.update_user('Generating Supervised Measured Vector Averages')
            labels = np.array(TMG.Layers[0].adata.obs['Type'])
            cell_types = np.unique(labels)
            dredfish_mu = np.zeros([TMG.Layers[0].adata.X.shape[1],cell_types.shape[0]])
            for i,ct in enumerate(cell_types):
                m = labels==ct
                dredfish_mu[:,i] = TMG.Layers[0].adata.X[m,:].mean(0)
            m,r,c = basicu.diag_matrix_rows(dredfish_mu.T)
            fig, ax = plt.subplots(figsize=(10,10))  
            s = sns.heatmap(pd.DataFrame(m,index=cell_types[r]),cmap='coolwarm',center=0,vmin=-3,vmax=3,ax=ax)
            plt.title('Measured Supervised Vector Averages')
            fig = s.get_figure()
            fname = self.generate_fname('','Measured_Supervised_Vector_Averages',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            plt.close()

            """Reference Vector Averages"""
            self.update_user('Generating Reference Vector Averages')
            ref='allen_smrt_dpnmf'
            ref_path = pathu.get_path(ref, check=True)
            ref_data = anndata.read(ref_path)
            # Normalize 
            ref_data.X = basicu.normalize_fishdata(ref_data.X, norm_cell=True, norm_basis=True)
            labels = np.array(ref_data.obs['subclass_label'])
            ref_cell_types = np.unique(labels)
            ref_mu = np.zeros([ref_data.X.shape[1],ref_cell_types.shape[0]])
            for i,ct in enumerate(ref_cell_types):
                m = labels==ct
                ref_mu[:,i] = ref_data.X[m,:].mean(0)
            import seaborn as sns
            m,r,c = basicu.diag_matrix_rows(ref_mu.T)
            fig, ax = plt.subplots(figsize=(10,10))  
            s = sns.heatmap(pd.DataFrame(m,index=ref_cell_types[r]),cmap='coolwarm',center=0,vmin=-3,vmax=3,ax=ax)
            plt.title('Reference Vector Averages')
            fig = s.get_figure()
            fname = self.generate_fname('','Reference Vector Averages',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            plt.close()

            xy = TMG.Layers[0].XY
            x, y = xy[:,0], xy[:,1]
            c = TMG.Layers[0].Type
            df = pd.DataFrame(np.vstack([x,y,c]).T, columns=['x', 'y', 'c'])
            kwargs = dict(s=1, edgecolor='none')
            fig, ax = plt.subplots(figsize=(10,12.5))
            plt.title('Supervised Label Transfer')
            for ct in np.unique(c):
                m = c==ct
                plt.scatter(x[m],y[m],s=0.05,marker='x',label=ct)
            # sns.scatterplot(data=df, x='x', y='y', hue='c', ax=ax, **kwargs)
            ax.set_aspect('equal')
            ax.axis('off')
            # ax.legend(ncol=5, bbox_to_anchor=(1,0.5), loc='center left', fontsize=10)
            fname = self.generate_fname('','Supervised_Label_Transfer',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            self.data.obs['predicted_cell_type'] = np.array(TMG.Layers[0].Type)
            plt.close()

            fig, ax = plt.subplots(figsize=(10,12.5))
            # plt.title('Supervised Label Transfer')
            # plt.title('Supervised Label Transfer')
            for ct in np.unique(c):
                plt.scatter(0,0,50,label=ct)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.legend(ncol=5, loc='center', fontsize=10)
            fname = self.generate_fname('','Supervised_Label_Transfer_legend',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            self.data.obs['predicted_cell_type'] = np.array(TMG.Layers[0].Type)
            plt.close()

            x, y = self.data.obs['stage_x'],self.data.obs['stage_y']
            c = self.data.obs['leiden']
            df = pd.DataFrame(np.vstack([x,y,c]).T, columns=['x', 'y', 'c'])
            kwargs = dict(s=1, edgecolor='none')
            fig, ax = plt.subplots(figsize=(10,12.5))
            plt.title('Unupervised Clustering')
            for ct in np.unique(c):
                m = c==ct
                plt.scatter(x[m],y[m],s=0.05,marker='x',label=ct)
            # sns.scatterplot(data=df, x='x', y='y', hue='c', ax=ax, **kwargs)
            ax.set_aspect('equal')
            ax.axis('off')
            # ax.legend(ncol=5, bbox_to_anchor=(1,0.5), loc='center left', fontsize=10)
            fname = self.generate_fname('','Unupervised_Clustering',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            self.data.obs['predicted_cell_type'] = np.array(TMG.Layers[0].Type)
            plt.close()

            fig, ax = plt.subplots(figsize=(10,12.5))
            # plt.title('Supervised Label Transfer')
            # plt.title('Supervised Label Transfer')
            for ct in np.unique(c):
                plt.scatter(0,0,50,label=ct)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.legend(ncol=5, loc='center', fontsize=10)
            fname = self.generate_fname('','Unupervised_Clustering_legend',model_type,dtype='png')
            figure_list.append(fname)
            fig.savefig(fname) #SAVE
            plt.close()

            fname = self.generate_fname('',model_type,'qc_data',dtype='h5ad')
            self.data.X = self.data.layers['raw'].copy()
            self.data.write(fname)
            
            image1 = Image.open(figure_list[0])
            im1 = image1.convert('RGB')
            imagelist = [Image.open(fname).convert('RGB') for fname in figure_list if not fname==figure_list[0]]
            im1.save(self.generate_fname('','QC',model_type,dtype='pdf'),save_all=True, append_images=imagelist)