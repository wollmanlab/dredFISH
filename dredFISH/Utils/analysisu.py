import os
import shutil
from dredFISH.Analysis.TissueGraph import *
from dredFISH.Analysis.Classification import *
from dredFISH.Visualization.Viz import *
from dredFISH.Registration.Registration import *
from dredFISH.Utils import fileu, basicu, ccfu
import gc
import traceback

def create_taxonomies(tax_basepath,bad_bits=[]):
    cell_tax_to_build = ['class', 'subclass', 'supertype']
    parcel_tax_to_build = ['parcellation_division', 'parcellation_structure', 'parcellation_substructure']#, 'parcellation_index']
    taxs_to_build = cell_tax_to_build + parcel_tax_to_build

    tax_mapping = {'cluster' : 'supertype', 
                'supertype' : 'subclass',
                'subclass' : 'class'}

    feature_mats = dict()
    types_dict = dict()
    upstream_type_vec = dict()
    types_color_dict = dict()
    types_std = dict()

    # Now create the feature_matrix of these taxonomies
    seq_adata = anndata.read_h5ad(pathu.get_path('allen_wmb_tree'))
    seq_adata = seq_adata[:,np.isin(seq_adata.var.index, bad_bits, invert=True)].copy()

    for tx in cell_tax_to_build:
        tx_clr = f"{tx}_color" 
        feature_mat_by_type = seq_adata.obs.groupby(tx).apply(lambda x: seq_adata[x.index].X.mean(axis=0))
        feature_mat_by_type.sort_index(inplace=True)
        feature_std_by_type = seq_adata.obs.groupby(tx).apply(lambda x: seq_adata[x.index].X.std(axis=0))
        feature_std_by_type.sort_index(inplace=True)
        types_std[tx] = np.vstack(np.array(feature_std_by_type))
        feature_mats[tx] = np.vstack(np.array(feature_mat_by_type))
        types_dict[tx] = feature_mat_by_type.index
        types_color_dict[tx] = [seq_adata.obs[seq_adata.obs[tx] == cls][tx_clr].iloc[0] for cls in types_dict[tx]]
        if tx in tax_mapping.keys(): 
            type_vec = seq_adata.obs[tx]
            upstream_type_vec[tx] = seq_adata.obs[tax_mapping[tx]]
            # Create a mapping between unique types and their corresponding upstream types
            type_to_upstream_type_map = dict(zip(type_vec, upstream_type_vec[tx]))
            # convert
            upstream_type_vec[tx] = [type_to_upstream_type_map[typ] for typ in types_dict[tx]]

    # Extract unique classes and their corresponding colors
    for tx in cell_tax_to_build:
        if tx in tax_mapping.keys():
            Tx = Taxonomy(name=tx, 
                        basepath = tax_basepath,
                        Types=types_dict[tx], 
                        feature_mat=feature_mats[tx],
                        rgb_codes=types_color_dict[tx],
                        upstream_tax = tax_mapping[tx],
                        upstream_types = upstream_type_vec[tx])
        else:   
            Tx = Taxonomy(name=tx, 
                        basepath = tax_basepath,
                        Types=types_dict[tx], 
                        feature_mat=feature_mats[tx],
                        rgb_codes=types_color_dict[tx])
        Tx.adata.layers['std']=types_std[tx]
        Tx.save()

    # create taxonomies based on onthologies for index level and the three Allen levels 
    onto = ccfu.get_ccf_term_onthology()
    for tx in parcel_tax_to_build:
        ix = onto[tx].unique()
        types_dict[tx] = onto.loc[ix,"acronym"].values
        clrs = onto.loc[ix,"hex_org"].values
        clrs = [clr.zfill(6) for clr in clrs]
        types_color_dict[tx] = clrs
        feature_mats[tx] = np.zeros((len(ix),1))
        types_std[tx] = np.zeros((len(ix),1))

    # add parcellation index taxonomy
    parcel_tax_to_build = parcel_tax_to_build + ["parcellation_index"]
    onto = ccfu.get_ccf_term_onthology(original=True)
    tx = "parcellation_index"
    ix = onto.index
    types_dict[tx] = onto.loc[ix,"acronym"].values
    clrs = onto.loc[ix,"hex_org"].values
    clrs = [clr.zfill(6) for clr in clrs]
    types_color_dict[tx] = clrs
    feature_mats[tx] = np.zeros((len(ix),1))
    types_std[tx] = np.zeros((len(ix),1))

    metadata = pd.read_csv('/orangedata/ExternalData/Allen_WMB_2024Mar06/metadata/MERFISH-C57BL6J-638850-CCF/20230830/views/cell_metadata_with_parcellation_annotation.csv')

    # Update the colors in types_color_dict for 'parcellation_division' using the mapping from parcellation_division_to_color
    parcellation_levels = ['parcellation_division', 'parcellation_structure', 'parcellation_substructure']
    for level in parcellation_levels:
        parcellation_to_color = dict(zip(metadata[level], metadata[level + '_color']))
        updated_colors = [parcellation_to_color[item] if item in parcellation_to_color else color 
                        for item, color in zip(types_dict[level], types_color_dict[level])]
        types_color_dict[level] = updated_colors


    # Extract unique classes and their corresponding colors
    for tx in parcel_tax_to_build:
        if tx in tax_mapping.keys():
            Tx = Taxonomy(name=tx, 
                        basepath = tax_basepath,
                        Types=types_dict[tx], 
                        feature_mat=feature_mats[tx],
                        rgb_codes=types_color_dict[tx],
                        upstream_tax = tax_mapping[tx])
        else:   
            Tx = Taxonomy(name=tx, 
                        basepath = tax_basepath,
                        Types=types_dict[tx], 
                        feature_mat=feature_mats[tx],
                        rgb_codes=types_color_dict[tx])
        Tx.adata.layers['std']=types_std[tx]
        Tx.save()


def analyze_mouse_brain_data(animal,
                            project_path='/scratchdata1/Images2024/Zach/MouseBrainAtlas',
                            analysis_path='/scratchdata1/MouseBrainAtlases_V1',
                            verbose=False):
    """ 
    Analyze mouse brain data and perform various analysis steps.
    
    Parameters:
    - animal: str, the name of the animal to analyze
    - project_path: str, the path to the project directory (default: '/scratchdata1/Images2024/Zach/MouseBrainAtlas')
    - analysis_path: str, the path to the analysis directory (default: '/scratchdata1/MouseBrainAtlases_V1')
    - verbose: bool, whether to print verbose output (default: False)
    """
    try:
        bad_bits = ['RS0109_cy5','RSN9927.0_cy5','RS0468_cy5','RS643.0_cy5','RS156.0_cy5','RS0237_cy5']

        if animal == 'Tax':
            tax_path = os.path.join(analysis_path, "Taxonomies")
            if not os.path.exists(tax_path):
                create_taxonomies(os.path.join(analysis_path, "Taxonomies"),bad_bits=bad_bits)
            return

        # Create analysis directory for the animal
        input_df = fileu.create_input_df(project_path, animal)
        basepath = os.path.join(analysis_path, animal)

        # Check if the animal is already analyzed, if so overwrite it
        if os.path.exists(basepath):
            if verbose:
                fileu.update_user(f"Animal {animal} already analyzed, overwriting",verbose=True)
            shutil.rmtree(basepath)
        else:
            os.mkdir(basepath, mode=0o777)
        
        # Create TissueMultiGraph object
        TMG = TissueMultiGraph(basepath=basepath, input_df=input_df, redo=True)

        fileu.update_user(f"Creating TMG for {animal}", verbose=True)
        TMG.create_cell_layer(bad_bits = bad_bits)
        print(TMG.Layers[0].adata)
        TMG.save()

        figure_path = os.path.join(basepath,'Figures')
        if not os.path.exists(figure_path): 
            os.mkdir(figure_path,mode=0o777)

        adata = TMG.Layers[0].adata.copy()
        for var in adata.var.index:
            c = np.array(adata.layers['classification_space'][:,np.isin(adata.var.index,[var])])
            vmin,vmax = np.percentile(c,[5,95])
            bit = f"Measurement : {var} vmin: {round(vmin,2)} vmax: {round(vmax,2)}"
            n_columns = np.min([6,adata.obs['Slice'].unique().shape[0]])
            n_rows = math.ceil(adata.obs['Slice'].unique().shape[0]/n_columns)
            fig,axs = plt.subplots(n_rows,n_columns,figsize=[n_columns*3,n_rows*3],dpi=300)
            fig.patch.set_facecolor((1, 1, 1, 0))
            fig.suptitle(f"{animal} {bit}", color='black')
            if adata.obs['Slice'].unique().shape[0]==1:
                axs = [axs]
            else:
                axs = axs.ravel()
            for ax in axs:
                ax.axis('off')
            for i,section in tqdm(enumerate(sorted(adata.obs['Slice'].unique())),desc=f"{var} Visualization"):
                m = (adata.obs['Slice']==section)
                temp_data = adata[m,:].copy()
                c = np.array(temp_data.layers['classification_space'][:,np.isin(adata.var.index,[var])])
                ax = axs[i]
                ax.set_title(section, color='black')
                ax.axis('off')
                ax.axis('off')
                im = ax.scatter(temp_data.obs['ccf_z'],temp_data.obs['ccf_y'],c=c,s=0.5,marker=',', edgecolors='none', linewidths=0,cmap='jet',vmin=vmin,vmax=vmax)
            plt.savefig(os.path.join(figure_path,f"{animal} {bit.split(var)[0]} {var}.png"))
            plt.close()
        fnames = []
        out_fname = os.path.join(figure_path,f"{animal} Measurement.gif")
        if os.path.exists(out_fname):
            os.remove(out_fname)
        # Create a GIF writer object
        with imageio.get_writer(out_fname, fps=1) as writer:
            for var in tqdm(adata.var.index):
                bit = f"Measurement : {var} "
                fname = os.path.join(figure_path,f"{animal} {bit.split(var)[0]} {var}.png")
                image = imageio.imread(fname)
                writer.append_data(image)

        seq_adata = anndata.read_h5ad(pathu.get_path('allen_wmb_tree'))
        pallette = dict(zip(seq_adata.obs[level],seq_adata.obs[f"{level}_color"]))
        del seq_adata

        TMG.update_user(f"Harmonizing to Reference and Classifying", verbose=True)
        X = np.zeros_like(TMG.Layers[0].adata.X)
        y = np.array(["Unknown"] * TMG.Layers[0].adata.shape[0], dtype=object)
        for section in TMG.unqS:
            print(section)
            m = TMG.Layers[0].adata.obs['Slice'] == section
            if np.sum(m) == 0:
                continue
            idx = np.where(m)[0]
            adata = TMG.Layers[0].adata[idx]
            scale = SingleCellAlignmentLeveragingExpectations(adata, 
                    tax_name='iterative_spatial_assisted_label_transfer',
                    ref='allen_wmb_tree', 
                    spatial_ref='spatial_reference',
                    ref_level='subclass', 
                    model='logreg',
                    out_path='',
                    batch_name='section_index',
                    save_fig=False,
                    neuron=None,
                    weighted=False,
                    verbose=False,
                    kde_kernel = (0.25,0.1,0.1),
                    binary_spatial_thresh=None,
                    use_prior=True)
            # scale.reference = reference_adata.copy()
            scale.train()
            features,labels = scale.classify()
            X[idx] = features
            y[idx] = labels
        TMG.Layers[0].adata.layers['harmonized'] = X
        TMG.Layers[0].adata.obs['subclass'] = y
        TMG.Layers[0].adata.uns['subclass_colors'] = TMG.Layers[0].adata.obs['subclass'].map(pallette)
        print(TMG.Layers[0].adata)
        TMG.save()
        gc.collect()

        level = 'subclass'
        adata = TMG.Layers[0].adata#.copy()
        bit = f"Supervised : {level}"
        n_columns = np.min([6,adata.obs['Slice'].unique().shape[0]])
        n_rows = math.ceil(adata.obs['Slice'].unique().shape[0]/n_columns)
        fig,axs = plt.subplots(n_rows,n_columns,figsize=[n_columns*5,n_rows*5],dpi=300)
        fig.patch.set_facecolor((1, 1, 1, 0))
        fig.suptitle(f"{animal} {bit}", color='black')
        if adata.obs['Slice'].unique().shape[0]==1:
            axs = [axs]
        else:
            axs = axs.ravel()
        for ax in axs:
            ax.axis('off')
        for i,section in tqdm(enumerate(sorted(adata.obs['Slice'].unique())),desc=f"{level} Visualization"):
            m = (adata.obs['Slice']==section)
            temp_data = adata[m,:].copy()
            c = np.array(temp_data.obs[level+'_color'])
            ax = axs[i]
            ax.set_title(section, color='black')
            ax.axis('off')
            ax.axis('off')
            im = ax.scatter(temp_data.obs['ccf_z'],temp_data.obs['ccf_y'],c=c,s=0.5,marker=',', edgecolors='none', linewidths=0)
        plt.savefig(os.path.join(figure_path,f"{animal} {bit.split(level)[0]} {level}.png"))
        plt.close()

        adata = TMG.Layers[0].adata.copy()
        for var in adata.var.index:
            c = np.array(adata.layers['harmonized'][:,np.isin(adata.var.index,[var])])
            vmin,vmax = np.percentile(c,[5,95])
            bit = f"Harmonized : {var} vmin: {round(vmin,2)} vmax: {round(vmax,2)}"
            n_columns = np.min([6,adata.obs['Slice'].unique().shape[0]])
            n_rows = math.ceil(adata.obs['Slice'].unique().shape[0]/n_columns)
            fig,axs = plt.subplots(n_rows,n_columns,figsize=[n_columns*3,n_rows*3],dpi=300)
            fig.patch.set_facecolor((1, 1, 1, 0))
            fig.suptitle(f"{animal} {bit}", color='black')
            if adata.obs['Slice'].unique().shape[0]==1:
                axs = [axs]
            else:
                axs = axs.ravel()
            for ax in axs:
                ax.axis('off')
            for i,section in tqdm(enumerate(sorted(adata.obs['Slice'].unique())),desc=f"{var} Visualization"):
                m = (adata.obs['Slice']==section)
                temp_data = adata[m,:].copy()
                c = np.array(temp_data.layers['harmonized'][:,np.isin(adata.var.index,[var])])
                ax = axs[i]
                ax.set_title(section, color='black')
                ax.axis('off')
                ax.axis('off')
                im = ax.scatter(temp_data.obs['ccf_z'],temp_data.obs['ccf_y'],c=c,s=0.5,marker=',', edgecolors='none', linewidths=0,cmap='jet',vmin=vmin,vmax=vmax)
            plt.savefig(os.path.join(figure_path,f"{animal} {bit.split(var)[0]} {var}.png"))
            plt.close()
        fnames = []
        out_fname = os.path.join(figure_path,f"{animal} Harmonized.gif")
        if os.path.exists(out_fname):
            os.remove(out_fname)
        # Create a GIF writer object
        with imageio.get_writer(out_fname, fps=1) as writer:
            for var in tqdm(adata.var.index):
                bit = f"Harmonized : {var} "
                fname = os.path.join(figure_path,f"{animal} {bit.split(var)[0]} {var}.png")
                image = imageio.imread(fname)
                writer.append_data(image)

        adata = TMG.Layers[0].adata.copy()
        bit = f"Supervised 3D : {level}"
        fig = plt.figure(figsize=[10,10],dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')
        ax.set_facecolor((1, 1, 1, 0))
        plt.title(f"{animal} {bit}", color='black')
        adata.obs['ccf_x'] = -10*adata.obs['ccf_x']
        adata.obs['ccf_y'] = -1*adata.obs['ccf_y']
        adata.obs['ccf_z'] = -1*adata.obs['ccf_z']
        for i,section in tqdm(enumerate(TMG.unqS),desc=f"{level} Visualization"):
            m = (adata.obs['Slice']==section)
            temp_data = adata[m,:].copy()
            c = np.array(temp_data.obs[level+'_color'])
            # ax.scatter(temp_data.obs['ccf_z'],temp_data.obs['ccf_x'],temp_data.obs['ccf_y'],c='k',s=0.1,marker=',')
            ax.scatter(temp_data.obs['ccf_z'],temp_data.obs['ccf_x'],temp_data.obs['ccf_y'],c=c,s=0.5,marker=',', edgecolors='none', linewidths=0)
        x = adata.obs['ccf_z']
        y = adata.obs['ccf_x']
        z = adata.obs['ccf_y']

        zoom_factor = 0.6
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        max_range = max_range*zoom_factor
        x_offset_scale = 1
        y_offset_scale = 1
        z_offset_scale = 1
        ax.set_xlim(x.mean() - x_offset_scale*max_range, x.mean() + x_offset_scale*max_range)
        ax.set_ylim(y.mean() - y_offset_scale*max_range, y.mean() + y_offset_scale*max_range)
        ax.set_zlim(z.mean() - z_offset_scale*max_range, z.mean() + z_offset_scale*max_range)

        # Set view angle
        ax.view_init(elev=25, azim=45)  # Adjust these values to change the view angle
        plt.tight_layout()
        plt.savefig(os.path.join(figure_path,f"{animal} {bit.split(level)[0]} {level}.png"))
        plt.close()

        tax_basepath = os.path.join(analysis_path, "Taxonomies")

        """ Check if taxonomies have been made and if not make them """
        if not os.path.exists(tax_basepath):
            TMG.update_user(f"Creating  Taxonomies", verbose=True)
            create_taxonomies(tax_basepath,bad_bits=bad_bits)

        cell_tax_to_build = ['subclass']#['class', 'subclass', 'supertype']
        parcel_tax_to_build = ['parcellation_division', 'parcellation_structure', 'parcellation_substructure', 'parcellation_index']
        taxs_to_build = cell_tax_to_build + parcel_tax_to_build
        Taxonomies = dict()

        # Load taxonomies
        TMG.update_user(f"Loading Taxonomies", verbose=True)
        for tx in taxs_to_build:
            Taxonomies[tx] = Taxonomy(tx, basepath=tax_basepath)
            Taxonomies[tx].load()

        TMG.update_user(f"Building Spatial Graph", verbose=True)
        TMG.Layers[0].build_spatial_graph(save_knn=True)
        TMG.save()

        # Add type information to cells based on cell taxonomies
        for tx in cell_tax_to_build:
            TMG.add_type_information(0, TMG.Layers[0].adata.obs[tx], Taxonomies[tx])
        TMG.update_user(f"Creating Merged Layer for Cell Labels", verbose=True)
        ccf_df, parcellation_label_type_df = ccfu.retrieve_CCF_info(TMG.Layers[0].XYZ, TMG.Layers[0].Section)

        # Create merged layer for parcellation labels
        TMG.update_user(f"Creating Merged Layer for Parcellation Labels", verbose=True)
        TMG.create_merged_layer(replace=True, layer_type="parcellation_label", Labels=ccf_df["parcellation_label"])
        parcel_layer_id = TMG.find_layer_by_name("parcellation_label")

        # Add type information to parcellation labels based on parcel taxonomies
        TMG.update_user(f"Adding Type Information to Parcellation Labels", verbose=True)
        for tx in parcel_tax_to_build:
            TMG.add_type_information(parcel_layer_id, parcellation_label_type_df[tx], Taxonomies[tx])
        TMG.save()

        # Create higher level parcellations
        TMG.update_user(f"Creating Merged Layer for Higher Level Parcellations", verbose=True)
        for tx in parcel_tax_to_build:
            TMG.create_merged_layer(base_layer_id=parcel_layer_id, layer_type=tx, tax_name=tx)

        for tx in cell_tax_to_build:
            TMG.create_merged_layer(layer_type=tx, tax_name=tx)
        TMG.save()


        geoms_to_make = ["parcellation_label", "parcellation_division", 'parcellation_structure', 'parcellation_substructure', 'parcellation_index']
        all_layers = [l.layer_type for l in TMG.Layers]
        
        # Add and save merged geometries
        TMG.update_user(f"Adding and Saving Merged Geometries", verbose=True)
        for lyr in geoms_to_make:
            print(lyr)
            TMG.add_and_save_merged_geoms(all_layers.index(lyr))
        TMG.save()

        # Generate Figures For Inspection ToDo


        TMG.update_user("Completed", verbose=verbose)
    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        print(animal)
        print('Failed')
        TMG.update_user("Failed", verbose=verbose)
        TMG.update_user(str(e), verbose=verbose)
        TMG.update_user(str(error_message), verbose=verbose)

