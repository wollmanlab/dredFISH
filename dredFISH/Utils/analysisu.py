import os
import shutil
from dredFISH.Analysis.TissueGraph import *
from dredFISH.Analysis.Classification import *
from dredFISH.Visualization.Viz import *
from dredFISH.Registration.Registration import *
from dredFISH.Utils import fileu, basicu, ccfu
import gc

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
        
    bad_bits = ['RS0109_cy5','RSN9927.0_cy5','RS0468_cy5','RS643.0_cy5','RS156.0_cy5','RS0237_cy5']

    if animal == 'Tax':
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

    TMG.update_user(f"Unsupervised Clustering", verbose=True)
    # Perform unsupervised clustering using graphLeiden algorithm
    TMG.Layers[0].adata.obs = graphLeiden(TMG.Layers[0].adata, verbose=verbose).classify(resolution=5).obs
    print(TMG.Layers[0].adata)
    TMG.save()
    gc.collect()

    TMG.update_user(f"Filtering Using Unsupervised Clustering", verbose=True)
    # Filter cells using unsupervised clustering labels
    TMG.Layers[0].adata = filterUsingUnsupervised(TMG.Layers[0].adata, label='leiden', verbose=verbose)
    print(TMG.Layers[0].adata)
    TMG.save()
    gc.collect()

    TMG.update_user(f"Classify as Neuron or Non Neuron", verbose=True)
    # Classify cells as neuron or non-neuron
    TMG.Layers[0].adata.obs = NeuronClassifier(verbose=verbose,bad_bits=bad_bits).classify(TMG.Layers[0].adata).obs
    print(TMG.Layers[0].adata)
    TMG.save()
    gc.collect()

    TMG.update_user(f"Classify Neuron and Non Neuron Separately", verbose=True)
    # Split classification into separate categories
    TMG.Layers[0].adata.obs = splitClassification(TMG.Layers[0].adata, ref_levels=['class', 'subclass', 'supertype'], weighted=False, verbose=verbose).obs
    print(TMG.Layers[0].adata)
    TMG.save()
    gc.collect()

    tax_basepath = os.path.join(analysis_path, "Taxonomies")

    """ Check if taxonomies have been made and if not make them """
    if not os.path.exists(tax_basepath):
        TMG.update_user(f"Creating  Taxonomies", verbose=True)
        create_taxonomies(tax_basepath,bad_bits=bad_bits)

    cell_tax_to_build = ['class', 'subclass', 'supertype']
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

    # Create higher level parcellations
    TMG.update_user(f"Creating Merged Layer for Higher Level Parcellations", verbose=True)
    for tx in parcel_tax_to_build:
        TMG.create_merged_layer(base_layer_id=parcel_layer_id, layer_type=tx, tax_name=tx)

    for tx in cell_tax_to_build:
        TMG.create_merged_layer(layer_type=tx, tax_name=tx)

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

