import os
import shutil
from dredFISH.Analysis.TissueGraph import *
from dredFISH.Analysis.Classification import *
from dredFISH.Visualization.Viz import *
from dredFISH.Registration.Registration import *
from dredFISH.Utils import fileu, basicu, ccfu


def analyze_mouse_brain_data(animal,
                            project_path='/scratchdata1/Images2024/Zach/MouseBrainAtlas',
                            analysis_path='/scratchdata1/MouseBrainAtlases_V0',
                            verbose=False):
    """ 
    Analyze mouse brain data and perform various analysis steps.
    
    Parameters:
    - animal: str, the name of the animal to analyze
    - project_path: str, the path to the project directory (default: '/scratchdata1/Images2024/Zach/MouseBrainAtlas')
    - analysis_path: str, the path to the analysis directory (default: '/scratchdata1/MouseBrainAtlases_V0')
    - verbose: bool, whether to print verbose output (default: False)
    """
        
    # Create analysis directory for the animal
    input_df = fileu.create_input_df(project_path, animal)
    basepath = os.path.join(analysis_path, animal)
    
    # Create TissueMultiGraph object
    TMG = TissueMultiGraph(basepath=basepath, input_df=input_df, redo=True)
    # Check if the animal is already analyzed, if so overwrite it
    if os.path.exists(basepath):
        if verbose:
            TMG.update_user(f"Animal {animal} already analyzed, overwriting")
        shutil.rmtree(basepath)

    fileu.update_user(f"Creating TMG for {animal}", verbose=verbose)
    TMG.create_cell_layer()
    TMG.save()

    TMG.update_user(f"Creating Classification Space", verbose=verbose)
    # Add classification space to the first layer of TMG
    TMG.Layers[0].adata.layers['classification_space'] = basicu.robust_zscore(np.array(TMG.Layers[0].adata.layers['normalized']).copy())
    TMG.save()

    TMG.update_user(f"Unsupervised Clustering", verbose=verbose)
    # Perform unsupervised clustering using graphLeiden algorithm
    TMG.Layers[0].adata = graphLeiden(TMG.Layers[0].adata, verbose=verbose).classify(resolution=5)
    TMG.save()

    TMG.update_user(f"Filtering Using Unsupervised Clustering", verbose=verbose)
    # Filter cells using unsupervised clustering labels
    TMG.Layers[0].adata = filterUsingUnsupervised(TMG.Layers[0].adata, label='leiden', verbose=verbose)
    TMG.save()

    TMG.update_user(f"Classify as Neuron or Non Neuron", verbose=verbose)
    # Classify cells as neuron or non-neuron
    TMG.Layers[0].adata = NeuronClassifier(verbose=verbose).classify(TMG.Layers[0].adata)
    TMG.save()

    TMG.update_user(f"Classify Neuron and Non Neuron Separately", verbose=verbose)
    # Split classification into separate categories
    TMG.Layers[0].adata = splitClassification(TMG.Layers[0].adata, ref_levels=['class', 'subclass', 'supertype'], weighted=False, verbose=verbose)
    TMG.save()

    cell_tax_to_build = ['class', 'subclass', 'supertype']
    parcel_tax_to_build = ['parcellation_division', 'parcellation_structure', 'parcellation_substructure', 'parcellation_index']
    taxs_to_build = cell_tax_to_build + parcel_tax_to_build
    Taxonomies = dict()
    tax_basepath = os.path.join(analysis_path, "Taxonomies")
    
    # Load taxonomies
    TMG.update_user(f"Loading Taxonomies", verbose=verbose)
    for tx in taxs_to_build:
        Taxonomies[tx] = Taxonomy(tx, basepath=tax_basepath)
        Taxonomies[tx].load()

    TMG.update_user(f"Building Spatial Graph", verbose=verbose)
    TMG.Layers[0].build_spatial_graph(save_knn=True)
    TMG.save()

    # Add type information to cells based on cell taxonomies
    for tx in cell_tax_to_build:
        TMG.add_type_information(0, TMG.Layers[0].adata.obs[tx], Taxonomies[tx])
    TMG.update_user(f"Creating Merged Layer for Cell Labels", verbose=verbose)
    ccf_df, parcellation_label_type_df = ccfu.retrieve_CCF_info(TMG.Layers[0].XYZ, TMG.Layers[0].Section)

    # Create merged layer for parcellation labels
    TMG.update_user(f"Creating Merged Layer for Parcellation Labels", verbose=verbose)
    TMG.create_merged_layer(replace=True, layer_type="parcellation_label", Labels=ccf_df["parcellation_label"])
    parcel_layer_id = TMG.find_layer_by_name("parcellation_label")

    # Add type information to parcellation labels based on parcel taxonomies
    TMG.update_user(f"Adding Type Information to Parcellation Labels", verbose=verbose)
    for tx in parcel_tax_to_build:
        TMG.add_type_information(parcel_layer_id, parcellation_label_type_df[tx], Taxonomies[tx])

    # Create higher level parcellations
    TMG.update_user(f"Creating Merged Layer for Higher Level Parcellations", verbose=verbose)
    for tx in parcel_tax_to_build:
        TMG.create_merged_layer(base_layer_id=parcel_layer_id, layer_type=tx, tax_name=tx)

    for tx in cell_tax_to_build:
        TMG.create_merged_layer(layer_type=tx, tax_name=tx)

    geoms_to_make = ["parcellation_label", "parcellation_division", 'parcellation_structure', 'parcellation_substructure', 'parcellation_index']
    all_layers = [l.layer_type for l in TMG.Layers]
    
    # Add and save merged geometries
    TMG.update_user(f"Adding and Saving Merged Geometries", verbose=verbose)
    for lyr in geoms_to_make:
        print(lyr)
        TMG.add_and_save_merged_geoms(all_layers.index(lyr))

    TMG.save()

    # Generate Figures For Inspection ToDo


    TMG.update_user("Completed", verbose=verbose)

