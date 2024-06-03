
import os
import numpy as np
import pandas as pd
import nrrd
from scipy.ndimage import label,grey_dilation



# the gloval path where CCF reference is saved. 
ccf_path = "/scratchdata1/MouseBrainAtlases/Taxonomies/CCF_Ref"

def create_annotation_from_cells():
    files = ["/orangedata/ExternalData/Allen_WMB_2024Mar06/metadata/Zhuang-ABCA-1-CCF/20230830/ccf_coordinates.csv",
         "/orangedata/ExternalData/Allen_WMB_2024Mar06/metadata/Zhuang-ABCA-2-CCF/20230830/ccf_coordinates.csv",
         "/orangedata/ExternalData/Allen_WMB_2024Mar06/metadata/Zhuang-ABCA-3-CCF/20230830/ccf_coordinates.csv",
         "/orangedata/ExternalData/Allen_WMB_2024Mar06/metadata/Zhuang-ABCA-4-CCF/20230830/ccf_coordinates.csv"]

    # Read and concatenate dataframes from the list of files
    dataframes = [pd.read_csv(file) for file in files]
    ccf_df = pd.concat(dataframes, ignore_index=True)

    Xref = np.array(ccf_df['z'])
    Yref = np.array(ccf_df["y"])
    Zref = np.array(ccf_df["x"]) 

    Xref_int = (Xref * 1000 / 25).astype(int)
    Yref_int = (Yref * 1000 / 25).astype(int)
    Zref_int = (Zref * 1000 / 25).astype(int)

    unq_parcel_index = ccf_df['parcellation_index'].unique()
    ccf_parcel_index = np.zeros((528, 320, 456))

    for id in unq_parcel_index:
        ix = np.flatnonzero(ccf_df['parcellation_index']==id)
        ccf_parcel_index[Zref_int[ix],Yref_int[ix],Xref_int[ix]]=id

    ccf_dense,header = nrrd.read(os.path.join(ccf_path,"CCFv3_annotation_25.nrrd"))

    crosstab_result = pd.crosstab(ccf_dense[ccf_parcel_index>0], ccf_parcel_index[ccf_parcel_index>0])

    # Calculate the percentage of ccf_parcel_index pixels mapped to ccf_dense
    crosstab_result_arr = np.array(crosstab_result)
    percentage_mapping = crosstab_result_arr / crosstab_result_arr.sum(axis=0, keepdims=True)

    # Find all ccf_dense values where the mapping percentage is >= 0.8
    dense_mapped = percentage_mapping >= 0.8
    mapped_indices = np.where(dense_mapped)

    dense_mapped_values = np.array(crosstab_result.index[mapped_indices[0]])
    parcel_mapped_values = np.array(crosstab_result.columns[mapped_indices[0]])

    # Find all unique ccf_dense values
    all_dense_values = np.unique(ccf_dense[ccf_dense>0])

    # Determine which ccf_dense values are not mapped
    dense_values_not_mapped = np.setdiff1d(all_dense_values, dense_mapped_values)
    dense_values_not_mapped = np.setdiff1d(dense_values_not_mapped,[0])

    # Create a mapping dictionary between values of ccf_dense and ccf_parcel_index for specific values
    mapping_dict = dict(zip(dense_mapped_values, parcel_mapped_values))

    # Create a new 3D array `ccf_labeled` with the same shape as `ccf_dense`
    ccf_labeled = np.zeros_like(ccf_dense)

    # Iterate over each value in `dense_values_mapped`
    for dense_value,parcel_index_value in mapping_dict.items():
        ccf_labeled[ccf_dense == dense_value] = parcel_index_value
    
    np.save(os.path.join(ccf_path,'swapped_annotation.npy'),ccf_labeled)
    np.save(os.path.join(ccf_path,'annotation.npy'),ccf_parcel_index)


def get_annotation_matrix():
    file_name = os.path.join(ccf_path,'swapped_annotation.npy')

    # Read the NRRD file
    annotation = np.load(file_name)
    return annotation


def get_pivot_df():
    pivot_df = pd.read_csv(os.path.join(ccf_path,"parcellation_to_parcellation_term_membership_acronym.csv"))
    # Drop the 'organ' and 'category' columns from pivot_df
    pivot_df.drop(columns=['organ', 'category'], inplace=True)

    # Rename columns 'division', 'structure', and 'substructure' to 'parcellation_division', 'parcellation_structure', and 'parcellation_substructure'
    pivot_df.rename(columns={'division': 'parcellation_division', 'structure': 'parcellation_structure', 'substructure': 'parcellation_substructure'}, inplace=True)

    return pivot_df


def create_neighbor_binary_image(image):
    z, y, x = image.shape
    binary_image = np.zeros_like(image, dtype=bool)
    
    # Define shifts for 26-connectivity in 3D
    shifts = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dz == dy == dx == 0)]

    for dz, dy, dx in shifts:
        shifted_image = np.roll(image, shift=(dz, dy, dx), axis=(0, 1, 2))
        mask = (image > 0) & (shifted_image > 0)
        different_label_mask = mask & (image != shifted_image)
        binary_image |= different_label_mask

    return binary_image

def fill_borders_vectorized(labeled_components, border_mask):
    # Create a mask of the non-border areas (where labeled_components is non-zero)
    non_border_mask = labeled_components > 0

    # Use grey_dilation to propagate the maximum label value into the zero-valued border areas
    # Only apply dilation where the border_mask is True
    dilation_structure = np.ones((3, 3, 3))  # 3D dilation structure for 26-connectivity
    filled_borders = grey_dilation(labeled_components, footprint=dilation_structure, mode='nearest')

    # Now, combine the original labeled components with the filled borders
    # Only update the border areas
    filled_image = labeled_components.copy()
    filled_image[border_mask] = filled_borders[border_mask]

    return filled_image

def create_and_save_labeled_annotation():
    annotation = get_annotation_matrix()
    annotation_borders = create_neighbor_binary_image(annotation)
    annotation_without_borders = annotation.copy()
    annotation_without_borders[annotation_borders]=0

    structure = np.ones((3, 3, 3), dtype=int)
    (labeled_components,n_lbls) = label(annotation_without_borders,structure=structure)

    labeled_components = fill_borders_vectorized(labeled_components,annotation_borders)
    np.save(os.path.join(ccf_path,'ccf_parcel_index.npy'),annotation)
    np.save(os.path.join(ccf_path,'ccf_parcel_label.npy'),labeled_components)

    return (annotation,labeled_components)


def retrieve_CCF_info(XYZ):
    """
    Data driven assignment of CCF types (division, structure, substructure)
    additionaly output includes lowest level index + unique label per connected component region

    function loads data from drive for index, labels, and pivot table converted
    data was created with resolytion of 25um (one of Allen SDK options)
    """
    
    # load key data to use for assignments
    resolution=25
    index_file_path = os.path.join(ccf_path, 'ccf_parcel_index.npy')
    if not os.path.exists(index_file_path):
        ccf_parcel_index,ccf_parcel_label = create_and_save_labeled_annotation()
    else: 
        ccf_parcel_index = np.load(os.path.join(ccf_path, 'ccf_parcel_index.npy'))
        ccf_parcel_label = np.load(os.path.join(ccf_path, 'ccf_parcel_label.npy'))
        
    ccf_pivot_table = get_pivot_df()

    # create output dataframe
    out_df = pd.DataFrame()

    # Convert XY from millimeters to micrometers and then to indices
    X_indices = np.round(XYZ[:, 0] * 1000 / resolution).astype(int)
    Y_indices = np.round(XYZ[:, 1] * 1000 / resolution).astype(int)
    Z_indices = np.round(XYZ[:, 2] * 1000 / resolution).astype(int)

    # Ensure indices are within the bounds of the annotation array
    X_indices = np.clip(X_indices, 0, ccf_parcel_index.shape[2] - 1)
    Y_indices = np.clip(Y_indices, 0, ccf_parcel_index.shape[1] - 1)
    Z_indices = np.clip(Z_indices, 0, ccf_parcel_index.shape[0] - 1) 

    out_df["parcellation_index"] =  ccf_parcel_index[Z_indices, Y_indices, X_indices]
    out_df["parcellation_label"] =  ccf_parcel_label[Z_indices, Y_indices, X_indices]

    parcilation_levels = ["parcellation_division",
                          "parcellation_structure",
                          "parcellation_substructure"]
    
    for tx in parcilation_levels: 
        converter = dict(zip(ccf_pivot_table['parcellation_index'],ccf_pivot_table[tx]))
        out_df[tx] =  out_df['parcellation_index'].map(converter)

    return out_df





