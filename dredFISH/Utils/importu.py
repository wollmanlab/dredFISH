"""
Import utilities for converting spatial transcriptomics data to TissueMultiGraph format.

This module provides functions to load and convert various spatial transcriptomics 
data formats into the TissueMultiGraph format used by the dredFISH analysis pipeline.

Supported formats:
    - Xineum (10x Genomics spatial transcriptomics)
    - Additional formats can be added as needed

Functions:
    load_xenium_data: Convert Xineum data to TMG
    create_input_df_xenium: Create input dataframe for Xineum TMG construction
    load_gene_expression_matrix_xenium: Load Xineum gene expression data
    load_cell_metadata_xenium: Load Xineum cell metadata
    create_cell_boundaries_geom_xenium: Create geometry objects from Xineum boundaries
"""

import pandas as pd
import numpy as np
import os
import json
import scipy.io
import scipy.sparse
import anndata
from shapely.geometry import Polygon
from typing import Dict, Optional
import warnings

from dredFISH.Analysis.TissueGraph import TissueMultiGraph, Taxonomy, Geom, TissueGraph


def load_xenium_data(xenium_path: str, 
                    output_path: str,
                    min_transcripts: int = 50,
                    min_genes: int = 20,
                    use_clustering: str = 'gene_expression_graphclust',
                    create_geometries: bool = True,
                    redo: bool = False) -> TissueMultiGraph:
    """
    Load Xineum spatial transcriptomics data and convert to TissueMultiGraph.
    
    Parameters
    ----------
    xenium_path : str
        Path to the Xineum output directory
    output_path : str
        Path where TMG data will be saved
    min_transcripts : int, default 50
        Minimum transcript count per cell for filtering
    min_genes : int, default 20
        Minimum gene count per cell for filtering
    use_clustering : str, default 'gene_expression_graphclust'
        Which clustering result to use for initial cell typing
    create_geometries : bool, default True
        Whether to create cell boundary geometries
    redo : bool, default False
        Whether to recreate TMG if it already exists
        
    Returns
    -------
    TissueMultiGraph
        The loaded/created TMG object
    """
    
    print(f"Loading Xineum data from: {xenium_path}")
    
    # Load experiment metadata
    experiment_info = _load_experiment_metadata_xenium(xenium_path)
    print(f"Experiment: {experiment_info['run_name']}")
    print(f"Region: {experiment_info['region_name']}")
    print(f"Cells detected: {experiment_info['num_cells']}")
    
    # Create input dataframe for TMG
    input_df = create_input_df_xenium(xenium_path, experiment_info)
    
    # Initialize TMG
    tmg = TissueMultiGraph(basepath=output_path, 
                          input_df=input_df,
                          redo=redo)
    
    if not redo and len(tmg.Layers) > 0:
        print("TMG already exists and redo=False. Loading existing TMG.")
        return tmg
    
    # Load and process data
    print("Loading gene expression matrix...")
    adata = load_gene_expression_matrix_xenium(xenium_path)
    
    print("Loading cell metadata...")
    cell_metadata = load_cell_metadata_xenium(xenium_path)
    
    print("Loading clustering results...")
    clustering_data = _load_clustering_results_xenium(xenium_path, use_clustering)
    
    # Merge data and filter cells
    print("Merging and filtering data...")
    adata = _merge_and_filter_data_xenium(adata, cell_metadata, clustering_data, 
                                         min_transcripts, min_genes)
    
    print(f"After filtering: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Add section information (Xineum typically has one section per run)
    # Create section name compatible with TMG expectations (animal_X.Y format)
    # Use region name as animal and a simple numeric identifier
    section_name = f"{experiment_info['region_name']}_0.0"
    adata.obs['section_name'] = section_name
    adata.obs['Slice'] = section_name
    
    # Add spatial coordinates
    adata.obsm['XY'] = np.array(adata.obs[['x_centroid', 'y_centroid']])
    
    # Create TissueGraph from adata
    print("Creating cell layer...")
    _create_cell_layer_from_adata(tmg, adata, build_spatial_graph=True)
    
    # Add taxonomy and cell type information
    print("Adding cell type taxonomy...")
    _create_cell_taxonomy_xenium(tmg, clustering_data)
    
    # Create geometries if requested
    if create_geometries:
        print("Creating cell boundary geometries...")
        create_cell_boundaries_geom_xenium(tmg, xenium_path)
    
    # Save TMG
    print("Saving TMG...")
    tmg.save()
    
    print("TMG creation completed!")
    return tmg


def create_input_df_xenium(xenium_path: str, experiment_info: Dict) -> pd.DataFrame:
    """Create input dataframe for Xineum TMG construction."""
    input_df = pd.DataFrame({
        'animal': [experiment_info.get('slide_id', 'unknown')],
        'section_acq_name': [experiment_info['region_name']],
        'registration_path': [xenium_path],  # For compatibility
        'processing': ['xenium_output'],
        'dataset': [experiment_info['run_name']],
        'dataset_path': [os.path.dirname(xenium_path)]
    })
    return input_df


def load_gene_expression_matrix_xenium(xenium_path: str) -> anndata.AnnData:
    """
    Load gene expression matrix from Xineum output.
    
    Parameters
    ----------
    xenium_path : str
        Path to the Xineum output directory
        
    Returns
    -------
    anndata.AnnData
        AnnData object with gene expression matrix
    """
    # Load matrix
    matrix_path = os.path.join(xenium_path, 'cell_feature_matrix', 'matrix.mtx.gz')
    matrix = scipy.io.mmread(matrix_path).T.tocsr()  # Transpose to cells x genes
    
    # Load barcodes (cell IDs)
    barcodes_path = os.path.join(xenium_path, 'cell_feature_matrix', 'barcodes.tsv.gz')
    barcodes = pd.read_csv(barcodes_path, header=None, names=['cell_id'])
    
    # Load features (genes)
    features_path = os.path.join(xenium_path, 'cell_feature_matrix', 'features.tsv')
    features = pd.read_csv(features_path, sep='\t', header=None, 
                          names=['gene_id', 'gene_name', 'feature_type'])
    
    # Create AnnData object
    adata = anndata.AnnData(X=matrix, dtype=np.float32)
    adata.obs.index = barcodes['cell_id']
    adata.var.index = features['gene_name']
    adata.var['gene_id'] = features['gene_id'].values
    adata.var['feature_type'] = features['feature_type'].values
    
    # Store raw counts
    adata.layers['raw'] = adata.X.copy()
    
    return adata


def load_cell_metadata_xenium(xenium_path: str) -> pd.DataFrame:
    """
    Load cell metadata from Xineum cells.parquet file.
    
    Parameters
    ----------
    xenium_path : str
        Path to the Xineum output directory
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cell metadata including spatial coordinates and metrics
    """
    cells_path = os.path.join(xenium_path, 'cells.parquet')
    cell_data = pd.read_parquet(cells_path)
    
    # No need to rename columns - the data already has appropriate column names
    # The original data already has 'total_counts', 'cell_area', 'nucleus_area'
    
    return cell_data


def create_cell_boundaries_geom_xenium(tmg: TissueMultiGraph, xenium_path: str):
    """
    Create cell boundary geometries from Xineum boundary data.
    
    Parameters
    ----------
    tmg : TissueMultiGraph
        The TMG object to add geometries to
    xenium_path : str
        Path to the Xineum output directory
    """
    boundaries_path = os.path.join(xenium_path, 'cell_boundaries.parquet')
    
    if not os.path.exists(boundaries_path):
        print("Warning: Cell boundaries file not found, skipping geometry creation")
        return
    
    print("Loading cell boundaries...")
    boundaries = pd.read_parquet(boundaries_path)
    
    # Group by cell_id and create polygons
    cell_polygons = {}
    
    for cell_id, group in boundaries.groupby('cell_id'):
        # Get vertices for this cell
        vertices = group[['vertex_x', 'vertex_y']].values
        
        # Create polygon
        try:
            polygon = Polygon(vertices)
            if polygon.is_valid:
                cell_polygons[cell_id] = polygon
        except Exception as e:
            print(f"Warning: Could not create polygon for cell {cell_id}: {e}")
            continue
    
    print(f"Created {len(cell_polygons)} cell polygons")
    
    # Create Geom object for the section
    section_name = tmg.Layers[0].unqS[0]  # Get section name from cell layer
    
    # Convert dict to list in same order as cells in TMG
    cell_order = tmg.Layers[0].names  # Get cell order from TMG
    polygon_list = []
    
    for cell_id in cell_order:
        if cell_id in cell_polygons:
            polygon_list.append(cell_polygons[cell_id])
        else:
            # Create a small placeholder polygon for missing boundaries
            x, y = tmg.Layers[0].XY[tmg.Layers[0].names == cell_id][0]
            placeholder = Polygon([(x-1, y-1), (x+1, y-1), (x+1, y+1), (x-1, y+1)])
            polygon_list.append(placeholder)
    
    # Create and save geometry
    geom = Geom(geom_type='cell', polys=polygon_list, 
                basepath=tmg.basepath, section=section_name)
    geom.save()
    
    # Initialize Geoms list if not exists
    if not hasattr(tmg, 'Geoms') or tmg.Geoms is None:
        tmg.Geoms = [None] * len(tmg.unqS)
    
    # Add to TMG
    section_idx = tmg.unqS.index(section_name)
    if tmg.Geoms[section_idx] is None:
        tmg.Geoms[section_idx] = {}
    tmg.Geoms[section_idx]['cell'] = geom
    
    # Update mapping
    tmg.geom_to_layer_type_mapping['cell'] = 'cell'
    tmg.layer_to_geom_type_mapping['cell'] = 'cell'
    
    print("Cell boundary geometries created and saved")


# Private helper functions for Xineum data processing
def _load_experiment_metadata_xenium(xenium_path: str) -> Dict:
    """Load experiment metadata from experiment.xenium file."""
    metadata_file = os.path.join(xenium_path, 'experiment.xenium')
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata


def _load_clustering_results_xenium(xenium_path: str, clustering_method: str) -> pd.DataFrame:
    """Load clustering results for cell type assignment."""
    clustering_path = os.path.join(xenium_path, 'analysis', 'clustering', 
                                  clustering_method, 'clusters.csv')
    
    if os.path.exists(clustering_path):
        clustering_data = pd.read_csv(clustering_path)
        clustering_data = clustering_data.rename(columns={'Barcode': 'cell_id'})
        return clustering_data
    else:
        print(f"Warning: Clustering file not found at {clustering_path}")
        return pd.DataFrame()


def _merge_and_filter_data_xenium(adata: anndata.AnnData, 
                                 cell_metadata: pd.DataFrame,
                                 clustering_data: pd.DataFrame,
                                 min_transcripts: int,
                                 min_genes: int) -> anndata.AnnData:
    """Merge cell metadata and clustering data with expression matrix and apply filters."""
    # Merge cell metadata
    cell_metadata_indexed = cell_metadata.set_index('cell_id')
    
    # Find common cells
    common_cells = adata.obs.index.intersection(cell_metadata_indexed.index)
    print(f"Common cells between expression and metadata: {len(common_cells)}")
    
    # Subset to common cells
    adata = adata[common_cells].copy()
    cell_metadata_subset = cell_metadata_indexed.loc[common_cells]
    
    # Add metadata to adata.obs
    for col in cell_metadata_subset.columns:
        adata.obs[col] = cell_metadata_subset[col]
    
    # Add clustering data if available
    if not clustering_data.empty:
        clustering_indexed = clustering_data.set_index('cell_id')
        common_clustered_cells = adata.obs.index.intersection(clustering_indexed.index)
        if len(common_clustered_cells) > 0:
            adata.obs.loc[common_clustered_cells, 'Cluster'] = clustering_indexed.loc[common_clustered_cells, 'Cluster']
            adata.obs['Type'] = adata.obs['Cluster'].fillna(-1).astype(int)
        else:
            print("Warning: No overlap between expression data and clustering results")
            adata.obs['Type'] = 0  # Default type
    else:
        adata.obs['Type'] = 0  # Default type
    
    # Apply filters
    # Filter cells by transcript count (use the correct column name)
    transcript_filter = adata.obs['transcript_counts'] >= min_transcripts
    print(f"Cells passing transcript filter ({min_transcripts}+): {transcript_filter.sum()}")
    
    # Filter cells by gene count (non-zero genes)
    gene_counts = (adata.X > 0).sum(axis=1).A1
    gene_filter = gene_counts >= min_genes
    print(f"Cells passing gene filter ({min_genes}+): {gene_filter.sum()}")
    
    # Combine filters
    combined_filter = transcript_filter & gene_filter
    print(f"Cells passing both filters: {combined_filter.sum()}")
    
    adata = adata[combined_filter].copy()
    
    # Add additional required fields for TMG
    adata.obs['node_size'] = 1.0
    adata.obs['label'] = adata.obs.index
    
    return adata


def _create_cell_taxonomy_xenium(tmg: TissueMultiGraph, clustering_data: pd.DataFrame):
    """Create and add cell type taxonomy to TMG from Xineum clustering results."""
    if clustering_data.empty:
        # Create simple taxonomy with single type
        types = ['Unknown']
        taxonomy = Taxonomy(name='xenium_types', basepath=tmg.basepath,
                           Types=types, feature_mat=np.ones((1, 1)))
    else:
        # Create taxonomy from clustering results
        unique_clusters = sorted(clustering_data['Cluster'].unique())
        types = [f'Cluster_{i}' for i in unique_clusters]
        
        # Create simple feature matrix (identity matrix for now)
        n_types = len(types)
        feature_mat = np.eye(n_types)
        
        taxonomy = Taxonomy(name='xenium_graphclust', basepath=tmg.basepath,
                           Types=types, feature_mat=feature_mat)
    
    # Add taxonomy to TMG
    tmg.Taxonomies.append(taxonomy)
    tmg.layer_taxonomy_mapping[0] = 0  # Cell layer uses this taxonomy
    tmg.save_taxonomies = True


def _create_cell_layer_from_adata(tmg: TissueMultiGraph, 
                                 adata: anndata.AnnData, 
                                 build_spatial_graph: bool = True, 
                                 build_feature_graph: bool = False):
    """Create cell layer directly from AnnData object."""
    # Check if cell layer already exists
    for TG in tmg.Layers:
        if TG.layer_type == "cell":
            tmg.update_user("!!`cell` layer already exists; return...")
            return
    
    # Create TissueGraph directly from adata
    TG = TissueGraph(adata=adata,
                     basepath=tmg.basepath,
                     layer_type="cell",
                     redo=True)
    
    # Add observations and init size
    if 'node_size' not in adata.obs.columns:
        TG.node_size = np.ones((adata.shape[0],))
    
    # Add XY and section information 
    if 'XY' in adata.obsm:
        TG.XY = adata.obsm['XY']
    
    if 'Slice' in adata.obs.columns:
        TG.Section = np.array(adata.obs['Slice'])
    
    # Build spatial graph if requested
    if build_spatial_graph:
        tmg.update_user('building spatial graphs')
        TG.build_spatial_graph()
    
    # Build feature graph if requested  
    if build_feature_graph:
        tmg.update_user('building feature graphs')
        TG.build_feature_graph(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
    
    # Add layer to TMG
    tmg.Layers.append(TG)
    tmg.update_user('done with create_cell_layer_from_adata') 