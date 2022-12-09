import numpy as np
# Basic parameters of imaging
bitmap = [('RS0109_cy5', 'hybe2', 'FarRed'),
         ('RS0175_cy5', 'hybe3', 'FarRed'),
         ('RS0237_cy5', 'hybe4', 'FarRed'),
         ('RS0307_cy5', 'hybe5', 'FarRed'),
         ('RS0332_cy5', 'hybe6', 'FarRed'),
         ('RS0384_atto565', 'hybe10', 'FarRed'),
         ('RS0406_atto565', 'hybe11', 'FarRed'),
         ('RS0451_atto565', 'hybe12', 'FarRed'),
         ('RS0468_atto565', 'hybe13', 'FarRed'),
         ('RS0548_atto565', 'hybe14', 'FarRed'),
         ('RS64.0_atto565', 'hybe15', 'FarRed'),
         ('RS156.0_alexa488', 'hybe19', 'FarRed'),
         ('RS278.0_alexa488', 'hybe20', 'FarRed'),
         ('RS313.0_alexa488', 'hybe21', 'FarRed'),
         ('RS643.0_alexa488', 'hybe22', 'FarRed'),
         ('RS740.0_alexa488', 'hybe23', 'FarRed'),
         ('RS810.0_alexa488', 'hybe24', 'FarRed'),
         ('RSN9927.0_cy5', 'hybe7', 'FarRed'),
         ('RSN2336.0_cy5', 'hybe8', 'FarRed'),
         ('RSN1807.0_cy5', 'hybe9', 'FarRed'),
         ('RSN4287.0_atto565', 'hybe16', 'FarRed'),
         ('RSN1252.0_atto565', 'hybe17', 'FarRed'),
         ('RSN9535.0_atto565', 'hybe18', 'FarRed'),
          ('RS0095_cy5', 'hybe1', 'FarRed'),
          ('PolyT', 'hybe25', 'FarRed')]


nbits = len(bitmap)
parameters = {}
# """ Old Microscope Setup"""
# parameters['camera_direction'] = [-1,-1] # NEW [-1,1] # OLD [-1,-1]
# parameters['pixel_size'] = 0.327 # um 490 or 330
# parameters['stitch_rotate'] = 1 # NEW 0
# parameters['stitch_flipud'] = False# NEW False
# parameters['stitch_fliplr'] = True# NEW True
# parameters['flipxy'] = True
# parameters['register_stitch_reference'] = False

""" New Microscope Setup"""
parameters['camera_direction'] = [-1,1] # NEW [-1,1] # OLD [-1,-1]
parameters['pixel_size'] =0.490# 0.490#0.327#0.490 # um 490 or 330
parameters['stitch_rotate'] = 0# NEW0 # NEW 0
parameters['stitch_flipud'] = False# NEW False
parameters['stitch_fliplr'] = True# NEW True
parameters['flipxy'] = False# NEW False # OLD True
parameters['register_stitch_reference'] = True

parameters['segment_gpu'] = False
parameters['fishdata']='fishdata_2022Dec08'
parameters['QC_pixel_size'] = 2 # um
parameters['diameter'] = 8 #15 # um
parameters['segment_diameter'] = parameters['diameter']/parameters['pixel_size']
parameters['nucstain_channel'] = 'DeepBlue'
parameters['nucstain_acq'] = 'hybe1'
parameters['total_channel'] = 'FarRed'
parameters['total_acq'] = 'all' #'hybe25'
parameters['overwrite'] = False #False
parameters['segment_overwrite'] = False #False
parameters['non_cell_overwrite'] = False #False
parameters['vector_overwrite'] = False #False
parameters['batches'] = 2000 #"Number of batches"
parameters['ncpu'] = 3#5 #"Number of threads"
parameters['nregions'] = 2#4 #"Number of Regions/Sections"
parameters['results'] = 'Results' #"Path to save data"
parameters['outpath'] = '/bigstore/GeneralStorage/Data/dredFISH/' #"Path to save data"
parameters['resolution'] = 100 #"resolution to round centroid before naming regions"
parameters['flip'] = False #"Flip Section when centering and rotating"
parameters['non_cell_percentile'] = 5 # Percentile of Image that can be assumed to be non cell
parameters['save_processed_images'] = False
parameters['save_stitched_images'] = False
parameters['nuclei_size_threshold'] = parameters['segment_diameter']*2
parameters['section'] = ''
parameters['signal_method'] = 'median'
parameters['ratio'] = parameters['pixel_size']/parameters['QC_pixel_size']
parameters['visualize'] = False 
parameters['n_pixels']=np.array([2448, 2048])
parameters['border'] = 1000
parameters['segment_type'] = 'cyto2'#'cyto2'
parameters['highpass_sigma'] = 25
parameters['highpass_function'] = 'rolling_ball'
parameters['highpass_smooth_function'] = 'rolling_ball'
parameters['strip'] = False
parameters['highpass_smooth'] = 1
parameters['FF'] = True
parameters['model_types'] = ['total','nuclei','cytoplasm']