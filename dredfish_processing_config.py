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
          ('RS0095_cy5', 'hybe1', 'FarRed')]


nbits = len(bitmap)
parameters = {}
parameters['camera_direction'] = [-1,-1] # NEW [-1,1] # OLD [-1,-1]
parameters['flipxy'] = False
parameters['fishdata']='fishdata_2022Jun21'
parameters['segment_gpu'] = False
parameters['pixel_size'] = 0.330 # um 490 or 330
parameters['QC_pixel_size'] = 5 # um
parameters['diameter'] = 15 # um
parameters['segment_diameter'] = parameters['diameter']/parameters['pixel_size']
parameters['nucstain_channel'] = 'DeepBlue'
parameters['nucstain_acq'] = 'infer'
parameters['total_channel'] = 'FarRed'
parameters['total_acq'] = 'infer'
parameters['overwrite'] = False
parameters['stitch_rotate'] = 1 # NEW 0
parameters['stitch_flipud'] = False# NEW False
parameters['stitch_fliplr'] = True# NEW True
parameters['batches'] = 2000 #"Number of batches"
parameters['ncpu'] = 10 #"Number of threads"
parameters['nregions'] = 6#4 #"Number of Regions/Sections"
parameters['outpath'] = '/bigstore/GeneralStorage/Data/dredFISH/' #"Path to save data"
parameters['resolution'] = 100 #"resolution to round centroid before naming regions"