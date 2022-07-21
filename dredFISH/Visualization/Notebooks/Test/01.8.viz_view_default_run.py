"""Example run
"""
from dredFISH.Visualization import viz_cell_layer

# mode = 'preview'
mode = 'view'
# mode = 'plot-only'

# TMG
# basepth = '/bigstore/GeneralStorage/Data/dredFISH/Dataset1-t3'
# basepth = '/bigstore/GeneralStorage/Data/dredFISH/Dataset1-t4'
basepth = '/bigstore/GeneralStorage/Data/dredFISH/Dataset1-t5'

# define a line to split things into hemi-coronal sections
split_lines = [
    # [(0,0),(1,1)],
    [(550, -6000), (200, 2000)],
    [(200, 2000), (550, -6000)],
    # [(300, 2000), (550, -6000)],
]

viz_cell_layer.main(mode, basepth, split_lines)