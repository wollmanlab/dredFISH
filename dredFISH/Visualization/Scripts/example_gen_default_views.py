#!/usr/bin/env python
"""Example run
"""
from dredFISH.Visualization import viz_cell_layer

# mode = 'preview' # need a notebook to see; save nothing
# mode = 'view' # go from the beginning to the end
# mode = 'analysis-only' # analysis only; no plots
mode = 'plot-only' # plots only; assuming analysis is done

# TMG
basepth = '/bigstore/GeneralStorage/Data/dredFISH/Dataset1-t5'

# define a line to split things into hemi-coronal sections
split_lines = [
    # [(0,0),(1,1)],
    [(550, -6000), (200, 2000)],
    [(200, 2000), (550, -6000)],
    # [(300, 2000), (550, -6000)],
]

viz_cell_layer.main(mode, basepth, split_lines, 
                    compile_pdf=True,
                    pdf_kwargs={'title': "dredFISH default analysis",
                                'author': "Fangming",
                                }
                    )
