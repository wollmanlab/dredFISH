#!/usr/bin/env python
"""Example run
"""
from dredFISH.Visualization import viz_cell_layer
# mode = 'preview' # need a notebook to see; save nothing
# mode = 'view' # go from the beginning to the end
# mode = 'analysis-only' # analysis only; no plots
mode = 'view' # plots only; assuming analysis is done

# TMG
basepth = "/bigstore/GeneralStorage/Data/dredFISH/NN1_v2_S2"

# define a line to split things into hemi-coronal sections
split_lines = [
     [(-19600, -10000),(-20000,-4000)],
]

viz_cell_layer.main(mode, basepth, split_lines, 
                    compile_pdf=True,
                    pdf_kwargs={'title': "NN Sep12 section 2",
                                'author': "Fangming",
                                }
                    )
