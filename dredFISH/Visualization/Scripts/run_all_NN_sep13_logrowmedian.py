#!/usr/bin/env python
"""Example run
"""
from dredFISH.Visualization import viz_cell_layer
# mode = 'preview' # need a notebook to see; save nothing
# mode = 'view' # go from the beginning to the end
# mode = 'analysis-only' # analysis only; no plots
# mode = 'plot-only' # analysis only; no plots
# mode = 'compile-only' # analysis only; no plots

mode = 'view'
redo = False
basepth = "/bigstore/GeneralStorage/Data/dredFISH/NN1_v2_S2_logrowmedian" # TMG
split_lines = [[(-19600, -10000),(-20000,-4000)]] # hemi-sphere line

viz_cell_layer.main(mode, basepth, split_lines, 
                    compile_pdf=True,
                    pdf_kwargs={'title': "NN Sep13 section 2",
                                'author': "Fangming",
                                },
                    tmg_kwargs=dict(
                                   norm='logrowmedian',
                                   norm_cell=True,
                                   norm_basis=True,
                                   ),
                    redo=redo,
                    )


mode = 'view'
redo = False
basepth = "/bigstore/GeneralStorage/Data/dredFISH/NN1_v2_S2_logrowmedian_nobasisnorm" # TMG
split_lines = [[(-19600, -10000),(-20000,-4000)]] # hemi-sphere line

viz_cell_layer.main(mode, basepth, split_lines, 
                    compile_pdf=True,
                    pdf_kwargs={'title': "NN Sep13 section 2",
                                'author': "Fangming",
                                },
                    tmg_kwargs=dict(
                                   norm='logrowmedian',
                                   norm_cell=True,
                                   norm_basis=False, # no basis norm to test if Igor's thing has a chance to work
                                   ),
                    redo=redo,
                    )
