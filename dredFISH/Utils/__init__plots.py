"""
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import seaborn as sns
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')
mpl.rcParams['pdf.fonttype'] = 42 # editable text in matplotlib
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
# PercentFormat = mtick.FuncFormatter(lambda y, _: '{:.1%}'.format(y))

# empty rectangle (for legend)
EMPTY_RECTANGLE = mpl.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
		                                 visible=False)
DEFAULT_COLORBAR_KWS = {'fraction': 0.05, 
						'shrink': 0.4, 
						'aspect': 5, 
						}

sns.set_style('ticks', rc={'axes.grid':True})
sns.set_context('talk')
