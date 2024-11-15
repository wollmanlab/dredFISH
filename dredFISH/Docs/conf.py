# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, '../..')
sys.path.append('..')

# -- Project information -----------------------------------------------------

project = 'dredFISH'
copyright = '2022, Wollman lab'
author = 'Wollman lab'


# -- mock imports ---------------------------------------------------
autodoc_mock_imports = [
    'numpy', 
    'pandas', 
    'igraph', 
    'pynndescent',
    'scipy', 
    'scipy.stats', 
    'scipy.optimize',
    'scipy.spatial', 
    'scipy.spatial.distance',
    'scipy.spatial.transform',
    'scipy.sparse.csgraph', 
    'scipy.ndimage',
    'scipy.cluster',
    'scipy.cluster.hierarchy',
    'rasterio',
    'rasterio.features',
    'IPython',
    'skimage',
    'skimage.morphology',
    'skimage.filters',
    'shapely',
    'shapely.geometry',
    'shapely.ops',
    'scanpy',
    'anndata', 

    'sklearn',
    'sklearn.metrics',
    'sklearn.decomposition',
    'sklearn.neighbors',
    'sklearn.manifold',

    'matplotlib', 
    'matplotlib.pyplot',
    'matplotlib.gridspec',
    'matplotlib.cm',
    'matplotlib.colors',
    'matplotlib.collections',
    'matplotlib.patches',

    'colorcet',
    'seaborn',

    'colormath',
    'colormath.color_objects',
    'colormath.color_conversions',

    'PIL',
    'pyemd',
    'leidenalg',
    'tqdm',
    'metadata',
    'ashlar',
    'ashlar.utils',
    'cv2',
    'cellpose',
    'pywt',
    # 'torch',  # this was commented out because of class inheritance issues in sphinx
    # 'torch.nn',
    # 'torch.utils',
    # 'torch.utils.data',
    'zarr',

    'nupack',
    'Bio',
]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinxcontrib.plantuml'
]

napoleon_use_admonition_for_examples = True

plantuml = 'java -Djava.awt.headless=true -jar Docs/plantuml-1.2022.5.jar'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxdoc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
