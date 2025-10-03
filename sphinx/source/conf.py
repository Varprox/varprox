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
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'varprox'
copyright = '2025, Arthur Marmin and Frédéric Richard'
author = 'Arthur Marmin and Frédéric Richard'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.githubpages',
              'sphinx.ext.autodoc',
              'nbsphinx',
              'sphinx.ext.intersphinx',
              # 'sphinx_gallery.gen_gallery',
              'sphinxcontrib.bibtex']
autodoc_member_order = 'bysource'

# Bibliography.
bibtex_bibfiles = ['./biblio.bib']
bibtex_encoding = 'latin'
bibtex_default_style = 'unsrt'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
show_authors = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_static_path = []

# sphinx_gallery_conf = {
#      'examples_dirs': ['../../examples/'],  # path to your example scripts
#      'gallery_dirs': ['auto_examples/'], # path to where to save gallery
# }

# Configuration for cross-references
intersphinx_mapping = {'python': ('http://docs.python.org/3', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('http://matplotlib.org/stable', None)}

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False 
numpydoc_xref_param_type = True 
numpydoc_attributes_as_param_list = False 
