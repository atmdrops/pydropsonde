# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HALO-DROPS'
copyright = '2023, Geet George'
author = 'Geet George'
release = 'v0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.duration',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'nbsphinx',
    "autodoc2",
    ]

myst_enable_extensions =[
    "colon_fence",
    "dollarmath"
]
autodoc2_packages = [
   {
        "path": "../../src/halodrops/",
    },
]
autodoc2_render_plugin = "myst"
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

# Set link name generated in the top bar.
html_title = 'HALO-DROPS'