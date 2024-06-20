import os
import sys

sys.path.insert(0, os.path.abspath('../../annime'))

project = 'Annime'
copyright = '2024, Avgustin Zhigalov'
author = 'Avgustin Zhigalov'
release = '0.1.0.1'

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
