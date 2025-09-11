import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "KenazLBM"
author = "Graham W. Johnson"
extensions = [
    "myst_parser",       # Enables Markdown in Sphinx
    "sphinx.ext.autodoc" # API documentation
]
html_theme = "sphinx_rtd_theme"
# html_theme = "furo"

# Allow .md files alongside .rst
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# # Add the _static folder
# html_static_path = ['_static']

# # Add custom CSS file
# def setup(app):
#     app.add_css_file('custom.css')