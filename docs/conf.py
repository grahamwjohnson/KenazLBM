import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "My Project"
author = "Your Name"
extensions = [
    "myst_parser",       # Enables Markdown in Sphinx
    "sphinx.ext.autodoc" # API documentation
]
html_theme = "sphinx_rtd_theme"

# Allow .md files alongside .rst
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
