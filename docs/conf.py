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
import requests

def get_github_contents(repo, path=""):
    """
    Fetch the contents of a directory in a GitHub repository.
    """
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

def download_file(url, local_path):
    """
    Download a file from a URL to a local path.
    """
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, 'wb') as f:
        f.write(response.content)

def run_download():
    repo = "AlexanderAivazidis/cell2fate_notebooks"
    base_path = ""
    base_url = "https://raw.githubusercontent.com/"
    contents = get_github_contents(repo, base_path)

    for content in contents:
        # Check if content is a file and ends with .ipynb
        if content['type'] == 'file' and content['name'].endswith('.ipynb'):
            notebook_path = content['path']
            notebook_url = f"{base_url}{repo}/main/{notebook_path}"
            local_path = os.path.join(base_path, os.path.basename(notebook_path))
            # Ensure the local directory exists
            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            print(f"Downloading {notebook_url} to {local_path}")
            download_file(notebook_url, local_path)
        elif content['type'] == 'dir':
            # Recursively download notebooks in subdirectories
            sub_contents = get_github_contents(repo, content['path'])
            for sub_content in sub_contents:
                if sub_content['type'] == 'file' and sub_content['name'].endswith('.ipynb'):
                    notebook_path = sub_content['path']
                    notebook_url = f"{base_url}{repo}/main/{notebook_path}"
                    local_path = "./notebooks/"+notebook_path

                    # Ensure the local directory exists
                    local_dir = os.path.dirname(local_path)
                    if not os.path.exists(local_dir):
                        os.makedirs(local_dir)

                    print(f"Downloading {notebook_url} to {local_path}")
                    download_file(notebook_url, local_path)


# -- Project information -----------------------------------------------------

project = "cell2fate"
copyright = "2024, Alexander Aivazidis, Vitalii Kleshchevnikov, Bayraktar Lab"
author = "Alexander Aivazidis, Vitalii Kleshchevnikov, Bayraktar Lab"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):
    app.add_css_file("css/custom.css")
    run_download()


master_doc = "index"

napoleon_use_param = False
autodoc_member_order = "bysource"
