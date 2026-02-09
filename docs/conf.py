# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import logging
import os
import pathlib
import sys
from importlib.metadata import version

import dotenv
import requests
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

# -- Load environment vairs -----------------------------------------------------
# Note: To override, use environment variables (e.g. PLOT_GALLERY=True make html)
# Defaults will build API docs for
dotenv.load_dotenv()
doc_version = os.getenv("DOC_VERSION", "main")
plot_gallery = os.getenv("PLOT_GALLERY", False)
run_stale_examples = os.getenv("RUN_STALE_EXAMPLES", False)
filename_pattern = os.getenv("FILENAME_PATTERN", r"/[0-9]+.*\.py")
logging.info(doc_version, plot_gallery, run_stale_examples)

root = pathlib.Path(__file__).parent
physicsnemo = root.parent / "third_party" / "physicsnemo"
release = version("earth2studio")

sys.path.insert(0, root.parent.as_posix())
# Add current folder to use sphinxext.py
sys.path.insert(0, os.path.dirname(__file__))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
version = ".".join(release.split(".")[:2])
project = "Earth2Studio"
copyright = "2025, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_favicon",
    "myst_parser",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_gallery.gen_gallery",
]

source_suffix = [".rst", ".md"]
myst_enable_extensions = ["colon_fence"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "sphinxext.py", "Thumbs.db", ".DS_Store"]
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/nvidia-sphinx-theme.css",
]
html_theme_options = {
    "logo": {
        "text": "Earth2Studio",
        "image_light": "_static/NVIDIA-Logo-V-ForScreen-ForLightBG.png",
        "image_dark": "_static/NVIDIA-Logo-V-ForScreen-ForDarkBG.png",
    },
    "navbar_align": "content",
    "navbar_start": [
        "navbar-logo",
        "version-switcher",
    ],
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/NVIDIA/earth2studio/gh-pages/_static/switcher.json",
        "version_match": doc_version,  # Set DOC_VERSION env variable to change
    },
    "external_links": [
        {
            "name": "Recipes",
            "url": "https://github.com/NVIDIA/earth2studio/tree/main/recipes",
        },
        {
            "name": "Changelog",
            "url": "https://github.com/NVIDIA/earth2studio/blob/main/CHANGELOG.md",
        },
    ],
    "icon_links": [
        {
            # Label for this link
            "name": "Github",
            # URL where the link will redirect
            "url": "https://github.com/NVIDIA/earth2studio",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
}
favicons = ["favicon.ico"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# https://sphinx-gallery.github.io/stable/configuration.html
sphinx_gallery_conf = {
    "examples_dirs": "../examples/",
    "gallery_dirs": "examples",
    "plot_gallery": plot_gallery,
    "filename_pattern": filename_pattern,
    "image_srcset": ["1x"],
    "within_subsection_order": FileNameSortKey,
    "run_stale_examples": run_stale_examples,
    "backreferences_dir": "modules/backreferences",
    "doc_module": ("earth2studio"),
    "reset_modules": (
        "matplotlib",
        "sphinxext.reset_torch",
        "sphinxext.reset_physicsnemo",
    ),
    "reset_modules_order": "both",
    "show_memory": False,
    "exclude_implicit_doc": {r"load_model", r"load_default_package"},
    "log_level": {"backreference_missing": "warning", "gallery_examples": "debug"},
}
