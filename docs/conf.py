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
import inspect
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
doc_public_build = os.getenv("DOC_PUBLIC_BUILD", "").lower() in ("1", "true", "yes")
plot_gallery = os.getenv("PLOT_GALLERY", "").lower() in ("1", "true", "yes")
run_stale_examples = os.getenv("RUN_STALE_EXAMPLES", "").lower() in ("1", "true", "yes")
filename_pattern = os.getenv("FILENAME_PATTERN", r"/[0-9]+.*\.py")
logging.info(doc_version, doc_public_build, plot_gallery, run_stale_examples)

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
copyright = "2026, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


def linkcode_resolve(domain, info):
    """Determine a GitHub URL corresponding to a Python object.

    Used by sphinx.ext.linkcode to generate [source] links that point directly
    to the source on GitHub instead of local _modules/ pages.

    Based on the common pattern used by NumPy, SciPy, and scikit-learn.
    See: https://github.com/scikit-learn/scikit-learn/blob/main/doc/sphinxext/github_link.py
    """
    if domain != "py":
        return None

    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname or not fullname:
        return None

    # Import the module and resolve the object
    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    # Unwrap decorated objects to get the original source
    try:
        obj = inspect.unwrap(obj)
    except ValueError:
        pass

    # Get the source file
    try:
        fn = inspect.getsourcefile(obj)
    except TypeError:
        fn = None
    if not fn:
        return None

    # Only link to files within the earth2studio package
    import earth2studio

    try:
        fn = os.path.relpath(fn, start=os.path.dirname(earth2studio.__file__))
    except ValueError:
        return None
    if fn.startswith(".."):
        return None

    # Get line number range
    try:
        source, lineno = inspect.getsourcelines(obj)
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    except (OSError, TypeError):
        linespec = ""

    return (
        f"https://github.com/NVIDIA/earth2studio/blob/{doc_version}/"
        f"earth2studio/{fn.replace(os.sep, '/')}{linespec}"
    )


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosummary",
    "sphinx_favicon",
    "myst_parser",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_gallery.gen_gallery",
    "sphinx_badges",
]

source_suffix = [".rst", ".md"]
myst_enable_extensions = ["colon_fence"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "sphinxext.py", "Thumbs.db", ".DS_Store"]
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_theme_options = {
    "public_docs_features": doc_public_build,
    # "logo": {
    #     "text": "Earth2Studio",
    #     "image_light": "_static/NVIDIA-Logo-V-ForScreen-ForLightBG.png",
    #     "image_dark": "_static/NVIDIA-Logo-V-ForScreen-ForDarkBG.png",
    # },
    "navbar_align": "content",
    "navbar_start": ["navbar-logo", "version-switcher", "navbar-nav"],
    "navbar_center": [],
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

# -- sphinx-badges configuration ----------------------------------------------

badges_position = "top"
badges_style = "square"

badges_group_labels = {
    "region": "Region",
    "class": "Class",
    "dataclass": "Data Class",
    "year": "Release",
    "product": "Product",
    "gpu": {"label": "Rec VRAM", "tooltip": "Recommended GPU Memory"},
}

badges_definitions = {
    # ── Region ───────────────────────────────────────────────────────────────
    "region:global": {
        "label": "Global",
        "icon": '<i class="fa-solid fa-globe"></i>',
        "color": "#0d47a1",
        "text_color": "#fff",
    },
    "region:na": {
        "label": "NA",
        "icon": '<i class="fa-solid fa-earth-americas"></i>',
        "color": "#1565c0",
        "text_color": "#fff",
        "tooltip": "North America",
    },
    "region:sa": {
        "label": "SA",
        "icon": '<i class="fa-solid fa-earth-americas"></i>',
        "color": "#1976d2",
        "text_color": "#fff",
        "tooltip": "South America",
    },
    "region:eu": {
        "label": "EU",
        "icon": '<i class="fa-solid fa-earth-europe"></i>',
        "color": "#1e88e5",
        "text_color": "#fff",
        "tooltip": "Europe",
    },
    "region:as": {
        "label": "AS",
        "icon": '<i class="fa-solid fa-earth-asia"></i>',
        "color": "#42a5f5",
        "text_color": "#000",
        "tooltip": "Asia",
    },
    "region:af": {
        "label": "AF",
        "icon": '<i class="fa-solid fa-earth-africa"></i>',
        "color": "#64b5f6",
        "text_color": "#000",
        "tooltip": "Africa",
    },
    "region:au": {
        "label": "AU",
        "icon": '<i class="fa-solid fa-earth-oceania"></i>',
        "color": "#90caf9",
        "text_color": "#000",
        "tooltip": "Australia",
    },
    # ── Class ──────────────────────────────────────────────────────────
    "class:nwc": {
        "label": "NWC",
        "color": "#198754",
        "text_color": "#fff",
        "tooltip": "Nowcasting",
    },
    "class:ds": {
        "label": "DS",
        "color": "#20c997",
        "text_color": "#000",
        "tooltip": "Downscaling",
    },
    "class:mrf": {
        "label": "MRF",
        "color": "#0dcaf0",
        "text_color": "#000",
        "tooltip": "Medium Range",
    },
    "class:s2s": {
        "label": "S2S",
        "color": "#6610f2",
        "text_color": "#fff",
        "tooltip": "Sub-Seasonal to Seasonal",
    },
    "class:da": {
        "label": "DA",
        "color": "#d63384",
        "text_color": "#fff",
        "tooltip": "Data Assimilation",
    },
    "class:cm": {
        "label": "CM",
        "color": "#ccff00",
        "text_color": "#000",
        "tooltip": "Climate Model",
    },
    # ── Data Class ───────────────────────────────────────────────────────────
    "dataclass:analysis": {
        "label": "",
        "icon": '<i class="fa-solid fa-chart-line"></i>',
        "color": "#0891b2",
        "text_color": "#fff",
        "tooltip": "Analysis / Initial Conditions",
    },
    "dataclass:reanalysis": {
        "label": "",
        "icon": '<i class="fa-solid fa-rotate"></i>',
        "color": "#1d4ed8",
        "text_color": "#fff",
        "tooltip": "Reanalysis",
    },
    "dataclass:observation": {
        "label": "",
        "icon": '<i class="fa-solid fa-binoculars"></i>',
        "color": "#15803d",
        "text_color": "#fff",
        "tooltip": "Direct Observations",
    },
    "dataclass:simulation": {
        "label": "",
        "icon": '<i class="fa-solid fa-microchip"></i>',
        "color": "#b451ed",
        "text_color": "#fff",
        "tooltip": "Numerical / AI Simulation",
    },
    # ── Publication year ─────────────────────────────────────────────────────
    "year:2021": {"label": "2021", "color": "#F2F2F2", "text_color": "#111"},
    "year:2022": {"label": "2022", "color": "#D9D9D9", "text_color": "#111"},
    "year:2023": {"label": "2023", "color": "#BDBDBD", "text_color": "#111"},
    "year:2024": {"label": "2024", "color": "#828282", "text_color": "#fff"},
    "year:2025": {"label": "2025", "color": "#4F4F4F", "text_color": "#fff"},
    "year:2026": {"label": "2026", "color": "#1A1A1A", "text_color": "#fff"},
    # ── Product (fields) ──────────────────────────────────────────────────────
    "product:wind": {
        "label": "",
        "icon": '<i class="fa-solid fa-wind"></i>',
        "color": "#4dabf7",
        "text_color": "#fff",
        "tooltip": "Surface Winds",
    },
    "product:precip": {
        "label": "",
        "icon": '<i class="fa-solid fa-cloud-rain"></i>',
        "color": "#339af0",
        "text_color": "#fff",
        "tooltip": "Surface Precipitation",
    },
    "product:temp": {
        "label": "",
        "icon": '<i class="fa-solid fa-temperature-half"></i>',
        "color": "#fd7e14",
        "text_color": "#fff",
        "tooltip": "Surface Temperature",
    },
    "product:atmos": {
        "label": "",
        "icon": '<i class="fa-solid fa-cloud"></i>',
        "color": "#74c0fc",
        "text_color": "#fff",
        "tooltip": "Atmosphere",
    },
    "product:ocean": {
        "label": "",
        "icon": '<i class="fa-solid fa-water"></i>',
        "color": "#1098ad",
        "text_color": "#fff",
        "tooltip": "Ocean",
    },
    "product:land": {
        "label": "",
        "icon": '<i class="fa-solid fa-mound"></i>',
        "color": "#a0522d",
        "text_color": "#fff",
        "tooltip": "Land/Soil",
    },
    "product:veg": {
        "label": "",
        "icon": '<i class="fa-solid fa-seedling"></i>',
        "color": "#40c057",
        "text_color": "#fff",
        "tooltip": "Vegetation",
    },
    "product:solar": {
        "label": "",
        "icon": '<i class="fa-solid fa-sun"></i>',
        "color": "#f59f00",
        "text_color": "#fff",
        "tooltip": "Solar",
    },
    "product:radar": {
        "label": "",
        "icon": '<i class="fa-solid fa-satellite-dish"></i>',
        "color": "#7048e8",
        "text_color": "#fff",
        "tooltip": "Radar",
    },
    "product:sat": {
        "label": "",
        "icon": '<i class="fa-solid fa-satellite"></i>',
        "color": "#1c7ed6",
        "text_color": "#fff",
        "tooltip": "Satellite",
    },
    "product:insitu": {
        "label": "",
        "icon": '<i class="fa-solid fa-tower-observation"></i>',
        "color": "#2f9e44",
        "text_color": "#fff",
        "tooltip": "In-Situ Observations",
    },
    # ── Minimum VRAM requirement ───────────────────────────────────────
    "gpu:96gb": {
        "label": "96 GB",
        "color": "#76b900",
        "text_color": "#fff",
        "tooltip": "96 GB VRAM (GH200, H200, B40, B100, B200)",
    },
    "gpu:80gb": {
        "label": "80 GB",
        "color": "#5aac44",
        "text_color": "#fff",
        "tooltip": "80 GB VRAM (A100 80GB, H100, GH200, H200, B40, B100, B200)",
    },
    "gpu:48gb": {
        "label": "48 GB",
        "color": "#4b9934",
        "text_color": "#fff",
        "tooltip": "48 GB VRAM (L40S, A100 80GB, H100, GH200, H200, B40, B100, B200)",
    },
    "gpu:40gb": {
        "label": "40 GB",
        "color": "#3d7a26",
        "text_color": "#fff",
        "tooltip": "40 GB VRAM (A100, L40S, H100, GH200, H200, B40, B100, B200)",
    },
    "gpu:24gb": {
        "label": "24 GB",
        "color": "#2d5c1e",
        "text_color": "#fff",
        "tooltip": "24 GB VRAM (A30, A100, L40S, H100, GH200, H200, B40, B100, B200)",
    },
}
