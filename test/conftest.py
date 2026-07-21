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

import os
import sys
from pathlib import Path

import pytest

from earth2studio.utils.imports import _check_package, _get_group_packages


def _group_available(group: str) -> bool:
    """Check if all packages in a dependency group are installed."""
    try:
        for spec in _get_group_packages(group):
            ok, _ = _check_package(spec)
            if not ok:
                return False
        return True
    except ValueError:
        return False


def _package_available(package: str) -> bool:
    """Check if a package is installed."""
    ok, _ = _check_package(package)
    return ok


# Mapping of test file patterns to required dependency groups or packages
# Files matching these patterns will be skipped during collection if deps missing
_TEST_DEPENDENCIES: dict[str, list[str]] = {
    # Data tests
    "test/data/test_cbottle.py": ["cbottle"],
    "test/data/test_goes_glm.py": ["netCDF4"],
    "test/data/test_ncep_conventional.py": ["pybufrkit"],
    "test/data/test_nnja.py": ["pybufrkit"],
    "test/data/test_ecmwf.py": ["data"],
    "test/data/test_cds.py": ["data"],
    "test/data/test_cams.py": ["data"],
    "test/data/test_cmip6.py": ["data"],
    "test/data/test_gdas.py": ["data"],
    "test/data/test_planetary_computer.py": ["data"],
    "test/data/test_hrrr.py": ["data"],
    "test/data/test_mrms.py": ["data"],
    "test/data/test_opera.py": ["h5py"],
    "test/data/test_jpss_atms.py": ["data"],
    "test/data/test_metop_iasi.py": ["data"],
    "test/data/test_metop_avhrr.py": ["data"],
    "test/data/test_metop_mhs.py": ["data"],
    "test/data/test_metop_amsua.py": ["data"],
    "test/data/test_meteosat_fci.py": ["data"],
    # Model dx tests
    "test/models/dx/test_cbottle_infill.py": ["cbottle"],
    "test/models/dx/test_cbottle_sr.py": ["cbottle"],
    "test/models/dx/test_cbottle_tc.py": ["cbottle"],
    "test/models/dx/test_corrdiff.py": ["corrdiff"],
    "test/models/dx/test_corrdiff_cmip6.py": ["corrdiff"],
    "test/models/dx/test_corrdiff_taiwan.py": ["corrdiff"],
    "test/models/dx/test_dlesym_v0_isccp_era5_precip.py": ["dlesym"],
    "test/models/dx/test_orbit2_precip.py": ["orbit"],
    "test/models/dx/test_precip_afno.py": ["precip-afno"],
    "test/models/dx/test_precip_afno_v2.py": ["precip-afno-v2"],
    "test/models/dx/test_solarradiation_afno.py": ["solarradiation-afno"],
    "test/models/dx/test_tc_tracking.py": ["cyclone"],
    "test/models/dx/test_wind_gust.py": ["windgust-afno"],
    # Model px tests
    "test/models/px/test_ace2.py": ["ace2"],
    "test/models/px/test_aifs.py": ["aifs"],
    "test/models/px/test_aifsens.py": ["aifsens"],
    "test/models/px/test_aifs2.py": ["aifs2"],
    "test/models/px/test_aifs2ens.py": ["aifs2ens"],
    "test/models/px/test_atlas.py": ["atlas"],
    "test/models/px/test_aurora.py": ["aurora"],
    "test/models/px/test_cbottle_video.py": ["cbottle"],
    "test/models/px/test_dlesym.py": ["dlesym"],
    "test/models/px/test_dlesym_v0_isccp_era5.py": ["dlesym"],
    "test/models/px/test_dlwp.py": ["dlwp"],
    "test/models/px/test_fcn.py": ["fcn"],
    "test/models/px/test_fcn3.py": ["fcn3"],
    "test/models/px/test_fengwu.py": ["fengwu"],
    "test/models/px/test_fuxi.py": ["fuxi"],
    "test/models/px/test_gencast_mini.py": ["gencast"],
    "test/models/px/test_graphcast.py": ["graphcast"],
    "test/models/px/test_interpcrpsdit.py": ["interp-crps-dit"],
    "test/models/px/test_interpmodafno.py": ["interp-modafno"],
    "test/models/px/test_pangu.py": ["pangu"],
    "test/models/px/test_sfno.py": ["sfno"],
    "test/models/px/test_stormcast.py": ["stormcast"],
    "test/models/px/test_stormscope.py": ["stormscope"],
    "test/models/px/test_dxwrapper.py": [
        "fcn3",
        "corrdiff",
        "precip-afno-v2",
        "solarradiation-afno",
    ],
    # Model da tests
    "test/models/da/test_da_healda.py": ["da-healda"],
    "test/models/da/test_da_interp.py": ["da-interp"],
    "test/models/da/test_da_sda_stormcast.py": ["da-stormcast"],
    # Serve tests
    "test/serve/client/test_serve_client.py": ["serve"],
    "test/serve/client/test_serve_e2client.py": ["serve"],
    "test/serve/client/test_serve_fsspec_utils.py": ["serve"],
    "test/serve/server/test_compare_crps.py": ["statistics"],
    "test/serve/server/test_server_cleanup.py": ["serve"],
    "test/serve/server/test_server_config.py": ["serve"],
    "test/serve/server/test_server_cpu_worker.py": ["serve"],
    "test/serve/server/test_server_health.py": ["serve"],
    "test/serve/server/test_server_main.py": ["serve"],
    "test/serve/server/test_server_object_storage.py": ["serve"],
    "test/serve/server/test_server_utils.py": ["serve"],
    "test/serve/server/test_server_worker.py": ["serve"],
    "test/serve/server/test_server_workflow.py": ["serve"],
    # Statistics tests
    "test/statistics/test_crps.py": ["statistics"],
    "test/statistics/test_lsd.py": ["statistics"],
    "test/statistics/test_ranks.py": ["statistics"],
    "test/statistics/test_energy_score.py": ["statistics"],
    # Perturbation tests
    "test/perturbation/test_spherical_gaussian.py": ["perturbation"],
    "test/perturbation/test_gaussian.py": ["perturbation"],
    # Utils tests
    "test/utils/test_interp.py": ["utils"],
}


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip collecting test files if their required dependencies are missing.

    This hook runs before Python imports the test module, preventing ImportError
    during collection for tests with optional dependencies.
    """
    # Convert to relative path for matching
    try:
        rel_path = collection_path.relative_to(Path.cwd())
    except ValueError:
        return None

    rel_str = str(rel_path)
    if rel_str not in _TEST_DEPENDENCIES:
        return None

    for dep in _TEST_DEPENDENCIES[rel_str]:
        # Check if it's a group name (exists in pyproject.toml) or a package
        if not _group_available(dep) and not _package_available(dep):
            print(
                f"WARNING: Ignoring {rel_str}: missing dependency '{dep}'",
                file=sys.stderr,
            )
            return True

    return None


def pytest_addoption(parser):
    parser.addoption(
        "--package-download",
        action="store_true",
        default=False,
        help="test auto model package downloads and generate cache",
    )
    parser.addoption(
        "--package",
        action="store_true",
        default=False,
        help="test model packages using CI cache ",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "package: mark test as requiring model package cache"
    )
    config.addinivalue_line(
        "markers",
        "package_download: mark test as requiring a model download from external store",
    )


def pytest_collection_modifyitems(config, items):

    # Skip tests whose optional dependency groups are missing.
    # pytest_ignore_collect handles directory traversal, this is for when individual
    # files are targeted when testing
    for item in items:
        try:
            rel_str = str(item.path.relative_to(Path.cwd()))
        except ValueError:
            continue
        if rel_str not in _TEST_DEPENDENCIES:
            continue
        for dep in _TEST_DEPENDENCIES[rel_str]:
            if not _group_available(dep) and not _package_available(dep):
                item.add_marker(
                    pytest.mark.skip(reason=f"missing dependency '{dep}' for {rel_str}")
                )
                break

    enable_packages = config.getoption("--package") or os.getenv(
        "EARTH2STUDIO_TEST_PACKAGES", ""
    ).strip().lower() in ("1", "true", "yes", "on")

    if not enable_packages:
        skip_model_package = pytest.mark.skip(
            reason="need --package option to run model package tests"
        )
        for item in items:
            if "package" in item.keywords:
                item.add_marker(skip_model_package)

    enable_download = config.getoption("--package-download") or os.getenv(
        "EARTH2STUDIO_DOWNLOAD_PACKAGES", ""
    ).strip().lower() in ("1", "true", "yes", "on")

    if not enable_download:
        skip_download = pytest.mark.skip(
            reason="need --package-download option to run package download tests"
        )
        for item in items:
            if "package_download" in item.keywords:
                item.add_marker(skip_download)
