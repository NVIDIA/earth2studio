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

"""Report generation entry point — aggregate scores and produce markdown.

Usage
-----
Single process (no GPU required)::

    python report.py

With a campaign config::

    python report.py campaign=fcn3_2024_monthly

Prerequisites
-------------
1. ``predownload.py`` with ``predownload.verification.enabled=true``
2. ``main.py`` (inference must have completed)
3. ``score.py`` (scoring must have completed)
"""

import hydra
from loguru import logger
from omegaconf import DictConfig
from src.report import generate_report


@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig) -> None:
    """Eval recipe report entry point — score aggregation and visualization."""
    logger.info("Starting report generation.")
    report_path = generate_report(cfg)
    logger.success(f"Report available at: {report_path}")


if __name__ == "__main__":
    main()
