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

"""Generate Earth2Studio example gallery Markdown before docs builds."""

from __future__ import annotations

import logging

from earth2studio_gallery.builder import GalleryBuilder
from earth2studio_gallery.config import GalleryConfig
from earth2studio_gallery.progress import ProgressEvent


def _log_progress(event: ProgressEvent) -> None:
    prefix = f"{event.example}: " if event.example else ""
    logging.info("%-9s %s%s", event.stage.upper(), prefix, event.message)


def main() -> None:
    """Render gallery Markdown and static assets."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    gallery = GalleryConfig.load(".")
    report = GalleryBuilder(gallery, progress=_log_progress).render()
    if report.failures:
        names = ", ".join(example.relative.as_posix() for example, _ in report.failures)
        raise RuntimeError(f"Gallery examples failed: {names}")


if __name__ == "__main__":
    main()
