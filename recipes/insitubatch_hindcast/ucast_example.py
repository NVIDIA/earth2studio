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

"""Drive a real checkpoint (U-CAST) off the insitubatch feed -- no predownload pass.

U-CAST takes 83 channels and a two-step history. Most of those channels are one level of a
pressure-level array, and ``WB2Lexicon`` already names each channel's array and level in the
``"array::level"`` spelling ``var_map`` accepts -- so the model describes its own inputs and
the feed resolves them onto the 11 arrays that actually hold them.

That is the point worth watching in the output: channels sharing an array share one read,
because a stored chunk holds every level of a step anyway.

Pulls a ~6.7 GB checkpoint on first run. Add ``--device cuda`` to land batches on the GPU
(U-CAST at ``--batch-size 4`` peaks around 6.3 GB).
"""

import argparse

import torch
from insitubatch import obstore_store

from earth2studio.data.insitu import InSituForecastFeed
from earth2studio.lexicon import WB2Lexicon
from earth2studio.models.px import UCast
from earth2studio.utils.coords import map_coords

WB2 = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)


def main() -> None:
    """Roll U-CAST forward over one window of init times and report the reads it cost."""
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2000, help="first init index")
    p.add_argument("--n-init", type=int, default=8, help="init times to stream")
    p.add_argument("--batch-size", type=int, default=4, help="init times per batch")
    p.add_argument("--steps", type=int, default=2, help="forecast steps to roll out")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = UCast.load_model(UCast.load_default_package()).to(args.device)
    ic = model.input_coords()

    feed = InSituForecastFeed(
        obstore_store(WB2, skip_signature=True),  # anonymous public read
        variables=ic["variable"],  # 83 channels
        var_map={v: WB2Lexicon.VOCAB[v] for v in ic["variable"]},  # "geopotential::500"
        lead_times=ic["lead_time"],  # [-12h, 0h] history window
        sample_range=(args.start, args.start + args.n_init),
        batch_size=args.batch_size,
        transpose_inner=True,  # WB2 is stored lon-major
        device=args.device,
    )

    requested = args.n_init * len(ic["lead_time"]) * len(ic["variable"])
    arrays = len({WB2Lexicon.VOCAB[v].split("::")[0] for v in ic["variable"]})
    print(
        f"{len(ic['variable'])} channels over {arrays} arrays; "
        f"{args.n_init} inits x {len(ic['lead_time'])} history steps "
        f"= {requested} requested field reads"
    )

    # Stream the whole window, so the decode count below covers the reads counted above.
    for batch, (x, coords) in enumerate(feed):
        x, coords = map_coords(x, coords, ic)
        init = str(coords["time"][0])[:16]
        with torch.inference_mode():
            for step, (fx, fc) in enumerate(model.create_iterator(x, coords)):
                if step == 0:
                    continue  # step 0 is the initial condition
                i = list(fc["variable"]).index("z500")
                lead = str(fc["lead_time"][0].astype("timedelta64[h]"))
                print(
                    f"  batch {batch} (from {init})  lead {lead:>9}  "
                    f"z500 mean {fx[0, 0, i].mean():.1f} m2/s2"
                )
                if step == args.steps:
                    break

    print(f"chunks decoded: {feed.dataset.cache_misses}")
    feed.dataset.close()


if __name__ == "__main__":
    main()
