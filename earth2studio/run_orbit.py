
from datetime import datetime
import numpy as np
import torch
from loguru import logger

from earth2studio.utils.time import to_time_array

from earth2studio.data import DataSource, prep_data_array
from earth2studio.io import IOBackend
from earth2studio.models.dx import OrbitGlobalPrecip
from collections import OrderedDict


def run(
    time: list[str] | list[datetime] | list[np.datetime64],
    orbit: OrbitGlobalPrecip,
    data: DataSource,
    io: IOBackend,
) -> IOBackend:

    logger.info("Running ORBIT-2 inference!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")

    orbit = orbit.to(device)

    time = to_time_array(time)
    x, coords = prep_data_array(
        data(time, orbit.input_coords()["variable"]), device=device
    )

    sst, coords = prep_data_array(
        data(time, "sst"), device=device
    )

    t2_sst_combined = torch.where(torch.isnan(sst[:,0]), x[:,0], sst[:,0])
    x[:,0] = t2_sst_combined

    for i in range(24):
        time_ = (np.array(time) + np.timedelta64(1, 'h') * (-1 * i)).astype(str)
        time_ = to_time_array(time_)
        p, coords = prep_data_array(
            data(time_, ["cp", "lsp", "t2m", "sst"]), device=device
        )

        #tuple of t (1, 4, 721, 1440) tensors
        batch_split = torch.split(p, 1)
        if i == 0:
            batch_p = torch.zeros((len(time), 24, p.shape[-3], p.shape[-2], p.shape[-1]), device=device)
        for j in range(len(batch_split)):
            batch_p[j,i] = batch_split[j][0]
                
        
    total_p_24hr = batch_p[:,:,0] + batch_p[:,:,1]
    total_p_24hr = torch.sum(total_p_24hr, dim=1)
    t2_sst_combined = torch.where(torch.isnan(batch_p[:,:,3]), batch_p[:,:,2], batch_p[:,:,3])
    t2_max = torch.max(t2_sst_combined, dim=1).values
    t2_min = torch.min(t2_sst_combined, dim=1).values
    total_p_24hr = total_p_24hr.unsqueeze(1)
    t2_max = t2_max.unsqueeze(1)
    t2_min = t2_min.unsqueeze(1)
    x = torch.cat((x,total_p_24hr,t2_max,t2_min), dim=1)
    logger.success(f"Fetched data from {data.__class__.__name__}")

    input_coords = OrderedDict({
        k: v for k, v in
        orbit.input_coords().items()
        if k != "batch" # remove placeholder batch dim
    })
    input_coords["time"] = time
    input_coords.move_to_end("time", last=False)

    total_coords = OrderedDict({
        k: v for k, v in
        orbit.output_coords(orbit.input_coords()).items()
        if k != "batch" # remove placeholder batch dim
    })
    total_coords["time"] = time
    total_coords.move_to_end("time", last=False)

    io.add_array(total_coords, total_coords["variable"], overwrite=True)
    logger.info("Inference Starting")
    x, output_coords = orbit(x, input_coords)
    io.write(x, output_coords, total_coords["variable"][0])
    logger.success("Inference complete")

    return io

