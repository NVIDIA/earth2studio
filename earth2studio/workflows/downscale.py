from collections import OrderedDict
from datetime import datetime
from typing import Literal

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array


# sphinx - diagnostic start
def diagnostic_downscale(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    diagnostic: DiagnosticModel,
    data: DataSource,
    io: IOBackend,
    output_coords: CoordSystem = OrderedDict({}),
    device: torch.device | None = None,
    output_prognostic: bool = False,
    shared_coords: Literal["prognostic", "diagnostic"] = "diagnostic"
) -> IOBackend:
    """Built in diagnostic workflow.
    This workflow creates a determinstic inference pipeline that couples a prognostic
    model with a diagnostic model.

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    nsteps : int
        Number of forecast steps
    prognostic : PrognosticModel
        Prognostic model
    diagnostic: DiagnosticModel
        Diagnostic model, must be on same coordinate axis as prognostic
    data : DataSource
        Data source
    io : IOBackend
        IO object
    output_coords: CoordSystem, optional
        IO output coordinate system override, by default OrderedDict({})
    device : torch.device, optional
        Device to run inference on, by default None
    output_prognostic: bool, optional
        Output variables from prognostic model in addition to those from
        the diagnostic, by default False. If True, variable names produced by
        both the prognostic and the diagnostic model will be returned from
        the diagnostic model.
    shared_coords:
        If the prognostic and diagnostic model have different ranges for the
        same coordinate axis, the version indicated by `shared_coords` will be
        used. Either `diagnostic` (default) or `prognostic`.

    Returns
    -------
    IOBackend
        Output IO object
    """
    # sphinx - diagnostic end
    logger.info("Running diagnostic workflow!")
    # Load model onto the device
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    diagnostic = diagnostic.to(device)
    # Fetch data from data source and load onto device
    prognostic_ic = prognostic.input_coords()
    prognostic_oc = prognostic.output_coords(prognostic_ic)
    diagnostic_ic = diagnostic.input_coords()
    diagnostic_oc = diagnostic.output_coords(diagnostic_ic)
    time = to_time_array(time)
    x, coords = fetch_data(
        source=data,
        time=time,
        variable=prognostic_ic["variable"],
        lead_time=prognostic_ic["lead_time"],
        device=device,
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")

    # Set up IO backend
    def get_total_coords(coords):
        total_coords = coords.copy()
        # Remove batch dimensions
        for key, value in coords.items():
            if value.shape == (0,):
                del total_coords[key]

        # Add time dimensions
        total_coords["time"] = time
        total_coords["lead_time"] = np.asarray(
            [
                prognostic_oc["lead_time"] * i
                for i in range(nsteps + 1)
            ]
        ).flatten()
        total_coords.move_to_end("lead_time", last=False)
        total_coords.move_to_end("time", last=False)

        # override coordinates given in output_coords
        for key, value in total_coords.items():
            if key == "variable" and "variable" in output_coords:
                var_names = [v for v in output_coords["variable"] if v in total_coords["variable"]]
                total_coords[key] = np.array(var_names)
            else:
                total_coords[key] = output_coords.get(key, value)

        return total_coords

    diagnostic_tc = get_total_coords(diagnostic_oc)
    if output_prognostic:
        prognostic_tc = get_total_coords(prognostic_oc)
        # Use variable from diagnostic if given by both diagnostic and prognostic
        prognostic_tc["variable"] = np.array(
            [v for v in prognostic_tc["variable"] if v not in diagnostic_tc["variable"]]
        )

        # Handle coordinate axes found in both prognostic and diagnostic
        for key in set(prognostic_tc.keys()) & set(diagnostic_tc.keys()) - {"variable"}:
            if shared_coords == "diagnostic":
                prognostic_tc[key] = diagnostic_tc[key]
            else:
                diagnostic_tc[key] = prognostic_tc[key]

        # Add prognostic variables
        io.add_array(
            {k: v for (k, v) in prognostic_tc.items() if k != "variable"}, 
            prognostic_tc["variable"]
        )
    
    # Add diagnostic variables
    io.add_array(
        {k: v for (k, v) in diagnostic_tc.items() if k != "variable"}, 
        diagnostic_tc["variable"]
    )

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, prognostic_ic)
    # Create prognostic iterator
    model = prognostic.create_iterator(x, coords)

    logger.info("Inference starting!")
    with tqdm(total=nsteps + 1, desc="Running inference") as pbar:
        for step, (x, coords) in enumerate(model):
            if output_prognostic:
                (x_out, coords_out) = map_coords(x, coords, prognostic_tc)
                io.write(*split_coords(x_out, coords_out))

            x, coords = map_coords(x, coords, diagnostic_ic)
            # Run diagnostic
            x, coords = diagnostic(x, coords)
            # Subselect domain/variables as indicated in output_coords
            x, coords = map_coords(x, coords, diagnostic_tc)
            io.write(*split_coords(x, coords))
            pbar.update(1)
            if step == nsteps:
                break

    logger.success("Inference complete")
    return io
