import random
from datetime import datetime

import numpy as np
import torch
import xarray as xr

from earth2studio.data import CMIP6, CMIP6MultiRealm, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.dx import CorrDiffCMIP6, CorrDiffCMIP6New  # noqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CorrDiffCMIP6New.load_model(
    Package("/localhome/local-ngeneva/cmip6_corrdiff")
)
model = model.to(device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

cmip6_kwargs = dict(
    experiment_id="ssp585",
    source_id="CanESM5",
    variant_label="r1i1p2f1",
    exact_time_match=True,
)
data = CMIP6MultiRealm(
    [CMIP6(table_id=t, **cmip6_kwargs) for t in ("day", "Eday", "SIday")]
)


time_arr = np.array([datetime(2037, 9, 6, 12)], dtype=np.datetime64)
x, coords = fetch_data(
    source=data,
    time=time_arr,
    lead_time=model.input_coords()["lead_time"],
    variable=np.asarray(model.input_variables),
    device=device,
)

out, coords = model(x, coords)

output_da = xr.DataArray(
    data=out.cpu().numpy(),
    coords=coords,
    dims=list(coords.keys()),
)
output_da.to_netcdf("output2.nc")
