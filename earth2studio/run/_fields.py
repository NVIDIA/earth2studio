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

"""Field mapping and output-layout helpers shared by workflows and runners."""

from collections import OrderedDict
from collections.abc import Generator, Mapping
from typing import Any

import numpy as np
import xarray as xr

from earth2studio.grids import E2S_CRS, E2S_GRID_ID, infer_grid, same_geometry
from earth2studio.io.utils import output_template
from earth2studio.models.px.utils import initial_condition
from earth2studio.utils.coords import E2S_STATISTICS, CoordSystem, coord_array_like


def _dimension_coords(x: xr.DataArray) -> CoordSystem:
    """Read dimension labels without materializing a coordinate signature."""
    return OrderedDict((dim, x.coords[dim].values) for dim in x.dims)


def map_field(x: xr.DataArray, target: xr.DataArray | CoordSystem) -> xr.DataArray:
    """Select the variables and domain of ``target`` from ``x``.

    Use it to fit fetched data or a model's output to another model's inputs, or to
    an output subset. ``batch``, ``time`` and ``lead_time`` are kept from ``x``,
    as are coordinates that ``target`` leaves empty. Exact contiguous selections
    are views; labels missing from a numeric coordinate select the nearest one.
    Curvilinear auxiliaries are never indexed as dimensions; model validation
    checks their geometry at the inference boundary.

    ``map_field`` selects and never regrids. Nearest selection is accepted only
    when ``x`` covers ``target``: a target label more than one source spacing
    beyond the source's range (a regional source on a global grid, or a
    -180..180 longitude source on a 0..360 grid) raises instead of copying edge
    values. Resolution is not checked, so a source coarser than the target (a
    1 degree source on a 0.25 degree grid) repeats neighbouring values. Fetch
    such data with ``fetch_data(target_grid=...)`` or wrap the source in
    :class:`~earth2studio.data.RegriddedSource`.

    Parameters
    ----------
    x : xr.DataArray
        Field to select from
    target : xr.DataArray | CoordSystem
        Signature to select, such as ``model.input_coords()``, or a mapping of
        coordinate names to labels

    Returns
    -------
    xr.DataArray
        ``x`` on ``target``'s coordinates, carrying ``target``'s grid metadata when
        ``target`` is a DataArray

    Raises
    ------
    ValueError
        If a label is missing from a non-numeric coordinate, a numeric coordinate
        of ``x`` does not cover ``target``'s range, or the
        geometry or CRS does not match a DataArray ``target``.
    """
    coordinates = (
        _dimension_coords(target) if isinstance(target, xr.DataArray) else target
    )
    for dim, values in coordinates.items():
        if dim in ("batch", "time", "lead_time") or len(values) == 0:
            continue
        source = x.coords[dim].values
        if np.array_equal(source, values):
            continue
        if source.ndim != 1 or np.asarray(values).ndim != 1:
            raise ValueError(f"Cannot map multidimensional coordinate {dim}")
        index = x.get_index(dim).get_indexer(values)
        if (index < 0).any():
            if not np.issubdtype(source.dtype, np.number):
                missing = np.asarray(values)[index < 0]
                raise ValueError(
                    f"Missing labels for coordinate {dim}: {missing.tolist()}; "
                    f"it has {source.tolist()}"
                )
            index = _nearest_covering(dim, source, np.asarray(values))
        selection = (
            slice(int(index[0]), int(index[-1]) + 1)
            if np.all(np.diff(index) == 1)
            else index
        )
        x = x.isel({dim: selection}).assign_coords({dim: values})
        if dim == "variable":
            statistics = coord_array_like(x).attrs.get(E2S_STATISTICS)
            x.attrs = dict(x.attrs)
            x.attrs.pop(E2S_STATISTICS, None)
            if statistics:
                x.attrs[E2S_STATISTICS] = statistics
    if isinstance(target, xr.DataArray) and "dims" in target.attrs:
        spatial_dims = set(target.attrs["dims"])
        for name, coordinate in target.coords.items():
            if not spatial_dims.intersection(coordinate.dims):
                continue
            if (
                name not in x.coords
                or x.coords[name].dims != coordinate.dims
                or not np.array_equal(x.coords[name], coordinate)
            ):
                raise ValueError(
                    f"Source geometry does not match target coordinate {name}"
                )
        actual_crs = x.attrs.get(E2S_CRS)
        if actual_crs is not None and actual_crs != target.attrs.get(E2S_CRS):
            raise ValueError("Source CRS does not match target CRS")
        x = x.copy(deep=False)
        for key in (
            "type",
            "dims",
            "shape",
            "topology",
            "crs",
            E2S_CRS,
            E2S_GRID_ID,
            "level",
            "nside",
            "ordering",
            "layout",
            "origin",
            "clockwise",
        ):
            x.attrs.pop(key, None)
            if key in target.attrs:
                x.attrs[key] = target.attrs[key]
    return x


_REGRID_HINT = (
    "map_field selects and never regrids; fetch with fetch_data(target_grid=...) "
    "or wrap the source in RegriddedSource"
)


def _nearest_covering(dim: str, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the nearest source index per target label, if ``source`` covers it.

    Raises when a target label lies more than one source spacing, measured at
    the nearer edge, beyond the source's range, which would otherwise copy edge
    values silently.
    """
    if len(source) == 0:
        raise ValueError(f"Source coordinate {dim} is empty; {_REGRID_HINT}")
    order = np.sort(source)
    if len(order) == 1:
        if not np.all(target == order[0]):
            raise ValueError(
                f"Source coordinate {dim} has the single label {order[0]}, which "
                f"does not match target labels {target.tolist()}; {_REGRID_HINT}"
            )
        return np.zeros(len(target), dtype=np.intp)
    below = target < order[0] - (order[1] - order[0])
    above = target > order[-1] + (order[-1] - order[-2])
    if below.any() or above.any():
        raise ValueError(
            f"{int(below.sum() + above.sum())} of {len(target)} target {dim} labels "
            f"lie beyond the source's range [{order[0]}, {order[-1]}] (target "
            f"range [{target.min()}, {target.max()}]); the source does not cover "
            "the target domain, so check its extent and conventions such as "
            f"longitude 0..360 versus -180..180; {_REGRID_HINT}"
        )
    index = np.abs(source[:, None] - target[None, :]).argmin(axis=0)
    return index


# Private alias kept for recipes written before ``map_field`` was public.
_map_field = map_field


def _require_unforced(prognostic: Any, workflow: str) -> None:
    """Reject forced models, whose iterators need forcing sources ``run.*`` lacks."""
    forcing_coords = getattr(prognostic, "forcing_coords", None)
    if callable(forcing_coords) and forcing_coords() is not None:
        raise ValueError(
            f"run.{workflow} does not supply forcing, but "
            f"{type(prognostic).__name__} declares forcing_coords(). Use "
            "PrognosticRunner with Pipeline, which fetches each forcing slot "
            "from the model's recommended sources or the forcing= argument."
        )


def _require_analysis_time_input(inputs: xr.DataArray) -> None:
    """Reject models whose input ends before the analysis time.

    The workflows write the initial condition and size stores from lead time
    zero, so a model initialized from an earlier background cannot run under
    them; ``PrognosticRunner`` handles such models.
    """
    leads = inputs.coords["lead_time"].values if "lead_time" in inputs.dims else None
    if leads is not None and leads[-1] < np.timedelta64(0, "s"):
        raise ValueError(
            f"Input lead_time {leads} ends before the analysis time, which "
            "run.deterministic, run.diagnostic and run.ensemble do not support; "
            "use PrognosticRunner with Pipeline"
        )


def _lead_times(
    signature: xr.DataArray, nsteps: int, step: np.timedelta64 | None = None
) -> np.ndarray:
    """Return a rollout's full lead-time extent from one step's declaration.

    ``step`` is the interval each model step advances, by default the
    declaration's last lead time (an input ending at zero).
    """
    leads = signature.coords["lead_time"].values
    step = leads[-1] if step is None else step
    return np.concatenate(
        [
            np.zeros(1, dtype=leads.dtype),
            *(leads + step * i for i in range(nsteps)),
        ]
    )


def _io_template(
    signature: xr.DataArray,
    nsteps: int,
    leading: Mapping[str, np.ndarray],
    output_coords: CoordSystem,
    lead_time: np.ndarray | None = None,
) -> xr.DataArray:
    """Plan an IO template for a rollout from one step's output declaration.

    ``output_coords`` restricts the stored domain and variables the way the
    workflows have always allowed: spatial and variable selections are applied to
    the signature, since :func:`output_template` only replaces non-spatial labels.
    The full rollout lead-time extent comes from ``_lead_times`` unless ``lead_time``
    is supplied (e.g. the diagnostic workflow stores the prognostic rollout extent).
    """
    # Selections along leading dims (time, ensemble) apply to `leading` below; the
    # signature only has its dynamic batch prefix there
    signature = map_field(
        signature,
        OrderedDict((k, v) for k, v in output_coords.items() if k in signature.dims),
    )
    leads = np.asarray(
        output_coords.get(
            "lead_time",
            lead_time if lead_time is not None else _lead_times(signature, nsteps),
        )
    )
    leading = {
        dim: np.asarray(output_coords.get(dim, values))
        for dim, values in leading.items()
    }
    # output_template rejects spatial replacements; map_field already applied them.
    # Only lead_time is a fixed-dim replacement here; time/ensemble are leading dims.
    if "lead_time" not in signature.dims:
        # A diagnostic may drop lead time; the store keeps the rollout's axis, so the
        # driver labels each write with it (see _with_lead_time)
        leading = {**leading, "lead_time": leads}
        return output_template(signature, leading=leading)
    return output_template(signature, leading=leading, coords={"lead_time": leads})


def _publishes_initial_condition(inputs: xr.DataArray, outputs: xr.DataArray) -> bool:
    """Whether an input slot's final frame can open the output slot's stream.

    It can when the input ends at the analysis time and the output holds the
    input's fields (see :func:`_holds_input_fields`). Models whose inputs are not
    among their outputs, or end before the analysis time, start at the first
    forecast.
    """
    if "lead_time" not in inputs.dims:
        return False
    # An input ending before zero is a background, not the analysis-time state.
    if inputs.coords["lead_time"].values[-1] < np.timedelta64(0, "s"):
        return False
    return _holds_input_fields(inputs, outputs)


def _holds_input_fields(inputs: xr.DataArray, outputs: xr.DataArray) -> bool:
    """Whether the output declares every input variable, on the input's grid.

    Outputs the input lacks, such as diagnostics, are allowed. This makes the
    input slot a state slot, whatever lead times it covers.
    """
    if "lead_time" not in inputs.dims or "variable" not in inputs.dims:
        return False
    variables = {str(v) for v in inputs.coords["variable"].values}
    if not variables <= {str(v) for v in outputs.coords["variable"].values}:
        return False
    try:
        return same_geometry(infer_grid(inputs), infer_grid(outputs))
    except Exception:  # noqa: BLE001 - not every signature carries a grid
        fixed = ("batch", "time", "lead_time", "variable")
        dims = [d for d in inputs.dims if d not in fixed]
        return dims == [d for d in outputs.dims if d not in fixed] and all(
            np.array_equal(inputs.coords[d].values, outputs.coords[d].values)
            for d in dims
            if d in inputs.coords and d in outputs.coords
        )


def _with_lead_time(x: xr.DataArray, lead_time: np.ndarray) -> xr.DataArray:
    """Restore the lead-time axis a diagnostic dropped, after the leading ``time`` axis."""
    if "lead_time" in x.dims:
        return x
    position = x.get_axis_num("time") + 1 if "time" in x.dims else 0
    return x.expand_dims(lead_time=np.atleast_1d(lead_time), axis=position)


def _rollout(
    prognostic: Any, x: xr.DataArray, publish_initial_condition: bool = True
) -> Generator[xr.DataArray, None, None]:
    """Iterate a rollout, publishing the initial condition as its first field.

    ``create_iterator`` yields forecasts only, so the initial condition is taken
    from ``x``. Pass ``publish_initial_condition=False`` when resuming a rollout
    whose initial condition has already been written.
    """
    iterator = prognostic.create_iterator(x)
    try:
        if publish_initial_condition:
            yield initial_condition(x)
        yield from iterator
    finally:
        iterator.close()
