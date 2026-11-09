from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


US_FLOW_UNITS = {"CFS", "GPM", "MGD"}


def _section_lines(lines: list[str], name: str) -> tuple[int, int]:
    section_start = None
    for index, line in enumerate(lines):
        match = re.match(r"^\s*\[([^]]+)\]", line)
        if match:
            if section_start is not None:
                return section_start, index
            if match.group(1).strip().upper() == name.upper():
                section_start = index + 1
    if section_start is not None:
        return section_start, len(lines)
    raise ValueError(f"SWMM input must contain a [{name}] section.")


def _parse_interval_minutes(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid SWMM rainfall interval {value!r}; expected HH:MM.")
    return int(parts[0]) * 60 + int(parts[1])


def _set_simulation_period(template_text: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    lines = template_text.splitlines()
    section_start, section_end = _section_lines(lines, "OPTIONS")
    updates = {
        "START_DATE": start.strftime("%m/%d/%Y"),
        "START_TIME": start.strftime("%H:%M:%S"),
        "REPORT_START_DATE": start.strftime("%m/%d/%Y"),
        "REPORT_START_TIME": start.strftime("%H:%M:%S"),
        "END_DATE": end.strftime("%m/%d/%Y"),
        "END_TIME": end.strftime("%H:%M:%S"),
    }
    found = set()
    for index in range(section_start, section_end):
        fields = lines[index].split()
        if fields and fields[0].upper() in updates:
            key = fields[0].upper()
            lines[index] = f"{fields[0]:<22}{updates[key]}"
            found.add(key)
    insertion = [f"{key:<22}{value}" for key, value in updates.items() if key not in found]
    lines[section_end:section_end] = insertion
    return "\n".join(lines) + "\n"


def inject_forecast_rainfall(
    template_text: str,
    gage_locations: pd.DataFrame,
    precipitation_mm: xr.DataArray,
    flow_units: str,
) -> str:
    """Replace selected SWMM rain-gage time series with member-specific forecast depths."""
    required = {"gage_name", "series_name", "lat", "lon"}
    if not required.issubset(gage_locations.columns):
        raise ValueError(f"Rain-gage CSV must contain columns {sorted(required)}.")
    if gage_locations["gage_name"].duplicated().any() or gage_locations["series_name"].duplicated().any():
        raise ValueError("Each forecast rain gage must have a unique gage_name and series_name.")
    interval_hours = float(precipitation_mm.attrs["accumulation_duration_hours"])
    interval_minutes = int(round(interval_hours * 60))
    lines = template_text.splitlines()
    gage_start, gage_end = _section_lines(lines, "RAINGAGES")
    referenced_series = {}
    gage_intervals = {}
    for line in lines[gage_start:gage_end]:
        fields = line.split()
        if not fields or fields[0].startswith(";"):
            continue
        if len(fields) < 6:
            raise ValueError(f"Malformed SWMM rain-gage row: {line}")
        if fields[0] in set(gage_locations["gage_name"].astype(str)):
            if fields[1].upper() != "VOLUME" or fields[4].upper() != "TIMESERIES":
                raise ValueError(f"Rain gage {fields[0]} must use VOLUME format and an inline TIMESERIES source.")
            referenced_series[fields[0]] = fields[5]
            gage_intervals[fields[0]] = _parse_interval_minutes(fields[2])

    if len(referenced_series) != len(gage_locations):
        missing = sorted(set(gage_locations["gage_name"].astype(str)) - set(referenced_series))
        raise ValueError(f"Rain gages not found in the SWMM [RAINGAGES] section: {missing}")
    if set(referenced_series.values()) != set(gage_locations["series_name"].astype(str)):
        raise ValueError("The series_name values in the rain-gage CSV must match the series referenced by [RAINGAGES].")
    if any(value != interval_minutes for value in gage_intervals.values()):
        raise ValueError(
            f"SWMM rain-gage intervals must match the forecast accumulation interval ({interval_hours:g} hours)."
        )

    series_names = set(gage_locations["series_name"].astype(str))
    _section_lines(lines, "TIMESERIES")
    new_rows = []
    for row in gage_locations.itertuples(index=False):
        point = precipitation_mm.sel(lat=float(row.lat), lon=float(row.lon), method="nearest")
        for lead_hours in precipitation_mm["lead_time_hours"].values:
            valid_time = pd.Timestamp(precipitation_mm["valid_time"].sel(lead_time_hours=lead_hours).values)
            value = float(point.sel(lead_time_hours=lead_hours).values)
            if not np.isfinite(value):
                raise ValueError(f"Forecast precipitation is missing for rain gage {row.gage_name} at {valid_time}.")
            if flow_units.upper() in US_FLOW_UNITS:
                value /= 25.4
            new_rows.append((str(row.series_name), valid_time, value))

    rewritten = []
    inside_timeseries = False
    inserted = False
    for line in lines:
        section = re.match(r"^\s*\[([^]]+)\]", line)
        if section:
            if inside_timeseries and not inserted:
                rewritten.extend(_format_timeseries_rows(new_rows))
                inserted = True
            inside_timeseries = section.group(1).strip().upper() == "TIMESERIES"
            rewritten.append(line)
            continue
        if inside_timeseries:
            fields = line.split()
            if fields and fields[0] in series_names:
                continue
        rewritten.append(line)
    if inside_timeseries and not inserted:
        rewritten.extend(_format_timeseries_rows(new_rows))
    return "\n".join(rewritten) + "\n"


def _format_timeseries_rows(rows) -> list[str]:
    output = []
    for series_name, valid_time, amount in rows:
        output.append(f"{series_name} {valid_time:%m/%d/%Y} {valid_time:%H:%M} {amount:.6f}")
    return output


def run_swmm_ensemble(
    forcing_path: str | Path,
    template_path: str | Path,
    gage_csv_path: str | Path,
    output_dir: str | Path,
    node_ids: list[str] | None = None,
) -> Path:
    """Run one SWMM simulation per Earth2 precipitation ensemble member."""
    try:
        from pyswmm import Nodes, Simulation
    except ImportError as exc:
        raise RuntimeError("Install the optional SWMM dependency with `pip install -e '.[swmm]'`.") from exc

    forcing_path, template_path, gage_csv_path = map(Path, (forcing_path, template_path, gage_csv_path))
    output_dir = Path(output_dir)
    if not template_path.exists():
        raise FileNotFoundError(f"Calibrated SWMM input file not found: {template_path}")
    with xr.open_dataset(forcing_path) as dataset:
        if "precipitation_mm" not in dataset:
            raise ValueError("Normalized forecast must contain 'precipitation_mm'.")
        precipitation = dataset["precipitation_mm"].load()
        if precipitation.attrs.get("units") != "mm":
            raise ValueError("Normalized precipitation must use millimeters.")
        gages = pd.read_csv(gage_csv_path)
        init_time = pd.Timestamp(dataset.attrs["forecast_initialization_time"])
        end_time = pd.Timestamp(dataset["valid_time"].max().values)
        template_text = _set_simulation_period(template_path.read_text(), init_time, end_time)
        options_start, options_end = _section_lines(template_text.splitlines(), "OPTIONS")
        option_rows = [line.split() for line in template_text.splitlines()[options_start:options_end] if line.strip() and not line.lstrip().startswith(";")]
        options = {fields[0].upper(): fields[1] for fields in option_rows if len(fields) > 1}
        flow_units = options.get("FLOW_UNITS", "CFS").upper()
        if flow_units not in US_FLOW_UNITS | {"CMS", "LPS", "MLD"}:
            raise ValueError(f"Unsupported SWMM FLOW_UNITS value: {flow_units}")

        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        member_count = precipitation.sizes["ensemble"]
        for member in range(member_count):
            member_forcing = precipitation.isel(ensemble=member)
            member_text = inject_forecast_rainfall(template_text, gages, member_forcing, flow_units)
            member_dir = output_dir / f"member_{member:03d}"
            member_dir.mkdir(parents=True, exist_ok=True)
            input_path = member_dir / "forecast.inp"
            input_path.write_text(member_text)
            with Simulation(str(input_path)) as simulation:
                available_nodes = Nodes(simulation)
                active_nodes = {node_id: available_nodes[node_id] for node_id in (node_ids or [])}
                if not active_nodes:
                    raise ValueError("Provide at least one monitored SWMM node with --nodes.")
                for _ in simulation:
                    for node_id, node in active_nodes.items():
                        rows.append({
                            "ensemble": member,
                            "time": simulation.current_time.isoformat(),
                            "node_id": node_id,
                            "depth": float(node.depth),
                            "flooding_rate": float(node.flooding),
                        })

    timeseries_path = output_dir / "swmm_node_timeseries.csv"
    results = pd.DataFrame(rows)
    results.to_csv(timeseries_path, index=False)
    summary = results.groupby(["ensemble", "node_id"], as_index=False).agg(
        maximum_depth=("depth", "max"),
        maximum_flooding_rate=("flooding_rate", "max"),
        flooded=("flooding_rate", lambda values: bool((values > 0).any())),
    )
    summary_path = output_dir / "swmm_member_summary.csv"
    summary.to_csv(summary_path, index=False)
    probability = summary.groupby("node_id", as_index=False).agg(
        probability_any_flooding=("flooded", "mean"),
        maximum_member_depth=("maximum_depth", "max"),
        ensemble_members=("ensemble", "nunique"),
    )
    probability.to_csv(output_dir / "swmm_flood_probability.csv", index=False)
    metadata = {
        "forcing": str(forcing_path.resolve()),
        "swmm_template": str(template_path.resolve()),
        "rain_gages": int(len(gages)),
        "ensemble_members": int(member_count),
        "flow_units": flow_units,
        "rain_depth_units": "inches" if flow_units in US_FLOW_UNITS else "millimeters",
        "precipitation_interval_hours": float(precipitation.attrs["accumulation_duration_hours"]),
        "depth_units": "feet" if flow_units in US_FLOW_UNITS else "meters",
        "flooding_rate_units": flow_units,
        "temporal_resolution_warning": "Six-hour forecast accumulations are not sub-hourly rainfall detail.",
        "interpretation": "Probability of node flooding conditional on the supplied SWMM model and forecast ensemble.",
    }
    (output_dir / "swmm_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return probability


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecast members through a calibrated SWMM model.")
    parser.add_argument("--forcing", type=Path, required=True, help="Normalized Earth2 precipitation NetCDF")
    parser.add_argument("--model", type=Path, required=True, help="Calibrated SWMM .inp template")
    parser.add_argument("--gages", type=Path, required=True, help="CSV with gage_name,series_name,lat,lon")
    parser.add_argument("--nodes", required=True, help="Comma-separated SWMM node IDs to monitor")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/swmm_forecast"))
    args = parser.parse_args()
    output = run_swmm_ensemble(
        args.forcing,
        args.model,
        args.gages,
        args.output_dir,
        [node.strip() for node in args.nodes.split(",") if node.strip()],
    )
    print(f"Saved node flooding probabilities to {output}")


if __name__ == "__main__":
    main()