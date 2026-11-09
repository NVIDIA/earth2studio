from __future__ import annotations

import argparse
from pathlib import Path

from .config import FIGURES_DIR, PROJECT_ROOT
from .data_acquisition import fetch_open_datasets, write_demo_rainfall_and_terrain
from .hydrology import compute_runoff_index, compute_susceptibility
from .plotting import generate_interactive_event_map, plot_high_resolution_terrain_context, plot_rainfall_and_susceptibility, plot_susceptibility_map
from .preprocessing import acquire_high_resolution_dem, build_antecedent_wetness, build_high_resolution_terrain_screening, build_terrain_layers, build_validation_points
from .real_event import process_real_event_data, run_real_event_pipeline
from .validation import summarize_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NYC flash-flood susceptibility workflow")
    parser.add_argument("--demo", action="store_true", help="Generate a synthetic but realistic demo dataset in lieu of full public downloads.")
    parser.add_argument("--real-ida", action="store_true", help="Run the real Ida, 2021-09-01 to 2021-09-02 case-study pipeline using NOAA, USGS, NLCD, and NYC 311 sources.")
    parser.add_argument("--skip-demo", action="store_true", help="Skip data generation and assume raw inputs already exist.")
    parser.add_argument("--download-only", action="store_true", help="Download the required public datasets without running the susceptibility model.")
    parser.add_argument("--high-resolution-terrain", action="store_true", help="Fetch a clipped USGS ~10 m DEM and generate terrain-context visualizations; this is not a rainfall-driven flood forecast.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.high_resolution_terrain:
        dem_path = acquire_high_resolution_dem()
        terrain_path = build_high_resolution_terrain_screening()
        context_figure = plot_high_resolution_terrain_context()
        interactive_map = generate_interactive_event_map()
        print(f"USGS high-resolution DEM: {dem_path}")
        print(f"Terrain-only screening layer: {terrain_path}")
        print(f"Terrain context figure: {context_figure}")
        print(f"Interactive event map: {interactive_map}")
        return

    if args.real_ida:
        manifest = run_real_event_pipeline(download_only=args.download_only)
        if args.download_only:
            print("Real-event data download manifest created.")
            for key, value in manifest.items():
                print(f"- {key}: {value}")
            return
        process_real_event_data()

    elif args.demo:
        write_demo_rainfall_and_terrain()
    else:
        fetch_open_datasets()

    if args.skip_demo and not Path(PROJECT_ROOT / "data" / "raw" / "rainfall" / "demo_rainfall_total.tif").exists():
        raise FileNotFoundError("No raw data found. Re-run with --demo or add your own datasets.")

    build_terrain_layers()
    build_antecedent_wetness()
    build_validation_points()
    compute_runoff_index()
    compute_susceptibility()
    summary = summarize_validation()
    fig1 = plot_susceptibility_map()
    fig2 = plot_rainfall_and_susceptibility()
    interactive_map = generate_interactive_event_map()

    print("Flash-flood susceptibility workflow completed.")
    print(f"Susceptibility map: {fig1}")
    print(f"Runoff and susceptibility figure: {fig2}")
    print(f"Interactive event map: {interactive_map}")
    print(summary)


if __name__ == "__main__":
    main()
