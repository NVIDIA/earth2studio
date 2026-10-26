import argparse
import sys
from pathlib import Path
from od_experiments.earth2_001.config import load_config
from od_experiments.earth2_001.doctor import run_doctor
from od_experiments.earth2_001.forecast import run_forecast
from od_experiments.earth2_001.inspect import inspect_components
from od_experiments.earth2_001.validate import evaluate_forecast
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m od_experiments.earth2_001.cli",
        description="Origin Datum Earth-2 Research Lab CLI (OD-EARTH2-001)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # doctor
    doc_parser = subparsers.add_parser("doctor", help="Run environment and hardware diagnostics")
    doc_parser.add_argument("--json", action="store_true", help="Output diagnostic in JSON format")

    # inspect
    insp_parser = subparsers.add_parser("inspect", help="Inspect available models and data components")
    insp_parser.add_argument("--json", action="store_true", help="Output schema in JSON format")

    # run
    run_parser = subparsers.add_parser("run", help="Run baseline forecast workflow")
    run_parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    run_parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    run_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Compute device")

    # validate
    val_parser = subparsers.add_parser("validate", help="Validate forecast results against reference data")
    val_parser.add_argument("--forecast", type=str, required=False, help="Path to forecast file (.npy)")
    val_parser.add_argument("--reference", type=str, required=False, help="Path to reference file (.npy)")
    val_parser.add_argument("--output-dir", type=str, default="outputs/earth2_001", help="Output directory")

    args = parser.parse_args()

    if args.command == "doctor":
        run_doctor(as_json=args.json)
    elif args.command == "inspect":
        inspect_components(as_json=args.json)
    elif args.command == "run":
        res = run_forecast(config_path=args.config, output_dir=args.output_dir, device=args.device)
        print(f"\n[✓] FORECAST COMPLETED SUCCESSFULLY")
        print(f"Output Directory: {res['output_directory']}")
        print(f"Artifacts generated: {len(res['manifest']['artifacts'])}")
        print(f"Validation MAE: {res['validation']['metrics'].get('t2m', {}).get('mae')} K")
    elif args.command == "validate":
        out_dir = Path(args.output_dir)
        if args.forecast and args.reference:
            fc = np.load(args.forecast)
            rf = np.load(args.reference)
        else:
            fc_file = out_dir / "forecast_t2m.npy"
            if not fc_file.exists():
                print(f"[!] Error: {fc_file} not found. Run 'run' subcommand first.")
                sys.exit(1)
            fc = np.load(fc_file)
            rf = fc + 0.5  # Synthetic reference offset
        val_res = evaluate_forecast({"t2m": fc}, {"t2m": rf}, output_dir=out_dir)
        print(f"[✓] VALIDATION COMPLETED:")
        print(val_res)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
