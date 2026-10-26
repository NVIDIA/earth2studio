import argparse
import sys
from pathlib import Path

import numpy as np

from od_experiments.earth2_001.config import load_config
from od_experiments.earth2_001.doctor import run_doctor
from od_experiments.earth2_001.forecast import run_forecast
from od_experiments.earth2_001.inspect import inspect_components
from od_experiments.earth2_001.validate import evaluate_forecast


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m od_experiments.earth2_001.cli",
        description="Origin Datum Earth-2 Research Lab CLI (OD-EARTH2-001)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # doctor
    doc_parser = subparsers.add_parser(
        "doctor", help="Run environment and hardware diagnostics"
    )
    doc_parser.add_argument(
        "--json", action="store_true", help="Output diagnostic in JSON format"
    )

    # inspect
    insp_parser = subparsers.add_parser(
        "inspect", help="Inspect available models and data components"
    )
    insp_parser.add_argument(
        "--json", action="store_true", help="Output schema in JSON format"
    )

    # run
    run_parser = subparsers.add_parser("run", help="Run baseline forecast workflow")
    run_parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yaml"
    )
    run_parser.add_argument(
        "--source",
        type=str,
        choices=["gfs", "random"],
        default=None,
        help="Override data source ('gfs' or 'random')",
    )
    run_parser.add_argument(
        "--output-dir", type=str, default=None, help="Custom output directory"
    )
    run_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )

    # validate
    val_parser = subparsers.add_parser(
        "validate", help="Validate forecast results against reference data"
    )
    val_parser.add_argument(
        "--forecast", type=str, required=False, help="Path to forecast file (.npy)"
    )
    val_parser.add_argument(
        "--reference", type=str, required=False, help="Path to reference file (.npy)"
    )
    val_parser.add_argument(
        "--lats",
        type=str,
        required=False,
        help="Path to latitude grid (.npy) for weighting",
    )
    val_parser.add_argument(
        "--output-dir", type=str, default="outputs/earth2_001", help="Output directory"
    )

    # api subcommands
    api_parser = subparsers.add_parser("api", help="Hosted NVIDIA FourCastNet API commands")
    api_sub = api_parser.add_subparsers(dest="api_command", help="API operations")

    api_sub.add_parser("doctor", help="Check NVIDIA API authentication and endpoint reachability")

    api_infer = api_sub.add_parser("infer", help="Execute hosted FourCastNet inference request")
    api_infer.add_argument("--input-id", type=int, default=0, help="Sample scenario index (0..4)")
    api_infer.add_argument("--variables", type=str, default="t2m", help="Comma-separated variables")
    api_infer.add_argument("--steps", type=int, default=1, help="Simulation steps (1 step = 6h)")
    api_infer.add_argument("--ensemble-size", type=int, default=1, help="Ensemble members (1..4)")
    api_infer.add_argument("--output-dir", type=str, default="outputs/earth2_001/api_live", help="Output directory")

    api_sub.add_parser("tests", help="Run the official 18-test live API verification suite")

    args = parser.parse_args()

    if args.command == "doctor":
        run_doctor(as_json=args.json)
    elif args.command == "inspect":
        inspect_components(as_json=args.json)
    elif args.command == "api":
        from od_experiments.earth2_001.api.artifact_processor import ForecastArtifactProcessor
        from od_experiments.earth2_001.api.fourcastnet_hosted import ForecastRequest, FourCastNetHostedClient

        client = FourCastNetHostedClient()

        if args.api_command == "doctor":
            print("[*] Checking NVIDIA Hosted FourCastNet API Configuration:")
            print(f"  • Endpoint: {client.endpoint}")
            print(f"  • Key: {client.get_masked_key()}")
            try:
                # Issue lightweight validation probe
                req = ForecastRequest(input_id=0, variables="t2m", simulation_length=1)
                resp = client.infer(req)
                print(f"  • Connection: SUCCESS (HTTP {resp.status_code})")
                print(f"  • NVCF Request ID: {resp.req_id}")
                print(f"  • NVCF Status: {resp.nvcf_status}")
            except Exception as e:
                print(f"  • Connection Error: {e}")
        elif args.api_command == "infer":
            req = ForecastRequest(
                input_id=args.input_id,
                variables=args.variables,
                simulation_length=args.steps,
                ensemble_size=args.ensemble_size,
            )
            print(f"[*] Submitting FourCastNet hosted inference (input_id={req.input_id}, vars={req.variables}, steps={req.simulation_length})...")
            resp = client.infer(req)
            out_path = Path(args.output_dir)
            proc = ForecastArtifactProcessor(out_path)
            artifacts = proc.process_response(req, resp, test_id="CLI")
            print(f"[✓] Inference successful (HTTP {resp.status_code}, NVCF Reqid: {resp.req_id})")
            print(f"Saved {len(artifacts)} artifact(s) to {out_path}:")
            for a in artifacts:
                print(f"  • {a.filename} (SHA-256: {a.sha256[:16]}..., size: {a.size_bytes} bytes)")
        elif args.api_command == "tests":
            import subprocess
            print("[*] Running official 18-test live API verification suite...")
            cmd = [sys.executable, "-m", "pytest", "od_experiments/tests/test_fourcastnet_live.py", "-v", "-s"]
            res = subprocess.run(cmd)
            sys.exit(res.returncode)
        else:
            api_parser.print_help()
    elif args.command == "run":
        res = run_forecast(
            config_path=args.config,
            output_dir=args.output_dir,
            device=args.device,
            override_source=args.source,
        )
        print(f"\n[✓] FORECAST COMPLETED SUCCESSFULLY")
        print(f"Classification: {res['classification']}")
        print(f"Output Directory: {res['output_directory']}")
        print(f"Artifacts generated: {len(res['manifest']['artifacts'])}")
        print(f"Validation Status: {res['validation'].get('validation_status')}")
        if res["validation"].get("validation_status") == "VALIDATED":
            for var, metrics in res["validation"]["metrics"].items():
                print(
                    f"  • {var.upper()}: MAE = {metrics['mae']}, RMSE = {metrics['rmse']}, Mean Bias = {metrics['mean_bias']}"
                )
        else:
            print(f"  Reason: {res['validation'].get('reason')}")
    elif args.command == "validate":
        out_dir = Path(args.output_dir)
        fc_dict = {}
        rf_dict = {}

        if args.forecast:
            fc_path = Path(args.forecast)
            if not fc_path.exists():
                print(f"[!] Error: Forecast file {fc_path} not found.")
                sys.exit(1)
            fc_dict["var"] = np.load(fc_path)
        else:
            fc_files = list(out_dir.glob("forecast_*.npy"))
            if not fc_files:
                print(
                    f"[!] Error: No forecast_*.npy files found in {out_dir}. Run 'run' subcommand first."
                )
                sys.exit(1)
            for f in fc_files:
                var_name = f.stem.replace("forecast_", "")
                fc_dict[var_name] = np.load(f)

        if args.reference:
            rf_path = Path(args.reference)
            if not rf_path.exists():
                print(f"[!] Error: Reference file {rf_path} not found.")
                sys.exit(1)
            rf_dict["var"] = np.load(rf_path)
        else:
            rf_files = list(out_dir.glob("reference_*.npy"))
            for f in rf_files:
                var_name = f.stem.replace("reference_", "")
                rf_dict[var_name] = np.load(f)

        lats = (
            np.load(args.lats)
            if args.lats and Path(args.lats).exists()
            else np.linspace(90, -90, 721)
        )

        val_res = evaluate_forecast(
            fc_dict, rf_dict if rf_dict else None, lats=lats, output_dir=out_dir
        )
        print(f"[✓] VALIDATION COMPLETED:")
        print(f"Status: {val_res.get('validation_status')}")
        if val_res.get("validation_status") == "VALIDATED":
            for var, metrics in val_res.get("metrics", {}).items():
                print(
                    f"  • {var.upper()}: MAE = {metrics['mae']}, RMSE = {metrics['rmse']}, Mean Bias = {metrics['mean_bias']}"
                )
        else:
            print(f"  Note: {val_res.get('reason')}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
