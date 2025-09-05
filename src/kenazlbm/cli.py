# src/kenazlbm/cli.py
import argparse
from .preprocess import preprocess_directory
from .core import run_bse, run_bsp, run_bsv, check_models, prefetch_models

def main(): 
    parser = argparse.ArgumentParser(description="KenazLBM CLI - run inference pipelines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add subparser
    # ----------------- prefetch_models command -----------------
    parser_prefetch = subparsers.add_parser("prefetch_models", help="Download and cache all pretrained models locally")
    parser_prefetch.add_argument("--force", action="store_true", help="Force redownload even if models exist")

    parser_check = subparsers.add_parser( "check_models", help="Check which pretrained models are cached in the Conda environment")

    parser_pre = subparsers.add_parser("preprocess", help="Preprocess input files: <dir>/<subject_id>/*.[edf|EDF]")
    parser_pre.add_argument("--input", type=str, required=True, help="Input directory formatted as: <dir>/<subject_id>/*.[edf|EDF]")
    parser_pre.add_argument("--eq_hrs", type=str, default=None, help="Optional time to use for equalization calculations (default: 24 hours)")
    parser_pre.add_argument("--checkpoint", type=str, default=0, help="Optional checkpoint to restart preprocessing from (default: 0, start from beginning; 1, after montage and filtering; 2, after equalization calculations)")

    parser_bse = subparsers.add_parser("run_bse", help="Run Brain-State Embedder inference")
    parser_bse.add_argument("--input", type=str, required=True, help="Input directory formatted as: <dir>/<subject_id>/preprocessed_epoched_data/*_bipole_scaled_filtered_data.pkl")

    parser_bsp = subparsers.add_parser("run_bsp", help="Run Brain-State Predictor inference")
    parser_bsp.add_argument("--input", type=str, required=True, help="Input directory formatted as: <dir>/<subject_id>/bse/*_bipole_scaled_filtered_data_BSE.pkl")

    parser_bsv = subparsers.add_parser("run_bsv", help="Run Brain-State Visualizer inference: <dir>/<subject_id>/*_pp_bse.pkl")
    parser_bsv.add_argument("--input", type=str, required=True, help="Input directory formatted as: <dir>/<subject_id>/bs*/*_bipole_scaled_filtered_data_BS*.pkl")

    args = parser.parse_args()
    if args.command == "prefetch_models": prefetch_models(force=args.force)
    elif args.command == "check_models": check_models()
    elif args.command == "preprocess": preprocess_directory(args.input, args.eq_hrs, args.checkpoint)
    elif args.command == "run_bse": run_bse(args.input)
    elif args.command == "run_bsp": run_bsp(args.input)
    elif args.command == "run_bsv": run_bsv(args.input)
    else: parser.print_help()

if __name__ == "__main__": main()
