# src/kenazlbm/cli.py
import argparse
from .preprocess import preprocess_directory
from .core import run_bse, run_bsp, run_bsv, check_models, prefetch_models, validate_directory_structure

def main():
    parser = argparse.ArgumentParser(description="KenazLBM CLI - run inference pipelines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add subparser
    # ----------------- prefetch_models command -----------------
    parser_prefetch = subparsers.add_parser("prefetch_models", help="Download and cache all pretrained models locally")
    parser_prefetch.add_argument("--force", action="store_true", help="Force redownload even if models exist")

    parser_check = subparsers.add_parser( "check_models", help="Check which pretrained models are cached in the Conda environment")

    parser_pre = subparsers.add_parser("preprocess", help="Preprocess input files: <dir>/<subject_id>/*.[edf|EDF]")
    parser_pre.add_argument("--input", type=str, required=True, help="Input directory: <dir>/<subject_id>/*.[edf|EDF]")
    parser_pre.add_argument("--output", type=str, default=None, help="Optional output directory (default: same as input)")

    parser_bse = subparsers.add_parser("run_bse", help="Run Brain-State Embedder inference: <dir>/<subject_id>/*_pp.pkl")
    parser_bse.add_argument("--input", type=str, required=True, help="Input directory: <dir>/<subject_id>/*_pp.pkl")
    parser_bse.add_argument("--output", type=str, default=None, help="Optional output directory (default: same as input)")

    parser_bsp = subparsers.add_parser("run_bsp", help="Run Brain-State Predictor inference: <dir>/<subject_id>/*_pp_bse.pkl")
    parser_bsp.add_argument("--input", type=str, required=True, help="Input directory: <dir>/<subject_id>/*_pp_bse.pkl")
    parser_bsp.add_argument("--output", type=str, default=None, help="Optional output directory (default: same as input)")

    parser_bsv = subparsers.add_parser("run_bsv", help="Run Brain-State Visualizer inference: <dir>/<subject_id>/*_pp_bse.pkl")
    parser_bsv.add_argument("--input", type=str, required=True, help="Input directory: <dir>/<subject_id>/*_pp_bse.pkl")
    parser_bsv.add_argument("--output", type=str, default=None, help="Optional output directory (default: same as input)")
    parser_bsv.add_argument("--file_pattern", type=str, default=None, help="Must specificy file_pattern because BSV can be run before or after BSP, thus can either do '*_pp_bse.pkl' or '*_pp_bse_bsp.pkl'")

    args = parser.parse_args()
    if args.command == "prefetch_models": prefetch_models(force=args.force)
    elif args.command == "check_models": check_models()
    elif args.command == "preprocess": validate_directory_structure(args.input, file_pattern="*.edf"); preprocess_directory(args.input, args.output)
    elif args.command == "run_bse": validate_directory_structure(args.input, file_pattern="*_pp.pkl"); run_bse(args.input, args.output)
    elif args.command == "run_bsp": validate_directory_structure(args.input, file_pattern="*_pp_bse.pkl"); run_bsp(args.input, args.output)
    elif args.command == "run_bsv": validate_directory_structure(args.input, file_pattern=args.file_pattern); run_bsv(args.input, args.file_pattern, args.output)
    else: parser.print_help()

if __name__ == "__main__": main()
