# src/kenazlbm/cli.py
import argparse
from .preprocess import preprocess_directory, validate_directory_structure
from .core import run_bse, run_bsp, run_bsv

def main():
    parser = argparse.ArgumentParser(description="KenazLBM CLI - run inference pipelines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_pre = subparsers.add_parser("preprocess", help="Preprocess input files: <dir>/<subject_id>/*.[edf|EDF]")
    parser_pre.add_argument("--input", type=str, required=True, help="Input directory: <dir>/<subject_id>/*.[edf|EDF]")
    parser_pre.add_argument("--output", type=str, default=None, help="Optional output directory (default: same as input)")

    parser_bse = subparsers.add_parser("run_bse", help="Run Brain-State Embedder inference: <dir>/<subject_id>/pp_*.pkl")
    parser_bse.add_argument("--input", type=str, required=True, help="Input directory: <dir>/<subject_id>/pp_*.pkl")
    parser_bse.add_argument("--output", type=str, default=None, help="Optional output directory (default: same as input)")

    parser_bsp = subparsers.add_parser("run_bsp", help="Run Brain-State Predictor inference: <dir>/<subject_id>/pp_bse_*.pkl")
    parser_bsp.add_argument("--input", type=str, required=True, help="Input directory: <dir>/<subject_id>/pp_bse_*.pkl")
    parser_bsp.add_argument("--output", type=str, default=None, help="Optional output directory (default: same as input)")

    parser_bsv = subparsers.add_parser("run_bsv", help="Run Brain-State Visualizer inference: <dir>/<subject_id>/pp_bse_bsp_*.pkl")
    parser_bsv.add_argument("--input", type=str, required=True, help="Input directory: <dir>/<subject_id>/pp_bse_bsp_*.pkl")
    parser_bsv.add_argument("--output", type=str, default=None, help="Optional output directory (default: same as input)")

    args = parser.parse_args()
    if args.command == "preprocess": validate_directory_structure(args.input); preprocess_directory(args.input, args.output)
    elif args.command == "run_bse": run_bse(args.input, args.output)
    elif args.command == "run_bsp": run_bsp(args.input, args.output)
    elif args.command == "run_bsv": run_bsv(args.input, args.output)
    else: parser.print_help()

if __name__ == "__main__": main()
