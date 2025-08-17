# src/kenazlbm/cli.py
import argparse
from .preprocess import preprocess_directory, validate_directory_structure
from .core import run_bse       # Brain-State Embedder (inference)
from .core import run_bsp       # Brain-State Predictor (inference)
from .core import run_bsv       # Brain-State Visualizer (inference)

def main():
    parser = argparse.ArgumentParser(description="KenazLBM CLI - run different functions from the package (inference only)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----------------- preprocess command -----------------
    parser_pre = subparsers.add_parser("preprocess", help="Preprocess input files on structured dorectory structure: <dir>/<subject_id>/*.[edf|EDF]")
    parser_pre.add_argument("--input", type=str, required=True, help="Path to input directiory (<dir>/<subject_id>/*.[edf|EDF])")
    parser_pre.add_argument("--output", type=str, default=None, help="Optional path to output directory (default: same as input directory)")

    # ----------------- run_bse command -----------------
    parser_bse = subparsers.add_parser("run_bse", help="Run Brain-State Embedder inference on structured directory: <dir>/<subject_id>/pp_*.pkl")
    parser_bse.add_argument("--input", type=str, required=True, help="Path to input directory with structure: <dir>/<subject_id>/pp_*.pkl")
    parser_pre.add_argument("--output", type=str, default=None, help="Optional path to output directory (default: same as input directory)")

    # ----------------- run_bsp command -----------------
    parser_bsp = subparsers.add_parser("run_bsp", help="Run Brain-State Predictor inference on structured directory: <dir>/<subject_id>/pp_bse_*.pkl")
    parser_bsp.add_argument("--input", type=str, required=True, help="Path to input directory with structure: <dir>/<subject_id>/pp_bse_*.pkl")
    parser_pre.add_argument("--output", type=str, default=None, help="Optional path to output directory (default: same as input directory)")

    # ----------------- run_bsv command -----------------
    parser_bsv = subparsers.add_parser("run_bsv", help="Run Brain-State Visualizer inference on structured directory: <dir>/<subject_id>/pp_bse_bsp_*.pkl")
    parser_bsv.add_argument("--input", type=str, required=True, help="Path to input directory with structure: <dir>/<subject_id>/pp_bse_bsp_*.pkl")
    parser_pre.add_argument("--output", type=str, default=None, help="Optional path to output directory (default: same as input directory)")

    # Parse args
    args = parser.parse_args()

    # Dispatch to correct function
    if args.command == "preprocess":
        validate_directory_structure(args.input)
        preprocess_directory(args.input, args.output)
    elif args.command == "run_bse":
        run_bse(args.data)
    elif args.command == "run_bsp":
        run_bsp(args.data)
    elif args.command == "run_bsv":
        run_bsv(args.input, args.verbose)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
