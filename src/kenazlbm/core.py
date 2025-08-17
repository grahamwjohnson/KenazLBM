# src/kenazlbm/core.py
import os, glob, re
import torch
from .hubconfig import _get_conda_cache, CONFIGS, _load_models

# ---- Global cache dir ----
ENV_PREFIX = os.environ.get("CONDA_PREFIX")
if ENV_PREFIX is None:
    raise EnvironmentError("No Conda environment detected. Activate your environment first.")
CACHE_DIR = os.path.join(ENV_PREFIX, "kenazlbm_models")
os.makedirs(CACHE_DIR, exist_ok=True)
# --------------------------

def prefetch_models(codename='commongonolek_sheldrake', force=False):
    """
    Prefetches all pretrained models and caches them in CACHE_DIR.
    """
    if os.listdir(CACHE_DIR) and not force:
        print(f"Models already exist in {CACHE_DIR}. Use --force to redownload.")
        return CACHE_DIR
    
    print(f"Caching KenazLBM models for '{codename}' in {CACHE_DIR} ...")

    # Override the hub cache location temporarily
    torch.hub._get_torch_home = lambda: CACHE_DIR

    # Load all models (BSE, Discriminator, BSP, BSV, SOM)
    _load_models(
        codename=codename,
        gpu_id='cpu',
        pretrained=True,
        load_bse=True,
        load_discriminator=True,
        load_bsp=True,
        load_bsv=True,
        load_som=True
    )

    print("All models prefetched and cached successfully.")
    return CACHE_DIR


def check_models(codename='commongonolek_sheldrake'):
    """
    Checks which pretrained models are present in CACHE_DIR and prints info.
    """
    if codename not in CONFIGS:
        print(f"Codename '{codename}' not found. Available: {list(CONFIGS.keys())}")
        return

    config = CONFIGS[codename]
    print(f"Checking cached models in: {CACHE_DIR}\n")

    model_files = [
        config.get('bse_weight_file'),
        config.get('disc_weight_file'),
        config.get('bsp_weight_file'),
        config.get('bsv_weight_file')
    ]

    for f in model_files:
        if f is None:
            continue
        path = os.path.join(CACHE_DIR, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"{f}: FOUND ({size_mb:.2f} MB)")
        else:
            print(f"{f}: MISSING")

    online_files = [
        config.get('som_file'),
        config.get('som_axis_file')
    ]

    for f in online_files:
        if f is None:
            continue
        print(f"{f}: ONLINE (not cached locally)")

import os
import glob
import re

def validate_directory_structure(input_root="raw", file_pattern="*_pp.pkl"):
    """
    Validates files exist under <input_root>/*/<file_pattern> and that filenames
    start with <subjectID>_MMDDYYYY_HHMMSSdd, with arbitrary suffix according
    to file_pattern.

    Args:
        input_root (str): Root directory (default "raw")
        file_pattern (str): Glob pattern for files (e.g., "*.edf", "*_pp.pkl")

    Raises:
        FileNotFoundError: If no files match.
        ValueError: If any filenames fail the naming convention.

    Returns:
        bool: True if all files are valid.
    """
    # Separate stem and extension
    stem, ext = os.path.splitext(file_pattern)
    if not ext:
        raise ValueError(f"File pattern must include an extension, got: {file_pattern}")
    ext = ext.lstrip(".")  # normalize extension

    # Convert file_pattern wildcards to regex
    # Escape other characters, replace * with .*
    stem_regex = re.escape(stem).replace(r"\*", ".*")

    # Regex for filename: <subjectID>_MMDDYYYY_HHMMSSdd + stem pattern + .ext
    filename_regex = re.compile(
        rf"^[A-Za-z0-9]+_\d{{8}}_\d{{8}}{stem_regex}\.{re.escape(ext)}$",
        re.IGNORECASE
    )

    # Collect all files under <input_root>/*/
    candidate_files = glob.glob(os.path.join(input_root, "*", "*"))
    all_files = [f for f in candidate_files if f.lower().endswith(f".{ext.lower()}")]

    if not all_files:
        raise FileNotFoundError(f"ERROR: No files found with extension '.{ext}' under {input_root}")

    invalid_files = [f for f in all_files if not filename_regex.match(os.path.basename(f))]

    if invalid_files:
        msg = "\n".join("  " + f for f in invalid_files)
        raise ValueError(
            f"ERROR: The following files have invalid names (expected <subjectID>_MMDDYYYY_HHMMSSdd{file_pattern[1:]}):\n{msg}"
        )

    print(f"All {len(all_files)} files have valid names and directory structure.")
    return True

def _check_cache_files(codename, keys):
    """Helper: check if given weight files exist in CACHE_DIR."""
    config = CONFIGS.get(codename, {})
    required_files = [config.get(k) for k in keys if config.get(k)]
    return all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in required_files)


def run_bse(in_dir, out_dir=None, codename='commongonolek_sheldrake'):
    """
    Run Brain-State Embedder (BSE) inference.

    Expects files in the format:
        <dir>/<subject_id>/*_pp.pkl

    Args:
        in_dir (str): Input directory containing preprocessed pickle files.
        out_dir (str, optional): Output directory to save results.
                                 If None, defaults to the input directory.
        codename (str): Model codename to load (default: commongonolek_sheldrake).

    Raises:
        FileNotFoundError: If the input directory does not exist.

    Notes:
        - Produces *_bse outputs like:
          <dir>/<subject_id>/*_bse.pkl
    """
    if out_dir is None:
        out_dir = in_dir
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    torch.hub._get_torch_home = lambda: CACHE_DIR

    if _check_cache_files(codename, ['bse_weight_file']):
        print(f"Using cached BSE model from {CACHE_DIR}")
    else:
        print(f"Downloading BSE model to {CACHE_DIR} ...")

    bse, _, _, _, _, _ = _load_models(
        codename=codename,
        gpu_id='cpu',
        pretrained=True,
        load_bse=True,
        load_discriminator=False,
        load_bsp=False,
        load_bsv=False,
        load_som=False
    )

    subject_dirs = [d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)]
    print(f"Found {len(subject_dirs)} subject(s): {[os.path.basename(d) for d in subject_dirs]}")

    for subj_path in subject_dirs:
        subject_id = os.path.basename(subj_path)
        out_subj_dir = os.path.join(out_dir, subject_id)
        os.makedirs(out_subj_dir, exist_ok=True)

        pp_files = glob.glob(os.path.join(subj_path, "*_pp.pkl"))
        print(f"Processing subject '{subject_id}' with {len(pp_files)} file(s).")
        for infile in pp_files:
            filename = os.path.splitext(os.path.basename(infile))[0]
            outfile = os.path.join(out_subj_dir, f"{filename}_bse.pkl")

            print(f"Running BSE on {infile} -> {outfile}")
            result = f"BSE output of {infile}"  # dummy inference
            with open(outfile, 'w') as f:
                f.write(result)


def run_bsp(in_dir, out_dir=None, codename='commongonolek_sheldrake'):
    """
    Run Brain-State Predictor (BSP) inference.

    Expects files in the format:
        <dir>/<subject_id>/*_pp_bse.pkl

    Args:
        in_dir (str): Input directory containing BSE output pickle files.
        out_dir (str, optional): Output directory to save results.
                                 If None, defaults to the input directory.
        codename (str): Model codename to load (default: commongonolek_sheldrake).

    Raises:
        FileNotFoundError: If the input directory does not exist.

    Notes:
        - Produces *_bsp outputs like:
          <dir>/<subject_id>/*_bsp.pkl
    """
    if out_dir is None:
        out_dir = in_dir
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    torch.hub._get_torch_home = lambda: CACHE_DIR

    if _check_cache_files(codename, ['bsp_weight_file']):
        print(f"Using cached BSP model from {CACHE_DIR}")
    else:
        print(f"Downloading BSP model to {CACHE_DIR} ...")

    _, _, bsp, _, _, _ = _load_models(
        codename=codename,
        gpu_id='cpu',
        pretrained=True,
        load_bse=False,
        load_discriminator=False,
        load_bsp=True,
        load_bsv=False,
        load_som=False
    )

    subject_dirs = [d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)]
    print(f"Found {len(subject_dirs)} subject(s): {[os.path.basename(d) for d in subject_dirs]}")

    for subj_path in subject_dirs:
        subject_id = os.path.basename(subj_path)
        out_subj_dir = os.path.join(out_dir, subject_id)
        os.makedirs(out_subj_dir, exist_ok=True)

        bse_files = glob.glob(os.path.join(subj_path, "*_pp_bse.pkl"))
        print(f"Processing subject '{subject_id}' with {len(bse_files)} file(s).")
        for infile in bse_files:
            filename = os.path.splitext(os.path.basename(infile))[0]
            outfile = os.path.join(out_subj_dir, f"{filename}_bsp.pkl")

            print(f"Running BSP on {infile} -> {outfile}")
            result = f"BSP output of {infile}"  # dummy inference
            with open(outfile, 'w') as f:
                f.write(result)


def run_bsv(in_dir, out_dir=None, codename='commongonolek_sheldrake'):
    """
    Run Brain-State Visualizer (BSV) inference.

    Expects files in the format:
        <dir>/<subject_id>/*_pp_bse.pkl

    Args:
        in_dir (str): Input directory containing BSE output pickle files.
        out_dir (str, optional): Output directory to save visualization results.
                                 If None, defaults to the input directory.
        codename (str): Model codename to load (default: commongonolek_sheldrake).

    Raises:
        FileNotFoundError: If the input directory does not exist.

    Notes:
        - Produces *_bsv outputs like:
          <dir>/<subject_id>/*_bsv.pkl
    """
    if out_dir is None:
        out_dir = in_dir
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    torch.hub._get_torch_home = lambda: CACHE_DIR

    if _check_cache_files(codename, ['bsp_weight_file', 'bsv_weight_file']):
        print(f"Using cached BSP + BSV models from {CACHE_DIR}")
    else:
        print(f"Downloading BSP + BSV models to {CACHE_DIR} ...")

    _, _, bsp, bsv, _, _ = _load_models(
        codename=codename,
        gpu_id='cpu',
        pretrained=True,
        load_bse=False,
        load_discriminator=False,
        load_bsp=True,
        load_bsv=True,
        load_som=False
    )

    subject_dirs = [d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)]
    print(f"Found {len(subject_dirs)} subject(s): {[os.path.basename(d) for d in subject_dirs]}")

    for subj_path in subject_dirs:
        subject_id = os.path.basename(subj_path)
        out_subj_dir = os.path.join(out_dir, subject_id)
        os.makedirs(out_subj_dir, exist_ok=True)

        bse_files = glob.glob(os.path.join(subj_path, "*_pp_bse.pkl"))
        print(f"Processing subject '{subject_id}' with {len(bse_files)} file(s).")
        for infile in bse_files:
            filename = os.path.splitext(os.path.basename(infile))[0]
            outfile = os.path.join(out_subj_dir, f"{filename}_bsv.pkl")

            print(f"Running BSV on {infile} -> {outfile}")
            result = f"BSV output of {infile}"  # dummy inference
            with open(outfile, 'w') as f:
                f.write(result)