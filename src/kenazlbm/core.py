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

def validate_directory_structure(input_root="raw", file_pattern="*.edf"):
    """
    Validates that files exist under: <input_root>/*/<file_pattern>
    and that filenames follow the format:
        <subjectID>_MMDDYYYY_HHMMSSdd.<ext>
    where <subjectID> is alphanumeric and <ext> is taken from file_pattern.

    Args:
        input_root (str): Root directory (default "raw")
        file_pattern (str): Glob pattern for files (e.g. "*.edf", "*_pp.pkl")

    Raises:
        FileNotFoundError: If no files match.
        ValueError: If any filenames fail the naming convention.

    Returns:
        bool: True if all files are valid.
    """
    # Build the glob pattern
    pattern = os.path.join(input_root, "*", file_pattern)
    all_files = glob.glob(pattern)

    if not all_files:
        raise FileNotFoundError(f"ERROR: No files found with pattern {pattern}")

    # Extract extension (everything after last dot in file_pattern)
    _, ext = os.path.splitext(file_pattern)
    if not ext:
        raise ValueError(f"File pattern must include an extension, got: {file_pattern}")
    ext_rx = re.escape(ext.lstrip("*."))  # normalize "*.edf" -> "edf"

    # Build regex: subjectID (alnum) + '_' + MMDDYYYY + '_' + HHMMSSdd + .ext
    filename_regex = re.compile(
        rf"^[A-Za-z0-9]+_\d{{8}}_\d{{8}}\.{ext_rx}$", re.IGNORECASE
    )

    invalid_files = [f for f in all_files if not filename_regex.match(os.path.basename(f))]

    if invalid_files:
        msg = "\n".join("  " + f for f in invalid_files)
        raise ValueError(
            "ERROR: The following files have invalid names (expected "
            f"<subjectID>_MMDDYYYY_HHMMSSdd.{ext_rx}):\n{msg}"
        )

    print(f"All {len(all_files)} files have valid names and directory structure.")
    return True


def preprocess(in_dir, out_dir=None):
    """
    Preprocess input files and save the results.

    Expects files in the format:
        <dir>/<subject_id>/*.[edf|EDF]

    Args:
        in_dir (str): Input directory containing raw EDF files.
        out_dir (str, optional): Output directory to save preprocessed files.
                                 If None, defaults to the input directory.

    Raises:
        FileNotFoundError: If the input directory does not exist.

    Notes:
        - Preprocessing should produce files like:
          <dir>/<subject_id>/*_pp.pkl
        - Replace placeholder logic with actual preprocessing implementation.
    """
    if out_dir is None:
        out_dir = in_dir

    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        

    # Example preprocessing logic (replace with actual)
    print(f"Preprocessing files in {in_dir} and saving to {out_dir}")


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
        - Produces files like:
          <dir>/<subject_id>/*_pp_bse.pkl
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
    # TODO: run inference with `bse`


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
        - Produces files like:
          <dir>/<subject_id>/*_pp_bsp.pkl
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
    # TODO: run inference with `bsp`


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
        - Produces visualization outputs (format TBD, e.g. images or reports).
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
    # TODO: run inference with `bsp` and `bsv`