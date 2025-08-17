# src/kenazlbm/core.py
import os
import torch
from .hubconfig import _get_conda_cache, CONFIGS,  _load_models

def prefetch_models(codename='commongonolek_sheldrake', env_prefix=None, force=False):
    """
    Prefetches all pretrained models and caches them in a folder inside the Conda environment.
    
    Args:
        codename (str): Model configuration to prefetch.
        env_prefix (str): Path to Conda environment. Defaults to current env.
        force (bool): If True, redownload even if models already exist.
    """
    import os, torch
    from .hubconfig import _load_models

    if env_prefix is None:
        env_prefix = os.environ.get('CONDA_PREFIX')
        if env_prefix is None:
            raise EnvironmentError("No Conda environment detected. Activate your environment first.")
    
    cache_dir = os.path.join(env_prefix, "kenazlbm_models")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check for existing files
    existing_files = os.listdir(cache_dir)
    if existing_files and not force:
        print(f"Models already exist in {cache_dir}. Use --force to redownload.")
        return cache_dir
    
    print(f"Caching KenazLBM models for '{codename}' in {cache_dir} ...")

    # Override the hub cache location temporarily
    torch.hub._get_torch_home = lambda: cache_dir  # forces torch to store checkpoints here

    # Load all models (BSE, Discriminator, BSP, BSV, SOM)
    bse, disc, bsp, bsv, som, config = _load_models(
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
    return cache_dir

def check_models(codename='commongonolek_sheldrake'):
    """
    Checks which pretrained models are present in the Conda cache and prints info.
    """
    if codename not in CONFIGS:
        print(f"Codename '{codename}' not found. Available: {list(CONFIGS.keys())}")
        return

    config = CONFIGS[codename]
    cache_dir = _get_conda_cache()
    print(f"Checking cached models in: {cache_dir}\n")

    model_files = [
        config.get('bse_weight_file'),
        config.get('disc_weight_file'),
        config.get('bsp_weight_file'),
        config.get('bsv_weight_file')
    ]

    for f in model_files:
        if f is None:
            continue
        path = os.path.join(cache_dir, f)
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

def preprocess(raw_dir):
    """
    Example preprocessing function.
    Replace with actual preprocessing logic.
    """
    print(f"Preprocessing input file: {raw_dir}")
    # Add your preprocessing code here

def run_bse(preprocessed_dir):
    print("Running BSE on the following directory:", preprocessed_dir)
    print("Need to code")

def run_bsp(postbse_dir):
    print("Running BSP on the following directory:", postbse_dir)
    print("Need to code")

def run_bsv(postbsp_dir):
    print("Running BSV on the following directory:", postbsp_dir)
    print("Need to code")