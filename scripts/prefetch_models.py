# src/kenazlbm/prefetch_models.py
import os
import torch
from kenazlbm.hubconfig import CONFIGS, _load_models

def prefetch_models(codename='commongonolek_sheldrake', env_prefix=None):
    """
    Prefetches all pretrained models and caches them in a folder inside the Conda environment.
    
    Args:
        codename (str): Model configuration to prefetch.
        env_prefix (str): Path to Conda environment. Defaults to current env.
    """
    if env_prefix is None:
        env_prefix = os.environ.get('CONDA_PREFIX')
        if env_prefix is None:
            raise EnvironmentError("No Conda environment detected. Activate your environment first.")
    
    cache_dir = os.path.join(env_prefix, "kenazlbm_models")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Caching KenazLBM models for '{codename}' in {cache_dir} ...")

    # Override the hub cache location temporarily
    torch.hub._get_torch_home = lambda: cache_dir  # forces torch to store checkpoints here

    # Load all models (BSE, Discriminator, BSP, BSV, SOM)
    bse, disc, bsp, bsv, som, config = _load_models(codename=codename, gpu_id='cpu',
                                                    pretrained=True,
                                                    load_bse=True,
                                                    load_discriminator=True,
                                                    load_bsp=True,
                                                    load_bsv=True,
                                                    load_som=True)
    print("All models prefetched and cached successfully.")
    return cache_dir
