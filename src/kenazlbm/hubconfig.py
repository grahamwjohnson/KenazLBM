import torch
import pickle
import requests
import os
from kenazlbm.BSE import BSE, Discriminator
from kenazlbm.BSP import BSP, BSV
from kenazlbm.ToroidalSOM_2 import ToroidalSOM_2


dependencies = ['torch', 'numpy']


CONFIGS = {
   'commongonolek_sheldrake': {


       # BSE Params
       'encode_token_samples': 1,
       'padded_channels': 256,
       'transformer_seq_length': 512,
       'max_seq_len': 512,
       'max_batch_size': 8,
       'n_layers': 16,
       'n_heads': 32,
       'multiple_of': 256,
       'ffn_dim_multiplier': 1.0,
       'attention_dropout': 0.1,
       'transformer_start_pos': 0,
       'transformer_dim': 1024,
       'encoder_transformer_activation': "silu",
       'top_dims': 1024,
       'hidden_dims': 1024,
       'latent_dim': 1024,
       'decoder_base_dims': 4096,
       'prior_mog_components': 8,
       'mean_lims': [-5, 5],
       'logvar_lims': [-5, 1],
       'gumbel_softmax_temperature_max': 0.05,
       'diag_mask_buffer_tokens': 16,
       'prior_initial_mean_spread': 3,
       'prior_initial_logvar': -2,
       'gp_sigma': 2.5,
       'gp_length_scale': 128,
       'crattn_num_heads': 8,
       'crattn_num_layers': 16,
       'crattn_max_seq_len': 1,
       'crattn_dropout': 0.1,
       'posterior_mogpredictor_hidden_dim_list': [2048, 1024, 512],
       'posterior_mogpredictor_dropout': 0.1,
       'classifier_hidden_dims': [2048, 1024, 512],
       'classifier_num_pats': 45,
       'classifier_dropout': 0.1,


       # Discriminator Params
       'disc_hidden_dims': [4096, 2048, 1024, 512, 256],


       # BSP Params
       'bse2p_chunk_size': 16*2*1024,
       'bse2p_transformer_seq_length': 512,
       'bsp_transformer_seq_length': 32,
       'bsp_latent_dim': 1024,
       'bsp_n_heads': 32,
       'bsp_n_layers': 16,
       'bsp_ffn_dim_multiplier': 0.8,
       'bsp_max_batch_size': 1,
       'bsp_max_seq_len': 32,
       'bsp_transformer_activation': "silu",
       'bsp_attention_dropout': 0.0,
       'bsp_transformer_start_pos': 0,
       'bsp2e_chunk_size': 16*2*1024,
       'bsp2e_transformer_seq_length': 512,


       # BSV Params
       'bsv_dims': [1024, 512, 256, 128, 8],


       # Kohonen/SOM Params
       'som_pca_init': False,
       'reduction': 'mean', # Keep at mean because currently using reparam in SOM training
       'som_epochs': 100,
       'som_batch_size': 1024,
       'som_lr': 0.5,
       'som_lr_min': 0.001,
       'som_lr_epoch_decay': 0.9397455978,
       'som_gridsize': 64,
       'som_sigma': 32,
       'som_sigma_min': 1,
       'som_sigma_epoch_decay': 0.96593632892,
       'som_rewin_seconds': 64,
       'som_stride_seconds': 16,


       # Weight files
       'bse_weight_file': 'bse_weights.pth',
       'disc_weight_file': 'disc_weights.pth',
       'bsp_weight_file': 'bsp_weights.pth',
       'bsv_weight_file': 'bsv_weights.pth',
       'som_file': 'som_file.pth',
       'som_axis_file': 'som_axis_file.pkl',
       'release_tag': 'v1.1.0'
   }
}


def _get_conda_cache():
    """Return path to Conda environment model cache folder"""
    env_prefix = os.environ.get("CONDA_PREFIX")
    if env_prefix is None:
        raise EnvironmentError("No Conda environment detected. Activate your environment first.")
    cache_dir = os.path.join(env_prefix, "kenazlbm_models")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def _cached_or_download(url, filename):
    """Check cache first, otherwise download"""
    cache_dir = _get_conda_cache()
    cached_path = os.path.join(cache_dir, filename)
    if os.path.exists(cached_path):
        return cached_path
    # Download
    try:
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location='cpu')
        torch.save(state_dict, cached_path)
        return cached_path
    except Exception as e:
        print(f"Error downloading {filename} from {url}: {e}")
        raise

def _load_models(codename='commongonolek_sheldrake', gpu_id='cpu', pretrained=True,
                 load_bse=True, load_discriminator=True, load_bsp=True, load_bsv=True,
                 load_som=True, **kwargs):
    if codename not in CONFIGS:
        raise ValueError(f"Codename '{codename}' not found. Available: {list(CONFIGS.keys())}")

    config = CONFIGS[codename].copy()
    config.update(kwargs)

    # --- Brain-State Embedder ---
    bse = None
    if load_bse:
        bse = BSE(gpu_id=gpu_id, **config)
        if pretrained and config.get('bse_weight_file') and config.get('release_tag'):
            url = f"https://github.com/grahamwjohnson/kenazlbm/releases/download/{config['release_tag']}/{config['bse_weight_file']}"
            try:
                cached_path = _cached_or_download(url, config['bse_weight_file'])
                state_dict = torch.load(cached_path, map_location='cpu')
                bse.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading BSE pretrained weights: {e}. Using random initialization.")

    # --- Discriminator ---
    disc = None
    if load_discriminator:
        disc = Discriminator(gpu_id=gpu_id, **config)
        if pretrained and config.get('disc_weight_file') and config.get('release_tag'):
            url = f"https://github.com/grahamwjohnson/kenazlbm/releases/download/{config['release_tag']}/{config['disc_weight_file']}"
            try:
                cached_path = _cached_or_download(url, config['disc_weight_file'])
                state_dict = torch.load(cached_path, map_location='cpu')
                disc.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading Discriminator pretrained weights: {e}")

    # --- Brain-State Predictor ---
    bsp = None
    if load_bsp:
        bsp = BSP(gpu_id=gpu_id, **config)
        if pretrained and config.get('bsp_weight_file') and config.get('release_tag'):
            url = f"https://github.com/grahamwjohnson/kenazlbm/releases/download/{config['release_tag']}/{config['bsp_weight_file']}"
            try:
                cached_path = _cached_or_download(url, config['bsp_weight_file'])
                state_dict = torch.load(cached_path, map_location='cpu')
                bsp.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading BSP pretrained weights: {e}")

    # --- Brain-State Visualizer ---
    bsv = None
    if load_bsv:
        bsv = BSV(gpu_id=gpu_id, **config)
        if pretrained and config.get('bsv_weight_file') and config.get('release_tag'):
            url = f"https://github.com/grahamwjohnson/kenazlbm/releases/download/{config['release_tag']}/{config['bsv_weight_file']}"
            try:
                cached_path = _cached_or_download(url, config['bsv_weight_file'])
                state_dict = torch.load(cached_path, map_location='cpu')
                bsv.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading BSV pretrained weights: {e}")

    # --- 2D SOM ---
    som = None
    if load_som:
        try:
            url = f"https://github.com/grahamwjohnson/kenazlbm/releases/download/{config['release_tag']}/{config['som_file']}"
            cached_path = _cached_or_download(url, config['som_file'])
            checkpoint = torch.load(cached_path, map_location='cpu')
            som = ToroidalSOM_2(**checkpoint)  # adjust to your init
            som.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Error loading SOM weights: {e}")

        try:
            # Axis file
            axis_url = f"https://github.com/grahamwjohnson/kenazlbm/releases/download/{config['release_tag']}/{config['som_axis_file']}"
            axis_cache_path = os.path.join(_get_conda_cache(), config['som_axis_file'])
            if not os.path.exists(axis_cache_path):
                r = requests.get(axis_url)
                r.raise_for_status()
                with open(axis_cache_path, "wb") as f:
                    f.write(r.content)
            with open(axis_cache_path, "rb") as f:
                som.axis_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading SOM plot info: {e}")

    return bse, disc, bsp, bsv, som, config

def load_lbm(codename='commongonolek_sheldrake', **kwargs):
    return _load_models(codename=codename, **kwargs)






