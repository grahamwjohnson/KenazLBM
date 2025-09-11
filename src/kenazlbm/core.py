# src/kenazlbm/core.py
import os, glob, re
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, barrier
import datetime
import pickle
import numpy as np
import pandas as pd
import sys
import matplotlib
import matplotlib.cm as cm
import matplotlib.pylab as pl
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import seaborn as sns
from matplotlib import colors


try:
    from .hubconfig import _get_conda_cache, CONFIGS, _load_models
except ImportError:
    # Fallback for running as a script
    from hubconfig import _get_conda_cache, CONFIGS, _load_models

# ---- Global cache dir ----
ENV_PREFIX = os.environ.get("CONDA_PREFIX")
if ENV_PREFIX is None:
    raise EnvironmentError("No Conda environment detected. Activate your environment first.")
CACHE_DIR = os.path.join(ENV_PREFIX, "kenazlbm_models")
os.makedirs(CACHE_DIR, exist_ok=True)
# --------------------------

class FileDataset(Dataset):
    def __init__(self, file_list, bse):
        self.file_list = file_list
        self.bse_samples = bse.encode_token_samples
        self.transformer_seq_length = bse.transformer_seq_length
        self.padded_channels = bse.padded_channels

        print(
            f"FileDataset initialized with: "
            f"BSE encode_token_samples: {self.bse_samples}, "
            f"BSE transformer_seq_length: {self.transformer_seq_length}, "
            f"BSE padded_channels: {self.padded_channels}"
        )
        print(len(file_list), "files found")

    def make_ch_index_vec(self, padded_positions, shuffled_channel_indices):
        """
        Returns a vector of length self.padded_channels where:
        - Entries at padded_positions are filled with shuffled_channel_indices.
        - All other entries are -1.
        """
        ch_index_vec = torch.ones(self.padded_channels, dtype=torch.long, device=padded_positions.device) * -1
        ch_index_vec[padded_positions] = shuffled_channel_indices
        return ch_index_vec

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Load the file's pickle
        with open(file_path, 'rb') as file:
            file_data = pickle.load(file)  # shape: [channels, time]

        num_channels = file_data.shape[0]
        total_samples = file_data.shape[1]
        seq_len = self.transformer_seq_length
        samples_per_step = self.bse_samples
        padded_channels = self.padded_channels

        # Calculate number of epochs (chunks)
        num_epochs = total_samples // (seq_len * samples_per_step)
        truncated_samples = num_epochs * seq_len * samples_per_step
        file_data = file_data[:, :truncated_samples]

        # Reshape to [channels, num_epochs, seq_len, samples_per_step]
        x = torch.tensor(file_data, dtype=torch.float32)
        x = x.view(num_channels, num_epochs, seq_len, samples_per_step)

        assert num_channels <= padded_channels, "More channels than padded_channels!"

        # Prepare output tensor [num_epochs, seq_len, padded_channels, samples_per_step]
        big_padded = torch.zeros(num_epochs, seq_len, padded_channels, samples_per_step, dtype=torch.float32)
        big_rand_ch_orders = torch.zeros(num_epochs, seq_len, padded_channels)

        # Shuffle ONCE per file for all epochs
        shuffled_channel_indices = torch.randperm(num_channels)
        padded_positions = torch.randperm(padded_channels)[:num_channels]

        # Fill big_rand_ch_orders for all epochs and time steps
        ch_index_vec = self.make_ch_index_vec(padded_positions, shuffled_channel_indices)
        big_rand_ch_orders[:, :, :] = ch_index_vec

        # Vectorized assignment for all epochs and time steps
        # x: [num_channels, num_epochs, seq_len, samples_per_step]
        # big_padded: [num_epochs, seq_len, padded_channels, samples_per_step]
        for src_idx, dst_pos in zip(shuffled_channel_indices, padded_positions):
            big_padded[:, :, dst_pos, :] = x[src_idx, :, :, :]
        
        return big_padded, file_path, big_rand_ch_orders

def prepare_ddp_dataloader(dataset: Dataset, batch_size: int, droplast=False, num_workers=0):

    if num_workers > 0:
        persistent_workers=True
        print("WARNING: num workers >0, have experienced odd errors...")

    else:
        persistent_workers=False

    sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,    
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
        drop_last=droplast,
        persistent_workers=persistent_workers
    ), sampler

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

def _check_cache_files(codename, keys):
    """Helper: check if given weight files exist in CACHE_DIR."""
    config = CONFIGS.get(codename, {})
    required_files = [config.get(k) for k in keys if config.get(k)]
    return all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in required_files)

def run_bse(in_dir, out_dir=None, codename='commongonolek_sheldrake'):
    """
    Run Brain-State Embedder (BSE) inference using distributed GPU processing.

    This function:
        - Loads a pretrained BSE model (from cache or downloads it if missing).
        - Expects preprocessed and epoched EEG/latent data in pickle files.
        - Spawns one subprocess per available GPU for Distributed Data Parallel (DDP) inference.
        - Saves BSE outputs for each subject.

    Expected input file structure:
        <in_dir>/<subject_id>/preprocessed_epoched_data/*_bipole_scaled_filtered_data.pkl

    Args:
        in_dir (str): Input directory containing preprocessed and epoched pickle files.
        out_dir (str, optional): Output directory to save results.
                                 Defaults to `in_dir` if None.
        codename (str): Model codename to load (default: 'commongonolek_sheldrake').

    Raises:
        FileNotFoundError: If the input directory does not exist.
        Exception: If no CUDA devices are found for GPU inference.

    Notes:
        - Produces *_bse outputs like:
            <out_dir>/<subject_id>/bse/*_bipole_scaled_filtered_data_PostBSEV.pkl
        - Automatically uses all available CUDA GPUs with one subprocess per device.
        - Uses torch.multiprocessing 'spawn' context for safe DDP initialization.
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

    # Set up DDP for inference if needed
    world_size = torch.cuda.device_count()
    cuda_available = torch.cuda.is_available()
    if cuda_available: 
        print("CUDA available:", cuda_available)
        print(f"Using world_size of {world_size} GPUs for Distributed Data Parallel (DDP) inference.")
    else: raise Exception ("No CUDA devices found. BSE inference requires at least one GPU.")
    
    # Spawn subprocesses with start/join 
    ctx = mp.get_context('spawn') # necessary to use context if have set_start_method anove?
    children = []
    for i in range(world_size):
        subproc = ctx.Process(target=bse_subprocess, args=(i, world_size, codename, in_dir, out_dir))
        children.append(subproc)
        subproc.start()

    for i in range(world_size):
        children[i].join()

# DDP subprocess for BSE (will run on each GPU)
def bse_subprocess(gpu_id, world_size, codename, in_dir, out_dir):

    # Initialize DDP 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=gpu_id, world_size=world_size, timeout=datetime.timedelta(minutes=999999))
    
    bse, disc, bsp, bsv, som, config = _load_models(
        codename=codename,
        gpu_id=gpu_id,
        pretrained=True,
        load_bse=True,
        load_discriminator=False,
        load_bsp=True,
        load_bsv=True,
        load_som=False)
    
    # Load all models onto GPU and setup DDP
    bse = bse.to(gpu_id)
    bse = DDP(bse, device_ids=[gpu_id])
    bse.eval()

    bsp = bsp.to(gpu_id)
    bsp = DDP(bsp, device_ids=[gpu_id])
    bsp.eval()

    bsv = bsv.to(gpu_id)
    bsv = DDP(bsv, device_ids=[gpu_id])
    bsv.eval()

    with torch.inference_mode():
        
        # Process subjects, distributing work across GPUs
        subject_dirs = [d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)]
        print(f"Found {len(subject_dirs)} subject(s): {[os.path.basename(d) for d in subject_dirs]}")

        # Iterate through subjects and make new dataloader for each
        for subj_path in subject_dirs:
            subject_id = os.path.basename(subj_path)
            in_epoched_dir = os.path.join(subj_path, "preprocessed_epoched_data")
            out_bsev_dir = os.path.join(out_dir, subject_id, "bsev")
            os.makedirs(out_bsev_dir, exist_ok=True)
            pp_files = glob.glob(os.path.join(in_epoched_dir, "*_bipole_scaled_filtered_data.pkl"))
            print(f"Processing subject '{subject_id}' with {len(pp_files)} file(s).")

            dataset = FileDataset(pp_files, bse.module)
            dataloader, _ = prepare_ddp_dataloader(dataset, batch_size=1, droplast=False, num_workers=0)

            count = 0
            for x, file_path, _ in dataloader: 
                filename = os.path.splitext(os.path.basename(file_path[0]))[0]
                outfile_bsev = os.path.join(out_bsev_dir, f"{filename}_PostBSEV.pkl")
                print(f"[GPU{gpu_id}: {count}/{len(dataloader)}]:\n{file_path[0]} -> \n{outfile_bsev}")

                # Run max batch size at a time
                bsize = bse.module.transformer_encoder.params.max_batch_size
                num_epochs = x.shape[1]

                bsev_z_all = torch.empty(x.shape[1], bsv.module.dims[-1], device=gpu_id)  # Preallocate BSV output
                for i in range(0, num_epochs, bsize):
                    x_batch = x[0,i:i+bsize, :, :, :].to(gpu_id)  # Move input to the correct GPU
                    
                    # BSE 
                    z_pseudobatch, _, _, _, _ = bse(x_batch, reverse=False) # No shift if not causal masking
                    post_bse_z = z_pseudobatch.reshape(-1, bse.module.transformer_seq_length * bse.module.encode_token_samples, bse.module.latent_dim).unsqueeze(1)

                    # BSE2P
                    _, _, post_bse2p_z = bsp.module.bse2p(post_bse_z)

                    # BSV 
                    _, _, _, bsev_z = bsv(post_bse2p_z)
                    bsev_z_all[i:i+bsize, :] = bsev_z.squeeze(1)  

                # Move result to CPU and convert to numpy & save pickle
                bsev_z_all = bsev_z_all.cpu().numpy()
                with open(outfile_bsev, 'wb') as f: pickle.dump(bsev_z_all, f)
            
                count += 1

    destroy_process_group()

def get_pat_seiz_datetimes(
    pat_id, 
    atd_file, 
    FBTC_bool=True, 
    FIAS_bool=True, 
    FAS_to_FIAS_bool=True,
    FAS_bool=True, 
    subclinical_bool=False, 
    focal_unknown_bool=True,
    unknown_bool=True, 
    non_electro_bool=False,
    artifact_bool=False,
    stim_fas_bool=False,
    stim_fias_bool=False,
    **kwargs
    ):

    failure_count = 0
    for i in range(5):
        try:
            atd_df = pd.read_csv(atd_file, sep=',', header='infer')
            pat_seizure_bool = (atd_df['Pat ID'] == pat_id) & (atd_df['Type'] == "Seizure")
            pat_seizurebool_AND_desiredTypes = pat_seizure_bool
            
            # Look for each seizure type individually & delete if not desired
            # seiz_type_list = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Subclinical', 'Focal, unknown awareness', 'Unknown', 'Non-electrographic']
            seiz_type_list = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Subclinical', 'Focal unknown awareness', 'Unknown', 'Non-electrographic', 'Artifact', 'Stim-FAS', 'Stim-FIAS']
            delete_seiz_type_bool_list = [FBTC_bool, FIAS_bool, FAS_to_FIAS_bool, FAS_bool, subclinical_bool, focal_unknown_bool, unknown_bool, non_electro_bool, artifact_bool, stim_fas_bool, stim_fias_bool]
            for i in range(0,len(seiz_type_list)):
                if delete_seiz_type_bool_list[i]==False:
                    find_str = seiz_type_list[i]
                    curr_bool = pat_seizure_bool & (atd_df.loc[:,'Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)'] == find_str)
                    pat_seizurebool_AND_desiredTypes[curr_bool] = False

            df_subset = atd_df.loc[pat_seizurebool_AND_desiredTypes, ['Type','Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)', 'Date (MM:DD:YYYY)', 'Onset String (HH:MM:SS)', 'Offset String (HH:MM:SS)']]
            
            pat_seiz_startdate_str = df_subset.loc[:,'Date (MM:DD:YYYY)'].astype(str).values.tolist() 
            pat_seiz_starttime_str = df_subset.loc[:,'Onset String (HH:MM:SS)'].astype(str).values.tolist()
            pat_seiz_stoptime_str = df_subset.loc[:,'Offset String (HH:MM:SS)'].astype(str).values.tolist()
            pat_seiz_types_str = df_subset.loc[:,'Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)'].astype(str).values.tolist()

            # Skip any lines that have nan/none or unknown time entries
            delete_list_A = [i for i, val in enumerate(pat_seiz_starttime_str) if (val=='nan' or val=='Unknown' or val=='None')]
            delete_list_B = [i for i, val in enumerate(pat_seiz_stoptime_str) if (val=='nan' or val=='Unknown' or val=='None')]
            delete_list = list(set(delete_list_A + delete_list_B))
            delete_list.sort()
            if len(delete_list) > 0:
                print(f"WARNING: deleting {len(delete_list)} seizure(s) out of {len(pat_seiz_startdate_str)} due to 'nan'/'none'/'Unknown' in master time sheet")
                print(f"Delete list is: {delete_list}")
                [pat_seiz_startdate_str.pop(del_idx) for del_idx in reversed(delete_list)]
                [pat_seiz_starttime_str.pop(del_idx) for del_idx in reversed(delete_list)]
                [pat_seiz_stoptime_str.pop(del_idx) for del_idx in reversed(delete_list)]
                [pat_seiz_types_str.pop(del_idx) for del_idx in reversed(delete_list)]

            # Initialize datetimes
            pat_seiz_start_datetimes = [0]*len(pat_seiz_starttime_str)
            pat_seiz_stop_datetimes = [0]*len(pat_seiz_stoptime_str)

            for i in range(0,len(pat_seiz_startdate_str)):
                sD_splits = pat_seiz_startdate_str[i].split(':')
                sT_splits = pat_seiz_starttime_str[i].split(':')
                start_time = datetime.time(
                                    int(sT_splits[0]),
                                    int(sT_splits[1]),
                                    int(sT_splits[2]))
                pat_seiz_start_datetimes[i] = datetime.datetime(int(sD_splits[2]), # Year
                                                    int(sD_splits[0]), # Month
                                                    int(sD_splits[1]), # Day
                                                    int(sT_splits[0]), # Hour
                                                    int(sT_splits[1]), # Minute
                                                    int(sT_splits[2])) # Second
                
                sTstop_splits = pat_seiz_stoptime_str[i].split(':')
                stop_time = datetime.time(
                                    int(sTstop_splits[0]),
                                    int(sTstop_splits[1]),
                                    int(sTstop_splits[2]))

                if stop_time > start_time: # if within same day (i.e. the TIME advances, no date included), assign same date to datetime, otherwise assign next day
                    pat_seiz_stop_datetimes[i] = datetime.datetime.combine(pat_seiz_start_datetimes[i], stop_time)
                else: 
                    pat_seiz_stop_datetimes[i] = datetime.datetime.combine(pat_seiz_start_datetimes[i] + datetime.timedelta(days=1), stop_time)

            return pat_seiz_start_datetimes, pat_seiz_stop_datetimes, pat_seiz_types_str

        except Exception as e:
            failure_count += 1
            print(f"[Attempt {i+1}] Plotting failed with error: {e}")
            print(f"Total failures so far: {failure_count}")

def preictal_label(atd_file, plot_preictal_color_sec, pat_ids, start_datetimes_input, stop_datetimes_input):
    
    '''
    PRE_ICTAL:
    
    Ictal data point is currently labeled as weight of 0
    Pre-ictal data point is labeled as a float between 0 and 1, where:
    - 0 = interictal (far from seizure)
    - 0.99999 = immediately before seizure (NOTE: ictal is labeled 0)

    '''
    data_window_preictal_score = np.zeros_like(pat_ids, dtype=float)
    data_window_ictal_score = np.zeros_like(pat_ids, dtype=float)

    # Generate numerical IDs for each unique patient, and give each datapoint an ID
    unique_ids = list(set(pat_ids))
    id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
    pat_idxs = [id_to_index[id] for id in pat_ids]

    # Initialzie all patient pre-ictal s
    num_ids = len(unique_ids)
    pat_seiz_start_datetimes = [-1] * num_ids
    pat_seiz_stop_datetimes = [-1] * num_ids
    pat_seiz_types_str = [-1] * num_ids

    # Iterate through unique pats and get all seizure info 
    for i in range(len(unique_ids)):
        id_curr = unique_ids[i]
        idx_curr = id_to_index[id_curr]
        pat_seiz_start_datetimes[i], pat_seiz_stop_datetimes[i], pat_seiz_types_str[i] = get_pat_seiz_datetimes(id_curr, atd_file) 

    # Iterate through every data window and get pre-ictal labels
    for i in range(len(pat_idxs)):

        print(f"Getting Pre-Ictal Weighting for Data Windows: {i}/{len(pat_idxs)}     ", end='\r')

        data_window_start = start_datetimes_input[i]
        data_window_stop = stop_datetimes_input[i]
        
        # PRE-ICTAL LABELING

        seiz_starts_curr, seiz_stops_curr, seiz_types_curr = pat_seiz_start_datetimes[pat_idxs[i]], pat_seiz_stop_datetimes[pat_idxs[i]], pat_seiz_types_str[pat_idxs[i]]

        for j in range(len(seiz_starts_curr)):
            seiz_start = seiz_starts_curr[j]
            seiz_stop = seiz_stops_curr[j]
            buffered_preictal_start = seiz_start - datetime.timedelta(seconds=plot_preictal_color_sec)

            # Compute time difference from seizure start (0 at start of preictal buffer, increasing as closer to seizure)
            # NOTE: IMPORTANT: Assign value of 0 for in seizure

            # Case where end of data window is in pre-ictal buffer (ignore start of preictal window)
            if data_window_stop < seiz_start and data_window_stop > buffered_preictal_start:
                dist_to_seizure = (seiz_start - data_window_stop).total_seconds()
                new_score = 1.0 - (dist_to_seizure / plot_preictal_color_sec)
                data_window_preictal_score[i] = max(data_window_preictal_score[i], new_score)  
                # Keep max score, and do NOT break because could find higher value

            # Case where end of the window is in the seizure
            elif data_window_stop > seiz_start and data_window_stop < seiz_stop:
                data_window_preictal_score[i] = 0 
                data_window_ictal_score[i] = 1
                break # Want to exclude seizures

            # Case where start of the window overlaps the preictal/ictal buffer, but end is past seizure end
            elif data_window_start > buffered_preictal_start and data_window_start < seiz_stop:
                data_window_preictal_score[i] = 0
                data_window_ictal_score[i] = 1
                break # Want to exclude seizures

    # Ensure values remain between 0 and 1
    return np.clip(data_window_preictal_score, 0, 1), np.clip(data_window_ictal_score, 0, 1), 

def filename_to_datetimes(list_file_names):
        start_datetimes = [datetime.datetime.min]*len(list_file_names)
        stop_datetimes = [datetime.datetime.min]*len(list_file_names)
        for i in range(0, len(list_file_names)):
            splits = list_file_names[i].split('_')
            aD = splits[1]
            aT = splits[2]
            start_datetimes[i] = datetime.datetime(int(aD[4:8]), int(aD[0:2]), int(aD[2:4]), int(aT[0:2]), int(aT[2:4]), int(aT[4:6]), int(int(aT[6:8])*1e4))
            bD = splits[4]
            bT = splits[5]
            stop_datetimes[i] = datetime.datetime(int(bD[4:8]), int(bD[0:2]), int(bD[2:4]), int(bT[0:2]), int(bT[2:4]), int(bT[4:6]), int(int(bT[6:8])*1e4))
        return start_datetimes, stop_datetimes

def plot_hex_grid(ax, data, title, cmap_str='viridis', vmin=None, vmax=None):
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # Remove axes

    cmap = matplotlib.colormaps[cmap_str]
    norm = pl.Normalize(vmin=vmin if vmin is not None else np.min(data), vmax=vmax if vmax is not None else np.max(data))

    # Hexagon geometry
    radius = 1.0  # circumradius
    width = 2 * radius  # not used directly but good to note
    height = np.sqrt(3) * radius  # vertical distance from flat to flat

    rows, cols = data.shape

    for i in range(rows):
        for j in range(cols):
            x = j * 1.5 * radius
            y = i * height + (j % 2) * (height / 2)

            # Scale the raw data to the range [0, 1] based on vmin and vmax
            face_color = cmap(norm(data[i, j]))

            hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius,
                                            orientation=np.radians(30),
                                            facecolor=face_color, alpha=0.7,
                                            edgecolor=face_color, linewidth=0.1)
            ax.add_patch(hexagon)

    x_extent = cols * 1.5 * radius
    y_extent = rows * height
    ax.set_xlim(-radius, x_extent + radius)
    ax.set_ylim(-radius, y_extent + height)
    ax.set_title(title)

    norm = pl.Normalize(vmin=vmin, vmax=vmax)  # Use vmin and vmax directly
    sm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = pl.colorbar(sm, ax=ax, label=title, shrink=0.92, pad=0) # Add a label to the colorbar for clarity
    cbar.ax.yaxis.set_label_position('left')

def rewindow_data(
    z: np.ndarray,
    file_windowsecs: int,
    file_stridesecs: int,
    rewin_windowsecs: int,
    rewin_strideseconds: int,
    reduction: str = 'mean',  # 'sum', 'mean', 'cat'
) -> np.ndarray:
    """
    Rewindows sequential latent samples (z) from an original
    windowing scheme to a new windowing scheme.

    Args:
        z: Array of windowed latent samples or MoG preds,
           shape [original_windows, latent_dim] or [original_windows, num_components].
        file_windowsecs: Window duration in seconds of the original data.
        file_stridesecs: Stride in seconds of the original data.
        rewin_windowsecs: Desired window duration in seconds.
        rewin_strideseconds: Desired stride in seconds.
        reduction: How to combine original windows inside the new window.
                   Options: 'mean', 'sum', 'cat'.

    Returns:
        rewin_z: Rewindowed array, shape:
                 - [new_windows, latent_dim] if reduction is 'mean' or 'sum'
                 - [new_windows, latent_dim * factor] if reduction is 'cat'
    """

    original_windows, latent_dim = z.shape

    # Validate divisibility
    if rewin_windowsecs % file_windowsecs != 0:
        raise ValueError("rewin_windowsecs must be a multiple of file_windowsecs.")
    if rewin_strideseconds % file_stridesecs != 0:
        raise ValueError("rewin_strideseconds must be a multiple of file_stridesecs.")
    if rewin_windowsecs < file_stridesecs and rewin_strideseconds > file_stridesecs:
        raise ValueError("Invalid window/stride combination.")

    # Start indices of new windows
    new_window_starts_in_original_samples = np.arange(
        0, original_windows * file_stridesecs, rewin_strideseconds
    )
    new_window_starts_in_original_windows = new_window_starts_in_original_samples // file_stridesecs

    # Number of new windows
    new_windows = (original_windows * file_stridesecs - (rewin_windowsecs - rewin_strideseconds)) // rewin_strideseconds
    new_windows = max(0, new_windows)

    # Allocate result (cat requires dynamic handling)
    if reduction in ['mean', 'sum']:
        rewin_z = np.zeros((new_windows, latent_dim), dtype=z.dtype)
    elif reduction == 'cat':
        factor = rewin_windowsecs // file_stridesecs
        rewin_z = np.zeros((new_windows, latent_dim * factor), dtype=z.dtype)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    # Fill
    for i in range(new_windows):
        start = new_window_starts_in_original_windows[i]
        end = start + (rewin_windowsecs // file_stridesecs)
        end = min(end, original_windows)

        valid = z[start:end]

        if valid.size == 0:
            continue

        if reduction == 'mean':
            rewin_z[i, :] = np.mean(valid, axis=0)
        elif reduction == 'sum':
            rewin_z[i, :] = np.sum(valid, axis=0)
        elif reduction == 'cat':
            # pad if shorter than expected
            padded = np.zeros((rewin_windowsecs // file_stridesecs, latent_dim), dtype=z.dtype)
            padded[:valid.shape[0]] = valid
            rewin_z[i, :] = padded.reshape(-1)

    return rewin_z

def toroidal_kohonen_subfunction_pytorch(
    atd_file,
    pat_ids_list,
    latent_z_windowed,
    start_datetimes_epoch,
    stop_datetimes_epoch,
    win_sec,
    stride_sec,
    savedir,
    som,
    plot_preictal_color_sec,
    sigma_plot=1,
    hits_log_view=True,
    umat_log_view=True,
    preictal_overlay_thresh=0.5,
    sleep_overlay_thresh=0.5,
    smooth_map_factor=1,
    **kwargs,
):
    """
    Perform toroidal SOM inference and visualization on latent time-series data.

    This function:
        - Cleans latent data of NaNs and flattens it for SOM processing.
        - Performs SOM inference to find Best Matching Units (BMUs) per data point.
        - Computes hit maps, pre-ictal/ictal density maps, and U-Matrix (distance map) on a hexagonal toroidal SOM grid.
        - Generates 2D overlay plots:
            * U-Matrix as a base layer.
            * Time progression overlay for a single patient (cubehelix colormap, opacity scaled by data density, 10–90% clipped).
            * Pre-ictal overlay on top (flare colormap, alpha fixed at 0.7).
        - Saves plots to the specified directory.

    Args:
        atd_file (str): CSV master time sheet for pre-ictal labeling.
        pat_ids_list (list of str): List of patient IDs corresponding to latent data files.
        latent_z_windowed (np.ndarray): Windowed latent vectors, shape [files, windows_per_file, latent_dim].
        start_datetimes_epoch (list of datetime): Start timestamps of each file/window.
        stop_datetimes_epoch (list of datetime): Stop timestamps of each file/window.
        win_sec (float): Window duration in seconds.
        stride_sec (float): Stride duration in seconds.
        savedir (str): Directory to save SOM outputs and figures.
        som (object): Pretrained SOM object with `.get_weights()` and `.find_bmu()`.
        plot_preictal_color_sec (float): Duration (seconds) to define pre-ictal data for overlay.
        sigma_plot (float, optional): Gaussian smoothing sigma for pre-ictal overlay (default: 1).
        hits_log_view (bool, optional): Whether to log-scale hit maps (default: True).
        umat_log_view (bool, optional): Whether to log-scale U-Matrix (default: True).
        preictal_overlay_thresh (float, optional): Threshold to show pre-ictal overlay (default: 0.5).
        sleep_overlay_thresh (float, optional): Threshold to show sleep overlay (not implemented in this snippet, default: 0.5).
        smooth_map_factor (float, optional): Smoothing factor for pre-ictal density map (default: 1).
        **kwargs: Optional keyword arguments:
            * plot_patient_id (str): Patient ID for the time progression overlay.
            * subsample_file_factor (int): Factor to subsample files (default: 1, use all files).

    Notes:
        - The function produces a single figure per patient showing:
            1. U-Matrix (bone_r colormap)
            2. Gist-earth/cubehelix time progression overlay with opacity scaled by node occupancy
            3. Pre-ictal flare overlay on top
        - Colorbars are clipped or scaled independently for visualization clarity.
        - Saved figure filename: "UMatrix_PreIctalOverlay_ToroidalSOM_patient_<patient_id>.jpg"
    """

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Check for NaNs in files
    delete_file_idxs = []
    for i in range(latent_z_windowed.shape[0]):
        if np.sum(np.isnan(latent_z_windowed[i, :, :])) > 0:
            delete_file_idxs = delete_file_idxs + [i]
            print(f"WARNING: Deleted file {start_datetimes_epoch[i]} that had NaNs")

    # Delete entries/files where there are NaNs
    latent_z_windowed = np.delete(latent_z_windowed, delete_file_idxs, axis=0)
    start_datetimes_epoch = [
        item for i, item in enumerate(start_datetimes_epoch) if i not in delete_file_idxs
    ]
    stop_datetimes_epoch = [
        item for i, item in enumerate(stop_datetimes_epoch) if i not in delete_file_idxs
    ]
    pat_ids_list = [
        item for i, item in enumerate(pat_ids_list) if i not in delete_file_idxs
    ]

    # Flatten data into [miniepoch, dim]
    latent_z_input = np.concatenate(latent_z_windowed, axis=0)
    pat_ids_input = [
        item for item in pat_ids_list for _ in range(latent_z_windowed[0].shape[0])
    ]
    start_datetimes_input = [
        item + datetime.timedelta(seconds=stride_sec * i)
        for item in start_datetimes_epoch
        for i in range(latent_z_windowed[0].shape[0])
    ]
    stop_datetimes_input = [
        item + datetime.timedelta(seconds=stride_sec * i)
        + datetime.timedelta(seconds=win_sec)
        for item in start_datetimes_epoch
        for i in range(latent_z_windowed[0].shape[0])
    ]

    # Get preictal weights and sleep stage
    preictal_float_input, ictal_float_input = preictal_label(
        atd_file,
        plot_preictal_color_sec,
        pat_ids_input,
        start_datetimes_input,
        stop_datetimes_input,
    )
    print("\nFinished gathering pre-ictal and sleep labels on all data windows")

    # Get model weights and coords
    weights = som.get_weights()
    hex_coords = som.get_hex_coords()
    grid_size = som.grid_size
    rows, cols = grid_size
    som_batch_size = som.batch_size
    som_device = som.device

    # Initialize maps
    preictal_sums = np.zeros(grid_size)
    ictal_sums = np.zeros(grid_size)
    hit_map = np.zeros(grid_size)
    neuron_patient_dict = {}

    # Keep BMUs for later plotting
    bmu_rows_all = []
    bmu_cols_all = []

    # SOM Inference on all data
    for i in range(0, len(latent_z_input), som_batch_size):
        print(
            f"Running all data windows through trained Kohonen Map: {i}/{int(len(latent_z_input))}                  ",
            end="\r",
        )

        batch = latent_z_input[i : i + som_batch_size]
        batch_patients = pat_ids_input[i : i + som_batch_size]
        batch_preictal_labels = preictal_float_input[i : i + som_batch_size]
        batch_ictal_labels = ictal_float_input[i : i + som_batch_size]

        batch = torch.tensor(batch, dtype=torch.float32, device=som_device)
        bmu_rows, bmu_cols = som.find_bmu(batch)
        bmu_rows, bmu_cols = bmu_rows.cpu().numpy(), bmu_cols.cpu().numpy()

        # Store BMUs
        bmu_rows_all.extend(list(bmu_rows))
        bmu_cols_all.extend(list(bmu_cols))

        # Update hit map
        np.add.at(hit_map, (bmu_rows, bmu_cols), 1)

        # Process preictal and ictal
        for j, (bmu_row, bmu_col) in enumerate(zip(bmu_rows, bmu_cols)):
            preictal_sums[bmu_row, bmu_col] += batch_preictal_labels[j]
            ictal_sums[bmu_row, bmu_col] += batch_ictal_labels[j]

        # Track patients
        for j, (bmu_row, bmu_col) in enumerate(zip(bmu_rows, bmu_cols)):
            if (bmu_row, bmu_col) not in neuron_patient_dict:
                neuron_patient_dict[(bmu_row, bmu_col)] = set()
            neuron_patient_dict[(bmu_row, bmu_col)].add(batch_patients[j])

    bmu_rows_all = np.array(bmu_rows_all, dtype=int)
    bmu_cols_all = np.array(bmu_cols_all, dtype=int)

    print("\nFinished Kohonen inference on all data")

    if hits_log_view:
        epsilon = np.finfo(float).eps
        hit_map = np.log(hit_map + epsilon)

    if np.max(preictal_sums) > np.min(preictal_sums):
        preictal_sums = (preictal_sums - np.min(preictal_sums)) / (
            np.max(preictal_sums) - np.min(preictal_sums)
        )
    if np.max(ictal_sums) > np.min(ictal_sums):
        ictal_sums = (ictal_sums - np.min(ictal_sums)) / (
            np.max(ictal_sums) - np.min(ictal_sums)
        )

    # Compute U-Matrix
    u_matrix_hex = np.zeros(grid_size)
    for i in range(rows):
        for j in range(cols):
            current_weight = weights[i, j]
            neighbor_distances = []

            if i % 2 == 0:
                neighbor_offsets = [
                    (0, 1),
                    (0, -1),
                    (-1, 0),
                    (-1, -1),
                    (1, 0),
                    (1, -1),
                ]
            else:
                neighbor_offsets = [
                    (0, 1),
                    (0, -1),
                    (-1, 1),
                    (-1, 0),
                    (1, 1),
                    (1, 0),
                ]

            for offset_row, offset_col in neighbor_offsets:
                ni = (i + offset_row + rows) % rows
                nj = (j + offset_col + cols) % cols
                neighbor_weight = weights[ni, nj]
                distance = np.linalg.norm(current_weight - neighbor_weight)
                neighbor_distances.append(distance)

            u_matrix_hex[i, j] = np.mean(neighbor_distances) if neighbor_distances else 0

    if umat_log_view:
        epsilon = np.finfo(float).eps
        u_matrix_hex = np.log(u_matrix_hex + epsilon)

    # Smooth preictal sums
    preictal_sums_smoothed = gaussian_filter(preictal_sums, sigma=smooth_map_factor)
    if np.max(preictal_sums_smoothed) > np.min(preictal_sums_smoothed):
        preictal_sums_smoothed = (preictal_sums_smoothed - np.min(preictal_sums_smoothed)) / (
            np.max(preictal_sums_smoothed) - np.min(preictal_sums_smoothed)
        )

    # -----------------------------
    # Plotting
    # -----------------------------
    fig_overlay, ax_overlay = pl.subplots(figsize=(15, 10))

    # -----------------------------
    # Subtitle with number of tokens and hours
    # -----------------------------
    num_tokens = latent_z_input.shape[0]
    coverage_sec = (num_tokens - 1) * stride_sec + win_sec
    coverage_hours = coverage_sec / 3600.0
    subtitle_text = f"{num_tokens:,} tokens ({coverage_hours:.1f} hours at {win_sec:.1f}s per token with stride of {stride_sec:.1f}s)"

    # Plot U-Matrix base
    plot_hex_grid(
        ax_overlay,
        u_matrix_hex,
        f"Toroidal SOM U-Matrix with Overlays\n{subtitle_text}",
        cmap_str="bone_r",
        vmin=np.min(u_matrix_hex),
        vmax=np.max(u_matrix_hex) if np.max(u_matrix_hex) > 0 else 1,
    )

    # -----------------------------
    # gist_earth time gradient overlay
    # -----------------------------
    plot_patient_id = kwargs.get(
        "plot_patient_id", pat_ids_list[0] if len(pat_ids_list) > 0 else None
    )

    if plot_patient_id is not None:
        start_dt_arr = np.array(start_datetimes_input)
        patient_mask = np.array([p == plot_patient_id for p in pat_ids_input], dtype=bool)
        patient_indices = np.where(patient_mask)[0]

        if patient_indices.size > 0:
            patient_bmu_rows = bmu_rows_all[patient_indices]
            patient_bmu_cols = bmu_cols_all[patient_indices]
            times = np.array([start_dt_arr[idx].timestamp() for idx in patient_indices])
            times_rel = times - times.min()
            times_frac = times_rel / times_rel.max() if times_rel.max() > 0 else np.zeros_like(times_rel)
            times_mapped = 0.1 + 0.8 * times_frac  # clip 10-90%

            per_node_mean_time = np.full(grid_size, np.nan, dtype=float)
            per_node_counts = np.zeros(grid_size, dtype=int)
            for r, c, t in zip(patient_bmu_rows, patient_bmu_cols, times_mapped):
                per_node_counts[r, c] += 1
                if np.isnan(per_node_mean_time[r, c]):
                    per_node_mean_time[r, c] = t
                else:
                    cur = per_node_mean_time[r, c]
                    n = per_node_counts[r, c]
                    per_node_mean_time[r, c] = (cur * (n - 1) + t) / n

            max_count = per_node_counts.max() if per_node_counts.max() > 0 else 1
            alpha_min, alpha_max = 0.12, 0.85
            cmap_gist = pl.get_cmap("cubehelix")

            radius = 1.0
            height = np.sqrt(3) * radius
            for i in range(rows):
                for j in range(cols):
                    if per_node_counts[i, j] > 0:
                        tm = per_node_mean_time[i, j]
                        face_color = cmap_gist(tm)
                        alpha_node = alpha_min + (alpha_max - alpha_min) * (
                            per_node_counts[i, j] / max_count
                        )
                        face_color = (
                            face_color[0],
                            face_color[1],
                            face_color[2],
                            alpha_node,
                        )

                        x = j * 1.5 * radius
                        y = i * height + (j % 2) * (height / 2)
                        hexagon = patches.RegularPolygon(
                            (x, y),
                            numVertices=6,
                            radius=radius,
                            orientation=np.radians(30),
                            facecolor=face_color,
                            alpha=None,
                            edgecolor=None,
                            linewidth=0,
                        )
                        ax_overlay.add_patch(hexagon)

            # Clip cubehelix colormap between 0.1 and 0.9
            cmap_cube_clipped = colors.LinearSegmentedColormap.from_list(
                "cubehelix_clipped",
                pl.cm.cubehelix(np.linspace(0.1, 0.8, 256))
            )

            norm_time = pl.Normalize(vmin=0.0, vmax=1.0)
            sm_time = pl.cm.ScalarMappable(cmap=cmap_cube_clipped, norm=norm_time)
            sm_time.set_array([])

            cbar = pl.colorbar(sm_time, ax=ax_overlay, shrink=0.92, pad=0.01)
            cbar.set_label(f"Time progression for patient {plot_patient_id} (start → end)")
            cbar.ax.yaxis.set_label_position('left')  # move label to the left side

            # Full 0.0–1.0 ticks preserved
            cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(['Start', '', 'Mid', '', 'End'])

        else:
            print(f"Warning: patient {plot_patient_id} not found. Skipping gist_earth overlay.")
    else:
        print("No patient selected for gist_earth overlay. Skipping.")

    # -----------------------------
    # Pre-Ictal overlay (flare) on top
    # -----------------------------
    overlay_preictal = np.clip(preictal_sums_smoothed, 0.0, 1.0)
    rows, cols = overlay_preictal.shape
    radius = 1.0
    height = np.sqrt(3) * radius
    cmap_overlay = sns.color_palette("flare", as_cmap=True)
    norm_overlay = pl.Normalize(vmin=preictal_overlay_thresh, vmax=1.0)

    for i in range(rows):
        for j in range(cols):
            if overlay_preictal[i, j] >= preictal_overlay_thresh:
                x = j * 1.5 * radius
                y = i * height + (j % 2) * (height / 2)
                face_color = cmap_overlay(norm_overlay(overlay_preictal[i, j]))
                hexagon = patches.RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=radius,
                    orientation=np.radians(30),
                    facecolor=face_color,
                    alpha=0.7,
                    edgecolor=None,
                    linewidth=0,
                )
                ax_overlay.add_patch(hexagon)

    sm_overlay = pl.cm.ScalarMappable(cmap=cmap_overlay, norm=norm_overlay)
    sm_overlay.set_array([])
    cbar = pl.colorbar(sm_overlay, ax=ax_overlay, label="Pre-Ictal Density (Clipped & Smoothed)", shrink=0.92, pad=0.05)
    cbar.ax.yaxis.set_label_position('left')  # move label to the left side

    # Save
    savename_overlay = savedir + f"/UMatrix_PreIctalOverlay_ToroidalSOM_patient_{plot_patient_id}.jpg"
    pl.savefig(savename_overlay, dpi=600)

    print("SOM Finished Plotting\n")

def run_som(in_dir, atd_file=None, out_dir=None, codename='commongonolek_sheldrake', plot_preictal_color_sec=4*60*60,
            file_windowseconds=1, file_strideseconds=1):
    """
    Run Self-Organizing Map (SOM) inference on BSEV latent data.

    This function:
        - Loads BSEV pickle files for each subject in <in_dir>/<subject_id>/bsev/
        - Rewindows the latent data according to SOM configuration
        - Performs inference through a pretrained SOM model
        - Computes U-Matrix and pre-ictal overlays
        - Produces figures and saves SOM outputs per subject

    Expected input file format:
        <in_dir>/<subject_id>/bsev/*_bipole_scaled_filtered_data_PostBSEV.pkl

    Args:
        in_dir (str): Directory containing subject subfolders with BSEV pickle files.
        atd_file (str, optional): Master time sheet CSV for pre-ictal labels.
                                  If None, defaults to 'atd_file.csv' in the source directory.
        out_dir (str, optional): Output directory to save SOM results and plots.
                                 Defaults to the input directory if None.
        codename (str): Pretrained SOM model codename (default: 'commongonolek_sheldrake').
        plot_preictal_color_sec (int): Time window (seconds) to color pre-ictal data in plots (default: 4 hours).
        file_windowseconds (int): Window length in seconds for rewindowing raw latent data (default: 1s).
        file_strideseconds (int): Stride in seconds for rewindowing raw latent data (default: 1s).

    Raises:
        FileNotFoundError: If input directory does not exist or default atd_file.csv is missing.

    Notes:
        - Rewindowing is applied per SOM config (som_rewin_seconds, som_stride_seconds, reduction).
        - SOM outputs are saved in:
            <out_dir>/<subject_id>/som/*_bipole_scaled_filtered_data_PostBSEV_SOM.pkl
        - Plots include:
            - U-Matrix
            - Pre-ictal overlay (flare colormap)
            - Full data overlay (time progression, opacity by density)
    """
    if out_dir is None:
        out_dir = in_dir
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    if atd_file is None:
        print("WARNING: No master time sheet provided, so pre-ictal labels are used from default atd_file.csv in src/kenazlbm")
        atd_file = os.path.join(os.path.dirname(__file__), 'atd_file.csv')
        if not os.path.exists(atd_file):
            raise FileNotFoundError(f"Default atd_file.csv not found in {atd_file}. Please provide a valid atd_file.")

    torch.hub._get_torch_home = lambda: CACHE_DIR

    if _check_cache_files(codename, ['som_file', 'som_axis_file']):
        print(f"Using cached SOM model from {CACHE_DIR}")
    else:
        print(f"Downloading SOM model to {CACHE_DIR} ...")

    _, _, _, _, som, config = _load_models(
        codename=codename,
        gpu_id='cpu',
        pretrained=True,
        load_bse=False,
        load_discriminator=False,
        load_bsp=False,
        load_bsv=False,
        load_som=True)

    # Process subjects sequentially on CPU
    subject_dirs = [d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)]
    print(f"Found {len(subject_dirs)} subject(s): {[os.path.basename(d) for d in subject_dirs]}")
    for subj_path in subject_dirs:
        subject_id = os.path.basename(subj_path)
        in_bsev_dir = os.path.join(subj_path, "bsev")
        out_som_dir = os.path.join(out_dir, subject_id, "som")
        os.makedirs(out_som_dir, exist_ok=True)

        # Load BSEV files
        data_filepaths = glob.glob(os.path.join(in_bsev_dir, "*_bipole_scaled_filtered_data_PostBSEV.pkl"))
        print(f"\nProcessing subject '{subject_id}' with {len(data_filepaths)} BSEV file(s).")

        # Gather metadata for files
        build_start_datetimes, build_stop_datetimes = filename_to_datetimes([s.split("/")[-1] for s in data_filepaths]) # Get start/stop datetimes
        build_pat_ids_list = [s.split("/")[-1].split("_")[0] for s in data_filepaths] # Get the build pat_ids
        
        # Load a sentinal file to get data params and intitialzie variables
        with open(data_filepaths[0], "rb") as f: latent_data_fromfile = pickle.load(f)
        print(f"Original shape of data: {latent_data_fromfile.shape}")
        rewin_z_sentinel = rewindow_data( latent_data_fromfile, file_windowseconds, file_strideseconds, config['som_rewin_seconds'], config['som_stride_seconds'], reduction=config['reduction'])
        print(f"Rewindowed shape of data: {rewin_z_sentinel.shape}")
        print(f"REDUCTION: {config['reduction']}")

        # Inialize all_file variables based on sentinel variables
        w_z_allfiles = np.zeros([len(data_filepaths), rewin_z_sentinel.shape[0], rewin_z_sentinel.shape[1]], dtype=np.float32)
        print("Loading all latent data from files")
        for i in range(len(data_filepaths)):
            sys.stdout.write(f"\rLoading Pickles: {i}/{len(data_filepaths)}        ") 
            sys.stdout.flush() 
            try:
                with open(data_filepaths[i], "rb") as f: latent_data_fromfile = pickle.load(f)
                w_z_allfiles[i, :, :] = rewindow_data( latent_data_fromfile, file_windowseconds, file_strideseconds, config['som_rewin_seconds'], config['som_stride_seconds'], reduction=config['reduction'])
            except: print(f"Error loading {data_filepaths[i]}")

        # Print broad statistics about dataset
        print(f"Dataset general statistics:\n"
            f"Mean {np.mean(w_z_allfiles):.2f}\n"
            f"Std {np.std(w_z_allfiles):.2f}\n"
            f"Max {np.max(w_z_allfiles):.2f}\n"
            f"Min {np.min(w_z_allfiles):.2f}"
            )
            
        toroidal_kohonen_subfunction_pytorch(
            atd_file = atd_file,
            pat_ids_list=build_pat_ids_list,
            latent_z_windowed=w_z_allfiles,
            start_datetimes_epoch=build_start_datetimes,  
            stop_datetimes_epoch=build_stop_datetimes,
            win_sec=config['som_rewin_seconds'], 
            stride_sec=config['som_stride_seconds'], 
            savedir=out_som_dir,
            subsample_file_factor=1, # Use all files
            som=som,
            plot_preictal_color_sec = plot_preictal_color_sec)
    
if __name__ == "__main__":
    prefetch_models()
    check_models()
    # For Development and Debugging
    # run_bse('/home/graham/Downloads/test_raw2')
    run_som('/home/graham/Downloads/test_raw3')