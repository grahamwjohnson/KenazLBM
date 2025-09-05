# src/kenazlbm/core.py
import os, glob, re
import torch
from .hubconfig import _get_conda_cache, CONFIGS, _load_models
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, barrier
import datetime
import pickle

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
        with open(file_path, 'rb') as file: file_data = pickle.load(file)  # shape: [channels, time]

        x = torch.tensor(file_data, dtype=torch.float32)

        # Reshape into [channels, seq_len, samples_per_step]
        x = x.view(x.shape[0], -1, self.bse_samples)

        actual_channels = x.shape[0]
        assert actual_channels <= self.padded_channels, "More channels than padded_channels!"

        total_samps_in_file = x.shape[1]
        assert total_samps_in_file >= self.transformer_seq_length, "File too short for transformer_seq_length!"

        num_chunks_in_file = total_samps_in_file // self.transformer_seq_length
        print(f"File {os.path.basename(file_path)}: {actual_channels} channels, {total_samps_in_file} samples, {num_chunks_in_file} chunks")

        # Prepare output tensor [sequence, padded_channels, latent_dim]
        padded = torch.zeros(num_chunks_in_file, self.padded_channels, self.bse_samples, dtype=torch.float32)
        print("Padded shape:", padded.shape)

        # Randomize channel mapping for each time step independently
        rand_ch_orders = torch.zeros(num_chunks_in_file, self.padded_channels)

        # Shuffle ONCE per entire token sequence, so each BSP token has same channel order
        shuffled_channel_indices = torch.randperm(actual_channels)
        padded_positions = torch.randperm(self.padded_channels)[:actual_channels]

        # Assign tokens
        for t in range(num_chunks_in_file):

            # Assign channel orders to variable (done within FOR loop to ensure it is clear that they all have the same channel order)
            rand_ch_orders[t, :] = self.make_ch_index_vec(padded_positions, shuffled_channel_indices)

            # Assign data to variable
            for src_idx, dst_pos in zip(shuffled_channel_indices, padded_positions):
                padded[t, dst_pos, :] = x[src_idx, t, :]  # use the corresponding time step

        # Unsqueeze a dimension and transpose to be ready for BSE
        # [seq, padded_channels, FS] --> [seq, FS, padded_channel, 1]
        out = padded.permute(0, 2, 1).unsqueeze(3)

        return out, file_path, rand_ch_orders

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
    Run Brain-State Embedder (BSE) inference.

    Expects files in the format:
        <in_dir>/<subject_id>/preprocessed_epoched_data/*_bipole_scaled_filtered_data.pkl

    Args:
        in_dir (str): Input directory containing preprocessed and epoched pickle files.
        out_dir (str, optional): Output directory to save results.
                                 If None, defaults to the input directory.
        codename (str): Model codename to load (default: commongonolek_sheldrake).

    Raises:
        FileNotFoundError: If the input directory does not exist.

    Notes:
        - Produces *_bse outputs like:
          <out_dir>/<subject_id>/bse/*_bipole_scaled_filtered_data_BSE.pkl
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
        subproc = ctx.Process(target=bse_main, args=(i, world_size, bse, in_dir, out_dir))
        children.append(subproc)
        subproc.start()

    for i in range(world_size):
        children[i].join()

# DDP subprocess for BSE (will run on each GPU)
def bse_main(gpu_id, world_size, bse, in_dir, out_dir):

    # Initialize DDP 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=gpu_id, world_size=world_size, timeout=datetime.timedelta(minutes=999999))
    bse = bse.to(gpu_id)
    bse = DDP(bse, device_ids=[gpu_id])
    bse.eval()
    with torch.inference_mode():
        
        # Process subjects, distributing work across GPUs
        subject_dirs = [d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)]
        print(f"Found {len(subject_dirs)} subject(s): {[os.path.basename(d) for d in subject_dirs]}")

        # Iterate through subjects and make new dataloader for each
        for subj_path in subject_dirs:
            subject_id = os.path.basename(subj_path)
            in_epoched_dir = os.path.join(subj_path, "preprocessed_epoched_data")
            out_bse_dir = os.path.join(out_dir, subject_id, "bse")
            os.makedirs(out_bse_dir, exist_ok=True)
            pp_files = glob.glob(os.path.join(in_epoched_dir, "*_bipole_scaled_filtered_data.pkl"))
            print(f"Processing subject '{subject_id}' with {len(pp_files)} file(s).")

            dataset = FileDataset(pp_files, bse.module)
            dataloader, _ = prepare_ddp_dataloader(dataset, batch_size=1, droplast=False, num_workers=0)

            for x, file_path, rand_ch_orders in dataloader: 
                filename = os.path.splitext(os.path.basename(file_path))[0]
                outfile = os.path.join(out_bse_dir, f"{filename}_BSE.pkl")
                print(f"Running BSE on {file_path} -> {outfile}")




                # TODO: Actual BSE inference code here





                print(f"x shape {x.shape}", file_path, rand_ch_orders)

                result = f"BSE output of {file_path}"  # dummy inference

                with open(outfile, 'w') as f:
                    f.write(result)

    destroy_process_group()

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

            # TODO

            print(f"Running BSP on {infile} -> {outfile}")
            result = f"BSP output of {infile}"  # dummy inference
            with open(outfile, 'w') as f:
                f.write(result)

def run_bsv(in_dir, file_pattern, out_dir=None, codename='commongonolek_sheldrake'):
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

        in_files = glob.glob(os.path.join(subj_path, file_pattern))
        print(f"Processing subject '{subject_id}' with {len(in_files)} file(s).")
        for infile in in_files:
            filename = os.path.splitext(os.path.basename(infile))[0]
            outfile = os.path.join(out_subj_dir, f"{filename}_bsv.pkl")

            # TODO

            print(f"Running BSV on {infile} -> {outfile}")
            result = f"BSV output of {infile}"  # dummy inference
            with open(outfile, 'w') as f:
                f.write(result)


if __name__ == "__main__":
    # For Development and Debugging
    run_bse('/home/graham/Downloads/test_raw2')
    # run_bsp('/home/graham/Downloads/test_raw2')
    # run_bsv('/home/graham/Downloads/test_raw2', file_pattern="*_pp_bse.pkl")