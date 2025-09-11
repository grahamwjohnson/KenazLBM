# Model Usage

## Option 2: Python Scripts

To have more granular control of the model usage, get intermediate outputs like full size 1024-dimensional embeddings from the BSE, or forecast future emebeddings with the BSP, you can interact with the models directly in a python script as follows:

```python
import os
import torch
from kenazlbm.hubconfig import _load_models # make sure kenazlbm is installed

gpu_id = 0 # This example does not use Torch DDP

# ---- Configuration ----
ENV_PREFIX = os.environ.get("CONDA_PREFIX")
if ENV_PREFIX is None:
    raise EnvironmentError("No Conda environment detected. Activate your environment first.")

CACHE_DIR = os.path.join(ENV_PREFIX, "kenazlbm_models")
codename = 'commongonolek_sheldrake'
device = 'cpu'  # or 'cuda:0' if GPU available
# -----------------------

# Ensure cache exists
if not os.path.exists(CACHE_DIR):
    raise FileNotFoundError(f"Cache directory {CACHE_DIR} not found. Run prefetch_models() first.")

# Point torch.hub to the cached models
torch.hub._get_torch_home = lambda: CACHE_DIR

# Load all models from cache
bse, disc, bsp, bsv, som, config = _load_models(
    codename=codename,
    gpu_id=device,
    pretrained=True,
    load_bse=True,
    load_discriminator=True,
    load_bsp=True,
    load_bsv=True,
    load_som=True
)

# Load all models onto GPU and set to eval
bse = bse.to(gpu_id)
bse.eval()

bsp = bsp.to(gpu_id)
bsp.eval()

bsv = bsv.to(gpu_id)
bsv.eval()

# Now create a dataloader that gives x as [batch, num_epochs_in_file, 512, 256, 1]]
for x in dataloader: 

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


```