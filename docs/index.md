# Kenaz Large Brain-State Model (KenazLBM)

This is the step-by-step guide on how to install and use KenazLBM on raw intracranial EEG data. 

Assumptions: This guide is made on OS Ubuntu 22.04 with Nvidia GPUs, cannot guarentee any other compatability. 

This ReadTheDocs is based on the GitHub repository: https://github.com/grahamwjohnson/KenazLBM

# Conda Install

Follow directions at: https://www.anaconda.com/docs/getting-started/anaconda/install to install Anaconda3 on Ubuntu 22.04

# KenazLBM Installation

After conda is installed, install **KenazLBM** with following line:

```bash
conda env create -f https://raw.githubusercontent.com/grahamwjohnson/KenazLBM/main/environment.yml
```

# Running Command-line Interface with KenazLBM Models
```bash
conda activate lbm_env
```

All preprocessing and model runs may be conducted in same directory or seperate directories for each step. 

```bash
kenaxlbm prefetch_models
```

The BSE, BSP, and BSV pretrained models should now be downloaded and cached locally. To check, you can runn the following command:

```bash
kenazlbm check_models
```

The output should look like:
```bash
bse_weights.pth: FOUND (1021.75 MB)
disc_weights.pth: FOUND (58.57 MB)
bsp_weights.pth: FOUND (1600.60 MB)
bsv_weights.pth: FOUND (5.29 MB)
som_file.pth: ONLINE (not cached locally)
som_axis_file.pkl: ONLINE (not cached locally)
```