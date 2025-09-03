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



# Option 1: Running Command-line Interface with KenazLBM Models

File Format Assumptions:

.EDF format
Sampling frequency is multiple of 512 Hz
File name must be formatted as: "<subject_id>_<MMDDYYY>_<HHMMSSSS>" where SSSS is seconds and deciseconds. 
Example is "Epat27_02182020_17072099"

Directory structure must be formatted as:

```bash
parent_dir
    subject_id_0
        file0
        file1
        ...
    subject_id_1
        file0
        file1
        ...
```
    
NOTE: All preprocessing and model runs will be conducted in same directory.

```bash
conda activate lbm_env
```

The first step is to preprocess your data. This command will filter the data and histogram equalize it to prepare for input ot the BSE. The zero-centered histogram equalization (ZHE) scheme looks at the first 24 hours (default) present in your files (missing data included in time calculation), then applies the calculated equalization scheme to all data. To change the hours used for equalization clculation, pass in a different value for '24' below. Preprocessing may take multiple minutes per file for large EDF files (e.g. 5-10 GB) depending on CPU and RAM resources. 

This step will create files named as follows:

```bash
parent_dir
    subject_id_0
        <filename>_bipole_filtered.pkl  # This is before equalization
        preprocessed_epoched_data
            ...pkl  # These are the equalized data epoched to 1024 seconds
            ...pkl
            ...pkl
```

```bash
kenazlbm preprocess --input /path/to/parent_dir --eq_hrs 24


```bash
kenazlbm prefetch_models
```

The BSE, BSP, and BSV pretrained models should now be downloaded and cached locally. To check, you can run the following command:

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
Note that the 2 self-organizing map (SOM) files are not cached locally


# Option 2: Running KenazLBM Models With Python Scripts



