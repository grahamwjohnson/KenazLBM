#!/usr/bin/env python3
from kenazlbm.hubconfig import load_lbm

# Preload all models for the default codename
print("Prefetching all pretrained models...")
load_lbm(codename="commongonolek_sheldrake",
         pretrained=True,
         load_bse=True,
         load_discriminator=True,
         load_bsp=True,
         load_bsv=True,
         load_som=True,
         gpu_id='cpu')
print("All models downloaded and cached.")