# hne_image_utils

A python package providing utility functions for downstream analysis
and visualization of H&E images. This includes:
- hovernet_utils.py: parsing Hover-Net (cell segmentation and cell type calling) output
- img_utlis.py: generic functions for processing images
- mask_utils.py: functions for reading and intersecting (tissue) masks and (regional) annotations

Installation:
mask_utils.py requires (at least):
conda install -c conda-forge rasterio
conda install -c conda-forge shapely
conda install -c conda-forge geojson
conda install -c conda-forge openslide
conda install -c conda-forge descartes