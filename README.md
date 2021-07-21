# Data Analytics 2 Case Study

Segmentation of land use and land cover from satellite images

## Set Up

    python3 -m venv .venv # create new virtual environment

    source .venv/bin/activate # linux
    .\.venv/scripts/activate # windows

    pip install -r requirements.txt # install dependencies


## Structure
File / Folder | Content
------------ | -------------
📄 SatelliteSegmentation | Contains our main results and all steps you need to reproduce it.
📁 data_exploration | Includes individual exploratory and experimental approaches.
📁 cnn | Includes all approaches based on convolutional neural networks.
📁 alternative_models | Includes all approaches which are not based on convolutional neural networks like SVMs, kNNs, LightGBM etc.
📁 preprocessing | Includes our approaches to label patches with semi supervised learning.
📁 helper_functions | Includes functions for the sliding window approach and functions to plot the result.
