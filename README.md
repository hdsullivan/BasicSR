# BasicSR

[![LICENSE](https://img.shields.io/github/license/xinntao/basicsr.svg)](https://github.com/xinntao/BasicSR/blob/master/LICENSE.txt)

## Description

This is a fork of the original [BasicSR](https://github.com/XPixelGroup/BasicSR) which updates data handling for compatibility with additive manufacturing x-ray computed tomography data. Namely, this repository allows for comparing the impact of baseline super-resolution methods on detection of defects.

## Installation

1. Clone the repository: `git clone https://github.com/hdsullivan/BasicSR`
2. Create Anaconda environment: `conda env create -n basic_sr`
3. Install dependencies: `pip install requirements.txt`

## Usage

### Preprocess Data

Data can be preprocessed by updating the parameters in scripts\data_preparation\preprocess.py and running
``python scripts\data_preparation\preprocess.py``
This script loads high-resolution and low-resolution reconstructions, performs histogram clustering to map the data to the range (0, 1),
extracts paired slices, and saves the patches for training or testing.

User-defined parameters:
- TRAINING_DATA: Boolean indicating whether the data is for training or testing.
- SCALE: Integer indicating the scaling factor for the low-resolution reconstructions.
- MATERIAL: String indicating the material being reconstructed.
- BEAM_HARDENING: String indicating whether beam hardening correction is applied.
- NOISE: Integer indicating the level of noise in the reconstructions.
- BLUR: Integer indicating the level of blur in the reconstructions.
- VIEWS: Integer indicating the number of views in the reconstructions.
- SCAN_LENGTH: String indicating the scan length of the reconstructions.
- MAIN_DIR: String indicating the main directory where the data is stored.
- GT_PATCH_SIZE: Integer indicating the patch size for the ground truth reconstructions.
- GT_STEP_SIZE: Integer indicating the step size for extracting patches from the ground truth reconstructions.

The script performs the following steps:
1. Loads the high-resolution and low-resolution reconstructions from the specified paths.
2. Performs histogram clustering to map the reconstructions to the range (0, 1).
3. Extracts paired slices from the high-resolution and low-resolution reconstructions.
    - If the data is for training, the slices are first cropped to detected object using Canny edge detection.
4. If the data is for training:
    - Saves the training and validation patches for the ground truth and low-resolution reconstructions.
5. If the data is for testing:
    - Saves the entire testing slices for the ground truth and low-resolution reconstructions.

Note: The script assumes that the specified files exist and the GT_PATCH_SIZE is divisible by SCALE.

### Training and Testing Super-Resolution Method

The options directory contains YAML files with the training and testing parameters for each each network. In order to train a network, update the corresponding YAML file and run `python basicsr\train.py -opt options\train\<path_to_yml_file>`. Similarly, to test a network, update the corresponding YAML file and run `python basicsr\test.py -opt options\test\<path_to_yml_file>`.

As of 3/21/2024, only the EDSR YAML files have been updated for our data.

