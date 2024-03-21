"""
This script is used for data preprocessing in the context of super-resolution reconstruction.
It loads high-resolution and low-resolution reconstructions, performs histogram clustering to map the data to the range (0, 1),
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
"""

import os
import preprocess_utils
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

# ######
# USER-DEFINED PARAMETERS
# ######
TRAINING_DATA = True
SCALE=3
MATERIAL = 'Al'
BEAM_HARDENING = 'noBH'
NOISE = 0
BLUR = 0
VIEWS = 1066
SCAN_LENGTH = 'fullscan'
MAIN_DIR='S:/NAOA/Projects/AM-SuperResolution/datasets'
GT_PATCH_SIZE = 120
GT_STEP_SIZE = 60

blur_str = 'NoBlur' if BLUR == 0  else 'blur_' + str(BLUR)
noise_str = 'NoNoise' if NOISE == 0 else 'noise' + str(NOISE)
CAD_model = 'CADRotated_wFlaws_REC_FDK' if TRAINING_DATA else 'CAD_M0277_PB_REC_FDK'
if not TRAINING_DATA:
    print("WARNING: Currently only using CADRotated for GT, so GT and LR scans will not match.")

GT_PATH=MAIN_DIR + '/main_data/GT/' + BEAM_HARDENING + '/' + MATERIAL + '/CADRotated_wFlaws_REC_FDK_'+MATERIAL+'_CleanGT_noBH_2132_Views_fullscan.npz'
LR_PATH=MAIN_DIR + '/main_data/down_sampled/' + BEAM_HARDENING + '/' + str(SCALE) + 'X/Recons/'+CAD_model+'_'+MATERIAL+'_'+BEAM_HARDENING+'_'+str(SCALE)+'Xbinning_'+noise_str+'_'+blur_str+'_' + str(VIEWS) + '_Views_'+ SCAN_LENGTH + '.npz'

# SAVE_DIR_GT=MAIN_DIR + '/'+BEAM_HARDENING+'/'+MATERIAL+'/CADRotated_wFlaws_REC_FDK_'+MATERIAL+'_CleanGT_noBH_2132_Views_fullscan'+ '/X' + str(SCALE)
SAVE_DIR_GT=MAIN_DIR + '/'+BEAM_HARDENING+'/'+MATERIAL+'/'+CAD_model+'_'+MATERIAL+'_'+BEAM_HARDENING+'_X'+str(SCALE)+'binning_'+noise_str+'_'+blur_str+'_' + str(VIEWS) + '_Views_'+ SCAN_LENGTH + "/GT/"
SAVE_DIR_LR=MAIN_DIR + '/'+BEAM_HARDENING+'/'+MATERIAL+'/'+CAD_model+'_'+MATERIAL+'_'+BEAM_HARDENING+'_X'+str(SCALE)+'binning_'+noise_str+'_'+blur_str+'_' + str(VIEWS) + '_Views_'+ SCAN_LENGTH + "/LR/"

assert GT_PATCH_SIZE % SCALE == 0, "GT_PATCH_SIZE must be divisible by SCALE"
assert GT_STEP_SIZE % SCALE == 0, "GT_STEP_SIZE must be divisible by SCALE"
assert os.path.isfile(LR_PATH), "File not found: " + LR_PATH
assert os.path.isfile(GT_PATH), "File not found: " + GT_PATH

print("\nLoading high- and low-resolution reconstructions...")
gt = np.load(GT_PATH)['arr_0']
lr = np.load(LR_PATH)['arr_0']

print("\nMapping to (0, 1) using histogram clustering...")
gt, mp1_gt, mp2_gt = preprocess_utils.hist_cluster_fixed(gt)
lr, mp1_lr, mp2_lr = preprocess_utils.hist_cluster_fixed(lr)

print("\nExtracting paired slices...")
gt_list, lr_list = preprocess_utils.extract_paired_slices(gt, lr, SCALE, TRAINING_DATA, patch_thresh=GT_PATCH_SIZE, thresh_range=0.8)

preprocess_utils.make_dir(SAVE_DIR_GT)
preprocess_utils.make_dir(SAVE_DIR_LR)

for idx in range(len(gt_list)):
    gt_img = gt_list[idx]
    lr_img = lr_list[idx]

    cv2.imwrite(os.path.join(SAVE_DIR_GT, 'slice_' + str(SCALE * idx) + '.tif'), gt_img)
    cv2.imwrite(os.path.join(SAVE_DIR_LR, 'slice_' + str(SCALE * idx) + '.tif'), lr_img)

if TRAINING_DATA:
    print("\nSaving training and validation patches...")

    preprocess_utils.make_dir(os.path.join(SAVE_DIR_GT, 'train'))
    preprocess_utils.make_dir(os.path.join(SAVE_DIR_GT, 'val'))
    preprocess_utils.make_dir(os.path.join(SAVE_DIR_LR, 'train'))
    preprocess_utils.make_dir(os.path.join(SAVE_DIR_LR, 'val'))

    gt_train, gt_test, lr_train, lr_test = train_test_split(gt_list, lr_list, test_size=0.2, random_state=42)

    preprocess_utils.save_patches(gt_train, os.path.join(SAVE_DIR_GT, 'train'), GT_PATCH_SIZE, GT_STEP_SIZE)
    preprocess_utils.save_patches(gt_test, os.path.join(SAVE_DIR_GT, 'val'), GT_PATCH_SIZE, GT_STEP_SIZE)
    preprocess_utils.save_patches(lr_train, os.path.join(SAVE_DIR_LR, 'train'), GT_PATCH_SIZE // SCALE, GT_STEP_SIZE // SCALE)
    preprocess_utils.save_patches(lr_test, os.path.join(SAVE_DIR_LR, 'val'), GT_PATCH_SIZE // SCALE, GT_STEP_SIZE // SCALE)

else:
    print("\nSaving testing patches...")

    preprocess_utils.save_patches(gt_list, SAVE_DIR_GT, GT_PATCH_SIZE, GT_STEP_SIZE)
    preprocess_utils.save_patches(lr_list, SAVE_DIR_LR, GT_PATCH_SIZE // SCALE, GT_STEP_SIZE // SCALE)


