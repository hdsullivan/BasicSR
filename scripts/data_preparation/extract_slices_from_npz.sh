# SCALE=3
# MAIN_DIR=S:/NAOA/Projects/AM-SuperResolution/datasets
# GT_PATH=$MAIN_DIR/main_data/GT/noBH/Al/CAD_M0s277_PB_REC_FDK_Al_CleanGT_noBH_145_Views_shortANDsparse.npz
# LR_PATH=$MAIN_DIR/main_data/down_sampled/CADRotated_wFlaws_REC_FDK_Al_noBH_X$SCALE'binning_NOnoise_NOblur_145_Views_shortANDsparse'
# SAVE_DIR_GT=$MAIN_DIR/noBH/Al/CADRotated_wFlaws_REC_FDK_145_Views_shortANDsparse/X$SCALE/GT
# SAVE_DIR_LR=$MAIN_DIR/noBH/Al/CADRotated_wFlaws_REC_FDK_145_Views_shortANDsparse/X$SCALE/LR_NOnoise_NOblur


conda activate basic_sr

# python extract_slices_from_npz.py --gt_npz $GT_PATH --lr_npz $LR_PATH --save_dir_gt $SAVE_DIR_GT --save_dir_lr $SAVE_DIR_LR --scale $SCALE

python scripts/data_preparation/extract_slices_from_npz.py --gt_npz S:/NAOA/Projects/AM-SuperResolution/datasets/main_data/GT/noBH/Al/CADRotated_wFlaws_REC_FDK_Al_CleanGT_noBH_2132_Views_fullscan.npz --lr_npz S:/NAOA/Projects/AM-SuperResolution/datasets/main_data/down_sampled/noBH/2X/Al/145view_Shortscan/NoNoise_NoBlur/CADRotated_wFlaws_REC_FDK_Al_noBH_2Xbinning_NOnoise_NOblur_145_Views_shortANDsparse.npz --save_dir_gt S:/NAOA/Projects/AM-SuperResolution/datasets/noBH/Al/X2/GT --save_dir_lr S:/NAOA/Projects/AM-SuperResolution/datasets/noBH/Al/X2/CADRotated_wFlaws_REC_FDK_145_Views_shortANDsparse/LR_NOnoise_NOblur --scale 2

# python scripts/data_preparation/extract_slices_from_npz.py --gt_npz S:/NAOA/Projects/AM-SuperResolution/datasets/main_data/GT/noBH/Al/CADRotated_wFlaws_REC_FDK_Al_CleanGT_noBH_145_Views_shortANDsparse.npz --lr_npz S:/NAOA/Projects/AM-SuperResolution/datasets/main_data/down_sampled/noBH/3X/Recons/CADRotated_wFlaws_REC_FDK_Al_noBH_3Xbinning_NOnoise_NOblur_145_Views_shortANDsparse.npz --save_dir_gt S:/NAOA/Projects/AM-SuperResolution/datasets/noBH/Al/CADRotated_wFlaws_REC_FDK_145_Views_shortANDsparse/X3/GT --save_dir_lr S:/NAOA/Projects/AM-SuperResolution/datasets/noBH/Al/CADRotated_wFlaws_REC_FDK_145_Views_shortANDsparse/X3/LR_NOnoise_NOblur --scale 3

# python scripts/data_preparation/extract_slices_from_npz.py --gt_npz S:/NAOA/Projects/AM-SuperResolution/datasets/main_data/GT/noBH/Al/CADRotated_wFlaws_REC_FDK_Al_CleanGT_noBH_145_Views_shortANDsparse.npz --lr_npz S:/NAOA/Projects/AM-SuperResolution/datasets/main_data/down_sampled/noBH/4X/Recons/CADRotated_wFlaws_REC_FDK_Al_noBH_4Xbinning_NOnoise_NOblur_145_Views_shortANDsparse.npz --save_dir_gt S:/NAOA/Projects/AM-SuperResolution/datasets/noBH/Al/CADRotated_wFlaws_REC_FDK_145_Views_shortANDsparse/X4/GT --save_dir_lr S:/NAOA/Projects/AM-SuperResolution/datasets/noBH/Al/CADRotated_wFlaws_REC_FDK_145_Views_shortANDsparse/X4/LR_NOnoise_NOblur --scale 4