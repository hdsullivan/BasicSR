import os
import numpy as np
import cv2
import argparse
from sklearn.model_selection import train_test_split

def extract_paired_slices_from_npz(gt_npz, lr_npz, save_dir_gt, save_dir_lr, scale, training_data = True, slices = None):
    """Extract slices from npz files.

    Args:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
        compression_level (int): Compression level.
    """
    if training_data:
        os.makedirs(os.path.join(save_dir_gt, 'train'), exist_ok=True)
        os.makedirs(os.path.join(save_dir_gt, 'val'), exist_ok=True)
        os.makedirs(os.path.join(save_dir_lr, 'train'), exist_ok=True)
        os.makedirs(os.path.join(save_dir_lr, 'val'), exist_ok=True)
    else:
        os.makedirs(os.path.join(save_dir_gt), exist_ok=True)
        os.makedirs(os.path.join(save_dir_lr), exist_ok=True)

    gt = np.load(gt_npz)['arr_0']
    lr = np.load(lr_npz)['arr_0']

    gt = gt[:, :(scale*lr.shape[1]), :(scale*lr.shape[1])]
    lr = lr[:, :, :lr.shape[1]]

    gt_list = []
    lr_list = []
    if slices is None:
        slices = np.arange(lr.shape[0])
    for slice in slices:
        img_lr = lr[slice, ...]
        img_lr_norm = map_image_to_range(img_lr)
        lr_list.append(img_lr_norm)

        img_gt = gt[scale * slice, ...]
        img_gt_norm = map_image_to_range(img_gt)
        gt_list.append(img_gt_norm)

        print(f'Loaded pair for slice {scale * slice}')

    if training_data:
        gt_train, gt_test, lr_train, lr_test = train_test_split(gt_list, lr_list, test_size=0.2, random_state=42)

        gt_train_mean = np.mean(gt_train)
        gt_test_mean = np.mean(gt_test)
        lr_train_mean = np.mean(lr_train)
        lr_test_mean = np.mean(lr_test)

        idx = 0
        for img_lr, img_gt in zip(lr_train, gt_train):
            cv2.imwrite(os.path.join(save_dir_lr, 'train', f'{idx}.png'), img_lr - lr_train_mean)
            cv2.imwrite(os.path.join(save_dir_gt, 'train', f'{idx}.png'), img_gt - gt_train_mean)
            idx += 1
        idx = 0
        for img_lr, img_gt, slice in zip(lr_test, gt_test, slices):
            cv2.imwrite(os.path.join(save_dir_lr, 'val', f'{idx}.png'), img_lr - lr_test_mean)
            cv2.imwrite(os.path.join(save_dir_gt, 'val', f'{idx}.png'), img_gt - gt_test_mean)
            idx += 1
    else:
        gt = np.array(gt_list)
        lr = np.array(lr_list)

        gt_mean = np.mean(gt)
        lr_mean = np.mean(lr)

        idx = 0
        for img_lr, img_gt in zip(lr, gt):
            cv2.imwrite(os.path.join(save_dir_lr, f'{scale * idx}.png'), img_lr - lr_mean)
            cv2.imwrite(os.path.join(save_dir_gt, f'{scale * idx}.png'), img_gt - gt_mean)
            idx += 1

    print('Done.')

def map_image_to_range(image):
    """Map image to (0, 255) range.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image mapped to (0, 255) range.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    mapped_image = (image - min_val) * (255 / (max_val - min_val))
    return mapped_image.astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract slices from npz files.")

    parser.add_argument("--gt_npz", type=str, help="Path to the ground truth npz file.")
    parser.add_argument("--lr_npz", type=str, help="Path to the low resolution npz file.")
    parser.add_argument("--save_dir_gt", type=str, help="Path to the save folder for ground truth slices.")
    parser.add_argument("--save_dir_lr", type=str, help="Path to the save folder for low resolution slices.")
    parser.add_argument("--scale", type=int, default=1, help="Scale of the slices.")
    parser.add_argument("--slices", nargs="+", type=int, default=None, help="Specify the slices you want to extract. Default is all slices.")

    args = parser.parse_args()

    extract_paired_slices_from_npz(args.gt_npz, args.lr_npz, args.save_dir_gt, args.save_dir_lr, args.scale, args.slices)
