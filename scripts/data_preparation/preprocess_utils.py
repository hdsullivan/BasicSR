import os
import numpy as np
import cv2
import scipy
from os import path as osp
import shutil

def hist_cluster_fixed(recon, peaks=[5000, 6000], type='float16', ret_th=0.00001, sub_sampling_factor=10):
    """
    Perform histogram clustering on the input reconstruction image.

    Args:
        recon (ndarray): The input reconstruction image.
        peaks (list, optional): The peak values used for clustering. Defaults to [5000, 15000].
        type (str, optional): The type of the output image. Defaults to 'uint16'.
        ret_th (int, optional): The threshold value for the output image. Defaults to 0.
        sub_sampling_factor (int, optional): The factor used for subsampling the input image. Defaults to 10.

    Returns:
        tuple: A tuple containing the output image, the median value of the pixels below the threshold,
               and the median value of the pixels above the threshold.
    """
    peak_1 = peaks[0]
    peak_2 = peaks[1]

    output = scipy.ndimage.zoom(recon, 1 / sub_sampling_factor, order=0)

    lt_indices = np.where(output < ret_th)
    ge_indices = np.where(output >= ret_th)

    mp1 = np.median(output[lt_indices])
    mp2 = np.median(output[ge_indices])

    output = calc_output(recon, peak_1, peak_2, mp1, mp2)
    if type == 'uint16':
        output = np.clip(output, 0, 65535).astype(type)
    elif type == 'float32' or type == 'float16':
        output = output / 65535.0
        output = np.clip(output, 0.0, 1.0)
        # if type == 'float16':
        #     output = np.float16(output)

    return output, mp1, mp2


import numpy as np

def calc_output(x, p1, p2, mp1, mp2):
    """
    Parameters:
    x (ndarray): Input array.
    p1 (float): Lower bound of the output range.
    p2 (float): Upper bound of the output range.
    mp1 (ndarray): Lower bound of the input range.
    mp2 (ndarray): Upper bound of the input range.

    Returns:
    ndarray: Output array with values scaled and shifted according to the given parameters.
    """
    dt = x.dtype
    x = x.astype('float32')
    output = np.subtract(x, mp1.astype(dt), out=x)
    scale = (p2 - p1) / (mp2 - mp1)
    output *= scale
    output += p1
    output = output.astype(dt)
    return output

def make_dir(dir_path):
    """
    Create a directory at the specified path if it doesn't exist.
    If the directory already exists, prompt the user to overwrite it or skip.

    Args:
        dir_path (str): The path of the directory to be created.

    Raises:
        ValueError: If the user provides an invalid input (not 'Y' or 'N').

    Returns:
        None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'\nCreated directory: {dir_path}')
    else:
        print(f'\nFolder {dir_path} already exists.')
        user_response = input('Do you want to overwrite it? Y/N\n')
        if user_response.lower() == 'y':
            print(f'Overwriting...')
            shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            print("     Done overwriting directory.")
        elif user_response.lower() == 'n':
            print(f'Skipping...')
        else:
            raise ValueError('Wrong input. Only accepts Y/N.')

def save_patches(img_list, save_folder, crop_size, step, compression_level=3):
    """Crop images into subimages and save them.

    Args:
        img_list (list): List of input images.
        save_folder (str): Path to the folder where subimages will be saved.
        crop_size (int): Size of the subimages.
        step (int): Step size for cropping.
        compression_level (int, optional): Compression level for saving images. Defaults to 3.
    """

    idx = 0
    for img in img_list:
        img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
        img_rgb[..., 0] = img
        img_rgb[..., 1] = img
        img_rgb[..., 2] = img
        process_img(img_rgb, str(idx) + ".tif", crop_size, step, save_folder, compression_level)
        idx += 1


def process_img(img, img_name, crop_size, step, save_folder, compression_level):
    """Process an image by cropping it into patches.

    This function takes an input image and crops it into patches of a specified size. The cropped patches are saved
    in a specified folder with a specific compression level.

    Args:
        img (numpy.ndarray): The input image to be processed.
        img_name (str): The name of the image file.
        crop_size (int): The size of the patches to be cropped.
        step (int): The step size for the overlapped sliding window.
        save_folder (str): The path to the folder where the cropped patches will be saved.
        compression_level (int): The compression level for saving the patches as PNG files.

    Returns:
        str: A process information string displayed in the progress bar.
    """
    idx_str, extension = osp.splitext(osp.basename(img_name))

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > crop_size:
        h_space = np.append(h_space, h - crop_size)

    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > crop_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(save_folder, f'{idx_str}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, compression_level])


def extract_paired_slices(gt, lr, scale, cropped_to_object, thresh_range=0.8, patch_thresh=120):
    """
    Extract slices from npz files.

    Args:
        gt (numpy.ndarray): The ground truth array.
        lr (numpy.ndarray): The low-resolution array.
        scale (int): The scaling factor.
        cropped_to_object (bool): Flag indicating whether to crop to the object.
        thresh_range (float, optional): The threshold range for image intensity. Defaults to 0.8.
        patch_thresh (int, optional): The threshold for width and height of detected object. Defaults to 120.

    Returns:
        list: A list of ground truth slices.
        list: A list of low-resolution slices.
    """
    gt = gt[:, :(scale * lr.shape[1]), :(scale * lr.shape[1])]
    lr = lr[:, :, :lr.shape[1]]

    gt_list = []
    lr_list = []

    for slice in range(lr.shape[0]):
        img_lr = lr[slice, ...]
        img_gt = gt[scale * slice, ...]

        if cropped_to_object:
            img_gt, img_lr = crop_to_object(img_gt, img_lr, scale, slice, thresh_range, patch_thresh)

        if img_gt is not None and img_lr is not None:
            gt_list.append(img_gt)
            lr_list.append(img_lr)

    return gt_list, lr_list


def crop_to_object(img_gt, img_lr, scale, slice, thresh_range, patch_thresh):
    """
    Crop the image to the object using Canny edge detection.

    Args:
        img_gt (numpy.ndarray): The ground truth image.
        img_lr (numpy.ndarray): The low-resolution image.
        scale (int): The scaling factor.
        slice (int): The slice number.
        thresh_range (float): The threshold range for image intensity.
        patch_thresh (int): The threshold for width and height of detected object.

    Returns:
        numpy.ndarray: The cropped ground truth image.
        numpy.ndarray: The cropped low-resolution image.
    """
    if np.max(img_gt) - np.min(img_gt) > thresh_range:
        img_gt_uint = (255.0 * img_gt).astype(np.uint8)
        med_val = np.median(img_gt_uint)
        lower = int(max(0 ,0.7*med_val))
        upper = int(min(255,1.3*med_val))
        edged = cv2.Canny(img_gt_uint, lower, upper)
        contours,_ = cv2.findContours(edged, 1, 1)

        if len(contours) > 0:
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)

            if w > patch_thresh and h > patch_thresh:
                print(f'Found Object in slice {scale * slice} with bbox of size = {w}x{h} pixels.')
                img_gt = img_gt[y:y+h, x:x+w]
                img_lr = img_lr[y // scale:y // scale + h // scale, x // scale:x // scale + w // scale]

                return img_gt, img_lr
    return None, None