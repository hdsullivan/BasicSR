import os
import numpy as np
import cv2
import argparse

def extract_slices_from_npz(npz_file, save_folder, slices = None):
    """Extract slices from npz files.

    Args:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
        compression_level (int): Compression level.
    """
    os.makedirs(save_folder, exist_ok=True)
    data = np.load(npz_file)['arr_0']
    if slices is None:
        slices = np.arange(data.shape[0])
    for slice in slices:
        img = data[slice, ...]
        # cv2.imwrite(os.path.join(save_folder, f'{slice}.tif'), img)
        img_norm = map_image_to_range(img)
        cv2.imwrite(os.path.join(save_folder, f'{slice}_norm.png'), img_norm)
        print(f'Saved {slice}_norm.png.')

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

    parser.add_argument("--npz_file", type=str, help="Path to the npz file.")
    parser.add_argument("--save_folder", type=str, help="Path to the save folder.")
    parser.add_argument("--slices", nargs="+", type=int, default=None, help="Specify the slices you want to extract. Default is all slices.")

    args = parser.parse_args()

    extract_slices_from_npz(args.npz_file, args.save_folder, args.slices)
