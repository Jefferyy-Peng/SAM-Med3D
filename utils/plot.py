import os
import matplotlib.pyplot as plt
from os.path import join

def plot_segmentation2D(img3D, prev_masks, gt3D, save_path, slice_axis=2, image_dice=None):
    """
        Plot each slice of a 3D image, its corresponding previous mask, and ground truth mask.

        Parameters:
        img3D (numpy.ndarray): The 3D image array of shape (depth, height, width).
        prev_masks (numpy.ndarray): The 3D array of previous masks of shape (depth, height, width).
        gt3D (numpy.ndarray): The 3D array of ground truth masks of shape (depth, height, width).
        slice_axis (int): The axis along which to slice the image (0=depth, 1=height, 2=width).
        """
    os.makedirs(save_path, exist_ok=True)
    # Determine the number of slices based on the selected axis
    num_slices = img3D.shape[slice_axis]

    # Iterate over each slice
    for n in range(num_slices):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
        # Slicing the numpy array along the specified axis
        if slice_axis == 0:
            img_slice = img3D[n, :, :]
            prev_mask_slice = prev_masks[n, :, :]
            gt_slice = gt3D[n, :, :]
        elif slice_axis == 1:
            img_slice = img3D[:, n, :]
            prev_mask_slice = prev_masks[:, n, :]
            gt_slice = gt3D[:, n, :]
        else:
            img_slice = img3D[:, :, n]
            prev_mask_slice = prev_masks[:, :, n]
            gt_slice = gt3D[:, :, n]

        if img_slice.max() == 0:
            continue

        # Plot image slice
        ax = axes[0]
        ax.imshow(img_slice, cmap='gray')
        ax.set_title(f'Slice {n + 1} - Image')
        ax.axis('off')

        # Plot previous mask slice
        cmap = plt.cm.get_cmap('viridis', 2)
        cmap.colors[0, 3] = 0
        ax = axes[1]
        ax.imshow(img_slice, cmap='gray')
        ax.imshow(prev_mask_slice, cmap=cmap, alpha=0.5)
        ax.set_title(f'Slice {n + 1} - Predict Mask')
        ax.axis('off')

        # Plot ground truth slice
        cmap = plt.cm.get_cmap('viridis', 2)
        cmap.colors[0, 3] = 0
        ax = axes[2]
        ax.imshow(img_slice, cmap='gray')
        ax.imshow(gt_slice, cmap=cmap, alpha=0.5)
        ax.set_title(f'Slice {n + 1} - Ground Truth')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(join(save_path, f'slice_{n}'))
        plt.close()