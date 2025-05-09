# This script plots the ROI masks superimposed on the mean image.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
with pynwb.NWBHDF5IO(file=h5_file, mode='r') as io:
    nwb = io.read()

    mean_image = nwb.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg'].data[:]
    plane_segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
    
    # Get image dimensions (height, width)
    # The nwb-file-info showed TwoPhotonSeries.dimension as [512 796] (height, width)
    # Pixel masks are (y, x, weight) or (row, col, weight) relative to this.
    img_dims = tuple(nwb.acquisition['TwoPhotonSeries'].dimension[:]) # (height, width)

    # Create an empty array for the combined ROI masks
    combined_roi_mask_img = np.zeros(img_dims, dtype=float)

    num_rois = len(plane_segmentation.id)

    for i in range(num_rois):
        pixel_mask = plane_segmentation['pixel_mask'][i] # This gives (y, x, weight) tuples
        for y, x, weight in pixel_mask:
            if 0 <= int(y) < img_dims[0] and 0 <= int(x) < img_dims[1]:
                 # Use weight directly, or simply mark as 1 if binary mask is preferred
                combined_roi_mask_img[int(y), int(x)] = max(combined_roi_mask_img[int(y), int(x)], weight)
    
    plt.figure(figsize=(10, 10 * img_dims[0]/img_dims[1] if img_dims[1] > 0 else 10))
    plt.imshow(mean_image, cmap='gray', aspect='auto')
    # Overlay ROI masks with some transparency
    # Using a hot colormap for masks and only showing non-zero pixels
    masked_roi_img = np.ma.masked_where(combined_roi_mask_img == 0, combined_roi_mask_img)
    plt.imshow(masked_roi_img, cmap='hot', alpha=0.6, aspect='auto') 
    plt.title('ROI Masks Superimposed on Mean Image')
    plt.axis('off')
    plt.savefig('explore/roi_masks_on_mean_image.png')
    plt.close()

print("Saved explore/roi_masks_on_mean_image.png")