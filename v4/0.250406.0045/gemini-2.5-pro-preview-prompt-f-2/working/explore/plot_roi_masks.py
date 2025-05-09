# This script loads an NWB file and plots the max projection image with ROI masks.

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

    # Get max projection image
    max_proj_img = nwb.processing['ophys']['Backgrounds_0'].images['max_proj'].data[:]

    # Get ROI pixel masks
    plane_segmentation = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
    roi_pixel_masks = []
    for i in range(len(plane_segmentation.id)):
        pixel_mask = plane_segmentation['pixel_mask'][i]
        # Convert pixel mask from list of (y, x, weight) to a 2D array
        img_shape = nwb.acquisition['TwoPhotonSeries'].dimension[:] # This is [width, height]
        # The pixel_mask often comes as y,x,weight. The dimension is [width, height].
        # So we need to be careful with indexing.
        # Let's assume a consistent (height, width) for the image representation internally
        # The nwb_file_info output says TwoPhotonSeries.dimension is [512 796]
        # Let's assume that's [width, height] based on typical imaging conventions.
        # However, matplotlib imshow expects (row, col) which is (height, width).
        # Max_proj_img also has shape (796, 512). So this seems to be (height, width)
        mask_array = np.zeros((img_shape[1], img_shape[0]))
        for y, x, weight in pixel_mask:
            # Ensure indices are within bounds. Max proj is (796, 512)
            if 0 <= int(x) < img_shape[1] and 0 <= int(y) < img_shape[0]: # (height, width)
                 mask_array[int(x), int(y)] = weight # Using x for row, y for col based on typical mask output.
        roi_pixel_masks.append(mask_array)

    # Create a composite mask image (max of all ROI masks)
    if roi_pixel_masks:
        composite_mask = np.max(np.array(roi_pixel_masks), axis=0)
    else:
        composite_mask = np.zeros_like(max_proj_img)


    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(max_proj_img, cmap='gray')
    ax.imshow(composite_mask, cmap='jet', alpha=0.5) # Superimpose masks with transparency
    ax.set_title("Max Projection with ROI Masks")
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")
    plt.savefig("explore/roi_masks.png")
    plt.close()

print("Saved plot to explore/roi_masks.png")