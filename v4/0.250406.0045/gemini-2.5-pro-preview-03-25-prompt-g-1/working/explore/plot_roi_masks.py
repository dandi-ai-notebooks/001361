# This script plots the ROI masks superimposed on the mean image.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # For colormap, not styling

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get ROI table and select first 3 cell ROIs
roi_table = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
is_cell_column = roi_table["iscell"][:]
cell_indices = np.where(is_cell_column[:, 0] == 1)[0]
selected_roi_indices = cell_indices[:3]
selected_roi_ids = roi_table.id[selected_roi_indices]

# Get mean image
mean_img = nwb.processing["ophys"]["Backgrounds_0"].images["meanImg"].data[:]
# Get imaging plane dimensions for mask reconstruction
# The dimension is [height, width] or [num_rows, num_cols]
img_dims = nwb.acquisition["TwoPhotonSeries"].dimension[:] # e.g. [512 796]

# Create a composite image of selected ROI masks
roi_masks_combined = np.zeros(img_dims, dtype=float) # Use img_dims from TwoPhotonSeries
for i, roi_idx in enumerate(selected_roi_indices):
    pixel_mask_data = roi_table["pixel_mask"][roi_idx] # This is a list of [y, x, weight]
    for y, x, weight in pixel_mask_data:
        if 0 <= int(y) < img_dims[0] and 0 <= int(x) < img_dims[1]: # Check bounds
             # Add weight, can adjust how masks are combined if needed later
            roi_masks_combined[int(y), int(x)] = max(roi_masks_combined[int(y), int(x)], weight)


# Create plot: mean image with ROI masks overlaid
# No seaborn styling for image plots
plt.figure(figsize=(10, 10))
plt.imshow(mean_img, cmap='gray', aspect='auto')
# Overlay the combined ROI masks. Use a colormap that's visible on gray and has transparency.
# 'viridis' has good perceptual properties. Using alpha for transparency.
plt.imshow(roi_masks_combined, cmap=sns.cm.rocket_r, alpha=0.5, aspect='auto') # Use rocket_r for better visibility
plt.title(f"Mean Image with ROI Masks (IDs: {', '.join(map(str, selected_roi_ids))})")
plt.xlabel("X Pixels")
plt.ylabel("Y Pixels")
plt.colorbar(label="Max ROI Weight")
plt.savefig("explore/roi_masks_plot.png")
plt.close()

io.close()
print("Saved explore/roi_masks_plot.png")