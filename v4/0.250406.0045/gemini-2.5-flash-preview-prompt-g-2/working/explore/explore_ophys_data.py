# This script explores the optical physiology data and visualizes fluorescence traces and ROIs

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get ophys data
ophys_module = nwb.processing["ophys"]
fluorescence = ophys_module.data_interfaces["Fluorescence"]
roi_response_series = fluorescence.roi_response_series["plane0"]

# Get fluorescence data (limit to first 10000 samples and first 10 ROIs)
fluorescence_data = roi_response_series.data[0:10000, 0:10]

# Generate timestamps using starting_time and rate
starting_time = roi_response_series.starting_time
rate = roi_response_series.rate
num_samples = fluorescence_data.shape[0]
timestamps = starting_time + np.arange(num_samples) / rate

# Plot fluorescence traces for a few ROIs
plt.figure(figsize=(12, 6))
for i in range(fluorescence_data.shape[1]):
    plt.plot(timestamps, fluorescence_data[:, i] + i * 50, label=f'ROI {i}') # Offset for visualization clarity
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (lumens)')
plt.title('Fluorescence Traces for First 10 ROIs')
plt.legend()
plt.savefig("explore/fluorescence_traces.png")
print("Fluorescence traces plot saved to explore/fluorescence_traces.png")

# Explore ImageSegmentation and ROIs
image_segmentation = ophys_module.data_interfaces["ImageSegmentation"]
plane_segmentation = image_segmentation.plane_segmentations["PlaneSegmentation"]

# Get ROIs as a DataFrame and print the first few rows
roi_df = plane_segmentation.to_dataframe()
print("\nFirst 5 rows of PlaneSegmentation DataFrame:")
print(roi_df.head())

# Visualize pixel masks for a few ROIs
plt.figure(figsize=(6, 6))
max_image_shape = (512, 796) # Assuming shape from NWB file info
roi_masks = np.zeros(max_image_shape)
num_rois_to_visualize = 5
for i in range(num_rois_to_visualize):
    if i < len(plane_segmentation["pixel_mask"]):
        mask_data = plane_segmentation["pixel_mask"][i]
        for x, y, wt in mask_data:
            if 0 <= y < max_image_shape[0] and 0 <= x < max_image_shape[1]:
                roi_masks[int(y), int(x)] += wt

plt.imshow(roi_masks, cmap='hot')
plt.title(f'Pixel Masks for First {num_rois_to_visualize} ROIs (Superimposed)')
plt.axis('off')
plt.savefig("explore/roi_masks.png")
print(f"ROI masks plot saved to explore/roi_masks.png")


io.close()