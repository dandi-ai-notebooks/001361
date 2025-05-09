# This script loads and plots fluorescence traces for a few ROIs and visualizes ROI locations on the mean image.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get fluorescence data and ROI table
fluorescence_ts = nwb.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['plane0']
roi_table = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']

# Load fluorescence data for a few ROIs
# Select the first 5 ROIs for plotting
n_rois_to_plot = 5
fluorescence_data_subset = fluorescence_ts.data[:, :n_rois_to_plot]
# Generate timestamps from the TwoPhotonSeries rate and starting_time
two_photon_series = nwb.acquisition["TwoPhotonSeries"]
rate = two_photon_series.rate
starting_time = two_photon_series.starting_time
num_frames = fluorescence_ts.data.shape[0]
timestamps = starting_time + np.arange(num_frames) / rate

# Plot fluorescence traces
plt.figure()
plt.plot(timestamps, fluorescence_data_subset)
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (lumens)')
plt.title(f'Fluorescence Traces for First {n_rois_to_plot} ROIs')
plt.savefig('explore/fluorescence_traces.png')
plt.close()

# Load the mean image
mean_img = nwb.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg'].data[:]

# Get ROI centroids for iscell == 1
iscell_data = roi_table['iscell'][:]
pixel_masks = roi_table['pixel_mask'][:]

roi_centroids = []
for i, mask in enumerate(pixel_masks):
    # Check if the ROI is classified as a cell and if the mask is not empty and has the expected shape (n, >=2)
    # Assuming mask is a list of [x, y, value] or similar triplets for each pixel in the mask
    if iscell_data[i, 0] == 1 and mask.shape and mask.shape[0] > 0 and mask.shape[1] >= 2:
        # Calculate centroid as the mean of x and y coordinates
        x_coords = mask[:, 0]
        y_coords = mask[:, 1]
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        roi_centroids.append((centroid_x, centroid_y))

roi_centroids = np.array(roi_centroids)

# Plot mean image with ROI centroids
plt.figure()
plt.imshow(mean_img, cmap='gray', origin='lower') # Use origin='lower' as pixel coordinates are typically (x, y) from bottom-left
if roi_centroids.shape[0] > 0:
    plt.plot(roi_centroids[:, 0], roi_centroids[:, 1], 'ro', markersize=5, alpha=0.5) # 'ro' for red circles
plt.title('Mean Image with IsCell ROI Centroids')
plt.xlabel('Image X')
plt.ylabel('Image Y')
plt.savefig('explore/roi_centroids_on_meanimg.png')
plt.close()

io.close()