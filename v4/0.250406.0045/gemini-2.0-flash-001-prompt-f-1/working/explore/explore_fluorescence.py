import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This script explores the fluorescence data for cells vs non-cells in the NWB file.

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the fluorescence data
plane_segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
iscell_data = plane_segmentation.iscell[:]
iscell_df = pd.DataFrame(iscell_data, columns=['iscell', 'probcell'])

fluorescence = nwb.processing['ophys'].data_interfaces['Fluorescence']
roi_response_series = fluorescence.roi_response_series['plane0']
fluorescence_data = roi_response_series.data[:]

# Separate fluorescence data for cells and non-cells
cell_indices = iscell_df[iscell_df['iscell'] > 0.5].index
noncell_indices = iscell_df[iscell_df['iscell'] <= 0.5].index

cell_fluorescence = fluorescence_data[:, cell_indices]
noncell_fluorescence = fluorescence_data[:, noncell_indices]

# Calculate mean fluorescence for cells and non-cells
mean_cell_fluorescence = np.mean(cell_fluorescence, axis=1)
mean_noncell_fluorescence = np.mean(noncell_fluorescence, axis=1)

# Plot the mean fluorescence for cells and non-cells
plt.figure(figsize=(10, 6))
plt.plot(mean_cell_fluorescence, label='Cells')
plt.plot(mean_noncell_fluorescence, label='Non-cells')
plt.xlabel('Time (frame)')
plt.ylabel('Mean Fluorescence (lumens)')
plt.title('Mean Fluorescence for Cells vs. Non-cells')
plt.legend()
plt.savefig('explore/fluorescence_cells_vs_noncells.png')
plt.close()