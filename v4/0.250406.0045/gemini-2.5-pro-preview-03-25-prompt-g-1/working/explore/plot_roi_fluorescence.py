# This script plots the fluorescence traces for a few selected ROIs.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get ROI fluorescence data (Deconvolved)
fluorescence_series = nwb.processing["ophys"]["Deconvolved"].roi_response_series["plane0"]
fluorescence_data = fluorescence_series.data
timestamps = np.arange(fluorescence_data.shape[0]) / fluorescence_series.rate

# Get ROI table and select first 3 cell ROIs
roi_table = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
is_cell_column = roi_table["iscell"][:]  # Load iscell data into memory
cell_indices = np.where(is_cell_column[:, 0] == 1)[0] # iscell is the first column
selected_roi_indices = cell_indices[:3]
selected_roi_ids = roi_table.id[selected_roi_indices]

# Create plot
sns.set_theme()
plt.figure(figsize=(15, 7))
for i, roi_idx in enumerate(selected_roi_indices):
    roi_id = selected_roi_ids[i]
    plt.plot(timestamps, fluorescence_data[:timestamps.shape[0], roi_idx], label=f"ROI {roi_id}")

plt.xlabel("Time (s)")
plt.ylabel(f"Fluorescence ({fluorescence_series.unit})")
plt.title("Fluorescence Traces for Selected ROIs (Deconvolved)")
plt.legend()
plt.savefig("explore/roi_fluorescence_plot.png")
plt.close()

io.close()
print("Saved explore/roi_fluorescence_plot.png")