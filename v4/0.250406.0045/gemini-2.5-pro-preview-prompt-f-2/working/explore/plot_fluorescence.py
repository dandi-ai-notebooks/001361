# This script loads an NWB file and plots fluorescence traces for a few selected ROIs.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
with pynwb.NWBHDF5IO(file=h5_file, mode='r') as io:
    nwb = io.read()

    fluorescence_data = nwb.processing['ophys']['Fluorescence']['plane0'].data
    timestamps = np.arange(fluorescence_data.shape[0]) / nwb.processing['ophys']['Fluorescence']['plane0'].rate

    # Select a few ROIs to plot
    rois_to_plot = [0, 10, 20, 30, 40] # Example ROI indices
    roi_ids = nwb.processing['ophys']['Fluorescence']['plane0'].rois.table.id[:]

    plt.figure(figsize=(15, 8))
    for i, roi_idx in enumerate(rois_to_plot):
        if roi_idx < fluorescence_data.shape[1]:
            plt.plot(timestamps[:5000], fluorescence_data[:5000, roi_idx] + i * 1.5, label=f"ROI {roi_ids[roi_idx]}") # Plot first 5000 timepoints for clarity and add offset for visualization
    plt.xlabel("Time (s)")
    plt.ylabel("Fluorescence (arbitrary units, offset for clarity)")
    plt.title("Fluorescence Traces for Selected ROIs (First 5000 Timesteps)")
    plt.legend(title="ROI ID")
    plt.grid(True)
    plt.savefig("explore/fluorescence_traces.png")
    plt.close()

print("Saved plot to explore/fluorescence_traces.png")