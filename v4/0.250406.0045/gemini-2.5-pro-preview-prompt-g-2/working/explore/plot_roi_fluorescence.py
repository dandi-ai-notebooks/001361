# This script plots the fluorescence traces for a few selected ROIs.
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

    fluorescence_data = nwb.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['plane0']
    
    # Select a few ROIs to plot (e.g., first 5) and a time window
    num_rois_to_plot = 5
    # Plot data for the first 500 time points (approx 32 seconds)
    time_points_to_plot = 500 
    
    selected_rois_data = fluorescence_data.data[:time_points_to_plot, :num_rois_to_plot]
    
    sampling_rate = fluorescence_data.rate
    timestamps = np.arange(time_points_to_plot) / sampling_rate

    plt.figure(figsize=(15, 7))
    for i in range(num_rois_to_plot):
        # Offset traces for better visualization
        plt.plot(timestamps, selected_rois_data[:, i] + i * np.nanmax(selected_rois_data[:,i]) * 1.5, label=f'ROI {i+1}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence (a.u. + offset)')
    plt.title(f'Fluorescence Traces for First {num_rois_to_plot} ROIs (First {time_points_to_plot/sampling_rate:.2f} s)')
    # plt.legend() # Optionally add legend if needed, removed faulty line
    plt.yticks([]) # Remove y-axis ticks as traces are offset
    plt.savefig('explore/roi_fluorescence_traces.png')
    plt.close()

print("Saved explore/roi_fluorescence_traces.png")