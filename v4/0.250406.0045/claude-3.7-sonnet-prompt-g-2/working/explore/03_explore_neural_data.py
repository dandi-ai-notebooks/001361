"""
This script explores the neural activity data from the two-photon calcium imaging.
It examines the fluorescence data, ROIs, and correlations between neural activity
and behavioral variables.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Configure seaborn
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get ophys data
ophys = nwb.processing["ophys"]
fluorescence = ophys.data_interfaces["Fluorescence"]
deconvolved = ophys.data_interfaces["Deconvolved"]
image_segmentation = ophys.data_interfaces["ImageSegmentation"]

# Get ROI information
plane_seg = image_segmentation.plane_segmentations["PlaneSegmentation"]
print(f"Number of ROIs: {len(plane_seg.id.data[:])}")
print(f"Columns in ROI table: {plane_seg.colnames}")

# Get iscell data to filter for real cells vs neuropil
iscell_data = plane_seg.iscell.data[:]
real_cells = np.where(iscell_data[:, 0] == 1)[0]
print(f"Number of real cells (iscell=1): {len(real_cells)}")

# Get fluorescence traces
fluor_plane0 = fluorescence.roi_response_series["plane0"]
fluor_data = fluor_plane0.data[:]
fluor_rate = fluor_plane0.rate

# Get background images
backgrounds = ophys.data_interfaces["Backgrounds_0"]
mean_img = backgrounds.images["meanImg"].data[:]
max_proj = backgrounds.images["max_proj"].data[:]

# Plot mean and max projection images
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
plt.imshow(mean_img, cmap='gray')
plt.title('Mean Image')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(max_proj, cmap='gray')
plt.title('Max Projection')
plt.colorbar()

plt.tight_layout()
plt.savefig('explore/roi_images.png')
plt.close()

# Sample neural data (first 2000 timepoints, 10 random real cells)
sample_time = 2000
if len(real_cells) > 10:
    sample_neurons = np.random.choice(real_cells, 10, replace=False)
else:
    sample_neurons = real_cells

# Get sampled fluorescence data
sampled_fluor = fluor_data[:sample_time, sample_neurons]

# Get corresponding timestamps (approximate from rate)
timestamps = np.arange(sample_time) / fluor_rate

# Plot fluorescence traces for sample neurons
plt.figure(figsize=(14, 10))
for i, neuron_idx in enumerate(sample_neurons):
    offset = i * np.std(sampled_fluor[:, 0]) * 3  # Offset for visualization
    plt.plot(timestamps, sampled_fluor[:, i] + offset, label=f'Neuron {neuron_idx}')
    
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.) + offset')
plt.title('Fluorescence Traces for Sample Neurons')
plt.grid(True, alpha=0.3)
plt.savefig('explore/fluorescence_traces.png')
plt.close()

# Get behavioral data to correlate with neural activity
behavior = nwb.processing["behavior"]
behavioral_ts = behavior.data_interfaces["BehavioralTimeSeries"]

# Get position and speed data
position_ts = behavioral_ts.time_series["position"]
speed_ts = behavioral_ts.time_series["speed"]
reward_zone_ts = behavioral_ts.time_series["reward_zone"]

# Downsample behavioral data to match imaging data if needed
behav_data_rate = 1 / np.mean(np.diff(position_ts.timestamps[:100]))
print(f"Behavior data rate: {behav_data_rate:.2f} Hz")
print(f"Imaging data rate: {fluor_rate:.2f} Hz")

# Since the rates might differ, we need to find corresponding behavior timepoints
# for our sample fluorescence data
fluor_times = timestamps  # Our sampled fluorescence timestamps
position_times = position_ts.timestamps[:]

# We'll need to manually align the behavioral data to the fluorescence data
# since h5py datasets can't be indexed with unsorted indices
sample_position = np.zeros_like(fluor_times)
sample_speed = np.zeros_like(fluor_times)

# For each fluorescence timestamp, find the nearest behavior timestamp
for i, f_time in enumerate(fluor_times):
    # Find the index of the closest timestamp
    idx = np.argmin(np.abs(position_times - f_time))
    sample_position[i] = position_ts.data[idx]
    sample_speed[i] = speed_ts.data[idx]

# Plot neural activity of a single cell vs behavioral variables
example_cell_idx = sample_neurons[0]  # Choose the first sampled neuron
example_fluor = sampled_fluor[:, 0]  # Corresponding fluorescence trace

# Create figure with three subplots
plt.figure(figsize=(14, 12))
gs = GridSpec(3, 1, figure=plt.gcf())

# Plot fluorescence vs time
ax1 = plt.subplot(gs[0])
ax1.plot(fluor_times, example_fluor, 'b-', linewidth=1.5)
ax1.set_title(f'Fluorescence vs Time (Neuron {example_cell_idx})')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Fluorescence (a.u.)')

# Plot position vs time
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(fluor_times, sample_position, 'g-', linewidth=1.5)
ax2.set_title('Position vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (cm)')

# Plot speed vs time
ax3 = plt.subplot(gs[2], sharex=ax1)
ax3.plot(fluor_times, sample_speed, 'r-', linewidth=1.5)
ax3.set_title('Speed vs Time')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Speed (cm/s)')

plt.tight_layout()
plt.savefig('explore/neuron_vs_behavior.png')
plt.close()

# Calculate correlation between neural activity and position
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(sample_position, example_fluor, alpha=0.5)
plt.xlabel('Position (cm)')
plt.ylabel('Fluorescence (a.u.)')
plt.title(f'Neuron {example_cell_idx} Activity vs Position')

plt.subplot(1, 2, 2)
plt.scatter(sample_speed, example_fluor, alpha=0.5)
plt.xlabel('Speed (cm/s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title(f'Neuron {example_cell_idx} Activity vs Speed')

plt.tight_layout()
plt.savefig('explore/correlations.png')
plt.close()

# Plot fluorescence heatmap for multiple neurons
plt.figure(figsize=(14, 8))
sns.heatmap(sampled_fluor.T, cmap='viridis', 
            xticklabels=500, yticklabels=[f"Neuron {idx}" for idx in sample_neurons])
plt.xlabel('Time (frames)')
plt.ylabel('Neurons')
plt.title('Fluorescence Heatmap')
plt.tight_layout()
plt.savefig('explore/fluorescence_heatmap.png')
plt.close()

# Print correlations between neural activity and behavioral variables for all sampled neurons
print("\nCorrelations between neural activity and behavioral variables:")
for i, neuron_idx in enumerate(sample_neurons):
    neuron_fluor = sampled_fluor[:, i]
    position_corr = np.corrcoef(neuron_fluor, sample_position)[0, 1]
    speed_corr = np.corrcoef(neuron_fluor, sample_speed)[0, 1]
    print(f"Neuron {neuron_idx}: Position correlation = {position_corr:.3f}, Speed correlation = {speed_corr:.3f}")