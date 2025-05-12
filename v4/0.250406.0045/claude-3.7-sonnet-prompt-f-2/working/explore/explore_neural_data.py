# This script explores neural data from the calcium imaging recordings in the NWB file
# Looking at fluorescence signals and how they relate to behavior

import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

print("Accessing neural data...")

# Access fluorescence data paths
ophys_path = '/processing/ophys/'

# Get a list of data interfaces in the ophys processing module
print("Available data interfaces in ophys:")
for key in h5_file[ophys_path].keys():
    print(f"  - {key}")

# Get fluorescence data (this is the calcium imaging data)
# We'll use a small subset of the data for efficiency
time_window = slice(0, 2000)  # Take first 2000 timepoints
neuron_subset = slice(0, 50)  # Take first 50 neurons

fluorescence_data = h5_file[f'{ophys_path}Fluorescence/plane0/data'][time_window, neuron_subset]
# Get the sampling rate for the fluorescence data
rate = h5_file[f'{ophys_path}Fluorescence/plane0/starting_time'].attrs['rate']
timestamps = np.arange(fluorescence_data.shape[0]) / rate

# Get information about the ROIs
print("\nAccessing ROI information...")
roi_table_path = f'{ophys_path}ImageSegmentation/PlaneSegmentation/'

# Get the cell classification (neural ROIs vs non-neural components)
iscell_data = h5_file[f'{roi_table_path}iscell'][:]
print(f"iscell data shape: {iscell_data.shape}")
if iscell_data.shape[1] >= 2:  # If there's a confidence column
    cell_confidence = iscell_data[:, 1][neuron_subset]
    print(f"Cell confidence range: {np.min(cell_confidence)} to {np.max(cell_confidence)}")

# Print basic information about the fluorescence data
print(f"\nFluorescence data shape: {fluorescence_data.shape}")
print(f"Sampling rate: {rate} Hz")

# Plot mean fluorescence across all neurons over time
mean_fluorescence = np.mean(fluorescence_data, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(timestamps, mean_fluorescence)
plt.xlabel('Time (s)')
plt.ylabel('Mean Fluorescence (a.u.)')
plt.title('Mean Fluorescence Signal Across Neurons Over Time')
plt.savefig(os.path.join('explore', 'mean_fluorescence.png'))
plt.close()

# Plot fluorescence traces for a few individual neurons
plt.figure(figsize=(12, 8))
for i in range(min(5, fluorescence_data.shape[1])):
    plt.plot(timestamps, fluorescence_data[:, i] + i*4, label=f'Neuron {i}')
    
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.) + Offset')
plt.title('Fluorescence Traces for Individual Neurons')
plt.legend()
plt.savefig(os.path.join('explore', 'individual_neurons.png'))
plt.close()

# Create a heatmap of neural activity
plt.figure(figsize=(12, 8))
# Use a custom colormap centered at zero
cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                         ['blue', 'white', 'red'], 
                                         N=256)

# Normalize the data for better visualization
activity_normalized = (fluorescence_data - np.mean(fluorescence_data, axis=0)) / np.std(fluorescence_data, axis=0)
plt.imshow(activity_normalized.T, aspect='auto', cmap=cmap,
           extent=[0, timestamps[-1], 0, fluorescence_data.shape[1]])
plt.colorbar(label='Normalized Fluorescence')
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.title('Heatmap of Neural Activity')
plt.savefig(os.path.join('explore', 'neural_heatmap.png'))
plt.close()

# Now let's get some behavioral data to correlate with neural activity
behavior_path = '/processing/behavior/BehavioralTimeSeries/'

# Get position data
position_data = h5_file[f'{behavior_path}position/data'][:]
position_timestamps = h5_file[f'{behavior_path}position/timestamps'][:]

# Get reward zone data
reward_zone_data = h5_file[f'{behavior_path}reward_zone/data'][:]
reward_zone_timestamps = h5_file[f'{behavior_path}reward_zone/timestamps'][:]

# Get a subset of behavioral data that matches our neural data timeframe
max_time = timestamps[-1]
position_mask = position_timestamps <= max_time
position_subset = position_data[position_mask]
position_timestamps_subset = position_timestamps[position_mask]

reward_zone_mask = reward_zone_timestamps <= max_time
reward_zone_subset = reward_zone_data[reward_zone_mask]
reward_zone_timestamps_subset = reward_zone_timestamps[reward_zone_mask]

# Plot neural activity aligned with position
plt.figure(figsize=(12, 10))

# Top plot: mean neural activity
plt.subplot(3, 1, 1)
plt.plot(timestamps, mean_fluorescence)
plt.ylabel('Mean Fluorescence (a.u.)')
plt.title('Neural Activity and Behavior')
plt.grid(True, linestyle='--', alpha=0.7)

# Middle plot: position
plt.subplot(3, 1, 2)
plt.plot(position_timestamps_subset, position_subset)
plt.ylabel('Position (cm)')
plt.grid(True, linestyle='--', alpha=0.7)

# Bottom plot: reward zones
plt.subplot(3, 1, 3)
plt.plot(reward_zone_timestamps_subset, reward_zone_subset)
plt.xlabel('Time (s)')
plt.ylabel('Reward Zone')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join('explore', 'neural_and_behavior.png'))
plt.close()

# If we have enough neurons, let's try to find position-selective cells
if fluorescence_data.shape[1] >= 10:  # If we have at least 10 neurons
    print("\nLooking for position-selective neurons...")
    
    # For this analysis, we need to match neural timepoints with position timepoints
    # We'll use a simple approach by finding the nearest position timepoint for each neural timepoint
    matched_positions = np.zeros_like(timestamps)
    
    for i, t in enumerate(timestamps):
        # Find the closest position timestamp
        idx = np.abs(position_timestamps - t).argmin()
        matched_positions[i] = position_data[idx]
    
    # Correlation between neural activity and position
    position_correlations = np.zeros(fluorescence_data.shape[1])
    for i in range(fluorescence_data.shape[1]):
        position_correlations[i] = np.corrcoef(fluorescence_data[:, i], matched_positions)[0, 1]
    
    # Plot the top 3 position-correlated neurons
    top_indices = np.abs(position_correlations).argsort()[-3:][::-1]
    
    plt.figure(figsize=(12, 10))
    for i, idx in enumerate(top_indices):
        # Neural activity
        plt.subplot(3, 2, i*2+1)
        plt.plot(timestamps, fluorescence_data[:, idx])
        plt.ylabel(f'Neuron {idx} (a.u.)')
        plt.title(f'Position Correlation: {position_correlations[idx]:.2f}')
        
        # Scatter plot of activity vs position
        plt.subplot(3, 2, i*2+2)
        plt.scatter(matched_positions, fluorescence_data[:, idx], s=3, alpha=0.5)
        plt.xlabel('Position (cm)')
        plt.ylabel(f'Neuron {idx} (a.u.)')
    
    plt.tight_layout()
    plt.savefig(os.path.join('explore', 'position_selective_neurons.png'))
    plt.close()

print("Analysis complete. Neural data plots saved to the 'explore' directory.")