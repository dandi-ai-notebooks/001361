"""
This script explores the relationship between neural activity and rewards.
We'll analyze how neural activity changes around reward delivery times.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb

# Set up matplotlib defaults
plt.rcParams['figure.figsize'] = (12, 8)

# Load the remote NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("Loaded NWB file:", nwb.identifier)
print("Session:", nwb.session_id)
print("Subject:", nwb.subject.subject_id)

# Access neural activity data
ophys = nwb.processing["ophys"]

# Get fluorescence data
fluor = ophys.data_interfaces["Fluorescence"].roi_response_series["plane0"]
fluor_rate = fluor.rate
print(f"\nFluorescence sampling rate: {fluor_rate} Hz")
print(f"Fluorescence data shape: {fluor.data.shape}")  # (time, cells)

# Get ROI information
rois = fluor.rois.table
print(f"Number of ROIs: {len(rois.id[:])}")

# Check which ROIs are cells (vs not cells)
iscell = rois.iscell[:]
cell_indices = np.where(iscell[:, 0] == 1)[0]
num_cells = len(cell_indices)
print(f"Number of cells: {num_cells}")

# Get reward information from behavior data
behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"]
reward_ts = behavior.time_series["Reward"]
reward_timestamps = reward_ts.timestamps[:]

# Get position data too
position = behavior.time_series["position"]
position_timestamps = position.timestamps[:]

print(f"Number of reward events: {len(reward_timestamps)}")

# Create a time window around rewards to analyze neural activity
pre_reward_time = 2.0  # seconds before reward
post_reward_time = 4.0  # seconds after reward
time_window = pre_reward_time + post_reward_time
samples_per_window = int(time_window * fluor_rate)
times_around_reward = np.linspace(-pre_reward_time, post_reward_time, samples_per_window)

# Select a subset of cells for analysis (first 10 cells)
num_cells_to_analyze = 10
selected_cells = cell_indices[:min(num_cells_to_analyze, len(cell_indices))]
selected_cells.sort()  # Ensure they're sorted for h5py indexing

print(f"Selected cells (sorted): {selected_cells}")

# Create a neural time vector
neural_times = np.arange(fluor.data.shape[0]) / fluor_rate

# For each reward event, we'll collect neural activity in a window around the reward
all_responses = []
for reward_time in reward_timestamps:
    # Find the neural data indices around this reward time
    start_time = reward_time - pre_reward_time
    end_time = reward_time + post_reward_time
    
    # Skip rewards that would go out of bounds
    if start_time < neural_times[0] or end_time > neural_times[-1]:
        continue
    
    # Find the closest timepoints in the neural data
    start_idx = np.searchsorted(neural_times, start_time)
    end_idx = np.searchsorted(neural_times, end_time)
    
    # Adjust to ensure we have the right number of samples
    if end_idx - start_idx != samples_per_window:
        end_idx = start_idx + samples_per_window
        
    # Make sure we have enough data points
    if end_idx <= fluor.data.shape[0]:
        # Get the window of neural data for all selected cells
        window_data = fluor.data[start_idx:end_idx, selected_cells]
        all_responses.append(window_data)

# Convert to numpy array
if all_responses:
    all_responses = np.array(all_responses)  # Shape: (n_rewards, time_points, n_cells)
    print(f"Collected responses for {len(all_responses)} reward events")
    print(f"Response data shape: {all_responses.shape}")
    
    # Calculate the average response across all rewards
    avg_response = np.mean(all_responses, axis=0)  # Shape: (time_points, n_cells)
    
    # Plot the average neural response around reward times
    plt.figure(figsize=(10, 6))
    plt.title("Average Neural Response Around Reward Times", fontsize=16)
    
    for i, cell_idx in enumerate(selected_cells):
        plt.plot(times_around_reward, avg_response[:, i], label=f"Cell {cell_idx}")
    
    plt.axvline(x=0, color='r', linestyle='--', label="Reward Delivery")
    plt.xlabel("Time Relative to Reward (s)")
    plt.ylabel("Fluorescence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("explore/avg_reward_response.png")
    plt.close()
    
    # Plot individual responses for a single example cell
    example_cell_idx = 0  # First cell in our selection
    example_cell_id = selected_cells[example_cell_idx]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Individual Responses Around Rewards: Cell {example_cell_id}", fontsize=16)
    
    # Plot the first 20 reward responses (or fewer if we have less)
    for i in range(min(20, len(all_responses))):
        plt.plot(times_around_reward, all_responses[i, :, example_cell_idx], 'k-', alpha=0.3)
    
    # Plot the average in red
    plt.plot(times_around_reward, avg_response[:, example_cell_idx], 'r-', linewidth=2, label="Average")
    plt.axvline(x=0, color='b', linestyle='--', label="Reward Delivery")
    plt.xlabel("Time Relative to Reward (s)")
    plt.ylabel("Fluorescence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("explore/individual_reward_responses.png")
    plt.close()
    
    # Create a heatmap of responses for all analyzed cells
    plt.figure(figsize=(10, 8))
    plt.title("Neural Responses Around Reward Times", fontsize=16)
    
    # Transpose avg_response for the heatmap (cells as rows, time as columns)
    heatmap_data = avg_response.T  # Shape: (n_cells, time_points)
    
    # Normalize each cell's response for better visualization
    normalized_heatmap = np.zeros_like(heatmap_data)
    for i in range(len(selected_cells)):
        cell_min = np.min(heatmap_data[i, :])
        cell_max = np.max(heatmap_data[i, :])
        # Avoid division by zero
        if cell_max > cell_min:
            normalized_heatmap[i, :] = (heatmap_data[i, :] - cell_min) / (cell_max - cell_min)
        else:
            normalized_heatmap[i, :] = 0
    
    plt.imshow(normalized_heatmap, aspect='auto', cmap='viridis', 
               extent=[-pre_reward_time, post_reward_time, len(selected_cells)-0.5, -0.5])
    plt.colorbar(label="Normalized Response")
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label="Reward Delivery")
    plt.xlabel("Time Relative to Reward (s)")
    plt.ylabel("Cell Number")
    plt.yticks(np.arange(len(selected_cells)), [f"Cell {cell}" for cell in selected_cells])
    plt.tight_layout()
    plt.savefig("explore/reward_response_heatmap.png")
    plt.close()

    print("Analysis complete. Plots saved to the 'explore' directory.")
else:
    print("No valid reward responses collected.")