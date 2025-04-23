"""
This script explores the neural activity (calcium imaging) data in the NWB file.
We'll look at fluorescence traces, deconvolved activity, and correlate neural activity
with behavioral events.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb
import os

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
num_cells = np.sum(iscell[:, 0])
print(f"Number of cells: {num_cells}")
print(f"Number of non-cells: {len(iscell) - num_cells}")

# Get a random sample of cells to plot
np.random.seed(42)  # For reproducibility
num_cells_to_plot = 10
# Only select from ROIs identified as cells
cell_indices = np.where(iscell[:, 0] == 1)[0]
if len(cell_indices) > 0:
    selected_cells = np.random.choice(cell_indices, 
                                     size=min(num_cells_to_plot, len(cell_indices)), 
                                     replace=False)
else:
    # Fallback to random ROIs if no cells are marked
    selected_cells = np.random.choice(rois.id[:], 
                                     size=min(num_cells_to_plot, len(rois.id[:])), 
                                     replace=False)

# Sort the selected cells to ensure indices are in increasing order
# This is required for H5PY dataset indexing
selected_cells.sort()
print(f"Selected cells (sorted): {selected_cells}")

# Time window to plot (in seconds)
time_start = 0
time_window = 60  # seconds
samples_to_plot = int(time_window * fluor_rate)

# Load fluorescence data for all selected cells at once
# First load the time window we want
fluor_data_window = fluor.data[:samples_to_plot, :]
# Then extract the columns for the selected cells
fluor_traces = fluor_data_window[:, selected_cells]

# Create time vector in seconds
time_vector = np.arange(fluor_traces.shape[0]) / fluor_rate

# Plot fluorescence traces
plt.figure()
plt.suptitle("Fluorescence Traces for Sample Cells", fontsize=16)

for i, cell_idx in enumerate(selected_cells):
    # Offset each trace for better visualization
    offset = i * 3
    plt.plot(time_vector, fluor_traces[:, i] + offset, label=f"Cell {cell_idx}")

plt.xlabel("Time (s)")
plt.ylabel("Fluorescence + Offset")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("explore/fluorescence_traces.png")
plt.close()

# Get deconvolved activity data
deconv = ophys.data_interfaces["Deconvolved"].roi_response_series["plane0"]
print(f"\nDeconvolved data shape: {deconv.data.shape}")  # Should match fluor.data.shape

# Fetch deconvolved data for the time window
deconv_data_window = deconv.data[:samples_to_plot, :]
# Then extract the columns for the selected cells
deconv_traces = deconv_data_window[:, selected_cells]

# Plot deconvolved activity
plt.figure()
plt.suptitle("Deconvolved Activity for Sample Cells", fontsize=16)

for i, cell_idx in enumerate(selected_cells):
    # Offset each trace for better visualization
    offset = i * 3
    plt.plot(time_vector, deconv_traces[:, i] + offset, label=f"Cell {cell_idx}")

plt.xlabel("Time (s)")
plt.ylabel("Deconvolved Activity + Offset")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("explore/deconvolved_activity.png")
plt.close()

# Get behavior data to correlate with neural activity
behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"]

# Load all position data
position = behavior.time_series["position"]
position_data = position.data[:]
position_timestamps = position.timestamps[:]

print(f"\nPosition data shape: {position_data.shape}")
print(f"Position timestamp shape: {position_timestamps.shape}")

# Let's find the mean activity of each cell as a function of position (place fields)
# First let's bin positions into discrete bins
position_bins = np.linspace(-400, 400, 41)  # 40 bins from -400 to 400 cm
bin_centers = (position_bins[:-1] + position_bins[1:]) / 2
bin_width = position_bins[1] - position_bins[0]

# Find which fluorescence samples correspond to which position bins
neural_time_vector = np.arange(fluor.data.shape[0]) / fluor_rate

# Get position at each neural sample time by interpolation
interp_position = np.interp(neural_time_vector, position_timestamps, position_data)

# Discretize positions into bins
binned_positions = np.digitize(interp_position, position_bins) - 1
binned_positions = np.clip(binned_positions, 0, len(bin_centers) - 1)  # Ensure valid indices

# Compute the mean activity for a subset of cells at each position bin
num_cells_for_place_cells = 20
# Prioritize cells marked as cells if available
if len(cell_indices) > 0:
    place_cell_indices = cell_indices[:min(num_cells_for_place_cells, len(cell_indices))]
else:
    place_cell_indices = np.array(range(min(num_cells_for_place_cells, len(rois.id[:]))))

# Make sure the indices are sorted
place_cell_indices.sort()
print(f"Place cell indices (first few): {place_cell_indices[:5]}...")

# Initialize place field arrays
n_place_cells = len(place_cell_indices)
n_bins = len(bin_centers)
place_fields = np.zeros((n_place_cells, n_bins))
place_fields_std = np.zeros_like(place_fields)

# For each cell
for i, cell_idx in enumerate(place_cell_indices):
    # Load the full fluorescence data for this cell
    cell_fluor_data = fluor.data[:, cell_idx]
    
    # For each position bin
    for j in range(n_bins):
        # Find time points where the animal was in this position bin
        bin_samples = np.where(binned_positions == j)[0]
        
        if len(bin_samples) > 0:
            # Calculate mean activity in this bin
            place_fields[i, j] = np.mean(cell_fluor_data[bin_samples])
            place_fields_std[i, j] = np.std(cell_fluor_data[bin_samples])

# Plot place fields for a subset of cells
num_place_cells_to_plot = min(6, n_place_cells)
place_cells_subset = place_cell_indices[:num_place_cells_to_plot]

fig, axes = plt.subplots(num_place_cells_to_plot, 1, figsize=(12, 2*num_place_cells_to_plot), sharex=True)
if num_place_cells_to_plot == 1:
    axes = [axes]  # Make it iterable if it's a single axis
plt.suptitle("Place Fields: Mean Fluorescence vs. Position", fontsize=16)

for i, (ax, cell_idx) in enumerate(zip(axes, place_cells_subset)):
    ax.plot(bin_centers, place_fields[i, :])
    ax.fill_between(bin_centers, 
                   place_fields[i, :] - place_fields_std[i, :],
                   place_fields[i, :] + place_fields_std[i, :],
                   alpha=0.3)
    ax.set_ylabel(f"Cell {cell_idx}\nFluor.")
    if i == len(axes) - 1:  # Only label the bottom plot
        ax.set_xlabel("Position (cm)")

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust for the suptitle
plt.savefig("explore/place_fields.png")
plt.close()

# Plot a heatmap of all place fields to look for spatial tuning
plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.title("Place Fields Heatmap", fontsize=14)
# Normalize each cell's activity for better visualization
normalized_place_fields = np.zeros_like(place_fields)
for i in range(n_place_cells):
    cell_min = np.min(place_fields[i, :])
    cell_max = np.max(place_fields[i, :])
    # Avoid division by zero
    if cell_max > cell_min:
        normalized_place_fields[i, :] = (place_fields[i, :] - cell_min) / (cell_max - cell_min)
    else:
        normalized_place_fields[i, :] = 0

# Plot the heatmap
plt.imshow(normalized_place_fields, aspect='auto', cmap='viridis')
plt.colorbar(label="Normalized Fluorescence")
plt.ylabel("Cell Number")
plt.xticks(np.arange(0, len(bin_centers), 5), 
          [f"{x:.0f}" for x in bin_centers[::5]])
plt.xlabel("Position (cm)")

# Also plot position histogram for reference
plt.subplot(212)
plt.title("Position Histogram", fontsize=14)
plt.hist(interp_position, bins=position_bins, alpha=0.7)
plt.xlabel("Position (cm)")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("explore/place_fields_heatmap.png")
plt.close()

# Let's try to identify place cells more systematically
# A simple approach is to compute how much the activity varies across positions
place_field_selectivity = np.zeros(n_place_cells)
for i in range(n_place_cells):
    place_field_selectivity[i] = np.max(place_fields[i, :]) - np.min(place_fields[i, :])

# Plot the selectivity of each cell
plt.figure()
plt.title("Place Field Selectivity", fontsize=14)
plt.bar(range(n_place_cells), place_field_selectivity)
plt.xlabel("Cell Index")
plt.ylabel("Selectivity (max - min)")
plt.tight_layout()
plt.savefig("explore/place_field_selectivity.png")
plt.close()

# For the cells with the highest selectivity, plot their place fields separately
num_top_cells = min(3, n_place_cells)
top_cell_indices = np.argsort(place_field_selectivity)[-num_top_cells:]

fig, axes = plt.subplots(num_top_cells, 1, figsize=(10, 3*num_top_cells), sharex=True)
if num_top_cells == 1:
    axes = [axes]  # Make it iterable if it's a single axis
plt.suptitle("Top Cells by Place Field Selectivity", fontsize=16)

for i, (ax, idx) in enumerate(zip(axes, top_cell_indices)):
    cell_idx = place_cell_indices[idx]
    ax.plot(bin_centers, place_fields[idx, :])
    ax.fill_between(bin_centers, 
                   place_fields[idx, :] - place_fields_std[idx, :],
                   place_fields[idx, :] + place_fields_std[idx, :],
                   alpha=0.3)
    ax.set_ylabel(f"Cell {cell_idx}\nFluor.")
    if i == len(axes) - 1:
        ax.set_xlabel("Position (cm)")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("explore/top_place_cells.png")
plt.close()

print("Analysis complete. Plots saved to the 'explore' directory.")