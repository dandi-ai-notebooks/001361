'''
Explore neural activity data from Dandiset 001361.
This script will examine calcium imaging data from hippocampal CA1 neurons 
and their relationship to the animal's position and behavior.
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set_theme()

# Print debug information at each step to help diagnose issues
DEBUG = True

try:
    # Load the NWB file
    url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()

    # Print basic information about the neural data
    print(f"Subject: {nwb.subject.subject_id}, Session: {nwb.session_id}")
    print(f"Experiment: {nwb.identifier}")
    print(f"Location: {nwb.imaging_planes['ImagingPlane'].location}")
    print(f"Indicator: {nwb.imaging_planes['ImagingPlane'].indicator}")

    # Get neural activity data
    fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["plane0"]
    deconvolved = nwb.processing["ophys"].data_interfaces["Deconvolved"].roi_response_series["plane0"]
    print(f"Neural data available for processing")

    # Get cell information
    cell_table = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    cell_df = cell_table.to_dataframe()
    
    # Debug: Check the structure of cell_df
    print("\nCell dataframe columns:", cell_df.columns.tolist())
    print("Sample of cell dataframe:")
    print(cell_df.head())
    
    # Extract the iscell information
    if 'iscell' in cell_df.columns:
        # Check if iscell is a Series containing arrays
        if hasattr(cell_df['iscell'].iloc[0], '__len__'):
            print("\niscell contains arrays")
            # Try to extract the first element from each array in iscell
            iscell_values = np.array([row[0] if len(row) > 0 else 0 for row in cell_df['iscell']])
        else:
            print("\niscell contains scalar values")
            iscell_values = cell_df['iscell'].values
    else:
        print("\niscell column not found, assuming all ROIs are cells")
        iscell_values = np.ones(len(cell_df), dtype=bool)
    
    # Check how many cells were detected
    print(f"\nTotal number of ROIs: {len(cell_df)}")
    print(f"Number of classified cells: {np.sum(iscell_values > 0)}")
    print(f"Neural data shape: {fluorescence.data.shape}")

    # Get behavioral data
    behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"]
    position = behavior.time_series["position"]
    speed = behavior.time_series["speed"]
    reward_zone = behavior.time_series["reward_zone"]
    trial_number = behavior.time_series["trial number"]

    # Get data and timestamps
    position_data = position.data[:]
    position_timestamps = position.timestamps[:]
    trial_number_data = trial_number.data[:]

    # Get fluorescence data for cells (not background)
    # Use only the first 1000 timepoints to keep the script quick
    time_limit = 1000
    f_data = fluorescence.data[:time_limit, :]  # Time x Cells
    f_timestamps = np.arange(time_limit) / fluorescence.rate + fluorescence.starting_time

    # Get corresponding position data
    # Find closest position timestamp for each fluorescence timestamp
    pos_indices = []
    for ts in f_timestamps:
        idx = np.argmin(np.abs(position_timestamps - ts))
        pos_indices.append(idx)

    matched_positions = position_data[pos_indices]
    matched_trials = trial_number_data[pos_indices]

    print(f"\nMatched data shapes:")
    print(f"Fluorescence: {f_data.shape}")
    print(f"Positions: {matched_positions.shape}")
    print(f"Trials: {matched_trials.shape}")

    # Plot the activity of a few example cells over time along with position
    n_examples = 5
    # Get indices of cells (use the iscell_values we calculated)
    example_indices = np.where(iscell_values > 0)[0][:n_examples]
    if len(example_indices) == 0:
        # If no cells were found, just use the first n_examples indices
        print("\nWarning: No cells classified with iscell > 0, using first ROIs as examples")
        example_indices = np.arange(min(n_examples, f_data.shape[1]))
    
    print(f"\nExample cell indices: {example_indices}")

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(n_examples+1, 1, 1)
    ax1.plot(f_timestamps, matched_positions, 'k')
    ax1.set_ylabel('Position (cm)')
    ax1.set_title('Mouse Position and Neural Activity')
    ax1.set_xticklabels([])

    for i, cell_idx in enumerate(example_indices):
        ax = plt.subplot(n_examples+1, 1, i+2, sharex=ax1)
        # Normalize the fluorescence data for better visualization
        cell_data = f_data[:, cell_idx]
        if np.max(cell_data) > np.min(cell_data):
            norm_f = (cell_data - np.min(cell_data)) / (np.max(cell_data) - np.min(cell_data))
        else:
            norm_f = np.zeros_like(cell_data)
        ax.plot(f_timestamps, norm_f)
        ax.set_ylabel(f'Cell {cell_idx}')
        if i < n_examples - 1:
            ax.set_xticklabels([])
        
    ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('explore/neural_activity_time.png')

    # Identify place cells by creating a spatial tuning curve for each cell
    # We'll bin positions and average fluorescence in each bin

    # Create position bins
    pos_bins = np.linspace(np.min(matched_positions), np.max(matched_positions), 40)
    bin_centers = (pos_bins[:-1] + pos_bins[1:]) / 2
    bin_indices = np.digitize(matched_positions, pos_bins) - 1
    bin_indices[bin_indices >= len(pos_bins)-1] = len(pos_bins) - 2  # Fix any out-of-bound indices

    # Create spatial tuning curves for all cells
    spatial_tuning = np.zeros((len(bin_centers), f_data.shape[1]))
    for i in range(len(bin_centers)):
        in_bin = bin_indices == i
        if np.sum(in_bin) > 0:  # Make sure there are timepoints in this bin
            spatial_tuning[i, :] = np.mean(f_data[in_bin, :], axis=0)

    # Find cells with strong spatial tuning
    # Compute the peak-to-baseline ratio for each cell
    baseline = np.percentile(spatial_tuning, 10, axis=0)
    peak = np.max(spatial_tuning, axis=0)
    peak_to_baseline = peak / (baseline + 1e-6)  # Avoid division by zero

    # Select top place cells
    n_place_cells = 20
    place_cell_indices = np.argsort(peak_to_baseline)[::-1][:n_place_cells]
    # Ensure they're classified as cells using our iscell_values
    place_cell_indices = [idx for idx in place_cell_indices if iscell_values[idx] > 0]
    if len(place_cell_indices) == 0:
        # If no cells were found, just use the top n_place_cells ROIs by peak-to-baseline
        place_cell_indices = np.argsort(peak_to_baseline)[::-1][:n_place_cells]
    
    print(f"\nTop place cell indices: {place_cell_indices}")

    # Plot spatial tuning curves for top place cells
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.flatten()

    for i, cell_idx in enumerate(place_cell_indices):
        if i >= len(axes) or i >= len(place_cell_indices):  # In case we have fewer than n_place_cells
            break
            
        # Normalize for better visualization
        tuning = spatial_tuning[:, cell_idx]
        if np.max(tuning) > np.min(tuning):
            norm_tuning = (tuning - np.min(tuning)) / (np.max(tuning) - np.min(tuning))
        else:
            norm_tuning = np.zeros_like(tuning)
        
        axes[i].plot(bin_centers, norm_tuning)
        axes[i].set_title(f'Cell {cell_idx}')
        
        # Show peak position with a vertical line
        peak_pos = bin_centers[np.argmax(norm_tuning)]
        axes[i].axvline(x=peak_pos, color='r', linestyle='--', alpha=0.5)
        
        if i % 5 == 0:  # Add y-axis label for leftmost plots
            axes[i].set_ylabel('Norm. Activity')
        
        if i >= 15:  # Add x-axis label for bottom plots
            axes[i].set_xlabel('Position (cm)')

    plt.tight_layout()
    plt.savefig('explore/place_cell_tuning.png')

    # Create a spatial heat map of neural activity
    plt.figure(figsize=(12, 10))

    # Sort cells by position of peak activity
    peak_pos = np.argmax(spatial_tuning, axis=0)
    sort_idx = np.argsort(peak_pos)

    # Only use cells classified as cells
    cell_sort_idx = [idx for idx in sort_idx if iscell_values[idx] > 0]
    if len(cell_sort_idx) == 0:
        # If no cells were found, use all ROIs
        cell_sort_idx = sort_idx

    # Normalize each cell's spatial tuning curve
    norm_tuning = np.zeros_like(spatial_tuning)
    for i in range(spatial_tuning.shape[1]):
        tuning = spatial_tuning[:, i]
        if np.max(tuning) > np.min(tuning):
            norm_tuning[:, i] = (tuning - np.min(tuning)) / (np.max(tuning) - np.min(tuning))

    # Create heatmap with cells sorted by peak position
    plt.pcolormesh(bin_centers, np.arange(len(cell_sort_idx)), 
                   norm_tuning[:, cell_sort_idx].T, cmap='viridis', shading='auto')
    plt.colorbar(label='Normalized Activity')
    plt.ylabel('Cell # (sorted by peak position)')
    plt.xlabel('Position (cm)')
    plt.title('Spatial Tuning of CA1 Neurons')

    # Mark the mean reward position with a vertical line
    reward_position = 252.0  # From our previous analysis
    plt.axvline(x=reward_position, color='r', linestyle='--', 
               label=f'Mean Reward Position: {reward_position:.1f} cm')
    plt.legend()

    plt.tight_layout()
    plt.savefig('explore/place_cell_heatmap.png')

    print("Neural data exploration complete!")
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()