# explore_06_plot_activity_vs_position.py
# This script plots neural activity (deconvolved) against the animal's position
# for a few selected ROIs.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

print("Attempting to load NWB file for activity vs. position plot...")

# Ensure the explore directory exists for saving plots
import os
if not os.path.exists('explore'):
    os.makedirs('explore')

io = None
remote_file = None
try:
    # Load NWB file
    url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
    print(f"Using URL: {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    # Get Deconvolved Ophys Data
    ophys_module = nwb.processing.get('ophys')
    deconvolved_interface = ophys_module.data_interfaces.get('Deconvolved')
    plane0_deconv = deconvolved_interface.roi_response_series.get('plane0')
    
    if plane0_deconv is None:
        print("Deconvolved/plane0 data not found. Exiting.")
        exit()

    roi_table = plane0_deconv.rois.table
    iscell_column = roi_table['iscell'].data[:] 
    cell_indices = np.where(iscell_column[:, 0] == 1)[0]

    # Select a few active cells for plotting (e.g., those that looked active in fluorescence plot)
    # Based on previous script, ROIs with IDs 2 and 6 were plotted. Let's try to find their indices.
    # ROI IDs in the table might not be 0-indexed if some ROIs were discarded by suite2p.
    # So, we'll use the indices from `cell_indices` that correspond to desired ROI_IDs if possible,
    # or simply take the first few from `cell_indices`.
    
    # Let's try to pick ROI IDs 2 and 6 if they are in `cell_indices` and `roi_table.id`
    # Note: roi_table.id are the actual IDs, cell_indices refers to rows in the table.
    
    rois_to_analyze_ids = [2, 6] # Desired ROI IDs
    rois_to_analyze_indices_in_table = []
    for target_id in rois_to_analyze_ids:
        found_idx = np.where(roi_table.id[:] == target_id)[0]
        if len(found_idx) > 0 and found_idx[0] in cell_indices:
            rois_to_analyze_indices_in_table.append(found_idx[0])
        elif len(cell_indices) > len(rois_to_analyze_indices_in_table): # Fallback
            # If specific ID isn't a 'cell' or not found, take next available cell from cell_indices
            # This fallback logic needs care to avoid duplicates if a desired ID isn't there
            # For simplicity, if ROIs 2 or 6 aren't good cells, we'll pick first few from cell_indices
            pass # Will handle below

    if not rois_to_analyze_indices_in_table or len(rois_to_analyze_indices_in_table) < 2:
        print(f"Could not find all specified ROI IDs {rois_to_analyze_ids} as cells. Using first 2 cells from cell_indices.")
        if len(cell_indices) >= 2:
            rois_to_analyze_indices_in_table = cell_indices[:2]
        elif len(cell_indices) == 1:
            rois_to_analyze_indices_in_table = cell_indices[:1]
        else:
            print("Not enough cells found to plot. Exiting.")
            exit()
            
    actual_roi_ids_to_plot = roi_table.id[rois_to_analyze_indices_in_table]
    print(f"Selected ROI indices in table: {rois_to_analyze_indices_in_table}")
    print(f"Selected ROI IDs for plotting: {list(actual_roi_ids_to_plot)}")

    deconv_data_selected_rois = plane0_deconv.data[:, rois_to_analyze_indices_in_table]
    
    num_timepoints_ophys = plane0_deconv.data.shape[0]
    rate_ophys = plane0_deconv.rate
    start_time_ophys = plane0_deconv.starting_time
    time_vector_ophys = np.arange(num_timepoints_ophys) / rate_ophys + start_time_ophys

    # Get Behavior Data (Position)
    behavior_module = nwb.processing.get('behavior')
    behavioral_ts_interface = behavior_module.data_interfaces.get('BehavioralTimeSeries')
    position_ts = behavioral_ts_interface.time_series.get('position')
    
    if position_ts is None:
        print("Position time series not found. Exiting.")
        exit()
        
    position_data = position_ts.data[:]
    position_timestamps = position_ts.timestamps[:]

    # Interpolate position data to match ophys timestamps
    # Ensure timestamps are monotonically increasing for interpolation
    if not np.all(np.diff(position_timestamps) > 0):
        print("Position timestamps are not strictly monotonically increasing. Attempting to sort.")
        sort_indices = np.argsort(position_timestamps)
        position_timestamps = position_timestamps[sort_indices]
        position_data = position_data[sort_indices]
        # Check again after sorting
        if not np.all(np.diff(position_timestamps) > 0):
            print("Position timestamps still not monotonic after sorting. Exiting.")
            exit()

    # Create an interpolation function. Handle edge cases by using bounds_error=False and fill_value.
    # Using last known position for extrapolation might be reasonable for short periods.
    interp_func = interp1d(position_timestamps, position_data, kind='nearest', bounds_error=False, fill_value=(position_data[0], position_data[-1]))
    position_interpolated = interp_func(time_vector_ophys)

    # Create plots
    sns.set_theme()
    num_selected_rois = deconv_data_selected_rois.shape[1]
    plt.figure(figsize=(10, 5 * num_selected_rois))

    for i in range(num_selected_rois):
        roi_id = actual_roi_ids_to_plot[i]
        activity = deconv_data_selected_rois[:, i]
        
        plt.subplot(num_selected_rois, 1, i + 1)
        
        # Bin position data
        num_bins = 50
        bins = np.linspace(np.min(position_interpolated), np.max(position_interpolated), num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate mean activity in each bin
        mean_activity_in_bin = np.zeros(num_bins)
        counts_in_bin = np.zeros(num_bins) # For occupancy
        
        digitized_position = np.digitize(position_interpolated, bins) -1 # -1 for 0-based indexing
        
        for bin_idx in range(num_bins):
            mask = (digitized_position == bin_idx)
            if np.any(mask):
                mean_activity_in_bin[bin_idx] = np.mean(activity[mask])
                counts_in_bin[bin_idx] = np.sum(mask) # Number of samples in bin
            else:
                mean_activity_in_bin[bin_idx] = np.nan # Or 0, if preferred for no activity
        
        # Normalize counts for occupancy plot (optional, but good for context)
        occupancy_normalized = counts_in_bin * (rate_ophys / np.sum(counts_in_bin)) # time spent in each bin

        # Plot mean activity
        ax1 = plt.gca()
        color = 'tab:red'
        ax1.set_xlabel(f"Position ({position_ts.unit})")
        ax1.set_ylabel(f"Mean Deconvolved Activity ({plane0_deconv.unit})", color=color)
        ax1.plot(bin_centers, mean_activity_in_bin, color=color, marker='o', linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Optionally, plot occupancy on a second y-axis
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Occupancy (s/bin)', color=color) 
        ax2.bar(bin_centers, occupancy_normalized, width=(bins[1]-bins[0])*0.8, alpha=0.5, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"ROI {roi_id}: Activity vs. Position")

    plt.tight_layout()
    plt.savefig("explore/activity_vs_position.png")
    plt.close()
    print("Saved activity_vs_position.png")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if io is not None:
        print("Closing NWB file.")
        io.close()
    if remote_file is not None:
        remote_file.close()

print("Activity vs. position plotting script finished.")