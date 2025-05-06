# explore_05_plot_fluorescence_traces.py
# This script loads fluorescence traces for a few selected ROIs and plots them.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Attempting to load NWB file for fluorescence trace plot...")

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

    ophys_module = nwb.processing.get('ophys')
    if ophys_module is None:
        print("Ophys processing module not found.")
    else:
        fluorescence_interface = ophys_module.data_interfaces.get('Fluorescence')
        if fluorescence_interface is None:
            print("Fluorescence interface not found.")
        else:
            plane0_fluor = fluorescence_interface.roi_response_series.get('plane0')
            if plane0_fluor is None:
                print("Fluorescence/plane0 not found.")
            else:
                print(f"Fluorescence/plane0 data shape: {plane0_fluor.data.shape}") # (timestamps, rois)
                
                roi_table = plane0_fluor.rois.table
                iscell_column = roi_table['iscell'].data[:] # Load all 'iscell' data (column 0 is 'iscell', column 1 is 'probcell')
                
                # Select ROIs that are classified as cells (iscell == 1)
                # The 'iscell' column from suite2p typically contains two values per ROI: [iscell_bool, prob_cell]
                # We are interested in the first value.
                cell_indices = np.where(iscell_column[:, 0] == 1)[0]
                
                if len(cell_indices) == 0:
                    print("No ROIs classified as cells found. Selecting first few ROIs instead.")
                    rois_to_plot_indices = np.arange(min(5, plane0_fluor.data.shape[1]))
                    roi_ids_to_plot = roi_table.id[rois_to_plot_indices]
                else:
                    print(f"Found {len(cell_indices)} ROIs classified as cells.")
                    rois_to_plot_indices = cell_indices[:min(5, len(cell_indices))] # Plot up to 5 cells
                    roi_ids_to_plot = roi_table.id[rois_to_plot_indices]

                print(f"Selected ROI indices for plotting: {rois_to_plot_indices}")
                print(f"Selected ROI IDs for plotting: {list(roi_ids_to_plot)}")

                if len(rois_to_plot_indices) > 0:
                    # Generate time vector
                    num_timepoints = plane0_fluor.data.shape[0]
                    rate = plane0_fluor.rate
                    starting_time = plane0_fluor.starting_time
                    time_vector = np.arange(num_timepoints) / rate + starting_time
                    
                    # Limit number of timepoints to plot for performance if very long
                    max_timepoints_to_plot = 3000 # e.g., plot ~200 seconds at ~15Hz
                    if num_timepoints > max_timepoints_to_plot:
                        time_indices_to_plot = np.linspace(0, num_timepoints - 1, max_timepoints_to_plot, dtype=int)
                        time_vector_subset = time_vector[time_indices_to_plot]
                        plot_title = f"Fluorescence Traces (Subset of Time, {len(rois_to_plot_indices)} ROIs)"
                    else:
                        time_indices_to_plot = slice(None) # equivalent to [:]
                        time_vector_subset = time_vector
                        plot_title = f"Fluorescence Traces ({len(rois_to_plot_indices)} ROIs)"

                    # Load data for selected ROIs and timepoints
                    # Data is (timepoints, rois)
                    # First, select all timepoints for the chosen ROIs to get a NumPy array
                    traces_data_all_time = plane0_fluor.data[:, rois_to_plot_indices]
                    # Then, select the desired timepoints from this NumPy array
                    traces_data = traces_data_all_time[time_indices_to_plot, :]
                    
                    sns.set_theme()
                    plt.figure(figsize=(15, 2 * len(rois_to_plot_indices)))
                    
                    for i, roi_idx_in_selection in enumerate(range(traces_data.shape[1])):
                        plt.subplot(len(rois_to_plot_indices), 1, i + 1)
                        plt.plot(time_vector_subset, traces_data[:, roi_idx_in_selection])
                        plt.ylabel(f"ROI {roi_ids_to_plot[i]}\n({plane0_fluor.unit})")
                        if i < len(rois_to_plot_indices) - 1:
                            plt.xticks([]) # Remove x_ticks for all but the last subplot
                    
                    plt.xlabel("Time (s)")
                    plt.suptitle(plot_title)
                    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
                    plt.savefig("explore/fluorescence_traces.png")
                    plt.close()
                    print("Saved fluorescence_traces.png")
                else:
                    print("No ROIs selected to plot.")
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

print("Fluorescence trace plotting script finished.")