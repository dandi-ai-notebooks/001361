# %% [markdown]
# # Exploring Dandiset 001361: A flexible hippocampal population code for experience relative to reward
#
# **Important Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This Dandiset, titled "A flexible hippocampal population code for experience relative to reward," contains data from 2-photon imaging and behavioral experiments in hippocampal area CA1 of mice navigating virtual reality environments. The research investigates how hippocampal ensembles encode experiences relative to changing reward locations.
#
# **Published in:** Sosa, Plitt, & Giocomo, "A flexible hippocampal population code for experience relative to reward," Nature Neuroscience.
#
# **Dandiset Description:**
#
# > 2-photon imaging and behavioral data from hippocampal area CA1 during virtual reality navigation in mice. Included in Sosa, Plitt, & Giocomo, "A flexible hippocampal population code for experience relative to reward," Nature Neuroscience.
# >
# > To reinforce rewarding behaviors, events leading up to and following rewards must be remembered. Hippocampal place cell activity spans spatial and non-spatial episodes, but whether hippocampal activity encodes entire sequences of events relative to reward is unknown. To test this, we performed two-photon imaging of hippocampal CA1 as mice navigated virtual environments with changing hidden reward locations. When the reward moved, a subpopulation of neurons updated their firing fields to the same relative position with respect to reward, constructing behavioral timescale sequences spanning the entire task. Over learning, this reward-relative representation became more robust as additional neurons were recruited, and changes in reward-relative firing often preceded behavioral adaptations following reward relocation. Concurrently, the spatial environment code was maintained through a parallel, dynamic subpopulation rather than through dedicated cell classes. These findings reveal how hippocampal ensembles flexibly encode multiple aspects of experience while amplifying behaviorally relevant information.
#
# **Citation:** Sosa, Marielena; Plitt, Mark H.; Giocomo, Lisa M. (2025) A flexible hippocampal population code for experience relative to reward (Version 0.250406.0045) [Data set]. DANDI Archive. https://doi.org/10.48324/dandi.001361/0.250406.0045
#
# **Link to Dandiset:** [https://dandiarchive.org/dandiset/001361/0.250406.0045](https://dandiarchive.org/dandiset/001361/0.250406.0045)

# %% [markdown]
# ## What this notebook covers
#
# This notebook will guide you through:
# 1.  Listing required Python packages.
# 2.  Loading the Dandiset metadata using the DANDI API.
# 3.  Selecting and loading an NWB (Neurodata Without Borders) file from the Dandiset.
# 4.  Exploring basic metadata and structure of the NWB file.
# 5.  Loading and visualizing a subset of 2-photon imaging data (calcium fluorescence traces of ROIs).
# 6.  Loading and visualizing a subset of behavioral data (e.g., position, licks).
# 7.  Displaying ROI masks.
# 8.  Summarizing findings and suggesting potential future analysis directions.

# %% [markdown]
# ## Required Packages
#
# This notebook requires the following Python packages. We assume they are already installed.
#
# *   `dandi` (for interacting with the DANDI Archive)
# *   `pynwb` (for reading NWB files)
# *   `h5py` (dependency for pynwb, for HDF5 file operations)
# *   `remfile` (for streaming remote files)
# *   `numpy` (for numerical operations)
# *   `matplotlib` (for plotting)
# *   `seaborn` (for enhanced visualizations)
# *   `itertools` (used for slicing iterators)

# %% [markdown]
# ## 1. Load Dandiset using DANDI API

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pynwb
import h5py
import remfile

# Set seaborn theme for plots
sns.set_theme()

# Connect to DANDI archive
client = DandiAPIClient()
dandiset_id = "001361"
dandiset_version = "0.250406.0045"
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Dandiset Description: {metadata.get('description', 'N/A')[:200]}...") # Print first 200 chars

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## 2. Load an NWB file
#
# We will select one NWB file from the Dandiset to explore. For this notebook, we will use the file `sub-m11/sub-m11_ses-03_behavior+ophys.nwb`.
#
# The asset ID for this file is `d77ea78a-8978-461d-9d11-3c5cef860d82`.
#
# We can construct the direct download URL for this asset as follows:
# `https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/`
#
# This URL will be used to stream the file data.

# %%
# NWB File URL (hard-coded based on the tool output for nwb-file-info)
nwb_file_url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
asset_id_for_neurosift = "d77ea78a-8978-461d-9d11-3c5cef860d82" # Extracted from the URL or asset listing

print(f"Loading NWB file from: {nwb_file_url}")

# Load the NWB file using remfile for remote access
try:
    remote_file = remfile.File(nwb_file_url)
    h5_file = h5py.File(remote_file, 'r') # Ensure 'r' mode for reading
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) # Ensure 'r' mode and load namespaces
    nwb = io.read()
    print("NWB file loaded successfully.")
except Exception as e:
    print(f"Error loading NWB file: {e}")
    nwb = None # Set nwb to None if loading fails to prevent further errors

if nwb:
    print(f"\nNWB File Identifier: {nwb.identifier}")
    print(f"Session Description: {nwb.session_description}")
    print(f"Session Start Time: {nwb.session_start_time}")
    print(f"Experimenter: {nwb.experimenter}")
    print(f"Subject ID: {nwb.subject.subject_id if nwb.subject else 'N/A'}")

# %% [markdown]
# ### Explore this NWB file on Neurosift
#
# You can interactively explore the structure and contents of this NWB file using Neurosift:
#
# [Explore on Neurosift](https_neurosift_app_nwb_url_https_api_dandyarchive_org_api_assets_d77ea78a-8978-461d-9d11-3c5cef860d82_download_dandisetId_001361_dandisetVersion_0_250406_0045)
#
# (Replace `_` with `/` and `.` in the Neurosift URL above if it doesn't render correctly as a link in your Jupyter environment - the link format should be https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/&dandisetId=001361&dandisetVersion=0.250406.0045)

# %% [markdown]
# ## 3. Summarize NWB File Contents
#
# The NWB file format organizes data into various groups. Key groups typically include:
#
# *   `acquisition`: Raw acquired data, often continuous signals like electrophysiology or imaging series.
#     *   In this file: `TwoPhotonSeries` contains the raw 2-photon imaging data.
# *   `processing`: Processed data derived from acquisition or other sources. This is where you'll often find fluorescence traces, spike times, behavioral events, etc.
#     *   `behavior`: Contains processed behavioral data.
#         *   `BehavioralTimeSeries`:
#             *   `position`: Animal's position.
#             *   `lick`: Lick events.
#             *   `Reward`: Reward delivery times and amounts.
#             *   `speed`: Animal's speed.
#             *   And others like `environment`, `trial_number`, `teleport`, etc.
#     *   `ophys`: Contains processed optical physiology data.
#         *   `Fluorescence`: Fluorescence traces for ROIs.
#             *   `plane0`: RoiResponseSeries for imaging plane 0.
#         *   `Deconvolved`: Deconvolved (e.g., estimated spiking activity) traces for ROIs.
#             *   `plane0`: RoiResponseSeries for imaging plane 0.
#         *   `ImageSegmentation`: Information about ROIs.
#             *   `PlaneSegmentation`: ROI definitions (pixel masks, etc.).
#         *   `Backgrounds_0`: Background images (e.g., `meanImg`, `Vcorr`, `max_proj`).
# *   `stimulus`: Information about stimuli presented. (May not be present in all files)
# *   `intervals`: Time intervals of interest, like trials.
# *   `general`: General metadata about the experiment, subject, session, etc.
#
# Let's check the main groups in our loaded NWB file:
#
# ```
# # (This is not a runnable cell, but shows what you would see if you printed nwb)
# NWBFile: /data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A
# Fields:
#   acquisition: {
#     TwoPhotonSeries <class 'pynwb.ophys.TwoPhotonSeries'>
#   }
#   analysis: {
#   }
#   devices: {
#     Microscope <class 'pynwb.device.Device'>
#   }
#   experiment_description: NIH funding information
#   experimenter: ['Mari Sosa']
#   file_create_date: [datetime.datetime(2025, 3, 12, 23, 45, 29, 830157, tzinfo=tzoffset(None, -25200))]
#   identifier: /data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A
#   imaging_planes: {
#     ImagingPlane <class 'pynwb.ophys.ImagingPlane'>
#   }
#   institution: Stanford University
#   intervals: {
#   }
#   lab: Giocomo Lab
#   processing: {
#     behavior <class 'pynwb.base.ProcessingModule'>,
#     ophys <class 'pynwb.base.ProcessingModule'>
#   }
#   session_description: processed suite2p data
#   session_id: 03
#   session_start_time: 2023-02-23 00:00:00-08:00
#   stimulus: {
#   }
#   subject: subject <class 'pynwb.file.Subject'>
#   timestamps_reference_time: 2023-02-23 00:00:00-08:00
# ```
#
# Based on the `nwb-file-info` tool output, we can expect the following structure for some key data:
# *   **Two-Photon Imaging Data:**
#     *   Raw data: `nwb.acquisition['TwoPhotonSeries'].data` (shape (1, 1, 1) - likely a placeholder or example in the metadata, actual data is usually much larger and external or linked)
#     *   Imaging plane dimensions: `nwb.acquisition['TwoPhotonSeries'].dimension[:]` (e.g., `[512, 796]`)
# *   **Processed Optical Physiology (ophys):**
#     *   Fluorescence traces: `nwb.processing['ophys']['Fluorescence']['plane0'].data` (e.g., shape `(19818, 349)` for 19818 timepoints, 349 ROIs)
#     *   Timestamps for fluorescence: `nwb.processing['ophys']['Fluorescence']['plane0'].timestamps` (or can be calculated from `starting_time` and `rate`)
#     *   ROI masks: `nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation'].pixel_mask`
#     *   ROI IDs: `nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation'].id[:]`
#     *   Mean image: `nwb.processing['ophys']['Backgrounds_0']['meanImg'].data`
# *   **Behavioral Data:**
#     *   Position: `nwb.processing['behavior']['BehavioralTimeSeries']['position'].data` and `.timestamps`
#     *   Lick events: `nwb.processing['behavior']['BehavioralTimeSeries']['lick'].data` and `.timestamps`
#     *   Speed: `nwb.processing['behavior']['BehavioralTimeSeries']['speed'].data` and `.timestamps`
#     *   Reward events: `nwb.processing['behavior']['BehavioralTimeSeries']['Reward'].data` and `.timestamps`

# %% [markdown]
# ## 4. Load and Visualize Data from the NWB File
#
# ### 4.1. Ophys Data: Fluorescence Traces
#
# We'll load fluorescence traces for a few ROIs over a short time window.
# The fluorescence data is located at `nwb.processing['ophys']['Fluorescence']['plane0']`.

# %%
if nwb:
    try:
        fluorescence_series = nwb.processing['ophys']['Fluorescence']['plane0']
        print(f"Fluorescence data shape: {fluorescence_series.data.shape}") # (num_timepoints, num_rois)
        print(f"Fluorescence timestamps shape: {fluorescence_series.timestamps.shape if hasattr(fluorescence_series, 'timestamps') and fluorescence_series.timestamps is not None else 'N/A, use rate'}")
        print(f"Fluorescence rate (Hz): {fluorescence_series.rate}")
        print(f"Fluorescence starting time (s): {fluorescence_series.starting_time}")

        num_rois_total = fluorescence_series.data.shape[1]
        num_timepoints_total = fluorescence_series.data.shape[0]

        # Select a few ROIs and a time window to plot
        rois_to_plot_indices = [0, 10, 20] # Example ROI indices
        if num_rois_total <= 20: # If very few ROIs, adjust
            rois_to_plot_indices = list(range(min(3, num_rois_total)))

        time_window_to_plot_seconds = 10  # seconds
        # Calculate time samples based on rate if timestamps are not directly used or are too large
        sampling_rate = fluorescence_series.rate
        num_samples_in_window = int(time_window_to_plot_seconds * sampling_rate)
        
        start_sample = 1000 # Starting sample index for plotting
        end_sample = min(start_sample + num_samples_in_window, num_timepoints_total) # Ensure we don't exceed total samples

        # Get ROI IDs if available, otherwise use indices
        plane_segmentation = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
        roi_ids = plane_segmentation.id[:]
        rois_to_plot_actual_ids = [roi_ids[i] for i in rois_to_plot_indices if i < len(roi_ids)]


        print(f"\nPlotting fluorescence for ROI indices {rois_to_plot_indices} (actual IDs: {rois_to_plot_actual_ids if rois_to_plot_actual_ids else 'N/A, using indices'})")
        print(f"Time window: samples {start_sample} to {end_sample} (approximately {time_window_to_plot_seconds} seconds)")

        # Load data for selected ROIs and time window
        # h5py datasets are sliced like [start_row:end_row, start_col:end_col]
        # For specific columns (ROIs): fluorescence_series.data[start_sample:end_sample, rois_to_plot_indices]
        # Note: Slicing with a list of indices for columns might be slow for remote files.
        # It's often better to load a contiguous block if possible, or load all and then select.
        # Given the potential size, we'll load only the selected ROIs.
        
        # Create time vector for the selected window
        # If timestamps are available and match data shape, prefer them
        if hasattr(fluorescence_series, 'timestamps') and fluorescence_series.timestamps is not None and fluorescence_series.timestamps.shape[0] == num_timepoints_total:
             time_vector = fluorescence_series.timestamps[start_sample:end_sample]
        else: # Calculate from rate and starting_time
            time_vector = fluorescence_series.starting_time + np.arange(start_sample, end_sample) / sampling_rate

        plt.figure(figsize=(15, 5))
        for i, roi_idx in enumerate(rois_to_plot_indices):
            if roi_idx < num_rois_total:
                # Load data for one ROI at a time to manage memory for remote files
                roi_trace = fluorescence_series.data[start_sample:end_sample, roi_idx]
                roi_label = f"ROI {roi_ids[roi_idx]}" if roi_idx < len(roi_ids) else f"ROI Index {roi_idx}"
                # Normalize or offset for better visualization if needed
                plt.plot(time_vector, roi_trace + i * np.percentile(roi_trace, 95), label=roi_label) # Offset traces

        plt.title(f"Fluorescence Traces (Samples {start_sample}-{end_sample})")
        plt.xlabel(f"Time (seconds from start: {time_vector[0]:.2f} s)")
        plt.ylabel("Fluorescence (arbitrary units, offset for clarity)")
        plt.legend(loc='upper right')
        sns.despine() # remove top and right spines
        plt.show()

    except KeyError as e:
        print(f"KeyError accessing ophys data: {e}. This data might not be available or path is incorrect.")
    except Exception as e:
        print(f"An error occurred during ophys data visualization: {e}")
else:
    print("NWB file not loaded, skipping ophys data visualization.")

# %% [markdown]
# ### 4.2. Behavioral Data
#
# Let's look at the animal's position and lick events over a segment of the recording.
# These are typically found in `nwb.processing['behavior']['BehavioralTimeSeries']`.

# %%
if nwb:
    try:
        behavior_ts = nwb.processing['behavior']['BehavioralTimeSeries']

        # --- Position Data ---
        if 'position' in behavior_ts:
            position_data = behavior_ts['position']
            pos_values = position_data.data[:] # Load all position data
            pos_timestamps = position_data.timestamps[:]

            # Select a time window (e.g., first 60 seconds of behavioral data)
            duration_seconds = 60
            
            # Find indices for the time window by comparing timestamps
            # Ensure pos_timestamps is not empty and is sorted
            if len(pos_timestamps) > 0:
                end_time_behavior = pos_timestamps[0] + duration_seconds
                indices_behavior = np.where(pos_timestamps <= end_time_behavior)[0]
                
                if len(indices_behavior) > 0:
                    selected_pos_timestamps = pos_timestamps[indices_behavior]
                    selected_pos_values = pos_values[indices_behavior]

                    plt.figure(figsize=(15, 4))
                    plt.plot(selected_pos_timestamps, selected_pos_values, label="Position")
                    plt.title(f"Animal Position (First {duration_seconds} seconds)")
                    plt.xlabel("Time (seconds)")
                    plt.ylabel(f"Position ({position_data.unit})")
                    plt.legend()
                    sns.despine()
                    plt.show()
                else:
                    print("No position data found in the selected time window.")
            else:
                print("Position timestamps are empty.")

        else:
            print("Position data not found in BehavioralTimeSeries.")

        # --- Lick Data ---
        if 'lick' in behavior_ts:
            lick_data_series = behavior_ts['lick']
            lick_values = lick_data_series.data[:] # Lick data (cumulative per frame)
            lick_timestamps = lick_data_series.timestamps[:]

            if len(lick_timestamps) > 0:
                # Use the same time window as for position for comparison
                end_time_behavior_lick = lick_timestamps[0] + duration_seconds
                indices_lick = np.where(lick_timestamps <= end_time_behavior_lick)[0]

                if len(indices_lick) > 0:
                    selected_lick_timestamps = lick_timestamps[indices_lick]
                    selected_lick_values = lick_values[indices_lick]
                    
                    # Lick data is often cumulative or binary, let's find changes for lick events
                    lick_events_times = selected_lick_timestamps[np.concatenate(([True], np.diff(selected_lick_values) > 0))]

                    plt.figure(figsize=(15, 4))
                    # Plot cumulative licks if desired
                    # plt.plot(selected_lick_timestamps, selected_lick_values, label="Cumulative Licks", alpha=0.5)
                    
                    # Plot lick events as vertical lines
                    if len(lick_events_times) > 0:
                        plt.vlines(lick_events_times, ymin=0, ymax=1, color='red', alpha=0.7, lw=2, label="Lick Events")
                        plt.title(f"Lick Events (First {duration_seconds} seconds)")
                    else:
                        plt.title(f"Lick Data (First {duration_seconds} seconds) - No lick events detected by diff")
                        plt.plot(selected_lick_timestamps, selected_lick_values, label="Lick Signal")


                    plt.xlabel("Time (seconds)")
                    plt.ylabel(f"Lick Status ({lick_data_series.unit})")
                    plt.ylim(-0.1, 1.1) # Assuming lick events are binary or for vlines
                    plt.legend()
                    sns.despine()
                    plt.show()
                else:
                    print("No lick data found in the selected time window.")
            else:
                print("Lick timestamps are empty.")
        else:
            print("Lick data not found in BehavioralTimeSeries.")

    except KeyError as e:
        print(f"KeyError accessing behavior data: {e}. This data might not be available or path is incorrect.")
    except Exception as e:
        print(f"An error occurred during behavioral data visualization: {e}")
else:
    print("NWB file not loaded, skipping behavioral data visualization.")


# %% [markdown]
# ### 4.3. Ophys Data: ROI Masks
#
# We can visualize the spatial footprints (masks) of the ROIs. These are stored in `nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']`. Each ROI has a `pixel_mask` which is a list of `(y, x, weight)` tuples for each pixel belonging to the ROI.
#
# We will create an image showing the outlines of a few selected ROIs.

# %%
if nwb:
    try:
        plane_segmentation = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
        if plane_segmentation:
            roi_ids = plane_segmentation.id[:]
            num_rois_total_masks = len(roi_ids)
            print(f"Total ROIs for masks: {num_rois_total_masks}")

            # Imaging plane dimensions are needed to create the image
            # These are in nwb.acquisition['TwoPhotonSeries'].dimension or nwb.imaging_planes['ImagingPlane'].dimension
            img_dims = None
            if 'TwoPhotonSeries' in nwb.acquisition and hasattr(nwb.acquisition['TwoPhotonSeries'], 'dimension'):
                img_dims = nwb.acquisition['TwoPhotonSeries'].dimension[:] # e.g., [512, 796] (height, width)
            elif 'ImagingPlane' in nwb.imaging_planes and hasattr(nwb.imaging_planes['ImagingPlane'], 'dimension'): # Fallback if not in TwoPhotonSeries
                 img_dims_obj = nwb.imaging_planes['ImagingPlane']
                 # Dimension might be a dataset or direct values, try to access it
                 if hasattr(img_dims_obj, 'dimension') and img_dims_obj.dimension is not None:
                     if hasattr(img_dims_obj.dimension, 'shape'): # It's a dataset
                         img_dims = img_dims_obj.dimension[:]
                     else: # Direct values
                         img_dims = img_dims_obj.dimension


            if img_dims is not None and len(img_dims) == 2:
                image_height, image_width = int(img_dims[0]), int(img_dims[1])
                print(f"Imaging plane dimensions: height={image_height}, width={image_width}")

                # Create a blank image to draw masks onto
                combined_mask_image = np.zeros((image_height, image_width), dtype=np.float32)

                rois_to_plot_mask_indices = [0, 10, 20, 30, 40] # Example ROI indices for masks
                if num_rois_total_masks <= 40: # If fewer ROIs, adjust
                     rois_to_plot_mask_indices = list(range(min(5, num_rois_total_masks)))
                
                print(f"Plotting masks for ROI indices: {rois_to_plot_mask_indices}")

                num_masks_plotted = 0
                for roi_idx in rois_to_plot_mask_indices:
                    if roi_idx < num_rois_total_masks:
                        pixel_mask_data = plane_segmentation['pixel_mask'][roi_idx] # This is a list of (x,y,weight) or (y,x,weight)
                        # Pixel_mask is often list of (x, y, weight) tuples or (pixel_index, weight)
                        # The structure (x_coord, y_coord, weight) or (y_coord, x_coord, weight) can vary.
                        # Suite2p convention is often (y,x) for coordinates.
                        # For nwb, pixel_mask is usually Nx3 where columns are x, y, weight or y, x, weight.
                        # Let's assume it's (y_coord, x_coord, weight) as often pixel_mask is (pixels, 3)
                        # and image is indexed image[y,x]
                        
                        # The `pixel_mask` in NWB is a list of (pixel_row, pixel_col, weight) tuples.
                        # Each row in `plane_segmentation['pixel_mask'][roi_idx]` refers to one pixel in the mask.
                        # `pixel_mask[i]` is an array like `[[y1, x1, w1], [y2, x2, w2], ...]`
                        
                        current_roi_mask_pixels = plane_segmentation.pixel_mask[roi_idx]
                        if isinstance(current_roi_mask_pixels, list): # If it's a list of lists for pixel coords/weights
                            for y, x, weight in current_roi_mask_pixels:
                                if 0 <= y < image_height and 0 <= x < image_width:
                                    combined_mask_image[int(y), int(x)] = max(combined_mask_image[int(y), int(x)], float(weight)) # Use max for superposition
                            num_masks_plotted +=1
                        elif isinstance(current_roi_mask_pixels, np.ndarray) and current_roi_mask_pixels.ndim == 2 and current_roi_mask_pixels.shape[1] == 3:
                            # Assuming it's an array of [y, x, weight]
                            for r_idx in range(current_roi_mask_pixels.shape[0]):
                                y, x, weight = current_roi_mask_pixels[r_idx, 0], current_roi_mask_pixels[r_idx, 1], current_roi_mask_pixels[r_idx, 2]
                                if 0 <= y < image_height and 0 <= x < image_width:
                                     combined_mask_image[int(y), int(x)] = max(combined_mask_image[int(y), int(x)], float(weight)) # Use max for superposition
                            num_masks_plotted +=1
                        else:
                            print(f"ROI {roi_ids[roi_idx]} pixel_mask has unexpected format: {type(current_roi_mask_pixels)}")


                if num_masks_plotted > 0 :
                    plt.figure(figsize=(8, 8 * (image_height / image_width) if image_width > 0 else 8))
                    # Do not use seaborn styling for images. Use a simple grayscale map.
                    plt.imshow(combined_mask_image, cmap='viridis', origin='upper', aspect='equal')
                    plt.title(f"Superimposed Masks for Selected ROIs (max projection)")
                    plt.xlabel("X pixels")
                    plt.ylabel("Y pixels")
                    plt.colorbar(label="Max Weight")
                    # Turn off seaborn theme for this plot specifically if it was globally set.
                    # This requires more complex context management for themes or reseting it.
                    # For simplicity, we'll rely on cmap not being too affected or accept seaborn's default grid.
                    # Or, one could reset specific rcParams altered by seaborn.
                    # For instance, sns.set_theme() adds a grid. To remove:
                    # plt.grid(False) # if sns added a grid
                    plt.show()
                else:
                    print("No masks were plotted. Check ROI indices or mask data format.")
            else:
                print("Could not determine imaging plane dimensions from NWB file.")
        else:
            print("PlaneSegmentation data not found.")
    except KeyError as e:
        print(f"KeyError accessing ROI mask data: {e}. This data might not be available or path is incorrect.")
    except Exception as e:
        print(f"An error occurred during ROI mask visualization: {e}")
else:
    print("NWB file not loaded, skipping ROI mask visualization.")

# %% [markdown]
# ## 5. Summary and Future Directions
#
# This notebook demonstrated how to:
# *   Connect to the DANDI archive and retrieve Dandiset metadata.
# *   Load an NWB file from the Dandiset, specifically `sub-m11/sub-m11_ses-03_behavior+ophys.nwb`.
# *   Access and understand some of its basic structure and metadata.
# *   Extract and visualize subsets of 2-photon imaging data (fluorescence traces).
# *   Extract and visualize subsets of behavioral data (position and licks).
# *   Display spatial ROI masks.
#
# ### Potential Future Directions:
#
# 1.  **Correlate Neural Activity with Behavior:** Analyze how neural fluorescence patterns change with respect to the animal's position, speed, or specific behavioral events like rewards or licks. For example, one could plot average fluorescence around reward times.
# 2.  **Place Cell Analysis:** If these are hippocampal recordings, investigate the presence of place cells by correlating neural activity with spatial location.
# 3.  **Trial-by-Trial Analysis:** If the data includes trial information (e.g., `nwb.intervals['trials']`), analyze how neural activity or behavior changes across trials or learning stages.
# 4.  **Population Dynamics:** Explore population-level neural dynamics using techniques like Principal Component Analysis (PCA) or other dimensionality reduction methods on the fluorescence traces.
# 5.  **Compare Across Sessions/Subjects:** Extend the analysis to other NWB files in the Dandiset to compare findings across different experimental sessions or animals.
# 6.  **Explore Deconvolved Data:** The file also contains `Deconvolved` traces in `nwb.processing['ophys']`. These represent an estimate of spiking activity and could be used for analyses more focused on discrete neural events.
# 7.  **Detailed Behavioral Analysis:** Further explore other behavioral variables available (e.g., `speed`, `reward_zone`, `teleport`) and their relationship with neural activity.
#
# Remember that data exploration is an iterative process. The visualizations here are starting points; deeper analysis will require more specific hypotheses and targeted computations.

# %% [markdown]
# This concludes the introductory exploration of Dandiset 001361.

# %%
if nwb and io:
    print("Closing NWB file.")
    io.close()
    # remote_file.close() # remfile.File objects might not have a close method or it might be handled by h5py.File closing.
                         # h5py.File.close() should handle closing the underlying remfile if it was passed correctly.
                         # The documentation for `remfile` would clarify this. Assuming h5py handles it.
if 'h5_file' in locals() and h5_file:
    try:
        h5_file.close() # Ensure h5_file is also closed if it's a separate object from io's file.
    except Exception as e:
        print(f"Error closing h5_file: {e}")
if 'remote_file' in locals() and remote_file:
    try:
        # Check if remote_file has a close method, common for file-like objects
        if hasattr(remote_file, 'close') and callable(remote_file.close):
            remote_file.close()
    except Exception as e:
        print(f"Error closing remote_file: {e}")

print("Notebook execution finished.")