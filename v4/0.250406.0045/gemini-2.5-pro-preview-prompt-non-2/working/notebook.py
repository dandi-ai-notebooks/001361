# %% [markdown]
# # Exploring Dandiset 001361: A flexible hippocampal population code for experience relative to reward
#
# **Disclaimer:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This notebook explores Dandiset [001361](https://dandiarchive.org/dandiset/001361/0.250406.0045), version 0.250406.0045.
#
# **Dandiset Name:** A flexible hippocampal population code for experience relative to reward
#
# **Description:** 2-photon imaging and behavioral data from hippocampal area CA1 during virtual reality navigation in mice. Included in Sosa, Plitt, &amp; Giocomo, "A flexible hippocampal population code for experience relative to reward," Nature Neuroscience.
#
# To reinforce rewarding behaviors, events leading up to and following rewards must be remembered. Hippocampal place cell activity spans spatial and non-spatial episodes, but whether hippocampal activity encodes entire sequences of events relative to reward is unknown. To test this, we performed two-photon imaging of hippocampal CA1 as mice navigated virtual environments with changing hidden reward locations. When the reward moved, a subpopulation of neurons updated their firing fields to the same relative position with respect to reward, constructing behavioral timescale sequences spanning the entire task. Over learning, this reward-relative representation became more robust as additional neurons were recruited, and changes in reward-relative firing often preceded behavioral adaptations following reward relocation. Concurrently, the spatial environment code was maintained through a parallel, dynamic subpopulation rather than through dedicated cell classes. These findings reveal how hippocampal ensembles flexibly encode multiple aspects of experience while amplifying behaviorally relevant information.
#
# This notebook will cover:
# - How to load the Dandiset using the DANDI API.
# - How to load a specific NWB file from the Dandiset.
# - How to access and summarize metadata from the NWB file.
# - How to load and visualize some of the data contained within the NWB file.

# %% [markdown]
# ## Required Packages
#
# The following Python packages are required to run this notebook. It is assumed that they are already installed on your system.
# - `dandi`
# - `pynwb`
# - `h5py`
# - `remfile`
# - `numpy`
# - `matplotlib`
# - `seaborn`

# %% [markdown]
# ## Loading the Dandiset

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import numpy asnp
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to DANDI archive
client = DandiAPIClient()
dandiset_id = "001361"
dandiset_version = "0.250406.0045"
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Loading an NWB File
#
# We will now load one of the NWB files from the Dandiset to explore its contents. We will use the file `sub-m11/sub-m11_ses-03_behavior+ophys.nwb`.
#
# The URL for this asset is constructed using its asset ID: `d77ea78a-8978-461d-9d11-3c5cef860d82`.

# %%
# Load the NWB file
# The URL is hard-coded here based on the information retrieved previously.
nwb_file_url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
print(f"Loading NWB file from: {nwb_file_url}")

remote_file = remfile.File(nwb_file_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Specify read-only mode
nwbfile = io.read()

print("\nNWB file loaded successfully.")
print(f"Identifier: {nwbfile.identifier}")
print(f"Session description: {nwbfile.session_description}")
print(f"Session start time: {nwbfile.session_start_time}")

# %% [markdown]
# You can also explore this NWB file interactively on NeuroSift:
# [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/&dandisetId=001361&dandisetVersion=draft](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/&dandisetId=001361&dandisetVersion=draft)

# %% [markdown]
# ## Summarizing NWB File Contents
#
# Let's look at the structure of the NWB file.

# %%
print("Key groups and datasets in the NWB file:")
print(f"- Acquisition: {list(nwbfile.acquisition.keys())}")
if nwbfile.processing:
    print(f"- Processing modules: {list(nwbfile.processing.keys())}")
    if "behavior" in nwbfile.processing:
        print(f"  - Behavior data interfaces: {list(nwbfile.processing['behavior'].data_interfaces.keys())}")
        if "BehavioralTimeSeries" in nwbfile.processing["behavior"].data_interfaces:
            print(f"    - BehavioralTimeSeries: {list(nwbfile.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series.keys())}")
    if "ophys" in nwbfile.processing:
        print(f"  - Ophys data interfaces: {list(nwbfile.processing['ophys'].data_interfaces.keys())}")
else:
    print("- No processing modules found.")
print(f"- Devices: {list(nwbfile.devices.keys())}")
print(f"- Imaging planes: {list(nwbfile.imaging_planes.keys())}")
print(f"- Subject information: {nwbfile.subject}")

# %% [markdown]
# ### Behavioral Data
#
# The NWB file contains several time series related to behavior. We can list them:

# %%
if "behavior" in nwbfile.processing and "BehavioralTimeSeries" in nwbfile.processing["behavior"].data_interfaces:
    behavior_ts = nwbfile.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
    print("Behavioral Time Series:")
    for ts_name, ts_data in behavior_ts.items():
        print(f"- {ts_name}: {ts_data.description} (shape: {ts_data.data.shape}, unit: {ts_data.unit})")
else:
    print("BehavioralTimeSeries not found in processing['behavior'].")

# %% [markdown]
# Let's visualize the mouse's position on the virtual linear track over a short period.
# We will load the first 500 data points and corresponding timestamps for `position`.

# %%
sns.set_theme() # Apply seaborn styling

if "behavior" in nwbfile.processing and \
   "BehavioralTimeSeries" in nwbfile.processing["behavior"].data_interfaces and \
   "position" in nwbfile.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series:

    position_ts = nwbfile.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"]
    num_points_to_plot = 500

    if len(position_ts.data) >= num_points_to_plot:
        position_data_subset = position_ts.data[:num_points_to_plot]
        position_timestamps_subset = position_ts.timestamps[:num_points_to_plot]

        plt.figure(figsize=(12, 4))
        plt.plot(position_timestamps_subset, position_data_subset)
        plt.xlabel(f"Time ({position_ts.timestamps_unit})")
        plt.ylabel(f"Position ({position_ts.unit})")
        plt.title(f"Mouse Position (First {num_points_to_plot} points)")
        plt.show()
    else:
        print(f"Not enough data points in 'position' to plot {num_points_to_plot} points. Available: {len(position_ts.data)}")
else:
    print("'position' time series not found in behavioral data.")

# %% [markdown]
# Let's also look at the lick data for the same time period.

# %%
if "behavior" in nwbfile.processing and \
   "BehavioralTimeSeries" in nwbfile.processing["behavior"].data_interfaces and \
   "lick" in nwbfile.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series:

    lick_ts = nwbfile.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["lick"]
    # We use the same number of points as for position for easier comparison, if available
    num_points_to_plot = 500

    if len(lick_ts.data) >= num_points_to_plot:
        lick_data_subset = lick_ts.data[:num_points_to_plot]
        lick_timestamps_subset = lick_ts.timestamps[:num_points_to_plot] # Assuming timestamps align or are similar rate

        plt.figure(figsize=(12, 4))
        plt.plot(lick_timestamps_subset, lick_data_subset)
        plt.xlabel(f"Time ({lick_ts.timestamps_unit})")
        plt.ylabel(f"Lick ({lick_ts.unit})")
        plt.title(f"Lick Data (First {num_points_to_plot} points)")
        plt.show()
    else:
        print(f"Not enough data points in 'lick' to plot {num_points_to_plot} points. Available: {len(lick_ts.data)}")
else:
    print("'lick' time series not found in behavioral data.")

# %% [markdown]
# ### Optical Physiology (Ophys) Data
#
# The NWB file also contains optical physiology data. Let's examine the available ophys data interfaces.

# %%
if "ophys" in nwbfile.processing:
    ophys_interfaces = nwbfile.processing["ophys"].data_interfaces
    print("Ophys Data Interfaces:")
    for name, interface in ophys_interfaces.items():
        print(f"- {name} (type: {type(interface).__name__})")
        if isinstance(interface, pynwb.ophys.Fluorescence):
            print("  Contains ROI response series:")
            for rrs_name, rrs_data in interface.roi_response_series.items():
                print(f"    - {rrs_name}: shape {rrs_data.data.shape}, rate {rrs_data.rate} Hz")
        elif isinstance(interface, pynwb.ophys.ImageSegmentation):
            print("  Contains plane segmentations:")
            for ps_name, ps_data in interface.plane_segmentations.items():
                print(f"    - {ps_name}: {len(ps_data.id)} ROIs")
        elif isinstance(interface, pynwb.base.Images):
            print("  Contains images:")
            for img_name in interface.images.keys():
                print(f"    - {img_name}")
else:
    print("Ophys processing module not found.")

# %% [markdown]
# #### ROI Fluorescence Traces
#
# We can plot the fluorescence traces for a few ROIs (Regions of Interest, likely individual cells).
# Let's look at the `Fluorescence` data, specifically `plane0`.

# %%
if "ophys" in nwbfile.processing and \
   "Fluorescence" in nwbfile.processing["ophys"].data_interfaces and \
   "plane0" in nwbfile.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series:

    fluorescence_rrs = nwbfile.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["plane0"]
    num_rois_to_plot = 3
    num_timepoints_to_plot = 500 # Plot a subset of timepoints to keep it manageable

    if fluorescence_rrs.data.shape[1] >= num_rois_to_plot and fluorescence_rrs.data.shape[0] >= num_timepoints_to_plot:
        roi_indices_to_plot = np.arange(num_rois_to_plot) # Plot first N ROIs
        actual_roi_ids = fluorescence_rrs.rois.table.id[:num_rois_to_plot]

        fluorescence_data_subset = fluorescence_rrs.data[:num_timepoints_to_plot, roi_indices_to_plot]

        # Generate timestamps for the subset
        # Timestamps are not directly stored per data point, but can be calculated from starting_time and rate
        time_vector = fluorescence_rrs.starting_time + np.arange(num_timepoints_to_plot) / fluorescence_rrs.rate

        plt.figure(figsize=(12, 6))
        for i, roi_idx in enumerate(roi_indices_to_plot):
            plt.plot(time_vector, fluorescence_data_subset[:, i], label=f"ROI ID {actual_roi_ids[i]}")

        plt.xlabel(f"Time ({fluorescence_rrs.starting_time_unit})")
        plt.ylabel(f"Fluorescence ({fluorescence_rrs.unit})")
        plt.title(f"Fluorescence Traces (First {num_rois_to_plot} ROIs, First {num_timepoints_to_plot} Timepoints)")
        plt.legend()
        plt.show()
    else:
        print(f"Not enough ROIs or timepoints in Fluorescence/plane0. Available ROIs: {fluorescence_rrs.data.shape[1]}, Available Timepoints: {fluorescence_rrs.data.shape[0]}")
else:
    print("Fluorescence/plane0 data not found.")

# %% [markdown]
# #### Image Segmentation and ROI Masks
#
# The `ImageSegmentation` interface provides information about the ROIs, including their pixel masks.
# Let's try to visualize the spatial layout of some ROIs. We'll use the `PlaneSegmentation` within `ImageSegmentation`.

# %%
# No seaborn styling for images
plt.style.use('default')

if "ophys" in nwbfile.processing and \
   "ImageSegmentation" in nwbfile.processing["ophys"].data_interfaces and \
   "PlaneSegmentation" in nwbfile.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations:

    plane_segmentation = nwbfile.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]

    # Get image dimensions from the imaging plane associated with the TwoPhotonSeries
    # This assumes there's a TwoPhotonSeries associated with this plane.
    # A more robust way would be to check plane_segmentation.imaging_plane or reference_images if available.
    # For this example, we'll try to get it from the TwoPhotonSeries if it exists, otherwise assume default.
    img_dims = None
    if "TwoPhotonSeries" in nwbfile.acquisition:
        ts_img = nwbfile.acquisition["TwoPhotonSeries"]
        if ts_img.dimension is not None and len(ts_img.dimension) >= 2:
             # Dimension usually is [height, width] or [depth, height, width]
             # For 2D imaging, it's [height, width]
            img_dims = ts_img.dimension[:2] # Assuming 2D for simplicity, might be [y,x]
            img_dims = [int(d) for d in img_dims] # Ensure they are integers
            print(f"Image dimensions from TwoPhotonSeries: {img_dims}")


    # Fallback or if we need to use reference images dimensions
    if img_dims is None and plane_segmentation.reference_images:
        # This part is more complex as reference_images might not directly give dimensions
        # or might be a list of images. For simplicity, we'll skip this for now.
        print("Could not robustly determine image dimensions from TwoPhotonSeries, and reference_images parsing is complex.")

    if img_dims is None:
        # If we still don't have dimensions, we might need to infer from pixel masks or a reference image.
        # Let's try to get it from a reference image if available (e.g. meanImg)
        if "ophys" in nwbfile.processing and "Backgrounds_0" in nwbfile.processing["ophys"].data_interfaces:
            backgrounds = nwbfile.processing["ophys"].data_interfaces["Backgrounds_0"]
            if "meanImg" in backgrounds.images:
                mean_image = backgrounds.images["meanImg"].data
                if mean_image is not None and hasattr(mean_image, 'shape'):
                    img_dims = mean_image.shape
                    print(f"Image dimensions from meanImg: {img_dims}")


    if img_dims:
        num_rois_to_show = min(10, len(plane_segmentation.id)) # Show up to 10 ROIs
        all_masks_image = np.zeros(img_dims, dtype=float)

        print(f"Attempting to plot masks for the first {num_rois_to_show} ROIs.")
        for i in range(num_rois_to_show):
            try:
                # pixel_mask is a list of (y, x, weight) tuples for each ROI
                pixel_mask_data = plane_segmentation["pixel_mask"][i]
                for y_coord, x_coord, weight in pixel_mask_data:
                    if 0 &lt;= y_coord &lt; img_dims[0] and 0 &lt;= x_coord &lt; img_dims[1]:
                         # Some masks might have weights > 1 due to Suite2p format, clip or normalize if necessary
                        all_masks_image[int(y_coord), int(x_coord)] = max(all_masks_image[int(y_coord), int(x_coord)], weight)
            except IndexError:
                print(f"Warning: Could not access pixel_mask for ROI index {i}. Skipping.")
                continue
            except Exception as e:
                print(f"Warning: Error processing pixel_mask for ROI index {i}: {e}. Skipping.")
                continue
        
        if np.any(all_masks_image > 0): # Check if any masks were actually added
            plt.figure(figsize=(8, 8))
            # Use np.max for overlaying, assuming weights are somewhat like probabilities or intensities
            # The previous loop already effectively did a max accumulation per pixel.
            plt.imshow(all_masks_image, cmap='viridis', aspect='auto', origin='lower')
            plt.title(f"Overlay of First {num_rois_to_show} ROI Masks")
            plt.xlabel("X pixel")
            plt.ylabel("Y pixel")
            plt.colorbar(label="Max Mask Weight")
            plt.show()
        else:
            print("No ROI masks could be plotted. The combined mask image is empty.")

    else:
        print("Could not determine image dimensions to plot ROI masks.")
        print("To visualize ROI masks, image dimensions are needed. These can often be found in:")
        print("- `nwbfile.acquisition['TwoPhotonSeries'].dimension`")
        print("- The shape of a reference image like `nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg'].data`")
        print("- `plane_segmentation.imaging_plane.grid_spacing` combined with pixel counts if available.")

else:
    print("ImageSegmentation/PlaneSegmentation data not found or image dimensions could not be determined.")


# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to:
# - Load a Dandiset and inspect its basic metadata and assets.
# - Load a specific NWB file from the Dandiset using its DANDI API URL.
# - Access and summarize key metadata and data structures within the NWB file, including behavioral time series and optical physiology data (fluorescence traces, ROI masks).
# - Create basic visualizations of position, lick data, and ROI fluorescence.
# - Attempt to visualize ROI masks overlayed on an image.
#
# **Potential Future Directions for Analysis:**
# - **Correlate neural activity with behavior:** Investigate how the fluorescence activity of individual neurons or populations relates to specific behavioral events (e.g., animal's position, speed, reward consumption, trial events).
# - **Analyze place cell properties:** If the data contains spatial information and neural activity from the hippocampus, one could analyze place cell characteristics (e.g., field size, stability, remapping across environments or learning).
# - **Population coding analysis:** Explore how ensembles of neurons collectively represent information about the environment, task variables, or internal states.
# - **Task-related modulation of activity:** Examine how neural activity changes across different phases of a trial or in response to specific task events (e.g., stimulus presentation, reward delivery).
# - **Cross-session comparisons:** If data from multiple sessions are available for the same subject/neurons, investigate learning-related changes in neural representations.
#
# Remember that this notebook provides a starting point. Deeper analysis would require more sophisticated techniques and careful consideration of the experimental design and data characteristics.

# %% [markdown]
# It's important to close the HDF5 file and the remfile object if you are done with them, especially in scripts.
# However, in a Jupyter notebook context, this is often omitted for brevity as the kernel shutdown usually handles it.
# For completeness:
# ```python
# # io.close()
# # h5_file.close()
# # remote_file.close()
# ```
# We will leave them open for now as the notebook execution might need them if cells are re-run.

# %%
print("Notebook execution finished.")