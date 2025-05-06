# %% [markdown]
# # Exploring Dandiset 001361: A flexible hippocampal population code for experience relative to reward
#
# **Important:** This notebook was AI-generated and has not been fully verified. Use caution when interpreting the code or results.
#
# ## Overview of the Dandiset
#
# This notebook explores Dandiset 001361, which contains 2-photon imaging and behavioral data from hippocampal area CA1 during virtual reality navigation in mice. The data is described in Sosa, Marielena; Plitt, Mark H.; Giocomo, Lisa M. (2025) "A flexible hippocampal population code for experience relative to reward," Nature Neuroscience. The study investigates how hippocampal activity encodes sequences of events relative to reward during virtual reality navigation.
#
# The Dandiset can be found on the DANDI Archive at https://dandiarchive.org/dandiset/001361/0.250406.0045.
#
# ## What this notebook covers
#
# This notebook demonstrates how to:
#
# *   Load the Dandiset metadata using the DANDI API.
# *   List the assets (files) available in the Dandiset.
# *   Load one of the NWB files in the Dandiset.
# *   Explore the structure of the NWB file.
# *   Visualize behavioral and electrophysiology data from the NWB file.
# *   Correlate position and neural activity.
#
# ## Required packages
#
# The following packages are required to run this notebook:
#
# *   `pynwb`
# *   `h5py`
# *   `remfile`
# *   `matplotlib`
# *   `numpy`
#
#
# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001361", "0.250406.0045")

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
# ## Loading an NWB file and exploring its contents
#
# In this section, we will load one of the NWB files in the Dandiset and explore its contents.
#
# We will load the file `sub-m11/sub-m11_ses-03_behavior+ophys.nwb`.
#
# Here's how to get the URL for the asset:

# %%
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ## Exploring the NWB file structure
#
# In this section, we will explore the structure of the NWB file to understand how the data is organized.

# %%
print(nwb)

# %% [markdown]
# The above output shows the structure of the NWB file, including the available data interfaces and processing modules.

# %% [markdown]
# The above nwb object contains all the data from the NWB file.
#
# Let's start by exploring the `TwoPhotonSeries` data. This data contains the raw imaging data from the two-photon microscope.

# %%
TwoPhotonSeries = nwb.acquisition["TwoPhotonSeries"]
TwoPhotonSeries

# %% [markdown]
# Now let's explore the position data. This data represents the position of the mouse in the virtual reality environment.

# %%
position = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"]
position

# %%
# The above shows the position data.
# Now let's visualize the position data.

import matplotlib.pyplot as plt
import numpy as np

position_data = position.data[:]
position_timestamps = position.timestamps[:]

plt.figure(figsize=(10, 5))
plt.plot(position_timestamps, position_data)
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.title("Position over Time")
plt.show()

# %% [markdown]
# The above plot show the position data over time. The mouse is moving back and forth in the virtual reality environment.

# %%
# Now let's explore the lick data. This data represents the licking behavior of the mouse.

lick = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["lick"]
lick

# %%
# The above shows the lick data.
# Now let's visualize the lick data.

lick_data = lick.data[:]
lick_timestamps = lick.timestamps[:]

plt.figure(figsize=(10, 5))
plt.plot(lick_timestamps, lick_data)
plt.xlabel("Time (s)")
plt.ylabel("Lick (AU)")
plt.title("Lick over Time")
plt.show()

# %% [markdown]
# The above plot shows the lick data over time. The mouse is licking at different times during the experiment.

# %%
# Now let's explore the fluorescence data. This data represents the activity of the neurons in the hippocampus.

fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["plane0"]
fluorescence

# %%
# The above shows the fluorescence data.
# Now let's visualize the fluorescence data.
# Handle missing timestamps

fluorescence_data = fluorescence.data[:]
if fluorescence.timestamps is not None:
    fluorescence_timestamps = fluorescence.timestamps[:]
else:
    starting_time = fluorescence.starting_time
    rate = fluorescence.rate
    num_frames = fluorescence_data.shape[0]
    fluorescence_timestamps = np.arange(num_frames) / rate + starting_time


num_rois = min(5, fluorescence_data.shape[1])
plt.figure(figsize=(10, 5))
for i in range(num_rois):
    plt.plot(fluorescence_timestamps, fluorescence_data[:, i], label=f"ROI {i}")

plt.xlabel("Time (s)")
plt.ylabel("Fluorescence (lumens)")
plt.title("Fluorescence over Time (First 5 ROIs)")
plt.legend()
plt.show()

# %% [markdown]
# The above plot shows the fluorescence data over time for the first 5 ROIs. Each ROI represents a different neuron in the hippocampus.
# There are a few spikes in fluorescence, particularly in ROI 1 around 650s, superimposed on a relatively constant background level of fluorescence for each ROI.

# %% [markdown]
# ## Correlating position and neural activity
#
# In this section, we will correlate the position data with the fluorescence data to see how the activity of the neurons relates to the position of the mouse.

# %%
# Correlate position and fluorescence data
position_reshaped = position_data[:len(fluorescence_timestamps)]
fluorescence_reshaped = fluorescence_data[:len(position_data), 0]

correlation = np.corrcoef(position_reshaped, fluorescence_reshaped)[0, 1]
print(f"Correlation between position and fluorescence: {correlation}")

# %% [markdown]
# # Summary and future directions
#
# This notebook has demonstrated how to load data from Dandiset 001361 and visualize some of the key data streams from one of the NWB files. The notebook has also demonstrated that the neural activity as captured by fluorescence imaging is correlated with the position of the mouse.
#
# Possible future directions for analysis include:
#
# *   Investigating the relationship between licking behavior and reward delivery.
# *   Performing spike sorting on the raw electrophysiology data to identify individual neurons and analyze their firing patterns in relation to position and reward.
# *   Analyzing the calcium imaging data to investigate the dynamics of neural populations during different phases of the experiment.
# *   Constructing place fields from the neural activity.