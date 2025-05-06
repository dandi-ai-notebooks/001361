# %% [markdown]
# # Exploring Dandiset 001361: A flexible hippocampal population code for experience relative to reward
# **Note**: This AI-generated notebook has not been fully verified. Use caution when interpreting code or results.

# %% [markdown]
# ## Overview of Dandiset
# Link: https://dandiarchive.org/dandiset/001361/0.250406.0045

# %% [markdown]
# ### Scientific Context
# The hippocampus contains place cells whose firing fields encode the position of the animal in space.
# This Dandiset contains two-photon calcium imaging and behavioral data acquired from hippocampal area CA1 during virtual navigation tasks with shifting reward locations, enabling analysis of how place cells adapt relative to reward locations.

# %% [markdown]
# This notebook will cover:
# - Retrieving Dandiset metadata and listing assets  
# - Selecting and loading one NWB file remotely  
# - Summarizing NWB file contents  
# - Visualizing behavioral time series (position, speed, reward events)  
# - Visualizing deconvolved ROI responses  
# - Summary and future directions  

# %% [markdown]
# ## Required Packages
# ```python
# from itertools import islice
# from dandi.dandiapi import DandiAPIClient
# import remfile
# import h5py
# import pynwb
# import numpy as np
# import matplotlib.pyplot as plt
# ```

# %% 
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001361", "0.250406.0045")

# Print basic metadata
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List first 5 assets
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Selecting an NWB File
# We will use the file `sub-m11/sub-m11_ses-03_behavior+ophys.nwb`.

# %% 
import remfile
import h5py
import pynwb

# Asset URL hard-coded from inspection
nwb_url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"

# Load remote NWB file
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ### NWB File Contents
# - **Session description**: `nwb.session_description`  
# - **Identifier**: `nwb.identifier`  
# - **Session start time**: `nwb.session_start_time`  
# - **Subject**: ID=`nwb.subject.subject_id`, species=`nwb.subject.species`, sex=`nwb.subject.sex`  
# - **Processing modules**: `list(nwb.processing.keys())`  

# %% 
# Print basic NWB metadata
print("Session description:", nwb.session_description)
print("Identifier:", nwb.identifier)
print("Session start:", nwb.session_start_time)
print("Subject:", nwb.subject.subject_id, nwb.subject.species, nwb.subject.sex)
print("Processing modules:", list(nwb.processing.keys()))

# %% [markdown]
# **NeuroSift link**:  
# https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/&dandisetId=001361&dandisetVersion=draft

# %% [markdown]
# ## Behavioral Time Series Visualization

# %% 
import numpy as np
import matplotlib.pyplot as plt

# Extract behavioral time series
behavior_ts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries']
position = behavior_ts.time_series['position']
speed = behavior_ts.time_series['speed']
reward = behavior_ts.time_series['Reward']

# Load arrays (subset if needed)
pos_t = position.timestamps[:]
pos_data = position.data[:]
speed_t = speed.timestamps[:]
speed_data = speed.data[:]
reward_t = reward.timestamps[:]

# Position over time
plt.figure(figsize=(10,4))
plt.plot(pos_t, pos_data, label='Position (cm)')
if len(reward_t) > 0:
    plt.scatter(reward_t, [pos_data.max()]*len(reward_t),
                color='red', marker='v', label='Reward')
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.title('Position over Time with Reward Events')
plt.legend()
plt.tight_layout()
plt.show()

# Speed over time
plt.figure(figsize=(10,4))
plt.plot(speed_t, speed_data, label='Speed (cm/s)')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Speed over Time')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Deconvolved ROI Response Visualization
# Plotting the first 5 ROIs from the deconvolved Fluorescence data (`plane0`).

# %% 
# Deconvolved ROI responses
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
# Subset: all timepoints, first 5 ROIs
roi_data = deconv.data[:, :5]
time = np.arange(roi_data.shape[0]) / deconv.rate

plt.figure(figsize=(10,6))
for idx in range(roi_data.shape[1]):
    plt.plot(time, roi_data[:, idx], label=f'ROI {idx}')
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (lumens)')
plt.title('Deconvolved ROI Responses for First 5 ROIs')
plt.legend(ncol=2, fontsize='small')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary and Future Directions
# This notebook demonstrated:
# - How to access and summarize Dandiset metadata and assets.  
# - Remote loading of an NWB file via DANDI API.  
# - Exploration of behavioral metrics and reward events.  
# - Visualization of calcium traces for selected ROIs.  
#
# Future analyses could include:
# - Event-triggered averaging around reward delivery.  
# - Spatial encoding and place field mapping.  
# - Comparative analyses across sessions or subjects.  
# - Integration of segmentation masks and ROI spatial footprints.  
# - Linking neural dynamics to behavioral states.