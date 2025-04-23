# %% [markdown]
# # Exploring Dandiset 001361: A Flexible Hippocampal Population Code for Experience Relative to Reward
#
# This notebook is AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of Dandiset 001361
#
# Dandiset ID: 001361
# Title: A flexible hippocampal population code for experience relative to reward
# Description: 2-photon imaging and behavioral data from hippocampal area CA1 during virtual reality navigation in mice.
# [Dandiset on DANDI Archive](https://dandiarchive.org/dandiset/001361)

# %% [markdown]
# ### Contents of this Notebook
# - Load and Explore the Dandiset using the DANDI API
# - Visualize TwoPhotonSeries and Behavioral Data
# - Conclusions and Future Directions

# %% [markdown]
# ## Required Packages
# - h5py
# - matplotlib
# - pynwb
# - numpy
# - remfile
# - seaborn

# %% [markdown]
# ## Load the Dandiset using DANDI API

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001361")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List the assets in the Dandiset
assets = list(dandiset.get_assets())
print(f"\nFound {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Visualizing Behavioral Data

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pynwb import NWBHDF5IO
from remfile import File

# Set the theme for seaborn
sns.set_theme()

url = 'https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/'
remote_file = File(url)
h5_file = h5py.File(remote_file)
io = NWBHDF5IO(file=h5_file)
nwb = io.read()

# Plot and visualize behavior speed data
behavior_data = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['speed']
speed_data = behavior_data.data[:]
speed_timestamps = behavior_data.timestamps[:]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(speed_timestamps, speed_data, label='Speed')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (cm/s)')
ax.set_title('Speed Over Time')
plt.legend()

# %% [markdown]
# ## Conclusions and Future Directions
# 
# The analysis provides insights into the subject's behavior. Future analysis could focus on correlating speed with other behavioral parameters or explore more detailed features of TwoPhotonSeries.