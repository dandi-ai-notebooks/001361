# %% [markdown]
"""
# Exploring Dandiset 001361: A Flexible Hippocampal Population Code for Experience Relative to Reward

**DISCLAIMER**: This notebook was AI-generated and has not been fully verified. Use caution interpreting code or results.

**Dandiset:** https://dandiarchive.org/dandiset/001361/0.250406.0045

**Overview:**  
- Load Dandiset metadata and list assets  
- Load an NWB file and inspect its structure  
- Visualize behavioral data (speed, position)  
- Summarize findings and suggest future directions
"""

# %% [markdown]
"""
## Required Packages

- itertools  
- dandi.dandiapi.DandiAPIClient  
- remfile  
- h5py  
- pynwb  
- matplotlib  
- seaborn  
"""

# %% 
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive and load metadata
client = DandiAPIClient()
dandiset = client.get_dandiset("001361", "0.250406.0045")
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List first 5 assets
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
"""
## Load NWB file session sub-m11_ses-03_behavior+ophys.nwb

This cell loads the NWB file from the DANDI archive and prints basic metadata.
"""

# %%
import remfile
import h5py
from pynwb import NWBHDF5IO

nwb_url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
print(f"Loading NWB file from: {nwb_url}")
remote = remfile.File(nwb_url)
h5f = h5py.File(remote, 'r')
io = NWBHDF5IO(file=h5f)
nwb = io.read()

print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

# %% [markdown]
"""
## NWB File Structure Summary

```
nwb
├── acquisitions
├── processing
│   ├── behavior
│   │   └── time_series: Reward, autoreward, environment, lick, position, reward_zone, scanning, speed, teleport, trial number, trial_start
│   └── ophys
│       ├── Deconvolved (RoiResponseSeries)
│       └── Fluorescence (RoiResponseSeries)
└── subject, devices, imaging_planes
```
"""

# %% [markdown]
"""
Explore this NWB file on NeuroSift:  
[Open in NeuroSift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/&dandisetId=001361&dandisetVersion=draft)
"""

# %% [markdown]
"""
## Visualize Behavioral Data

The following cell plots speed and position over the first 2,000 samples.
"""

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

bt = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries']
speed = bt.time_series['speed']
position = bt.time_series['position']

n_samples = 2000
t_speed = speed.timestamps[:n_samples]
d_speed = speed.data[:n_samples]
t_pos = position.timestamps[:n_samples]
d_pos = position.data[:n_samples]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(t_speed, d_speed, color='C0')
axes[0].set_title("Speed over time (first 2000 samples)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel(f"Speed ({speed.unit})")

axes[1].plot(t_pos, d_pos, color='C1')
axes[1].set_title("Position over time (first 2000 samples)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel(f"Position ({position.unit})")

plt.tight_layout()

# %% [markdown]
"""
## Summary and Future Directions

**Findings:**  
- Behavioral data show clear cyclic navigation patterns with varying speed and position across trials.  
- NWB file structure provides straightforward access to behavioral and optical physiology data.

**Future Directions:**  
- Visualize optical physiology (e.g., deconvolved fluorescence signals) for individual ROIs.  
- Correlate neural activity with behavioral metrics such as speed and position.  
- Investigate reward-related time series to study learning dynamics.
"""