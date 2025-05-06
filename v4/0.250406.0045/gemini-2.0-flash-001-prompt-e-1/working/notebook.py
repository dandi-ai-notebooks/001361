# %% [markdown]
# # Exploring Dandiset 001361: A flexible hippocampal population code for experience relative to reward

# %% [markdown]
# **Important:** This notebook was AI-generated and has not been fully verified. Use caution when interpreting the code or results.

# %% [markdown]
# ## Overview of Dandiset 001361
#
# This notebook provides an exploration of Dandiset 001361, titled "A flexible hippocampal population code for experience relative to reward." This Dandiset contains 2-photon imaging and behavioral data from hippocampal area CA1 during virtual reality navigation in mice. The data were collected by Sosa, Marielena; Plitt, Mark H.; Giocomo, Lisa M. and are related to their publication in Nature Neuroscience.
#
# Dandiset URL: [https://dandiarchive.org/dandiset/001361/0.250406.0045](https://dandiarchive.org/dandiset/001361/0.250406.0045)

# %% [markdown]
# ## What this notebook will cover
#
# This notebook will guide you through the process of:
#
# 1.  Loading the Dandiset metadata using the DANDI API.
# 2.  Loading a specific NWB file from the Dandiset.
# 3.  Exploring the contents of the NWB file, including behavioral and optical physiology data.
# 4.  Visualizing some of the data.
# 5.  Suggesting possible future directions for analysis.

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook. Please make sure they are installed in your environment.
#
# *   pynwb
# *   h5py
# *   remfile
# *   matplotlib
# *   numpy
# *   seaborn

# %% [markdown]
# ## Loading the Dandiset using the DANDI API

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
# In this section, we will load one of the NWB files from the Dandiset and explore its contents. We will focus on the file:
#
# `sub-m11/sub-m11_ses-03_behavior+ophys.nwb`
#
# This file contains both behavioral and optical physiology data.
#
# We will be using the following URL to access the file:
#
# `https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/`

# %%
import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

nwb # (NWBFile)
print(f"Session description: {nwb.session_description}") # (str) processed suite2p data
print(f"Identifier: {nwb.identifier}") # (str) /data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A
print(f"Session start time: {nwb.session_start_time}") # (datetime) 2023-02-23T00:00:00-08:00
print(f"Timestamps reference time: {nwb.timestamps_reference_time}") # (datetime) 2023-02-23T00:00:00-08:00
print(f"File create date: {nwb.file_create_date}") # (list) [datetime.datetime(2025, 3, 12, 23, 45, 29, 830157, tzinfo=tzoffset(None, -25200))]
print(f"Experimenter: {nwb.experimenter}") # (tuple) ['Mari Sosa']
# %% [markdown]
# You can explore this NWB file on neurosift using the following link:
#
# [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/&dandisetId=001361&dandisetVersion=draft](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/&dandisetId=001361&dandisetVersion=draft)

# %% [markdown]
# ### Acquisition: TwoPhotonSeries
#
# This section describes the TwoPhotonSeries data.

# %%
acquisition = nwb.acquisition
TwoPhotonSeries = acquisition["TwoPhotonSeries"]
TwoPhotonSeries # (TwoPhotonSeries)
print(f"Starting time: {TwoPhotonSeries.starting_time}") # (float64) 0.0
print(f"Rate: {TwoPhotonSeries.rate}") # (float64) 15.5078125
print(f"Resolution: {TwoPhotonSeries.resolution}") # (float64) -1.0
print(f"Comments: {TwoPhotonSeries.comments}") # (str) no comments
print(f"Description: {TwoPhotonSeries.description}") # (str) no description
print(f"Conversion: {TwoPhotonSeries.conversion}") # (float64) 1.0
print(f"Offset: {TwoPhotonSeries.offset}") # (float64) 0.0
print(f"Unit: {TwoPhotonSeries.unit}") # (str) volt
print(f"Starting time unit: {TwoPhotonSeries.starting_time_unit}") # (str) seconds
print(f"Dimension: {TwoPhotonSeries.dimension[:]}")
print(f"Format: {TwoPhotonSeries.format}") # (str) raw
print(f"Imaging plane description: {TwoPhotonSeries.imaging_plane.description}") # (str) standard
print(f"Excitation lambda: {TwoPhotonSeries.imaging_plane.excitation_lambda}") # (float64) 920.0
print(f"Imaging rate: {TwoPhotonSeries.imaging_plane.imaging_rate}") # (float64) 15.5078125
print(f"Indicator: {TwoPhotonSeries.imaging_plane.indicator}") # (str) GCaMP7f
print(f"Location: {TwoPhotonSeries.imaging_plane.location}") # (str) hippocampus, CA1
print(f"Conversion: {TwoPhotonSeries.imaging_plane.conversion}") # (float) 1.0
print(f"Unit: {TwoPhotonSeries.imaging_plane.unit}") # (str) meters
print(f"Origin coords unit: {TwoPhotonSeries.imaging_plane.origin_coords_unit}") # (str) meters
print(f"Grid spacing: {TwoPhotonSeries.imaging_plane.grid_spacing[:]}") # (Dataset) shape (2,); dtype float64
print(f"Grid spacing unit: {TwoPhotonSeries.imaging_plane.grid_spacing_unit}") # (str) microns
print(f"Device description: {TwoPhotonSeries.imaging_plane.device.description}") # (str) My two-photon microscope
print(f"Device manufacturer: {TwoPhotonSeries.imaging_plane.device.manufacturer}") # (str) Neurolabware
# %% [markdown]
# ### Processing: behavior
#
# This section describes the behavior data

# %%
processing = nwb.processing
behavior = processing["behavior"]
print(f"Behavior description: {behavior.description}") # (str) behavior data
data_interfaces = behavior.data_interfaces
BehavioralTimeSeries = data_interfaces["BehavioralTimeSeries"]
time_series = BehavioralTimeSeries.time_series
Reward = time_series["Reward"]
print(f"Reward resolution: {Reward.resolution}") # (float64) -1.0
print(f"Reward comments: {Reward.comments}") # (str) no comments
print(f"Reward description: {Reward.description}") # (str) reward delivery
print(f"Reward conversion: {Reward.conversion}") # (float64) 1.0
print(f"Reward offset: {Reward.offset}") # (float64) 0.0
print(f"Reward unit: {Reward.unit}") # (str) mL

autoreward = time_series["autoreward"]
print(f"Autoreward resolution: {autoreward.resolution}") # (float64) -1.0
print(f"Autoreward comments: {autoreward.comments}") # (str) no comments
print(f"Autoreward  description: {autoreward.description}") # (str) whether trial was automatically rewarded if the subject failed to lick
print(f"Autoreward conversion: {autoreward.conversion}") # (float64) 1.0
print(f"Autoreward offset: {autoreward.offset}") # (float64) 0.0
print(f"Autoreward unit: {autoreward.unit}") # (str) integer

environment = time_series["environment"]
print(f"Environment resolution: {environment.resolution}") # (float64) -1.0
print(f"Environment comments: {environment.comments}") # (str) no comments
print(f"Environment description: {environment.description}") # (str) Virtual reality environment
print(f"Environment conversion: {environment.conversion}") # (float64) 1.0
print(f"Environment offset: {environment.offset}") # (float64) 0.0
print(f"Environment unit: {environment.unit}") # (str) AU

lick = time_series["lick"]
print(f"Lick resolution: {lick.resolution}") # (float64) -1.0
print(f"Lick comments: {lick.comments}") # (str) no comments
print(f"Lick description: {lick.description}") # (str) lick detection by capacitive sensor, cumulative per imaging frame
print(f"Lick conversion: {lick.conversion}") # (float64) 1.0
print(f"Lick offset: {lick.offset}") # (float64) 0.0
print(f"Lick unit: {lick.unit}") # (str) AU

position = time_series["position"]
print(f"Position resolution: {position.resolution}") # (float64) -1.0
print(f"Position comments: {position.comments}") # (str) no comments
print(f"Position description: {position.description}") # (str) Position in a virtual linear track
print(f"Position conversion: {position.conversion}") # (float64) 1.0
print(f"Position offset: {position.offset}") # (float64) 0.0
print(f"Position unit: {position.unit}") # (str) cm

reward_zone = time_series["reward_zone"]
print(f"Reward zone resolution: {reward_zone.resolution}") # (float64) -1.0
print(f"Reward zone comments: {reward_zone.comments}") # (str) no comments
print(f"Reward zone description: {reward_zone.description}") # (str) reward zone entry (binary)
print(f"Reward zone conversion: {reward_zone.conversion}") # (float64) 1.0
print(f"Reward zone offset: {reward_zone.offset}") # (float64) 0.0
print(f"Reward zone unit: {reward_zone.unit}") # (str) integer

scanning = time_series["scanning"]
print(f"Scanning resolution: {scanning.resolution}") # (float64) -1.0
print(f"Scanning comments: {scanning.comments}") # (str) no comments
print(f"Scanning description: {scanning.description}") # (str) whether scanning occurred to collect ophys data
print(f"Scanning conversion: {scanning.conversion}") # (float64) 1.0
print(f"Scanning offset: {scanning.offset}") # (float64) 0.0
print(f"Scanning unit: {scanning.unit}") # (str) integer

speed = time_series["speed"]
print(f"Speed resolution: {speed.resolution}") # (float64) -1.0
print(f"Speed comments: {speed.comments}") # (str) no comments
print(f"Speed description: {speed.description}") # (str) the speed of the subject measured over time
print(f"Speed conversion: {speed.conversion}") # (float64) 1.0
print(f"Speed offset: {speed.offset}") # (float64) 0.0
print(f"Speed unit: {speed.unit}") # (str) cm/s

teleport = time_series["teleport"]
print(f"Teleport resolution: {teleport.resolution}") # (float64) -1.0
print(f"Teleport comments: {teleport.comments}") # (str) no comments
print(f"Teleport description: {teleport.description}") # (str) end of a trial, i.e. entry into the intertrial interval
print(f"Teleport conversion: {teleport.conversion}") # (float64) 1.0
print(f"Teleport offset: {teleport.offset}") # (float64) 0.0
print(f"Teleport unit: {teleport.unit}") # (str) integer

trial_number = time_series["trial number"]
print(f"Trial number resolution: {trial_number.resolution}") # (float64) -1.0
print(f"Trial number comments: {trial_number.comments}") # (str) no comments
print(f"Trial number description: {trial_number.description}") # (str) trial number, where each trial is a lap of the track
print(f"Trial number conversion: {trial_number.conversion}") # (float64) 1.0
print(f"Trial number offset: {trial_number.offset}") # (float64) 0.0
print(f"Trial number unit: {trial_number.unit}") # (str) integer

trial_start = time_series["trial_start"]
print(f"Trial start resolution: {trial_start.resolution}") # (float64) -1.0
print(f"Trial start comments: {trial_start.comments}") # (str) no comments
print(f"Trial start description: {trial_start.description}") # (str) start of a trial, i.e. entry to the linear track
print(f"Trial start conversion: {trial_start.conversion}") # (float64) 1.0
print(f"Trial start offset: {trial_start.offset}") # (float64) 0.0
print(f"Trial start unit: {trial_start.unit}") # (str) integer
# %% [markdown]
# ### Processing: ophys
#
# This section describes the ophys data

# %%
ophys = processing["ophys"]
print(f"Ophys description: {ophys.description}") # (str) optical physiology processed data
data_interfaces = ophys.data_interfaces
Backgrounds_0 = data_interfaces["Backgrounds_0"]
print(f"Backgrounds_0 description: {Backgrounds_0.description}") # (str) no description
images = Backgrounds_0.images
Vcorr = images["Vcorr"]
max_proj = images["max_proj"]
meanImg = images["meanImg"]

Deconvolved = data_interfaces["Deconvolved"]
roi_response_series = Deconvolved.roi_response_series
plane0 = roi_response_series["plane0"]
print(f"Plane0 starting time: {plane0.starting_time}") # (float64) 0.0
print(f"Plane0 rate: {plane0.rate}") # (float64) 15.5078125
print(f"Plane0 resolution: {plane0.resolution}") # (float64) -1.0
print(f"Plane0 comments: {plane0.comments}") # (str) no comments
print(f"Plane0 description: {plane0.description}") # (str) no description
print(f"Plane0 conversion: {plane0.conversion}") # (float64) 1.0
print(f"Plane0 offset: {plane0.offset}") # (float64) 0.0
print(f"Plane0 unit: {plane0.unit}") # (str) lumens

Fluorescence = data_interfaces["Fluorescence"]
roi_response_series = Fluorescence.roi_response_series
plane0 = roi_response_series["plane0"]
print(f"Plane0 starting time: {plane0.starting_time}") # (float64) 0.0
print(f"Plane0 rate: {plane0.rate}") # (float64) 15.5078125
print(f"Plane0 resolution: {plane0.resolution}") # (float64) -1.0
print(f"Plane0 comments: {plane0.comments}") # (str) no comments
print(f"Plane0 description: {plane0.description}") # (str) no description
print(f"Plane0 conversion: {plane0.conversion}") # (float64) 1.0
print(f"Plane0 offset: {plane0.offset}") # (float64) 0.0
print(f"Plane0 unit: {plane0.unit}") # (str) lumens

ImageSegmentation = data_interfaces["ImageSegmentation"]
plane_segmentations = ImageSegmentation.plane_segmentations
PlaneSegmentation = plane_segmentations["PlaneSegmentation"]
print(f"PlaneSegmentation description: {PlaneSegmentation.description}") # (str) suite2p output

Neuropil = data_interfaces["Neuropil"]
roi_response_series = Neuropil.roi_response_series
plane0 = roi_response_series["plane0"]
print(f"Plane0 starting time: {plane0.starting_time}") # (float64) 0.0
print(f"Plane0 rate: {plane0.rate}") # (float64) 15.5078125
print(f"Plane0 resolution: {plane0.resolution}") # (float64) -1.0
print(f"Plane0 comments: {plane0.comments}") # (str) no comments
print(f"Plane0 description: {plane0.description}") # (str) no description
print(f"Plane0 conversion: {plane0.conversion}") # (float64) 1.0
print(f"Plane0 offset: {plane0.offset}") # (float64) 0.0
print(f"Plane0 unit: {plane0.unit}") # (str) lumens

# %% [markdown]
# ### Devices & Imaging Planes
#

# %%
devices = nwb.devices
Microscope = devices["Microscope"]
print(f"Microscope description: {Microscope.description}") # (str) My two-photon microscope
print(f"Microscope manufacturer: {Microscope.manufacturer}") # (str) Neurolabware

imaging_planes = nwb.imaging_planes
ImagingPlane = imaging_planes["ImagingPlane"]
print(f"ImagingPlane description: {ImagingPlane.description}") # (str) standard
print(f"ImagingPlane excitation_lambda: {ImagingPlane.excitation_lambda}") # (float64) 920.0
print(f"ImagingPlane imaging_rate: {ImagingPlane.imaging_rate}") # (float64) 15.5078125
print(f"ImagingPlane indicator: {ImagingPlane.indicator}") # (str) GCaMP7f
print(f"ImagingPlane location: {ImagingPlane.location}") # (str) hippocampus, CA1
print(f"ImagingPlane conversion: {ImagingPlane.conversion}") # (float) 1.0
print(f"ImagingPlane unit: {ImagingPlane.unit}") # (str) meters
print(f"ImagingPlane origin_coords_unit: {ImagingPlane.origin_coords_unit}") # (str) meters
print(f"ImagingPlane grid_spacing: {ImagingPlane.grid_spacing[:]}")
print(f"ImagingPlane grid_spacing_unit: {ImagingPlane.grid_spacing_unit}") # (str) microns
print(f"ImagingPlane device description: {ImagingPlane.device.description}") # (str) My two-photon microscope
print(f"ImagingPlane device manufacturer: {ImagingPlane.device.manufacturer}") # (str) Neurolabware

# %% [markdown]
# ### Subject
#

# %%
print(f"Session id: {nwb.session_id}") # (str) 03
subject = nwb.subject
print(f"Subject age reference: {subject.age__reference}") # (str) birth
print(f"Subject sex: {subject.sex}") # (str) M
print(f"Subject species: {subject.species}") # (str) Mus musculus
print(f"Subject subject_id: {subject.subject_id}") # (str) m11
print(f"Subject date of birth: {subject.date_of_birth}") # (datetime) 2022-09-20T00:00:00-07:00

# %% [markdown]
# ## Loading and visualizing data
#

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# %% [markdown]
# ### Plotting position data
#

# %%
position_data = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"].data[:]
position_timestamps = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"].timestamps[:]

plt.figure(figsize=(10, 5))
plt.plot(position_timestamps, position_data)
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.title("Position vs. Time")
plt.show()

# %% [markdown]
# ### Plotting a subset of fluorescence data
#
# Here we plot a subset of the deconvolved fluorescence data for the first 5 cells.

# %%
deconvolved_data = nwb.processing["ophys"].data_interfaces["Deconvolved"].roi_response_series["plane0"].data[:]
num_cells = 5
plt.figure(figsize=(10, 5))
for i in range(num_cells):
    plt.plot(deconvolved_data[:, i], label=f"Cell {i}")
plt.xlabel("Time (frames)")
plt.ylabel("Deconvolved Fluorescence")
plt.title(f"Deconvolved Fluorescence for First {num_cells} Cells")
plt.legend()
plt.show()

# %% [markdown]
# ## Summary and future directions
#
# This notebook has provided a basic overview of how to load and explore data from Dandiset 001361. We have demonstrated how to:
#
# 1.  Connect to the DANDI archive and retrieve Dandiset metadata.
# 2.  Load an NWB file from the Dandiset.
# 3.  Explore the contents of the NWB file, including behavioral and optical physiology data.
# 4.  Visualize some of the data, such as position and deconvolved fluorescence.
#
# Possible future directions for analysis include:
#
# *   Performing more in-depth analysis of the behavioral data, such as identifying reward zones and analyzing the animal's behavior in relation to those zones.
# *   Analyzing the relationships between the behavioral and optical physiology data.
# *   Performing spike sorting on the raw electrophysiology data.
# *   Developing more sophisticated visualizations to better understand the data.