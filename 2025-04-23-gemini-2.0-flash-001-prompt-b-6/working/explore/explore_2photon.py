import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to load the NWB file and explore TwoPhotonSeries data

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the TwoPhotonSeries data
two_photon_series = nwb.acquisition["TwoPhotonSeries"]

# Get some metadata
rate = two_photon_series.rate
unit = two_photon_series.unit
description = two_photon_series.description
dimensions = two_photon_series.dimension[:]

print(f"Rate: {rate}")
print(f"Unit: {unit}")
print(f"Description: {description}")
print(f"Dimensions: {dimensions}")

# Load a small subset of the data
subset_size = 10
data_subset = two_photon_series.data[:subset_size, :, :]

# Print the shape of the subset
print(f"Shape of data subset: {data_subset.shape}")