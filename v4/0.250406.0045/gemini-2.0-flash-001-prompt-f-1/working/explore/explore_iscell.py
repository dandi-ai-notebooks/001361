import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This script explores the iscell data in the NWB file.

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the iscell data
plane_segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
iscell_data = plane_segmentation.iscell[:]
iscell_df = pd.DataFrame(iscell_data, columns=['iscell', 'probcell'])

# Print some summary statistics
print(f"Number of ROIs: {len(iscell_df)}")
print(f"Number of cells (iscell > 0.5): {len(iscell_df[iscell_df['iscell'] > 0.5])}")
print(f"Number of non-cells (iscell <= 0.5): {len(iscell_df[iscell_df['iscell'] <= 0.5])}")

# Plot the distribution of iscell values
plt.figure(figsize=(8, 6))
plt.hist(iscell_df['iscell'], bins=20)
plt.xlabel('iscell value')
plt.ylabel('Number of ROIs')
plt.title('Distribution of iscell values')
plt.savefig('explore/iscell_distribution.png')
plt.close()