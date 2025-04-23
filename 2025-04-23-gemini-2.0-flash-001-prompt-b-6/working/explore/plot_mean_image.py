import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Set seaborn theme for non-image plots
sns.set_theme()

# Script to load the NWB file and plot the mean image

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract mean image
mean_image = nwb.processing["ophys"].data_interfaces["Backgrounds_0"].images["meanImg"].data[:]

# Create the plot
plt.figure(figsize=(6, 6))
plt.imshow(mean_image, cmap="gray")
plt.colorbar(label="Intensity")
plt.title("Mean Image")
plt.savefig("explore/mean_image.png")
plt.close()