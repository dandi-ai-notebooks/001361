# This script loads the mean and max projection images from the NWB file at the provided remote URL
# and saves them as PNGs to the explore/ directory.
# These summary images represent the average and maximum fluorescence across frames, providing insight into image quality and field of view.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# URL to remote NWB file (keep hard-coded for reproducibility)
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

ophys = nwb.processing["ophys"]
Backgrounds_0 = ophys.data_interfaces["Backgrounds_0"]
images = Backgrounds_0.images

# Plot and save mean projection image
meanImg = np.array(images["meanImg"].data)
plt.figure(figsize=(8, 6))
plt.imshow(meanImg, cmap="gray")
plt.axis('off')
plt.title("Mean Fluorescence Projection")
plt.savefig("explore/mean_projection.png", bbox_inches='tight')
plt.close()

# Plot and save max projection image
max_proj = np.array(images["max_proj"].data)
plt.figure(figsize=(8, 6))
plt.imshow(max_proj, cmap="gray")
plt.axis('off')
plt.title("Max Fluorescence Projection")
plt.savefig("explore/max_projection.png", bbox_inches='tight')
plt.close()