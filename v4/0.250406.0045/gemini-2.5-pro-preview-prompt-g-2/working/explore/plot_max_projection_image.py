# This script plots the maximum projection image from the ophys data.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
with pynwb.NWBHDF5IO(file=h5_file, mode='r') as io:
    nwb = io.read()

    max_proj_image = nwb.processing['ophys'].data_interfaces['Backgrounds_0'].images['max_proj'].data[:]

    plt.figure(figsize=(8, 8))
    plt.imshow(max_proj_image, cmap='gray')
    plt.title('Maximum Intensity Projection Image')
    plt.axis('off') # Turn off axis numbers and ticks
    plt.savefig('explore/max_projection_image.png')
    plt.close()

print("Saved explore/max_projection_image.png")