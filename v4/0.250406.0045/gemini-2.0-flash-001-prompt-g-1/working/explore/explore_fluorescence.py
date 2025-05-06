import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# This script explores the Fluorescence data in the NWB file.

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the Fluorescence data
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["plane0"]
fluorescence_data = fluorescence.data[:]
if fluorescence.timestamps is not None:
    fluorescence_timestamps = fluorescence.timestamps[:]
else:
    starting_time = fluorescence.starting_time
    rate = fluorescence.rate
    num_frames = fluorescence_data.shape[0]
    fluorescence_timestamps = np.arange(num_frames) / rate + starting_time

# Plot the Fluorescence data for the first 5 ROIs
num_rois = min(5, fluorescence_data.shape[1])
plt.figure(figsize=(10, 5))
for i in range(num_rois):
    plt.plot(fluorescence_timestamps, fluorescence_data[:, i], label=f"ROI {i}")

plt.xlabel("Time (s)")
plt.ylabel("Fluorescence (lumens)")
plt.title("Fluorescence over Time (First 5 ROIs)")
plt.legend()
plt.savefig("explore/fluorescence_over_time.png")