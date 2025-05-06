import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# This script explores the position data in the NWB file.

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the position data
position = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['position']
position_data = position.data[:]
position_timestamps = position.timestamps[:]

# Plot the position data
plt.figure(figsize=(10, 5))
plt.plot(position_timestamps, position_data)
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.title('Position vs. Time')
plt.savefig('explore/position.png')
plt.close()

# Print some summary statistics
print(f"Number of position samples: {len(position_data)}")
print(f"Position data mean: {np.mean(position_data)}")
print(f"Position data std: {np.std(position_data)}")