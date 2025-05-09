# This script explores the position and speed behavioral data and plots them

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get behavioral data
behavior_ts = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"]
position = behavior_ts.time_series["position"]
speed = behavior_ts.time_series["speed"]

# Get data and timestamps (limit to first 10000 samples to avoid large downloads)
position_data = position.data[0:10000]
speed_data = speed.data[0:10000]
timestamps = position.timestamps[0:10000] # positions and speed share the same timestamps

# Create plots
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(timestamps, position_data)
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.title("Animal Position Over Time")

plt.subplot(2, 1, 2)
plt.plot(timestamps, speed_data)
plt.xlabel("Time (s)")
plt.ylabel("Speed (cm/s)")
plt.title("Animal Speed Over Time")

plt.tight_layout()
plt.savefig("explore/behavioral_data.png")
print("Behavioral data plot saved to explore/behavioral_data.png")

io.close()