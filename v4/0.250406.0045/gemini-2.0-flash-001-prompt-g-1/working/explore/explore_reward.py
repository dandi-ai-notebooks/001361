import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# This script explores the Reward data in the NWB file.

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the Reward data
reward = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["Reward"]
reward_data = reward.data[:]
reward_timestamps = reward.timestamps[:]

# Plot the Reward data
plt.figure(figsize=(10, 5))
plt.plot(reward_timestamps, reward_data)
plt.xlabel("Time (s)")
plt.ylabel("Reward (mL)")
plt.title("Reward over Time")
plt.savefig("explore/reward_over_time.png")