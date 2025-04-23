import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to explore the 'Reward' data in the NWB file.

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract reward data and timestamps
reward_data = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["Reward"].data[:]
reward_timestamps = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["Reward"].timestamps[:]

# Plot the reward data
plt.figure(figsize=(10, 6))
plt.plot(reward_timestamps, reward_data, marker='o', linestyle='-', markersize=4)
plt.xlabel("Time (s)")
plt.ylabel("Reward (mL)")
plt.title("Reward over Time")
plt.grid(True)
plt.savefig("explore/reward_over_time.png")

print("Reward data exploration complete. Plot saved to explore/reward_over_time.png")