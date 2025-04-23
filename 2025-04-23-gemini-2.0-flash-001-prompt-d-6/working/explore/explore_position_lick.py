import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to explore the relationship between 'position' and 'lick' data.

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract position and lick data and timestamps
position_data = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"].data[:]
position_timestamps = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"].timestamps[:]
lick_data = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["lick"].data[:]
lick_timestamps = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["lick"].timestamps[:]

# Ensure both datasets have the same timestamps
# For simplicity, let's take the minimum length of both datasets
min_length = min(len(position_data), len(lick_data))
position_data = position_data[:min_length]
lick_data = lick_data[:min_length]
position_timestamps = position_timestamps[:min_length]
lick_timestamps = lick_timestamps[:min_length]

# Plot position vs lick
plt.figure(figsize=(10, 6))
plt.scatter(position_data, lick_data, alpha=0.5)
plt.xlabel("Position (cm)")
plt.ylabel("Lick (AU)")
plt.title("Position vs Lick")
plt.grid(True)
plt.savefig("explore/position_vs_lick.png")

print("Position vs lick data exploration complete. Plot saved to explore/position_vs_lick.png")