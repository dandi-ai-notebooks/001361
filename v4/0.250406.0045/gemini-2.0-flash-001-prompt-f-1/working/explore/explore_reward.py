import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# This script explores the reward data in the NWB file.

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the position and reward data
position = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['position']
position_data = position.data[:]
position_timestamps = position.timestamps[:]
reward = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['Reward']
reward_data = reward.data[:]
reward_timestamps = reward.timestamps[:]

# Plot the position and reward data
plt.figure(figsize=(10, 5))
plt.plot(position_timestamps, position_data, label='Position')
plt.vlines(reward_timestamps, ymin=min(position_data), ymax=max(position_data), color='red', label='Reward')
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.title('Position and Reward vs. Time')
plt.legend()
plt.savefig('explore/reward.png')
plt.close()

# Print some summary statistics
print(f"Number of reward deliveries: {len(reward_data)}")