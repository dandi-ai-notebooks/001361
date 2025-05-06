"""
Script to explore behavioral time series data from the NWB file for sub-m11_ses-03_behavior+ophys.nwb.
Generates plots for position and speed over time, and marks reward events.
"""

import os
import matplotlib.pyplot as plt
import remfile
import h5py
import pynwb

# Ensure output directory exists
output_dir = os.path.dirname(__file__)
os.makedirs(output_dir, exist_ok=True)

# Load remote NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract behavioral time series
behavior_ts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries']
position = behavior_ts.time_series['position']
speed = behavior_ts.time_series['speed']
reward = behavior_ts.time_series['Reward']

# Convert to numpy arrays
pos_t = position.timestamps[:]
pos_data = position.data[:]
speed_t = speed.timestamps[:]
speed_data = speed.data[:]
reward_t = reward.timestamps[:]
reward_data = reward.data[:]

# Plot position vs time
plt.figure(figsize=(10, 4))
plt.plot(pos_t, pos_data, label='Position (cm)')
if len(reward_t) > 0:
    plt.scatter(reward_t, [max(pos_data)] * len(reward_t), color='red', marker='v', label='Reward')
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.title('Position over Time with Reward Events')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'position_vs_time.png'))
plt.close()

# Plot speed vs time
plt.figure(figsize=(10, 4))
plt.plot(speed_t, speed_data, label='Speed (cm/s)')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Speed over Time')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'speed_vs_time.png'))
plt.close()