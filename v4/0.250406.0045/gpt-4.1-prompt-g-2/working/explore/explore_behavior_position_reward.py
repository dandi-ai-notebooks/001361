# This script visualizes the position of the mouse on the virtual track and overlays reward events over time.
# The y-axis is position (cm), the x-axis is time (seconds).
# Red dots show reward delivery.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Position time series
position = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"]
pos_data = np.array(position.data[:])
pos_times = np.array(position.timestamps[:])

# Reward events
reward = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["Reward"]
reward_times = np.array(reward.timestamps[:])
reward_amount = np.array(reward.data[:])

plt.figure(figsize=(12, 6))
plt.plot(pos_times, pos_data, label='Position (cm)', color='blue', lw=1)
plt.scatter(reward_times, np.interp(reward_times, pos_times, pos_data),
                marker='o', color='red', s=40, label="Reward")
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.title("Mouse Position on Track and Reward Delivery")
plt.legend()
plt.tight_layout()
plt.savefig("explore/behavior_position_reward.png")
plt.close()