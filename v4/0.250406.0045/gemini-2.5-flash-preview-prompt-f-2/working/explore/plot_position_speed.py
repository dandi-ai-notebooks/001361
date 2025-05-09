# This script loads and plots the position and speed from the NWB file.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt

# Load
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get position and speed time series
position_ts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['position']
speed_ts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['speed']

# Load data
# Load a subset of data to avoid excessive memory usage for plotting
subset_size = 10000
position_data = position_ts.data[:subset_size]
speed_data = speed_ts.data[:subset_size]
timestamps = position_ts.timestamps[:subset_size]

# Create plot
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (cm)', color=color)
ax1.plot(timestamps, position_data, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Speed (cm/s)', color=color)  # we already handled the x-label with ax1
ax2.plot(timestamps, speed_data, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Position and Speed over Time (Subset)')
plt.savefig('explore/position_speed_plot.png')
plt.close(fig)

io.close()