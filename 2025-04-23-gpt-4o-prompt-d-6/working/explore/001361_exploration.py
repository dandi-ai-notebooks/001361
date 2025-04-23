# Script for exploring Dandiset 001361 NWB file focusing on behavior and optical physiology data
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from remfile import File
import seaborn as sns
from pynwb import NWBHDF5IO

# Set the theme for seaborn
sns.set_theme()

# Define the URL and load the NWB file
url = 'https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/'
remote_file = File(url)
h5_file = h5py.File(remote_file)
io = NWBHDF5IO(file=h5_file)
nwb = io.read()

# Create explore directory if it does not exist
if not os.path.exists('explore'):
    os.makedirs('explore')

# Check the shape of TwoPhotonSeries data
series_data = nwb.acquisition['TwoPhotonSeries'].data
if len(series_data.shape) > 1 and series_data.shape[0] > 1:
    # Assuming the first dimension is time if multiple data points exist
    series_data = series_data[:, 0, 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(0, len(series_data)), series_data, label='TwoPhotonSeries')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Amplitude')
    ax.set_title('TwoPhotonSeries Amplitude Over Time')
    plt.legend()
    plt.savefig('explore/two_photon_series.png')
    plt.close(fig)

# Plot and save Behavior data: Speed over time
behavior_data = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['speed']
speed_data = behavior_data.data[:]
speed_timestamps = behavior_data.timestamps[:]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(speed_timestamps, speed_data, label='Speed')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (cm/s)')
ax.set_title('Speed Over Time')
plt.legend()
plt.savefig('explore/behavior_speed.png')
plt.close(fig)