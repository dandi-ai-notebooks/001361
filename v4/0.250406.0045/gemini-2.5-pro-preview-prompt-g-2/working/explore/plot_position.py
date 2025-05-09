# This script plots the mouse's position over time from the NWB file.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
with pynwb.NWBHDF5IO(file=h5_file, mode='r') as io:
    nwb = io.read()

    position_data = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['position'].data[:]
    position_timestamps = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['position'].timestamps[:]

    plt.figure(figsize=(12, 6))
    plt.plot(position_timestamps, position_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (cm)')
    plt.title('Mouse Position Over Time')
    plt.savefig('explore/position_over_time.png')
    plt.close()

print("Saved explore/position_over_time.png")