# This script loads an NWB file and plots the animal's speed over time.

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

    # Get speed data and timestamps
    speed_data = nwb.processing['behavior']['BehavioralTimeSeries']['speed'].data[:]
    speed_timestamps = nwb.processing['behavior']['BehavioralTimeSeries']['speed'].timestamps[:]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(speed_timestamps, speed_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (cm/s)")
    plt.title("Mouse Speed Over Time")
    plt.grid(True)
    plt.savefig("explore/speed_over_time.png")
    plt.close()

print("Saved plot to explore/speed_over_time.png")