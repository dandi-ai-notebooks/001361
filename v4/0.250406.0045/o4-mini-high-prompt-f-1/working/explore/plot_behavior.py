#!/usr/bin/env python3
"""
Script to plot behavior data (speed and position) for the NWB file.
Generates two PNG files in explore/: behavior_speed.png and behavior_position.png.
"""
import os
import h5py
import remfile
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import seaborn as sns

sns.set_theme()

# URL for the selected NWB asset
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"

# Open remote NWB file and load via h5py
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access behavioral time series
bt = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries']
ts = bt.time_series
speed = ts['speed']
position = ts['position']

# Subset first N samples to limit memory and plotting time
n_samples = 2000
speed_t = speed.timestamps[:n_samples]
speed_d = speed.data[:n_samples]
position_t = position.timestamps[:n_samples]
position_d = position.data[:n_samples]

# Ensure output directory exists
os.makedirs('explore', exist_ok=True)

# Speed plot
plt.figure(figsize=(10, 4))
plt.plot(speed_t, speed_d)
plt.xlabel('Time (s)')
plt.ylabel(f"Speed ({speed.unit})")
plt.title('Speed over time (first 2000 samples)')
plt.tight_layout()
plt.savefig('explore/behavior_speed.png')
plt.close()

# Position plot
plt.figure(figsize=(10, 4))
plt.plot(position_t, position_d)
plt.xlabel('Time (s)')
plt.ylabel(f"Position ({position.unit})")
plt.title('Position over time (first 2000 samples)')
plt.tight_layout()
plt.savefig('explore/behavior_position.png')
plt.close()