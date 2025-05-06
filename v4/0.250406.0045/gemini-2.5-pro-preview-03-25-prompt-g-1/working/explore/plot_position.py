# This script plots the mouse's position over time from the NWB file.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Add mode='r'
nwb = io.read()

# Get position data
position_timeseries = nwb.processing["behavior"]["BehavioralTimeSeries"].time_series["position"]
position_data = position_timeseries.data[:]
position_timestamps = position_timeseries.timestamps[:]

# Create plot
sns.set_theme()
plt.figure(figsize=(12, 6))
plt.plot(position_timestamps, position_data)
plt.xlabel("Time (s)")
plt.ylabel(f"Position ({position_timeseries.unit})")
plt.title("Mouse Position Over Time")
plt.savefig("explore/position_plot.png")
plt.close()

io.close()
print("Saved explore/position_plot.png")