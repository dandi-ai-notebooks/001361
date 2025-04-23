import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to load the NWB file and plot the 'speed' data against time

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract speed and timestamps
speed_data = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["speed"].data[:]
speed_timestamps = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["speed"].timestamps[:]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(speed_timestamps, speed_data)
plt.xlabel("Time (s)")
plt.ylabel("Speed (cm/s)")
plt.title("Speed vs. Time")
plt.savefig("explore/speed_vs_time.png")
plt.close()