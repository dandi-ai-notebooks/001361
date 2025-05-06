import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# This script explores the lick data in the NWB file.

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the lick data
lick = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["lick"]
lick_data = lick.data[:]
lick_timestamps = lick.timestamps[:]

# Plot the lick data
plt.figure(figsize=(10, 5))
plt.plot(lick_timestamps, lick_data)
plt.xlabel("Time (s)")
plt.ylabel("Lick (AU)")
plt.title("Lick over Time")
plt.savefig("explore/lick_over_time.png")