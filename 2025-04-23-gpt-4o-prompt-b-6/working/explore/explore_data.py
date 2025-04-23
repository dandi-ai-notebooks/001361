# This script explores two-photon imaging data and behavioral data from a specified NWB file.
# It generates plots to visualize aspects of the TwoPhotonSeries and BehavioralTimeSeries

import matplotlib.pyplot as plt
import h5py
import remfile
import numpy as np
import seaborn as sns
import pynwb

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/a22cc1da-b5e8-4fea-a770-7b83a6e79656/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Function to plot and save data
def plot_and_save(data, title, ylabel, filename):
    sns.set_theme()
    plt.figure(figsize=(14, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

# Explore and visualize the TwoPhotonSeries data
tp_series = nwb.acquisition["TwoPhotonSeries"]
tp_series_data = tp_series.data[:, :, 0]  # First imaging plane
plot_and_save(tp_series_data.flatten(), "TwoPhotonSeries Imaging Data", "Voltage (V)", "explore/two_photon_imaging.png")

# Explore and visualize the Reward TimeSeries data
reward_data = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["Reward"].data[:]
plot_and_save(reward_data, "Reward TimeSeries Data", "Reward (mL)", "explore/reward_timeseries.png")

# Explore and visualize the Position TimeSeries data
position_data = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"].data[:]
plot_and_save(position_data, "Position TimeSeries Data", "Position (cm)", "explore/position_timeseries.png")

# Explore and visualize the Speed TimeSeries data
speed_data = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["speed"].data[:]
plot_and_save(speed_data, "Speed TimeSeries Data", "Speed (cm/s)", "explore/speed_timeseries.png")

io.close()