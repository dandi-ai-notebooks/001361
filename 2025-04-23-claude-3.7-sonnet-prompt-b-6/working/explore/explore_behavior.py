"""
This script explores the behavioral data in the NWB file.
We'll look at position, speed, reward, and trial information to understand 
the behavioral components of the experiment.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb

# Set up matplotlib defaults
plt.rcParams['figure.figsize'] = (12, 8)

# Load the remote NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("Loaded NWB file:", nwb.identifier)
print("Session:", nwb.session_id)
print("Subject:", nwb.subject.subject_id)
print("Species:", nwb.subject.species)
print("Sex:", nwb.subject.sex)

# Access behavioral data
behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"]

# Get a list of all behavioral timeseries
time_series_names = list(behavior.time_series.keys())
print("\nAvailable behavioral time series:")
for name in time_series_names:
    print(f"- {name}")

# Let's extract a subset of timestamps and position data to avoid loading too much data at once
# We'll use a slice of 5000 data points, which should be enough to see patterns
start_idx = 0
num_samples = 5000
position_data = behavior.time_series["position"].data[start_idx:start_idx+num_samples]
position_timestamps = behavior.time_series["position"].timestamps[start_idx:start_idx+num_samples]
speed_data = behavior.time_series["speed"].data[start_idx:start_idx+num_samples]
reward_zone_data = behavior.time_series["reward_zone"].data[start_idx:start_idx+num_samples]
trial_number_data = behavior.time_series["trial number"].data[start_idx:start_idx+num_samples]
trial_start_data = behavior.time_series["trial_start"].data[start_idx:start_idx+num_samples]
teleport_data = behavior.time_series["teleport"].data[start_idx:start_idx+num_samples]

# Create a figure with multiple subplots
fig, axes = plt.subplots(4, 1, sharex=True)

# Position plot
axes[0].plot(position_timestamps, position_data, 'b-')
axes[0].set_ylabel('Position (cm)')
axes[0].set_title('Animal Position Over Time')

# Speed plot
axes[1].plot(position_timestamps, speed_data, 'g-')
axes[1].set_ylabel('Speed (cm/s)')
axes[1].set_title('Animal Speed')

# Reward zone entry
axes[2].plot(position_timestamps, reward_zone_data, 'r-')
axes[2].set_ylabel('Reward Zone')
axes[2].set_title('Reward Zone Entry (Binary)')

# Trial number
axes[3].plot(position_timestamps, trial_number_data, 'k-')
axes[3].set_ylabel('Trial Number')
axes[3].set_xlabel('Time (s)')
axes[3].set_title('Trial Number Over Time')

plt.tight_layout()
plt.savefig("explore/behavior_timeseries.png")
plt.close()

# Let's also look at the relationship between position and trial starts
fig, ax = plt.subplots(1, 1)
# Plot position data
ax.plot(position_timestamps, position_data, 'b-', alpha=0.5)
# Overlay trial start events
trial_start_events = np.where(trial_start_data > 0)[0]
if len(trial_start_events) > 0:
    ax.scatter(position_timestamps[trial_start_events], position_data[trial_start_events], 
               color='green', marker='^', s=100, label='Trial Start')

# Overlay teleport events
teleport_events = np.where(teleport_data > 0)[0]
if len(teleport_events) > 0:
    ax.scatter(position_timestamps[teleport_events], position_data[teleport_events], 
               color='red', marker='v', s=100, label='Teleport/Trial End')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (cm)')
ax.set_title('Position with Trial Start and End Events')
ax.legend()
plt.tight_layout()
plt.savefig("explore/trial_events.png")
plt.close()

# Let's also look at a simple histogram of positions to see where the animal spends most time
fig, ax = plt.subplots(1, 1)
ax.hist(position_data, bins=50, alpha=0.7)
ax.set_xlabel('Position (cm)')
ax.set_ylabel('Count')
ax.set_title('Histogram of Animal Positions')
plt.tight_layout()
plt.savefig("explore/position_histogram.png")
plt.close()

# Extract reward information
# Reward events have their own time series with fewer data points
reward_data = behavior.time_series["Reward"].data[:]
reward_timestamps = behavior.time_series["Reward"].timestamps[:]

print(f"\nNumber of reward events: {len(reward_data)}")
print(f"First 5 reward timestamps: {reward_timestamps[:5]}")

# Let's summarize reward locations by finding the position at each reward time
reward_positions = []

# This is a naive approach and might not be exact, but gives an approximation of reward positions
for reward_time in reward_timestamps:
    # Find the closest timestamp in the position data
    idx = np.abs(position_timestamps - reward_time).argmin()
    if idx < len(position_data):
        reward_positions.append(position_data[idx])

# Plot histogram of reward positions if we found any
if reward_positions:
    fig, ax = plt.subplots(1, 1)
    ax.hist(reward_positions, bins=20, alpha=0.7)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Reward Positions')
    plt.tight_layout()
    plt.savefig("explore/reward_position_histogram.png")
    plt.close()