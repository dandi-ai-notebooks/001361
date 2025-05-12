"""
This script explores the behavioral data in the NWB file, focusing on variables 
like position, speed, reward timing, and trial information. It examines the structure
and content of the behavior data and creates visualizations to better understand
the experimental paradigm.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Configure seaborn
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get behavioral data
behavior = nwb.processing["behavior"]
behavioral_ts = behavior.data_interfaces["BehavioralTimeSeries"]

# Print available behavioral metrics
print("Available behavioral metrics:")
for name in behavioral_ts.time_series:
    ts = behavioral_ts.time_series[name]
    print(f"  - {name}: {ts.description} (unit: {ts.unit}, shape: {ts.data.shape})")

# Sample a subset of data for analysis (first 5000 time points)
sample_size = 5000
timestamps = behavioral_ts.time_series["position"].timestamps[:sample_size]
position = behavioral_ts.time_series["position"].data[:sample_size]
speed = behavioral_ts.time_series["speed"].data[:sample_size]
reward_zone = behavioral_ts.time_series["reward_zone"].data[:sample_size]
trial_number = behavioral_ts.time_series["trial number"].data[:sample_size]
trial_start = behavioral_ts.time_series["trial_start"].data[:sample_size]
teleport = behavioral_ts.time_series["teleport"].data[:sample_size]

# Get reward timestamps (these have a different shape)
reward_ts = behavioral_ts.time_series["Reward"].timestamps[:]
reward_values = behavioral_ts.time_series["Reward"].data[:]

# Create a figure for behavioral data visualization
plt.figure(figsize=(12, 10))
gs = GridSpec(4, 1, figure=plt.gcf(), height_ratios=[2, 1, 1, 1])

# Plot position over time
ax1 = plt.subplot(gs[0])
ax1.plot(timestamps, position, 'b-', alpha=0.7)
ax1.set_title('Position vs Time')
ax1.set_ylabel('Position (cm)')
ax1.set_xlabel('Time (s)')

# Mark trial starts, teleports, and reward zones
trial_start_times = timestamps[np.where(trial_start > 0)[0]]
teleport_times = timestamps[np.where(teleport > 0)[0]]
reward_zone_times = timestamps[np.where(reward_zone > 0)[0]]

for t in trial_start_times:
    ax1.axvline(x=t, color='g', linestyle='--', alpha=0.3)
for t in teleport_times:
    ax1.axvline(x=t, color='r', linestyle='--', alpha=0.3)
for t in reward_zone_times:
    ax1.axvline(x=t, color='m', linestyle=':', alpha=0.3)

# Add reward delivery times as vertical lines
for r in reward_ts:
    if r <= timestamps[-1]:  # Only include rewards within our sample time range
        ax1.axvline(x=r, color='gold', linestyle='-', linewidth=2, alpha=0.7)

# Plot speed over time
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(timestamps, speed, 'g-')
ax2.set_title('Speed vs Time')
ax2.set_ylabel('Speed (cm/s)')
ax2.set_xlabel('Time (s)')

# Plot trial number over time
ax3 = plt.subplot(gs[2], sharex=ax1)
ax3.plot(timestamps, trial_number, 'r-')
ax3.set_title('Trial Number vs Time')
ax3.set_ylabel('Trial Number')
ax3.set_xlabel('Time (s)')

# Plot reward zone entry over time
ax4 = plt.subplot(gs[3], sharex=ax1)
ax4.plot(timestamps, reward_zone, 'k-')
ax4.set_title('Reward Zone Entry vs Time')
ax4.set_ylabel('In Reward Zone')
ax4.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('explore/behavior_overview.png')

# Create a figure to visualize position distribution
plt.figure(figsize=(10, 6))
plt.hist(position, bins=50, alpha=0.7)
plt.title('Position Distribution')
plt.xlabel('Position (cm)')
plt.ylabel('Count')
plt.savefig('explore/position_distribution.png')

# Create a figure to visualize speed distribution
plt.figure(figsize=(10, 6))
plt.hist(speed, bins=50, alpha=0.7)
plt.title('Speed Distribution')
plt.xlabel('Speed (cm/s)')
plt.ylabel('Count')
plt.savefig('explore/speed_distribution.png')

# Print summary statistics
print("\nSummary statistics:")
print(f"Time period: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s ({timestamps[-1] - timestamps[0]:.2f}s)")
print(f"Number of trials in sample: {int(max(trial_number) - min(trial_number) + 1)}")
print(f"Position range: {min(position):.2f}cm to {max(position):.2f}cm")
print(f"Average speed: {np.mean(speed):.2f}cm/s")
print(f"Number of reward deliveries in sample: {len([r for r in reward_ts if r <= timestamps[-1]])}")