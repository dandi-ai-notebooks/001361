'''
Explore behavioral data from Dandiset 001361.
This script will look at position, speed, reward zones, and lick behavior
to understand the experimental paradigm.
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb

# Set up the plot style
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the dataset
print(f"Subject: {nwb.subject.subject_id}, Session: {nwb.session_id}")
print(f"Experiment: {nwb.identifier}")
print(f"Start time: {nwb.session_start_time}")
print(f"Location: {nwb.imaging_planes['ImagingPlane'].location}")
print(f"Indicator: {nwb.imaging_planes['ImagingPlane'].indicator}")
print(f"Imaging rate: {nwb.imaging_planes['ImagingPlane'].imaging_rate} Hz")

# Access behavioral data
behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"]
position = behavior.time_series["position"]
speed = behavior.time_series["speed"]
lick = behavior.time_series["lick"]
reward_zone = behavior.time_series["reward_zone"]
reward = behavior.time_series["Reward"]
trial_number = behavior.time_series["trial number"]
teleport = behavior.time_series["teleport"]

# Get data and timestamps for behavior
position_data = position.data[:]
position_timestamps = position.timestamps[:]
speed_data = speed.data[:]
lick_data = lick.data[:]
reward_zone_data = reward_zone.data[:]
trial_number_data = trial_number.data[:]
teleport_data = teleport.data[:]

# Get reward timestamps
reward_timestamps = reward.timestamps[:]

print(f"\nBehavioral data shape: {position_data.shape}")
print(f"Number of reward events: {len(reward_timestamps)}")
print(f"Total recording duration: {position_timestamps[-1] - position_timestamps[0]:.2f} seconds")

# Create plots for behavioral data
# For clarity, we'll just plot a subset of the data (first minute)
time_limit = 60  # First minute
idx = np.where(position_timestamps < position_timestamps[0] + time_limit)[0]

# Plot 1: Position over time
plt.figure(figsize=(10, 6))
plt.plot(position_timestamps[idx] - position_timestamps[0], position_data[idx])
# Mark reward zone entries
reward_zone_entries = np.where(np.diff(reward_zone_data[idx].astype(int)) == 1)[0]
if len(reward_zone_entries) > 0:
    plt.scatter(position_timestamps[idx][reward_zone_entries] - position_timestamps[0], 
                position_data[idx][reward_zone_entries], 
                color='red', s=50, zorder=3, label='Reward zone entry')
# Mark rewards
reward_in_window = [r for r in reward_timestamps if r < position_timestamps[0] + time_limit]
if len(reward_in_window) > 0:
    reward_indices = [np.argmin(np.abs(position_timestamps - r)) for r in reward_in_window]
    plt.scatter([position_timestamps[i] - position_timestamps[0] for i in reward_indices], 
                [position_data[i] for i in reward_indices], 
                color='green', s=80, marker='*', zorder=4, label='Reward delivery')
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.title('Position in Virtual Linear Track')
plt.legend()
plt.savefig('explore/position_time.png')

# Plot 2: Speed over time
plt.figure(figsize=(10, 4))
plt.plot(position_timestamps[idx] - position_timestamps[0], speed_data[idx])
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Running Speed')
plt.savefig('explore/speed_time.png')

# Plot 3: Trial structure (looking at teleports which indicate trial ends)
teleport_events = np.where(np.diff(teleport_data.astype(int)) == 1)[0]
trial_starts = np.where(np.diff(trial_number_data) != 0)[0]

print(f"\nNumber of trials (teleport events): {len(teleport_events)}")
print(f"Number of trial starts (based on trial number changes): {len(trial_starts)}")

# Plot all trials in a raster format
# Get all trials and their durations
trial_ids = np.unique(trial_number_data)
print(f"Trial IDs: {trial_ids}")

# For each trial, extract position vs time
plt.figure(figsize=(12, 8))
for trial in trial_ids:
    if trial == 0:  # Skip zero which might be pre-task
        continue
    trial_indices = np.where(trial_number_data == trial)[0]
    if len(trial_indices) == 0:
        continue
    trial_time = position_timestamps[trial_indices] - position_timestamps[trial_indices[0]]
    trial_pos = position_data[trial_indices]
    plt.plot(trial_time, trial_pos, alpha=0.5, linewidth=1)

plt.xlabel('Time in Trial (s)')
plt.ylabel('Position (cm)')
plt.title(f'Position Profiles for All Trials (n={len(trial_ids)-1})')
plt.savefig('explore/all_trials_position.png')

# Plot 4: Position distribution and reward zone
plt.figure(figsize=(10, 4))
plt.hist(position_data, bins=100, alpha=0.7)
# Mark where rewards occur
reward_positions = []
for r_time in reward_timestamps:
    idx = np.argmin(np.abs(position_timestamps - r_time))
    reward_positions.append(position_data[idx])
plt.axvline(np.mean(reward_positions), color='r', linestyle='--', 
            label=f'Mean Reward Position: {np.mean(reward_positions):.1f} cm')
plt.xlabel('Position (cm)')
plt.ylabel('Count')
plt.title('Position Distribution')
plt.legend()
plt.savefig('explore/position_distribution.png')

# Plot 5: Lick behavior relative to position
plt.figure(figsize=(10, 4))
# Bin positions and get average lick rate in each bin
bins = np.linspace(np.min(position_data), np.max(position_data), 100)
bin_indices = np.digitize(position_data, bins)
bin_lick_rates = [np.mean(lick_data[bin_indices == i]) for i in range(1, len(bins))]

plt.bar(bins[:-1], bin_lick_rates, width=np.diff(bins)[0], alpha=0.7)
plt.axvline(np.mean(reward_positions), color='r', linestyle='--', 
            label=f'Mean Reward Position: {np.mean(reward_positions):.1f} cm')
plt.xlabel('Position (cm)')
plt.ylabel('Average Lick Rate')
plt.title('Lick Rate vs. Position')
plt.legend()
plt.savefig('explore/lick_vs_position.png')

print("Behavioral exploration complete!")