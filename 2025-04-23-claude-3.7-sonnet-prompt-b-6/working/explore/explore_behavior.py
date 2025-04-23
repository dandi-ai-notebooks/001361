"""
Explore behavioral data from the NWB file.
This script extracts and visualizes behavioral data from the Dandiset,
including position, speed, licks, rewards, and trial structure.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for plots if it doesn't exist
os.makedirs('explore', exist_ok=True)

# Set up matplotlib style
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Subject: {nwb.subject.subject_id}")
print(f"Session: {nwb.session_id}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

# Extract behavioral data - use a subset to reduce data size
# Get time data for a subset of the data (first 5000 samples)
sample_size = 5000
timestamps = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"].timestamps[:sample_size]

# Extract position data
position = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["position"].data[:sample_size]

# Extract speed data
speed = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["speed"].data[:sample_size]

# Extract lick data
lick = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["lick"].data[:sample_size]

# Extract reward zone data
reward_zone = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["reward_zone"].data[:sample_size]

# Extract trial number data
trial_num = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["trial number"].data[:sample_size]

# Extract trial start and teleport data
trial_start = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["trial_start"].data[:sample_size]
teleport = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["teleport"].data[:sample_size]

# Get reward timestamps
reward_timestamps = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["Reward"].timestamps[:]
reward_data = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series["Reward"].data[:]

print(f"Position data shape: {position.shape}")
print(f"Speed data shape: {speed.shape}")
print(f"Lick data shape: {lick.shape}")
print(f"Reward zone data shape: {reward_zone.shape}")
print(f"Trial number data shape: {trial_num.shape}")
print(f"Number of rewards: {len(reward_timestamps)}")

# Create plots
# Fig 1: Position vs Time
plt.figure()
plt.plot(timestamps, position)
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.title('Mouse Position in Virtual Track')
plt.grid(True)
plt.savefig('explore/position_vs_time.png', dpi=150)
plt.close()

# Fig 2: Speed vs Time
plt.figure()
plt.plot(timestamps, speed)
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Mouse Speed')
plt.grid(True)
plt.savefig('explore/speed_vs_time.png', dpi=150)
plt.close()

# Fig 3: Lick and Reward Zone
plt.figure()
plt.plot(timestamps, lick, 'b', label='Lick')
plt.plot(timestamps, reward_zone, 'r', label='Reward Zone')
# Mark rewards on the plot
reward_in_range = [t for t in reward_timestamps if t <= timestamps[-1]]
if reward_in_range:
    plt.vlines(reward_in_range, 0, max(lick), 'g', label='Reward')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Licking Behavior and Reward Zone')
plt.legend()
plt.grid(True)
plt.savefig('explore/lick_reward.png', dpi=150)
plt.close()

# Fig 4: Trial structure
plt.figure()
plt.plot(timestamps, trial_num, label='Trial Number')
plt.plot(timestamps, trial_start, 'r', label='Trial Start')
plt.plot(timestamps, teleport, 'g', label='Teleport')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Trial Structure')
plt.legend()
plt.grid(True)
plt.savefig('explore/trial_structure.png', dpi=150)
plt.close()

# Fig 5: Position vs Trial Number
# Create a 2D histogram of position vs trial number
plt.figure()
trial_nums = np.unique(trial_num)
valid_trials = trial_nums[~np.isnan(trial_nums)]
if len(valid_trials) > 1:  # Ensure we have at least 2 trials
    # Create bins for position and trial number
    pos_bins = np.linspace(min(position), max(position), 50)
    trial_bins = np.linspace(min(valid_trials), max(valid_trials), len(valid_trials)+1)
    
    plt.hist2d(position, trial_num, bins=[pos_bins, trial_bins], cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel('Position (cm)')
    plt.ylabel('Trial Number')
    plt.title('Position vs Trial Number')
    plt.savefig('explore/position_vs_trial.png', dpi=150)
    plt.close()

print("Plots saved in the 'explore' directory.")