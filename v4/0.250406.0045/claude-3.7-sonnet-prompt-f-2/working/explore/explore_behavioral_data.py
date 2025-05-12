# This script explores behavioral data in the NWB file to understand the experimental task
# Looking at position, speed, reward zones, and trial structure

import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Access behavioral data
print("Accessing behavioral data...")
behavior_path = '/processing/behavior/BehavioralTimeSeries/'

# Get position, speed, and reward data
# Only get a subset of the data to keep the script efficient
sample_size = 10000  # Number of samples to analyze

# Get trial number data
trial_numbers_data = h5_file[f'{behavior_path}trial number/data'][:sample_size]
trial_numbers_timestamps = h5_file[f'{behavior_path}trial number/timestamps'][:sample_size]

# Get position data
position_data = h5_file[f'{behavior_path}position/data'][:sample_size]
position_timestamps = h5_file[f'{behavior_path}position/timestamps'][:sample_size]
position_unit = h5_file[f'{behavior_path}position/data'].attrs['unit']

# Get speed data
speed_data = h5_file[f'{behavior_path}speed/data'][:sample_size]
speed_timestamps = h5_file[f'{behavior_path}speed/timestamps'][:sample_size]
speed_unit = h5_file[f'{behavior_path}speed/data'].attrs['unit']

# Get reward zone data
reward_zone_data = h5_file[f'{behavior_path}reward_zone/data'][:sample_size]
reward_zone_timestamps = h5_file[f'{behavior_path}reward_zone/timestamps'][:sample_size]

# Get rewards data
# These might have a different number of timestamps than the other data
rewards_data = h5_file[f'{behavior_path}Reward/data'][:]
rewards_timestamps = h5_file[f'{behavior_path}Reward/timestamps'][:]
rewards_unit = h5_file[f'{behavior_path}Reward/data'].attrs['unit']

# Print basic information about the data
print(f"Position data shape: {position_data.shape}, unit: {position_unit}")
print(f"Speed data shape: {speed_data.shape}, unit: {speed_unit}")
print(f"Reward zone data shape: {reward_zone_data.shape}")
print(f"Rewards data shape: {rewards_data.shape}, unit: {rewards_unit}")
print(f"Trial numbers shape: {trial_numbers_data.shape}")

# Print the range of values for position and speed
print(f"Position range: {np.min(position_data)} to {np.max(position_data)} {position_unit}")
print(f"Speed range: {np.min(speed_data)} to {np.max(speed_data)} {speed_unit}")

# Print how many unique trial numbers we see
unique_trials = np.unique(trial_numbers_data)
print(f"Unique trial numbers: {unique_trials}")

# Plot position over time
plt.figure(figsize=(10, 6))
plt.plot(position_timestamps, position_data)
plt.xlabel('Time (s)')
plt.ylabel(f'Position ({position_unit})')
plt.title('Position over Time')
plt.savefig(os.path.join('explore', 'position_over_time.png'))
plt.close()

# Plot speed over time
plt.figure(figsize=(10, 6))
plt.plot(speed_timestamps, speed_data)
plt.xlabel('Time (s)')
plt.ylabel(f'Speed ({speed_unit})')
plt.title('Speed over Time')
plt.savefig(os.path.join('explore', 'speed_over_time.png'))
plt.close()

# Plot reward zones and rewards
plt.figure(figsize=(10, 6))
plt.plot(reward_zone_timestamps, reward_zone_data, label='Reward Zone')
plt.scatter(rewards_timestamps, np.ones_like(rewards_timestamps), color='red', label='Reward Delivery', marker='v', s=100)
plt.xlabel('Time (s)')
plt.ylabel('Reward Zone (binary)')
plt.title('Reward Zones and Rewards')
plt.legend()
plt.savefig(os.path.join('explore', 'reward_zones.png'))
plt.close()

# Plot position vs trial number as a 2D histogram 
plt.figure(figsize=(10, 6))
plt.hist2d(position_data, trial_numbers_data, bins=(100, max(2, len(unique_trials))), cmap='viridis')
plt.colorbar(label='Count')
plt.xlabel(f'Position ({position_unit})')
plt.ylabel('Trial Number')
plt.title('Position Distribution by Trial')
plt.savefig(os.path.join('explore', 'position_by_trial.png'))
plt.close()

# Plot a few example trials
example_trials = unique_trials[:min(3, len(unique_trials))]
plt.figure(figsize=(12, 6))
for trial in example_trials:
    trial_mask = (trial_numbers_data == trial)
    plt.plot(position_timestamps[trial_mask] - position_timestamps[trial_mask][0], 
             position_data[trial_mask], 
             label=f'Trial {int(trial)}')

plt.xlabel('Time from Trial Start (s)')
plt.ylabel(f'Position ({position_unit})')
plt.title('Position Trajectories for Example Trials')
plt.legend()
plt.savefig(os.path.join('explore', 'example_trials.png'))
plt.close()

print("Analysis complete. Plots saved to the 'explore' directory.")