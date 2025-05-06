# explore_02_plot_behavior.py
# This script loads behavioral data (position and speed) from the NWB file
# and generates plots.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Attempting to load NWB file for behavior plots...")

# Ensure the explore directory exists for saving plots
import os
if not os.path.exists('explore'):
    os.makedirs('explore')

io = None
remote_file = None
try:
    # Load NWB file
    url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
    print(f"Using URL: {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    behavior_module = nwb.processing.get('behavior')
    if behavior_module is None:
        print("Behavior processing module not found.")
    else:
        behavioral_ts_interface = behavior_module.data_interfaces.get('BehavioralTimeSeries')
        if behavioral_ts_interface is None:
            print("BehavioralTimeSeries interface not found.")
        else:
            # Set seaborn style
            sns.set_theme()

            # Plot position over time
            position_ts = behavioral_ts_interface.time_series.get('position')
            if position_ts is not None:
                print(f"Position data shape: {position_ts.data.shape}")
                print(f"Position timestamps shape: {position_ts.timestamps.shape}")
                
                # Load a subset of data if it's too large for quick plotting
                num_points_to_plot = min(len(position_ts.data), 5000) # Plot at most 5000 points
                indices = np.linspace(0, len(position_ts.data) - 1, num_points_to_plot, dtype=int)
                
                position_data = position_ts.data[indices]
                position_timestamps = position_ts.timestamps[indices]

                plt.figure(figsize=(12, 6))
                plt.plot(position_timestamps, position_data)
                plt.xlabel("Time (s)")
                plt.ylabel(f"Position ({position_ts.unit})")
                plt.title("Mouse Position Over Time (Subset)")
                plt.savefig("explore/position_over_time.png")
                plt.close() # Close the figure to free memory
                print("Saved position_over_time.png")
            else:
                print("Position time series not found.")

            # Plot speed over time
            speed_ts = behavioral_ts_interface.time_series.get('speed')
            if speed_ts is not None:
                print(f"Speed data shape: {speed_ts.data.shape}")
                print(f"Speed timestamps shape: {speed_ts.timestamps.shape}")

                num_points_to_plot = min(len(speed_ts.data), 5000) # Plot at most 5000 points
                indices = np.linspace(0, len(speed_ts.data) - 1, num_points_to_plot, dtype=int)

                speed_data = speed_ts.data[indices]
                speed_timestamps = speed_ts.timestamps[indices]
                
                plt.figure(figsize=(12, 6))
                plt.plot(speed_timestamps, speed_data)
                plt.xlabel("Time (s)")
                plt.ylabel(f"Speed ({speed_ts.unit})")
                plt.title("Mouse Speed Over Time (Subset)")
                plt.savefig("explore/speed_over_time.png")
                plt.close() # Close the figure to free memory
                print("Saved speed_over_time.png")
            else:
                print("Speed time series not found.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if io is not None:
        print("Closing NWB file.")
        io.close()
    if remote_file is not None:
        remote_file.close()

print("Behavior plotting script finished.")