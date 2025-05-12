# This script explores the structure of the NWB file to find the correct paths to data

import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Function to recursively print the structure of an H5 group with limits on recursion
def print_h5_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        try:
            shape = obj.shape
            dtype = obj.dtype
            print(f"Dataset: {name}, Shape: {shape}, Dtype: {dtype}")
        except Exception as e:
            print(f"Dataset: {name}, Error: {e}")

# Print the processing/behavior structure
print("Exploring processing/behavior structure:")
if 'processing' in h5_file and 'behavior' in h5_file['processing']:
    h5_file['processing/behavior'].visititems(print_h5_structure)

# Look specifically for time series data
print("\nExploring the behavior/BehavioralTimeSeries structure:")
if 'processing' in h5_file and 'behavior' in h5_file['processing'] and 'BehavioralTimeSeries' in h5_file['processing/behavior']:
    h5_file['processing/behavior/BehavioralTimeSeries'].visititems(print_h5_structure)

# Check if there's a time_series group
if 'processing/behavior/BehavioralTimeSeries/time_series' in h5_file:
    print("\nTime series in BehavioralTimeSeries:")
    for key in h5_file['processing/behavior/BehavioralTimeSeries/time_series']:
        print(f"- {key}")
        # Print subgroups/datasets
        for subkey in h5_file[f'processing/behavior/BehavioralTimeSeries/time_series/{key}']:
            try:
                item = h5_file[f'processing/behavior/BehavioralTimeSeries/time_series/{key}/{subkey}']
                if isinstance(item, h5py.Dataset):
                    shape = item.shape
                    dtype = item.dtype
                    print(f"  - {subkey}: Shape {shape}, Dtype: {dtype}")
                else:
                    print(f"  - {subkey} (Group)")
            except Exception as e:
                print(f"  - {subkey}, Error: {e}")
else:
    print("Could not find 'time_series' in the BehavioralTimeSeries group")
    # Let's check what's directly in the BehavioralTimeSeries group
    if 'processing/behavior/BehavioralTimeSeries' in h5_file:
        print("Direct children of BehavioralTimeSeries:")
        for key in h5_file['processing/behavior/BehavioralTimeSeries'].keys():
            print(f"- {key}")
    else:
        print("Could not find the BehavioralTimeSeries group")

# Try to find trial data
print("\nSearching for trial number data...")
for key in h5_file['/processing/behavior/BehavioralTimeSeries']:
    if 'trial' in key.lower():
        print(f"Found: {key}")
        try:
            if key in h5_file['/processing/behavior/BehavioralTimeSeries']:
                print(f"  Children of {key}:")
                for subkey in h5_file[f'/processing/behavior/BehavioralTimeSeries/{key}']:
                    print(f"  - {subkey}")
        except Exception as e:
            print(f"  Error exploring {key}: {e}")

# Close the file
h5_file.close()