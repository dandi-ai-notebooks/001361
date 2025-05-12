# This script explores the attributes of datasets in the NWB file

import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Access behavioral data
behavior_path = '/processing/behavior/BehavioralTimeSeries/'

# Function to print attributes of a dataset or group
def print_attributes(name):
    try:
        obj = h5_file[name]
        print(f"\nAttributes for {name}:")
        for attr_name, attr_value in obj.attrs.items():
            try:
                if isinstance(attr_value, bytes):
                    attr_value = attr_value.decode('utf-8')
                print(f"  {attr_name}: {attr_value}")
            except Exception as e:
                print(f"  {attr_name}: Error decoding: {e}")
    except Exception as e:
        print(f"Error accessing {name}: {e}")

# Print attributes for position
print_attributes(f'{behavior_path}position')
print_attributes(f'{behavior_path}position/data')

# Print attributes for speed
print_attributes(f'{behavior_path}speed')
print_attributes(f'{behavior_path}speed/data')

# Print attributes for reward
print_attributes(f'{behavior_path}Reward')
print_attributes(f'{behavior_path}Reward/data')

# Close the file
h5_file.close()