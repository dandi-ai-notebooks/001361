# This script explores the attributes of the neural data in the NWB file

import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

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

# Print attributes for Fluorescence and related groups/datasets
print_attributes('/processing/ophys/Fluorescence')
print_attributes('/processing/ophys/Fluorescence/plane0')
print_attributes('/processing/ophys/Fluorescence/plane0/data')
print_attributes('/processing/ophys/Fluorescence/plane0/starting_time')

# Look for rate or imaging_rate in other places
print("\nChecking for imaging rate information elsewhere:")
print_attributes('/acquisition/TwoPhotonSeries')
print_attributes('/processing/ophys/ImageSegmentation/PlaneSegmentation/imaging_plane')

# Close the file
h5_file.close()