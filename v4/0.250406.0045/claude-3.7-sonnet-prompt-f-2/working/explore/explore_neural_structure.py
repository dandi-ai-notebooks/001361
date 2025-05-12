# This script explores the structure of the neural data in the NWB file

import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Function to recursively print the structure of a group
def print_group_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        try:
            shape = obj.shape
            dtype = obj.dtype
            print(f"Dataset: {name}, Shape: {shape}, Dtype: {dtype}")
        except Exception as e:
            print(f"Dataset: {name}, Error: {e}")

# Explore the structure of the ophys processing module
ophys_path = '/processing/ophys'
print(f"Exploring structure of {ophys_path}:")
h5_file[ophys_path].visititems(print_group_structure)

# Explore specifically the Fluorescence data
print("\nExploring Fluorescence data structure:")
if 'Fluorescence' in h5_file[ophys_path]:
    h5_file[f'{ophys_path}/Fluorescence'].visititems(print_group_structure)
    
    # See what's in roi_response_series if it exists
    if 'roi_response_series' in h5_file[f'{ophys_path}/Fluorescence']:
        print("\nROI response series keys:")
        for key in h5_file[f'{ophys_path}/Fluorescence/roi_response_series']:
            print(f"  - {key}")
            
        # If plane0 exists, explore it
        if 'plane0' in h5_file[f'{ophys_path}/Fluorescence/roi_response_series']:
            print("\nExploring plane0:")
            h5_file[f'{ophys_path}/Fluorescence/roi_response_series/plane0'].visititems(print_group_structure)
else:
    print("Fluorescence not found in ophys processing module")

# Explore ImageSegmentation to find ROI information
print("\nExploring ImageSegmentation data structure:")
if 'ImageSegmentation' in h5_file[ophys_path]:
    h5_file[f'{ophys_path}/ImageSegmentation'].visititems(print_group_structure)
else:
    print("ImageSegmentation not found in ophys processing module")

# Close the file
h5_file.close()