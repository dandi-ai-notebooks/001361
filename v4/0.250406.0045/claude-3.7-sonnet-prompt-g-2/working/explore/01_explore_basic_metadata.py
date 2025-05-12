"""
This script loads the NWB file and explores the basic metadata to understand
the structure of the data, including session information, subject details,
and the types of data available.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure seaborn
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information
print("=== Basic NWB Information ===")
print(f"Session ID: {nwb.session_id}")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"File Create Date: {nwb.file_create_date}")
print(f"Experimenter: {nwb.experimenter}")
print("\n=== Subject Information ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Date of Birth: {nwb.subject.date_of_birth}")
print(f"Age Reference: {nwb.subject.age__reference}")

# Print structure information
print("\n=== NWB File Structure ===")
print("Acquisition:")
for key in nwb.acquisition:
    print(f"  - {key}")

print("\nProcessing Modules:")
for module_name in nwb.processing:
    module = nwb.processing[module_name]
    print(f"  - {module_name} ({module.description})")
    for interface_name in module.data_interfaces:
        print(f"    - {interface_name}")

# Print device information
print("\n=== Devices ===")
for device_name, device in nwb.devices.items():
    print(f"  - {device_name}: {device.description} (Manufacturer: {device.manufacturer})")

# Print imaging plane information
print("\n=== Imaging Planes ===")
for plane_name, plane in nwb.imaging_planes.items():
    print(f"  - {plane_name}")
    print(f"    - Description: {plane.description}")
    print(f"    - Location: {plane.location}")
    print(f"    - Indicator: {plane.indicator}")
    print(f"    - Excitation Lambda: {plane.excitation_lambda}")
    print(f"    - Imaging Rate: {plane.imaging_rate}")
    print(f"    - Grid Spacing: {plane.grid_spacing[:]}")