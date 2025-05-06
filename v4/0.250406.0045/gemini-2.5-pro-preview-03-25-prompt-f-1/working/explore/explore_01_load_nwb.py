# explore_01_load_nwb.py
# This script loads the NWB file and prints some basic metadata
# to confirm that the file can be accessed and to get an overview.

import pynwb
import h5py
import remfile
import numpy as np

print("Attempting to load NWB file...")

try:
    # Load NWB file
    url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
    print(f"Using URL: {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r') # Ensure read-only mode explicitly
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Ensure read-only mode explicitly
    nwb = io.read()

    print("NWB file loaded successfully.")
    print(f"Identifier: {nwb.identifier}")
    print(f"Session description: {nwb.session_description}")
    print(f"Session start time: {nwb.session_start_time}")
    print(f"Experimenter: {nwb.experimenter}")
    print(f"Subject ID: {nwb.subject.subject_id}")

    print("\nProcessing modules available:")
    for name in nwb.processing:
        print(f"- {name}")

    if 'ophys' in nwb.processing:
        ophys_module = nwb.processing['ophys']
        print("\nData interfaces in 'ophys' module:")
        for name in ophys_module.data_interfaces:
            print(f"- {name}")
            if name == 'Fluorescence':
                fluorescence_data = ophys_module.data_interfaces['Fluorescence']
                if 'plane0' in fluorescence_data.roi_response_series:
                    plane0_fluor = fluorescence_data.roi_response_series['plane0']
                    print(f"  - Fluorescence/plane0 data shape: {plane0_fluor.data.shape}")
                    print(f"  - Fluorescence/plane0 timestamps shape: Not directly available, uses rate: {plane0_fluor.rate} Hz")

            if name == 'Deconvolved':
                deconvolved_data = ophys_module.data_interfaces['Deconvolved']
                if 'plane0' in deconvolved_data.roi_response_series:
                    plane0_deconv = deconvolved_data.roi_response_series['plane0']
                    print(f"  - Deconvolved/plane0 data shape: {plane0_deconv.data.shape}")
                    print(f"  - Deconvolved/plane0 timestamps shape: Not directly available, uses rate: {plane0_deconv.rate} Hz")

            if name == 'ImageSegmentation':
                image_seg = ophys_module.data_interfaces['ImageSegmentation']
                if 'PlaneSegmentation' in image_seg.plane_segmentations:
                    plane_seg = image_seg.plane_segmentations['PlaneSegmentation']
                    print(f"  - ImageSegmentation/PlaneSegmentation table has {len(plane_seg.id)} ROIs.")
                    print(f"  - Imaging plane location: {plane_seg.imaging_plane.location}")


    if 'behavior' in nwb.processing:
        behavior_module = nwb.processing['behavior']
        print("\nData interfaces in 'behavior' module:")
        for name in behavior_module.data_interfaces:
            print(f"- {name}")
            if name == 'BehavioralTimeSeries':
                behavioral_ts = behavior_module.data_interfaces['BehavioralTimeSeries']
                print("  Time series in 'BehavioralTimeSeries':")
                for ts_name in behavioral_ts.time_series:
                    ts = behavioral_ts.time_series[ts_name]
                    print(f"  - {ts_name}: shape {ts.data.shape}, timestamps shape {ts.timestamps.shape if hasattr(ts, 'timestamps') and ts.timestamps is not None else 'N/A (uses rate)'}")
    
    print("\nChecking acquisition data (TwoPhotonSeries):")
    if "TwoPhotonSeries" in nwb.acquisition:
        tps = nwb.acquisition["TwoPhotonSeries"]
        print(f"TwoPhotonSeries data shape: {tps.data.shape}") # This was (1,1,1) in tools_cli output
        print(f"TwoPhotonSeries dimension: {tps.dimension[:] if hasattr(tps, 'dimension') and tps.dimension is not None else 'N/A'}")
        print(f"TwoPhotonSeries imaging_plane name: {tps.imaging_plane.name}")
        print(f"TwoPhotonSeries imaging_plane description: {tps.imaging_plane.description}")
        print(f"TwoPhotonSeries imaging_plane location: {tps.imaging_plane.location}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'io' in locals() and io is not None:
        print("Closing NWB file.")
        io.close()
    if 'remote_file' in locals() and remote_file is not None:
        remote_file.close()

print("Script finished.")