"""
Summary information for NWB file sub-m11_ses-03_behavior+ophys.nwb
Fetches basic session metadata, explores key behavioral time series (position, speed, reward, etc.), and reports overall shape of ROI/fluorescence data.
This script is a precursor to plotting and deep dives.
"""

import pynwb
import h5py
import remfile
import numpy as np

# -- NWB file parameters --
nwb_url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"

print("# Loading remote NWB file:", nwb_url)
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# --- Session metadata ---
print(f"Session ID: {nwb.session_id}")
print(f"Subject ID: {nwb.subject.subject_id}, Age: {getattr(nwb.subject, 'age__reference', 'n/a')}, Sex: {nwb.subject.sex}, Species: {nwb.subject.species}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Experimenter: {nwb.experimenter}")
print("-"*40)

# --- BehavioralTimeSeries (behavior) ---
behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
for k in behavior:
    ts = behavior[k]
    print(f"Behavioral Timeseries '{k}': {ts.description}; Unit: {ts.unit}; Data shape: {ts.data.shape}")
    # summary statistics for data
    try:
        arr = ts.data[:]
        min_v = np.nanmin(arr)
        max_v = np.nanmax(arr)
        mean_v = np.nanmean(arr)
        print(f"  Min: {min_v:.4f}, Max: {max_v:.4f}, Mean: {mean_v:.4f}")
    except Exception as e:
        print(f"  Could not compute stats for '{k}': {e}")
    # summary stats for timestamps
    try:
        tarr = ts.timestamps[:]
        print(f"  Timestamps: start={tarr[:1]}, end={tarr[-1:]}, count={tarr.size}")
    except Exception as e:
        print(f"  Could not access timestamps for '{k}': {e}")
    print("-"*30)

# --- Fluorescence data ---
try:
    fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["plane0"]
    print(f"Fluorescence 'plane0' data shape: {fluorescence.data.shape}, unit: {fluorescence.unit}")
    print(f"ROI count: {fluorescence.data.shape[1]}, Frames: {fluorescence.data.shape[0]}")
except Exception as e:
    print("Could not load fluorescence data:", e)

# --- ROI summary (PlaneSegmentation dynamic table) ---
try:
    ps_table = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["plane0"].rois.table
    df_head = ps_table.to_dataframe().head()
    print("ROI Segmentation table (first few rows):")
    print(df_head)
except Exception as e:
    print("Could not load ROI segmentation table:", e)

io.close()