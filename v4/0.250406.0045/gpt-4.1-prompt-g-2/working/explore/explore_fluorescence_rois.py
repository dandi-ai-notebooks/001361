# This script samples example ROI fluorescence traces and overlays ROI masks on the mean image.
# It saves two plots: individual ROI traces over time, and all ROI masks overlaid on the mean projection.
# This helps illustrate both cell activity and geometric segmentation.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Load fluorescence traces
plane = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["plane0"]
fluorescence = np.array(plane.data)
num_cells = fluorescence.shape[1]
num_frames = fluorescence.shape[0]

# Select 6 random cells to show
random.seed(0)
if num_cells >= 6:
    cell_indices = random.sample(range(num_cells), 6)
else:
    cell_indices = list(range(num_cells))

plt.figure(figsize=(10, 6))
for idx in cell_indices:
    plt.plot(np.arange(num_frames), fluorescence[:, idx], label=f'Cell {idx}')
plt.xlabel("Frame")
plt.ylabel("ΔF/F (arbitrary units)")
plt.title("Example ROI Fluorescence Traces (ΔF/F)")
plt.legend()
plt.tight_layout()
plt.savefig("explore/roi_traces.png")
plt.close()

# Plotting spatial masks over mean image
plane_seg = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
meanImg = np.array(nwb.processing["ophys"].data_interfaces["Backgrounds_0"].images["meanImg"].data)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(meanImg, cmap="gray")
for ii in range(len(plane_seg.id)):
    # pixel_mask: list of (y, x, weight) for this ROI
    pmask = plane_seg["pixel_mask"][ii]
    for y, x, v in pmask:
        ax.plot(x, y, 'r.', markersize=1, alpha=0.03)
ax.set_title("ROI Masks Overlaid on Mean Image")
ax.axis("off")
plt.savefig("explore/roi_masks_on_mean.png", bbox_inches='tight')
plt.close()