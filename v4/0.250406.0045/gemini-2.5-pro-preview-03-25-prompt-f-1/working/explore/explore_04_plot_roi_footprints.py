# explore_04_plot_roi_footprints.py
# This script loads ROI pixel masks from ImageSegmentation and plots them.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Not using seaborn for image plots

print("Attempting to load NWB file for ROI footprint plot...")

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

    ophys_module = nwb.processing.get('ophys')
    if ophys_module is None:
        print("Ophys processing module not found.")
    else:
        image_seg_interface = ophys_module.data_interfaces.get('ImageSegmentation')
        if image_seg_interface is None:
            print("ImageSegmentation interface not found.")
        else:
            plane_seg = image_seg_interface.plane_segmentations.get('PlaneSegmentation')
            if plane_seg is None:
                print("PlaneSegmentation not found in ImageSegmentation.")
            else:
                print(f"Found PlaneSegmentation with {len(plane_seg.id)} ROIs.")
                imaging_plane = plane_seg.imaging_plane
                # From tools_cli: TwoPhotonSeries.dimension: [512 796]
                # This should be the dimensions of the imaging plane
                plane_dimensions = nwb.acquisition['TwoPhotonSeries'].dimension[:] # (height, width)
                
                if plane_dimensions is None or len(plane_dimensions) != 2:
                    # Fallback if not available directly from TwoPhotonSeries.dimension
                    # Try to get from a summary image if available
                    backgrounds_interface = ophys_module.data_interfaces.get('Backgrounds_0')
                    if backgrounds_interface and 'meanImg' in backgrounds_interface.images:
                        plane_dimensions = backgrounds_interface.images['meanImg'].data.shape
                        print(f"Inferred plane dimensions from meanImg: {plane_dimensions}")
                    else:
                        print("Could not determine imaging plane dimensions. Using default 512x512.")
                        plane_dimensions = (512, 512) # A common default, but might be incorrect

                # Create an empty image to store the composite mask
                # The pixel_mask stores (x, y, weight) tuples. Coordinates are typically pixel indices.
                # Max x and y from pixel_mask can confirm dimensions, but usually they are 0-indexed
                # within the imaging plane dimensions.
                
                composite_mask_image = np.zeros(plane_dimensions, dtype=float) # Use float for weights

                num_rois = len(plane_seg.id)
                for i in range(num_rois):
                    pixel_mask_data = plane_seg['pixel_mask'][i] # This is a list of (y, x, weight) tuples
                    for y, x, weight in pixel_mask_data:
                        if 0 <= y < plane_dimensions[0] and 0 <= x < plane_dimensions[1]:
                             # Suite2p pixel_mask can have (pixel_idx, weight). Needs reshaping or careful indexing.
                             # The tools_cli output shows pixel_mask_index, suggesting it's complex.
                             # pynwb's PlaneSegmentation pixel_mask is a list of (x, y, weight) tuples.
                             # Here, assuming (y, x, weight) based on common image indexing (row, col)
                             # and typical pynwb access.
                             # For pixel_mask data structure as list of (WeightedPixel):
                             # each WeightedPixel is (y_coord, x_coord, weight)
                             # Let's assume (y, x, weight) is correct based on typical pynwb.
                             # If weight is relative, it might be used directly. If it's binary mask, any non-zero is 1.
                             # The user prompt says "image masks values range from 0 to 1".
                             # So, we should use the weight.
                            composite_mask_image[int(y), int(x)] = max(composite_mask_image[int(y), int(x)], weight)

                if num_rois > 0:
                    plt.figure(figsize=(10, 10 * plane_dimensions[0]/plane_dimensions[1] if plane_dimensions[1] > 0 else 10))
                    # Using np.max on the image masks means we take the max weight if pixels overlap.
                    plt.imshow(composite_mask_image, cmap='hot', interpolation='nearest', vmin=0, vmax=1) # Assuming weights 0-1
                    plt.title(f"All ROI Footprints ({num_rois} ROIs)")
                    plt.colorbar(label="Max ROI weight")
                    plt.axis('off')
                    plt.savefig("explore/roi_footprints.png")
                    plt.close()
                    print("Saved roi_footprints.png")
                else:
                    print("No ROIs found to plot.")

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

print("ROI footprint plotting script finished.")