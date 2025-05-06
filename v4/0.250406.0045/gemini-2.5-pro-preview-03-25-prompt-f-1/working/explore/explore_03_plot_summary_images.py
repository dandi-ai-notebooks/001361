# explore_03_plot_summary_images.py
# This script loads and displays summary images from the ophys module,
# specifically the mean fluorescence image and max projection image.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Not using seaborn for image plots

print("Attempting to load NWB file for summary image plots...")

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
        backgrounds_interface = ophys_module.data_interfaces.get('Backgrounds_0')
        if backgrounds_interface is None or not hasattr(backgrounds_interface, 'images'):
            print("Backgrounds_0 interface or its images not found.")
        else:
            images_dict = backgrounds_interface.images

            mean_img = images_dict.get('meanImg')
            if mean_img is not None:
                print(f"Mean image data shape: {mean_img.data.shape}")
                plt.figure(figsize=(8, 8))
                plt.imshow(mean_img.data[:], cmap='gray') # Use [:] to load data
                plt.title("Mean Fluorescence Image")
                plt.colorbar(label="Fluorescence intensity (a.u.)")
                plt.axis('off')
                plt.savefig("explore/mean_fluorescence_image.png")
                plt.close()
                print("Saved mean_fluorescence_image.png")
            else:
                print("meanImg not found in Backgrounds_0 images.")

            max_proj_img = images_dict.get('max_proj')
            if max_proj_img is not None:
                print(f"Max projection image data shape: {max_proj_img.data.shape}")
                plt.figure(figsize=(8, 8))
                plt.imshow(max_proj_img.data[:], cmap='gray') # Use [:] to load data
                plt.title("Max Projection Image")
                plt.colorbar(label="Max fluorescence intensity (a.u.)")
                plt.axis('off')
                plt.savefig("explore/max_projection_image.png")
                plt.close()
                print("Saved max_projection_image.png")
            else:
                print("max_proj not found in Backgrounds_0 images.")
            
            vcorr_img = images_dict.get('Vcorr')
            if vcorr_img is not None:
                print(f"Correlation image (Vcorr) data shape: {vcorr_img.data.shape}")
                # This image might be useful as well, will plot it.
                plt.figure(figsize=(8, 8))
                plt.imshow(vcorr_img.data[:], cmap='viridis') # Use [:] to load data
                plt.title("Pixel Correlation Image (Vcorr)")
                plt.colorbar(label="Correlation value")
                plt.axis('off')
                plt.savefig("explore/vcorr_image.png")
                plt.close()
                print("Saved vcorr_image.png")
            else:
                print("Vcorr image not found in Backgrounds_0 images.")


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

print("Summary image plotting script finished.")