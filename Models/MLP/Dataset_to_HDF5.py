# %%
import os
import numpy as np
import h5py
from PIL import Image

npy_dir = 'C:/Users/20202555/OneDrive - TU Eindhoven/Documents/M AI&ES - Year 1/Team-internship/Experimental_Data/Test'
masks_dir =  npy_dir + '/Masks'
npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
masks_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
original_resolution = (640, 480)  # Set your original resolution, e.g., (64, 64)

hdf5_path = 'Datasets/test2%.h5'  # Set your output HDF5 file path

# %%
with h5py.File(hdf5_path, 'w') as h5f:
    num_frames = 1300
    sample_size = max(1, int(num_frames * 0.02))  # Sample 1% of frames
    sampled_indices = np.linspace(num_frames//3, num_frames - 1, sample_size, dtype=int)

    prefix_counts = {}
    for npy_file in npy_files:
        npy_path = os.path.join(npy_dir, npy_file)
        arr = np.load(npy_path)
        print(arr.shape)

        arr_sampled = arr[sampled_indices]
        
        arr_reshaped = arr_sampled.reshape(-1, original_resolution[1], original_resolution[0])
        
        
        prefix = npy_file.split('_')[0]

        # Recount idx for each prefix
        if prefix not in prefix_counts:
            prefix_counts[prefix] = 0
        idx = prefix_counts[prefix]
        prefix_counts[prefix] += 1

        dataset_name = f'video_{prefix}_{idx}'
        h5f.create_dataset(dataset_name, data=arr_reshaped, compression="gzip")
        print(f"Added {npy_file} as dataset {dataset_name} to HDF5.")

    for mask_file in masks_files:
        mask_path = os.path.join(masks_dir, mask_file)
        mask_image = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_image)
        mask_array = (mask_array > 0).astype(np.uint8)

        prefix = mask_file.split('.')[0]
        h5f.create_dataset(f'mask_{prefix}', data=mask_array, compression="gzip")
        print(f"Added {mask_file} as dataset mask_{prefix} to HDF5.")
      

print("conversion complete.")

# %%
