import os
import torch
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from PIL import Image
from preprocessing import *

class ThermalDataset(Dataset):
    def __init__(self, file_paths, data_dir, mask_map, num_pixels=5000,
                 add_zero_padding=False, apply_fft=False, apply_wiener=False,
                 cutoff_frequency=1):
        """
        file_paths: list of .npy paths
        data_dir: directory where data is stored
        num_pixels: number of pixels to sample per video
        """
        self.file_paths = file_paths
        self.data_dir = data_dir
        self.num_pixels = num_pixels
        self.add_zero_padding = add_zero_padding
        self.apply_fft = apply_fft
        self.apply_wiener = apply_wiener
        self.cutoff_frequency = cutoff_frequency

        # mapping of masks to corresponding files
        self.mask_mapping = mask_map

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        
        # load data from file  
        file_path = os.path.join(self.data_dir, self.file_paths[idx])
        data = np.load(file_path).astype(np.float32)
        print(f"loaded {self.file_paths[idx]}")
        print("data shape: ", data.shape)

        # load mask (labels)
        file_name = os.path.basename(file_path)
        if "Composite plate" in file_path:
            mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["composite"])
        elif "Resin plates" in file_path:
            if "circular" in file_name:
                mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["circular"])
            elif "rec" in file_name:
                mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["rec"])
            elif "square" in file_name:
                mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["square"])
            elif "tri" in file_name:
                mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["tri"])
        else:
            raise ValueError(f"Unknown file path: {file_name}")
        # load mask (labels) as a PNG image
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = np.array(mask).astype(np.float32)  # Convert to numpy array
        print(f"Loaded mask with shape: {mask.shape}")

        mask = mask.reshape(480, 640)
        mask = np.where(mask > 0, 0, 1) # invert (0 for defect, 1 for no defect)
        
        # plot mask
        plt.imshow(mask, cmap='gray')
        plt.colorbar()
        plt.title('Defect Mask')
        plt.show()
       
        # randomly sample pixels
        pixel_ids = np.random.choice(307200, self.num_pixels, replace=False)
        data = data[:, pixel_ids]
        mask = mask.flatten()[pixel_ids]

        # center the data
        data -= np.mean(data, axis=0)

        # add zero padding
        if self.add_zero_padding:
            data = add_zero_padding(data, 2048 - data.shape[0])
            print("zero padded data shape: ", data.shape)

        # apply fft
        if self.apply_fft:
            t = np.linspace(0.5, data.shape[0]*0.5, data.shape[0])
            fft_data, freq = apply_fft(data, t)

            # apply low-pass filter to remove high frequencies
            cutoff_index = np.where(freq > self.cutoff_frequency)[0][0] 
            fft_data = fft_data[:cutoff_index]
            freq = freq[:cutoff_index]
            print("reduced fft data shape: ", fft_data.shape)

            # apply wiener filter
            if self.apply_wiener:
                fft_data = apply_wiener(fft_data)
                print("filtered fft data shape: ", fft_data.shape)
            
            print("fft data shape: ", fft_data.shape)

            return fft_data, mask, freq
        
        return data, mask
        


    




