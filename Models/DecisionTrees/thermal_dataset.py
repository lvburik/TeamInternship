import sys
import os
import re
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from PIL import Image

# add the path to the preprocessing module
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Preprocessing.preprocessing import *

class ThermalDataset(Dataset):
    def __init__(self, file_paths, data_dir, mask_map=None, sim=False, mask_dir=None, center_data=False,
                 add_zero_padding=False, apply_fft=False, cutoff_frequency=1,
                 apply_PCA=False, extract_patches=False, extract_cnn_patches=False):
        
        self.file_paths = file_paths                    # list of file paths within data_dir
        self.data_dir = data_dir                        # path to data directory 
        self.sim = sim                                  # whether the dataset is a simulation dataset 
        self.mask_dir = mask_dir                        # path to mask directory (for simulation dataset)  
        self.center_data = center_data                  # center the data
        self.add_zero_padding = add_zero_padding        # zeropad time series
        self.apply_fft = apply_fft                      # apply fft to data
        self.cutoff_frequency = cutoff_frequency        # cutoff frequency for fft
        self.apply_PCA = apply_PCA                      # apply PCA to data
        self.extract_patches = extract_patches          # extract patches from input image
        self.extract_cnn_patches = extract_cnn_patches  # extract patches for CNN training

        # mapping of masks to corresponding files
        self.mask_mapping = mask_map

    def __len__(self):
        # number of files in dataset
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # load data from file  
        file_path = os.path.join(self.data_dir, self.file_paths[idx])
        data = np.load(file_path, mmap_mode='r').copy()

         # load corresponding mask (labels)
        if self.sim:
            file_name = os.path.basename(file_path)

            match = re.search(r'_(\d+)\.npy$', file_name)
            
            sim_number = int(match.group(1))
            mask_filename = f"mask_{sim_number:02}.png"

            mask_path = os.path.join(self.data_dir, self.mask_dir, mask_filename)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            
        else:
            file_name = os.path.basename(file_path)
            if "Composite" in file_path:
                mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["composite"])
            elif "Resin" in file_path:
                if "circular" in file_name:
                    mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["circular"])
                elif "rec" in file_name:
                    mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["rec"])
                elif "square" in file_name:
                    mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["square"])
                elif "tri" in file_name:
                    mask_path = os.path.join(self.data_dir, "Masks", self.mask_mapping["tri"])
            else:
                raise FileNotFoundError(f"Unknown file path: {file_name}")
        
        # load mask (labels)
        mask = Image.open(mask_path).convert('L')  # convert to grayscale
        mask = np.array(mask).astype(np.float32)
        mask = mask.reshape(480, 640)
        mask = np.where(mask > 0, 1, 0) # binary mask

        # optional preprocessing steps
        if self.center_data:
            data -= np.mean(data, axis=0)

        if self.add_zero_padding:
            data = add_zero_padding(data, 2048 - data.shape[0])
            print("zero padded data shape: ", data.shape)

        if self.apply_fft:
            t = np.linspace(0.5, data.shape[0]*0.5, data.shape[0])
            fft_data, freq = apply_fft(data, t)

            # apply low-pass filter to remove high frequencies
            cutoff_index = np.where(freq > self.cutoff_frequency)[0][0] 
            fft_data = fft_data[:cutoff_index]
            freq = freq[:cutoff_index]
            print("reduced fft data shape: ", fft_data.shape)

            # save filtered fft data and frequencies
            np.save(os.path.splitext(file_name)[0] + "_fft", fft_data)
            np.save(os.path.splitext(file_name)[0] + "_freq", freq)
            
            print("fft data shape: ", fft_data.shape)

            return fft_data, mask, freq
        
        if self.extract_patches:
            data, mask = extract_patches(data, mask, patch_size=4)
            data = data.T
        
        if self.extract_cnn_patches:
            # pass in data.T if extract from fftpca data!
            data, mask = extract_cnn_patches(data, mask, patch_size=64, 
                                             overlap=0.75, neg_patch_prob=1, augment=False)
        
        if self.apply_PCA:
            if self.extract_cnn_patches:
                data = apply_PCA_patches(data, num_components=10)
            else:
                data = apply_PCA_SVD(data, num_components=10)
        
        data = np.abs(data)

        return data, mask