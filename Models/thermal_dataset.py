import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Preprocessing')))
from preprocessing import *

class ThermalDataset(Dataset):
    def __init__(self, file_paths, data_dir, mask_map, center_data=False,
                 add_zero_padding=False, apply_fft=False, cutoff_frequency=1,
                 apply_PCA=False, extract_patches=False, extract_cnn_patches=False):
        
        self.file_paths = file_paths
        self.data_dir = data_dir
        self.center_data = center_data
        self.add_zero_padding = add_zero_padding
        self.apply_fft = apply_fft
        self.cutoff_frequency = cutoff_frequency
        self.apply_PCA = apply_PCA
        self.extract_patches = extract_patches
        self.extract_cnn_patches = extract_cnn_patches

        # mapping of masks to corresponding files
        self.mask_mapping = mask_map

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        
        # load data from file  
        file_path = os.path.join(self.data_dir, self.file_paths[idx])
        data = np.load(file_path, mmap_mode='r').copy()
        #print(f"loaded {self.file_paths[idx]}")
        #print("data shape: ", data.shape)

        # load mask (labels)
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
            raise ValueError(f"Unknown file path: {file_name}")
        
        # load mask (labels)
        mask = Image.open(mask_path).convert('L')  # convert to grayscale
        mask = np.array(mask).astype(np.float32)
        #print(f"loaded mask with shape: {mask.shape}")

        mask = mask.reshape(480, 640)
        mask = np.where(mask > 0, 1, 0)
        
        # plot mask
        """plt.imshow(mask, cmap='gray')
        plt.colorbar()
        plt.title('Defect Mask')
        plt.show()"""

        # center the data
        if self.center_data:
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

            # save filtered fft data and frequencies
            np.save(os.path.splitext(file_name)[0] + "_fft", fft_data)
            np.save(os.path.splitext(file_name)[0] + "_freq", freq)
            
            print("fft data shape: ", fft_data.shape)

            return fft_data, mask, freq
        
        # extract patches from fft data
        if self.extract_patches:
            data, mask = extract_patches(data, mask, patch_size=4)
            data = data.T
            print("extracted patches shape: ", data.shape)
            print("extracted patches mask shape: ", mask.shape)
        
        # extract patches from fft data for CNN 
        if self.extract_cnn_patches:
            data, mask = extract_cnn_patches(data, mask, patch_size=128, overlap=0.5, neg_patch_prob=1)
            #print("extracted cnn patches shape: ", data.shape)
            #print("extracted cnn patches mask shape: ", mask.shape)
        
        # apply PCA to reduce dimensionality
        if self.apply_PCA:
            data = apply_PCA_SVD(data, num_components=10)
            print("PCA data shape: ", data.shape)
        
        data = np.abs(data)
        return data, mask
        


    




