import os
import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt

def apply_fft(data, t):
    # compute fast fourier transform
    fft_result = fft(data, axis=0)

    # compute frequencies
    freq = fftfreq(len(t), d=(t[1]-t[0]))

    return fft_result, freq

def add_zero_padding(data, num_zeroes):
    # add zero padding to time sequence (should be to nearest power of 2)
    # num_zeroes for exp_data should be 2048 - data.shape[0]
    return np.pad(data, ((0, num_zeroes), (0,0)))

def standardize(data):
    # standardize fft data
    fft_magnitudes = np.abs(data)
    mean = np.mean(fft_magnitudes, axis=0)
    std = np.std(fft_magnitudes, axis=0)
    standardized_data = (fft_magnitudes - mean)/std
    return standardized_data

def extract_peak_features(data, freq):
    # extract peak frequency and magnitude
    fft_magnitude = np.abs(data)
    peak_indices = np.argmax(fft_magnitude, axis=0)  # shape: (num_pixels,)
    peak_magnitudes = fft_magnitude[peak_indices, np.arange(fft_magnitude.shape[1])]
    peak_freqs = freq[peak_indices]
    
    features = np.stack((peak_freqs, peak_magnitudes), axis=1)  # shape: (num_pixels, 2)
    return features

def extract_patches(data, mask, patch_size=16):
    # reshape into (480, 640, num_freqs)
    data = data.T.reshape(480, 640, data.shape[0])

    patches = []
    labels = []
    # extract non-overlapping patches and their labels
    for i in range(0, 480, patch_size):
        for j in range(0, 640, patch_size):
            if i + patch_size <= 480 and j + patch_size <= 640:
                patch = data[i:i+patch_size, j:j+patch_size]
                # take mean magnitude for each frequency bin of pixels in patch
                patch = np.mean(patch, axis=(0,1))

                label = mask[i:i+patch_size, j:j+patch_size]
                label = 1 if np.mean(label) > 0.5 else 0

                patches.append(patch)
                labels.append(label)

    return np.array(patches), np.array(labels)

def apply_PCA(X , num_components):
     
    #Step-1
    X_centered = X - np.mean(X , axis = 1, keepdims=True)
     
    #Step-2
    cov_mat = np.cov(X_centered , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(X_centered, eigenvector_subset.T)

     
    return X_reduced

def apply_PCA_SVD(X, num_components):
    # Step-1: Mean center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Step-2: Compute SVD instead of covariance matrix
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Step-3: Select the top 'num_components' principal components
    eigenvector_subset = Vt[:num_components, :]  # Shape: (num_components, 307200)
    
    # Step-4: Project data onto the new subspace
    X_reduced = np.dot(X_centered, eigenvector_subset.T)  # Shape: (301, num_components)
    
    return X_reduced

